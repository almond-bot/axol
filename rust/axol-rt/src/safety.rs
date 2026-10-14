//! Shared SocketCAN transmit-stall protection.
//!
//! A powered-off Axol bus cannot ACK frames. Linux then fills the interface
//! TX queue and returns `ENOBUFS`; frames already accepted by that queue can
//! replay when power returns. Every Rust CAN owner uses this module so both
//! realtime control and maintenance traffic stop and purge identically.

use crate::can::CanSock;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const STALL_DETECT: Duration = Duration::from_secs(1);
const PURGE_DEDUPE: Duration = Duration::from_secs(3);
static LAST_PURGE: Mutex<Option<Instant>> = Mutex::new(None);

/// The provisioned bring-up script — `almond_axol.constants.CAN_BRINGUP_SCRIPT`.
/// Root-owned and outside the operator-writable state tree, which is why
/// `axol provision` grants the purge a NOPASSWD sudo rule for exactly this
/// path (`almond_axol.utils.can_purge`).
const BRINGUP_SCRIPT: &str = "/etc/almond-axol/can/startup.sh";
/// Where `axol can.setup` wrote the same script before the move to
/// `/etc`. Still honoured so a purge works on a robot that has not been
/// re-provisioned yet; `axol provision` deletes the root references to it.
const LEGACY_BRINGUP_SCRIPT: &str = ".almond/can/startup.sh";
/// The arm hub's USB reset — `almond_axol.constants.CAN_RESET_SCRIPT`. The
/// hub firmware keeps the frames it already accepted (up to the driver's 10
/// in flight per channel) through a link down/up and transmits them on the
/// next open, so on the arm buses a flap only defers the replay; this script
/// resets the device, then runs the bring-up script. Same grant as above.
const RESET_SCRIPT: &str = "/etc/almond-axol/can/reset_adapter.sh";
/// The two channels of the arm hub (`CAN_LEFT` / `CAN_RIGHT` in
/// `almond_axol.constants`), the only interfaces the reset script covers.
const ARM_HUB_IFACES: [&str; 2] = ["can_alm_axol_l", "can_alm_axol_r"];

fn is_tx_full(err: &io::Error) -> bool {
    matches!(err.raw_os_error(), Some(libc::ENOBUFS) | Some(libc::EAGAIN))
        || err.kind() == io::ErrorKind::WouldBlock
}

pub enum SendOutcome {
    Sent,
    /// Transient congestion: this frame was dropped.
    Dropped,
    /// Nothing has ACKed frames for `STALL_DETECT`.
    Stalled,
}

pub fn guarded_send(
    sock: &CanSock,
    id: u16,
    data: &[u8],
    enobufs_since: &mut Option<Instant>,
) -> io::Result<SendOutcome> {
    match sock.send(id, data) {
        Ok(()) => {
            *enobufs_since = None;
            Ok(SendOutcome::Sent)
        }
        Err(err) if is_tx_full(&err) => {
            let now = Instant::now();
            match *enobufs_since {
                None => {
                    *enobufs_since = Some(now);
                    Ok(SendOutcome::Dropped)
                }
                Some(t) if now.duration_since(t) >= STALL_DETECT => Ok(SendOutcome::Stalled),
                Some(_) => Ok(SendOutcome::Dropped),
            }
        }
        Err(err) => Err(err),
    }
}

fn run_root(args: &[&str]) -> io::Result<std::process::ExitStatus> {
    let mut cmd = if unsafe { libc::geteuid() } == 0 {
        let mut c = std::process::Command::new(args[0]);
        c.args(&args[1..]);
        c
    } else {
        let mut c = std::process::Command::new("sudo");
        c.arg("-n").args(args);
        c
    };
    cmd.stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
}

/// The bring-up script to flap with, provisioned location first.
///
/// Split out from [`purge_tx_queue`] so the preference order is testable: a
/// core running as root under systemd has `HOME=/root` and a manual run has
/// the operator's, so keying only off `$HOME` (as this did) missed the
/// script on every provisioned robot and silently fell back to the
/// single-interface flap below.
fn bringup_script_in(provisioned: &Path, home: Option<&Path>) -> Option<PathBuf> {
    if provisioned.is_file() {
        return Some(provisioned.to_path_buf());
    }
    home.map(|h| h.join(LEGACY_BRINGUP_SCRIPT))
        .filter(|p| p.is_file())
}

fn bringup_script() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    bringup_script_in(Path::new(BRINGUP_SCRIPT), home.as_deref())
}

/// Drop frames queued behind a dead bus.
///
/// On an arm-hub channel, USB-reset the hub first (`RESET_SCRIPT`): a flap
/// clears the kernel's queue but not the frames the hub firmware already
/// holds, which it transmits on the next open. Otherwise, or when the reset
/// cannot run, flap the interface — preferring the installed bring-up script
/// because the dual-channel adapter is most reliable when both channels are
/// flapped together. A purge performed
/// for the other arm within the last three seconds counts for this arm too.
///
/// Returns false when the flap could not be run at all — most often because
/// escalation failed: `run_root` uses `sudo -n`, so an operator account
/// without the provisioned NOPASSWD rule (and without a cached credential)
/// cannot purge. The callers say so in their stall report, and the next
/// session's bring-up refuses to enable into a queue that is still poisoned
/// (`almond_axol.cli.can.setup.purge_stale_tx`).
pub fn purge_tx_queue(iface: &str) -> bool {
    let mut last = LAST_PURGE.lock().unwrap();
    let reset = reset_script_for(iface, Path::new(RESET_SCRIPT));
    let script = bringup_script();
    if (reset.is_some() || script.is_some()) && last.is_some_and(|t| t.elapsed() < PURGE_DEDUPE) {
        return true;
    }
    if let Some(path) = &reset {
        // Falls through to the flap when the reset cannot run — most often a
        // robot whose sudo grant predates the script (`axol provision`).
        if run_root(&["bash", &path.to_string_lossy()]).is_ok_and(|st| st.success()) {
            *last = Some(Instant::now());
            return true;
        }
    }
    let result = match &script {
        Some(path) => run_root(&["bash", &path.to_string_lossy()]),
        None => run_root(&["ip", "link", "set", iface, "down"]).and_then(|st| {
            if st.success() {
                run_root(&["ip", "link", "set", iface, "up"])
            } else {
                Ok(st)
            }
        }),
    };
    match result {
        Ok(st) if st.success() => {
            *last = Some(Instant::now());
            true
        }
        _ => false,
    }
}

/// The reset script to purge `iface` with: only for an arm-hub channel, and
/// only once `axol can.setup` has installed it. Split out for the tests.
fn reset_script_for(iface: &str, provisioned: &Path) -> Option<PathBuf> {
    (ARM_HUB_IFACES.contains(&iface) && provisioned.is_file()).then(|| provisioned.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique scratch directory; the crate has no dev-dependencies.
    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "axol-rt-safety-{}-{tag}-{:?}",
            std::process::id(),
            std::thread::current().id(),
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join(".almond/can")).unwrap();
        dir
    }

    #[test]
    fn prefers_the_provisioned_script_over_the_operator_copy() {
        // The bug this guards: keying only off $HOME found the legacy copy on
        // an un-provisioned robot and *nothing* on a provisioned one, so every
        // e-stop purge fell back to the single-interface flap that wedges the
        // dual-channel adapter's RX path.
        let dir = scratch("both");
        let provisioned = dir.join("etc-startup.sh");
        std::fs::write(&provisioned, "#!/bin/bash\n").unwrap();
        std::fs::write(dir.join(LEGACY_BRINGUP_SCRIPT), "#!/bin/bash\n").unwrap();
        assert_eq!(
            bringup_script_in(&provisioned, Some(&dir)),
            Some(provisioned)
        );
    }

    #[test]
    fn falls_back_to_the_operator_copy_before_provisioning() {
        let dir = scratch("legacy");
        let legacy = dir.join(LEGACY_BRINGUP_SCRIPT);
        std::fs::write(&legacy, "#!/bin/bash\n").unwrap();
        assert_eq!(
            bringup_script_in(&dir.join("absent.sh"), Some(&dir)),
            Some(legacy)
        );
    }

    #[test]
    fn no_script_anywhere_leaves_the_ip_link_fallback() {
        let dir = scratch("none");
        assert_eq!(bringup_script_in(&dir.join("absent.sh"), Some(&dir)), None);
        assert_eq!(bringup_script_in(&dir.join("absent.sh"), None), None);
    }

    #[test]
    fn arm_hub_channels_purge_with_the_usb_reset() {
        let dir = scratch("reset");
        let reset = dir.join("reset_adapter.sh");
        std::fs::write(&reset, "#!/bin/bash\n").unwrap();
        for iface in ARM_HUB_IFACES {
            assert_eq!(reset_script_for(iface, &reset), Some(reset.clone()));
        }
    }

    #[test]
    fn other_buses_and_unset_up_hosts_keep_the_flap() {
        // The wheel/chest adapters are single-channel and not the hub; a
        // reset of the hub for their stall would drop a healthy arm session.
        let dir = scratch("noreset");
        let reset = dir.join("reset_adapter.sh");
        std::fs::write(&reset, "#!/bin/bash\n").unwrap();
        assert_eq!(reset_script_for("can_alm_axol_b", &reset), None);
        assert_eq!(reset_script_for("can0", &reset), None);
        assert_eq!(
            reset_script_for("can_alm_axol_l", &dir.join("absent.sh")),
            None
        );
    }
}
