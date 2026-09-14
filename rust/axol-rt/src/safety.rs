//! Shared SocketCAN transmit-stall protection.
//!
//! A powered-off Axol bus cannot ACK frames. Linux then fills the interface
//! TX queue and returns `ENOBUFS`; frames already accepted by that queue can
//! replay when power returns. Every Rust CAN owner uses this module so both
//! realtime control and maintenance traffic stop and purge identically.

use crate::can::CanSock;
use std::io;
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const STALL_DETECT: Duration = Duration::from_secs(1);
const PURGE_DEDUPE: Duration = Duration::from_secs(3);
static LAST_PURGE: Mutex<Option<Instant>> = Mutex::new(None);

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

/// Drop frames queued behind a dead bus by flapping the CAN interface.
///
/// Prefer the installed bring-up script because the dual-channel adapter is
/// most reliable when both channels are flapped together. A purge performed
/// for the other arm within the last three seconds counts for this arm too.
pub fn purge_tx_queue(iface: &str) -> bool {
    let mut last = LAST_PURGE.lock().unwrap();
    let script = std::env::var("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".almond/can/startup.sh"))
        .ok()
        .filter(|p| p.exists());
    if script.is_some() && last.is_some_and(|t| t.elapsed() < PURGE_DEDUPE) {
        return true;
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
