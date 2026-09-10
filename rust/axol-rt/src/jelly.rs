//! Realtime wheel controller for the Jelly mobile base.
//!
//! Python owns VR mapping and the optional lift. This service owns the wheel
//! motor lifecycle, command watchdog, axis snap, vector slew, x-drive mix,
//! gyro heading hold, park/unpark state machine, and every wheel CAN frame.
//!
//! # The wheel CAN timeout is the runaway safety layer
//!
//! Every wheel is armed with the Damiao loss-of-comms alarm (`TIMEOUT`
//! register, `Config::can_timeout_ms`, RAM only and readback-verified) at
//! enable; a wheel that goes that long without a command frame faults
//! `LOST_COMM` and torques off by itself, whether or not this process is
//! still running. The loop's job is then to never feed that alarm with
//! anything but a live motion command:
//!
//! - While the Python target is fresh (`Config::timeout`) every tick sends
//!   one frame per wheel — velocity, or the parked MIT hold — which is what
//!   keeps the alarm fed.
//! - When the target goes stale while *driving*, the wheels get one
//!   zero-velocity frame and then **silence**: no keepalive, no re-send of
//!   the stale command. The alarm trips and the wheels torque off. Once that
//!   has certainly happened (`can_timeout_ms` plus two ticks) and the wheels
//!   have stopped rolling they are re-enabled straight into the park hold
//!   (IMPEDANCE mode set *before* enable, so no velocity target can replay).
//! - A stationary hold is not a runaway risk, so a parked Jelly keeps its
//!   hold with or without a source — the hold frames are the one thing
//!   streamed unlinked, and they carry no velocity.
//! - When a fresh target arrives, wheels the alarm tripped (or that were
//!   torqued off any other way) are cleared and re-enabled (IMPEDANCE →
//!   enable → VELOCITY, so nothing stale replays) and the ramp restarts from
//!   rest. Other faults (over-current, thermal, under-voltage) are reported
//!   but never cleared blindly.

use crate::bringup;
use crate::can::CanSock;
use crate::proto::{self, MitRanges};
use crate::safety::{guarded_send, purge_tx_queue, SendOutcome};
use crate::txn;
use std::io::{self, Read, Write};
use std::net::Shutdown;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const IDS: [u8; 4] = [1, 2, 3, 4];
const PMAX: f64 = 400.0;
const PARK_MAX_SPEED: f64 = 0.5;
const DM_REG_WRITE: u8 = 0x55;
/// Damiao loss-of-comms alarm register: uint32, 50 µs ticks, 0 disables.
const DM_REG_TIMEOUT: u8 = 9;
const DM_TIMEOUT_MS_PER_TICK: f64 = 0.05;
/// Control-mode register values.
const MODE_MIT: u32 = 1;
const MODE_VELOCITY: u32 = 3;
/// Feedback status nibbles the loop clears and re-enables on its own: the two
/// the CAN timeout leaves behind. Anything else is a real fault that stays put
/// until an operator looks.
const STATUS_DISABLED: u8 = 0x0;
const STATUS_ENABLED: u8 = 0x1;
const STATUS_LOST_COMM: u8 = 0xD;
/// Minimum spacing between wheel re-enable attempts while the source is live
/// but a wheel keeps reporting a fault (dead bus, motor unpowered). Each
/// attempt is several CAN round trips, so it must not run every tick.
const WHEEL_RECOVER_MIN: Duration = Duration::from_secs(1);
/// Spacing of attempts to anchor tripped wheels into the park hold while no
/// source is attached (a base still rolling on a slope is retried until it
/// settles).
const PARK_RETRY: Duration = Duration::from_millis(200);
/// Per-wheel request/response budget for the explicit feedback polls used
/// while the wheels are not being streamed (tripped or torqued off).
const POLL_TIMEOUT: Duration = Duration::from_millis(20);
/// Settle between the register/enable writes of a re-enable sequence so the
/// motor applies each before the next arrives.
const ENABLE_STEP_SETTLE: Duration = Duration::from_millis(5);
static SIGNAL_STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    SIGNAL_STOP.store(true, Ordering::SeqCst);
}

#[derive(Clone, Copy, Debug)]
struct Config {
    max_speed: f64,
    turn_scale: f64,
    slew: f64,
    axis_snap_deg: f64,
    yaw_gain: f64,
    yaw_max: f64,
    hold_kp: f64,
    hold_kd: f64,
    frequency: f64,
    timeout: f64,
    can_timeout_ms: f64,
}

#[derive(Clone, Copy)]
struct Target {
    vx: f64,
    vy: f64,
    wz: f64,
    at: Instant,
}
#[derive(Clone, Copy)]
struct Yaw {
    rate: f64,
    at: Instant,
}
#[derive(Clone, Copy, Default)]
struct Inputs {
    target: Option<Target>,
    yaw: Option<Yaw>,
}

fn read_message(stream: &mut UnixStream) -> io::Result<Option<Vec<u8>>> {
    let mut header = [0u8; 4];
    match stream.read_exact(&mut header) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let n = u32::from_le_bytes(header) as usize;
    if n == 0 || n > 4096 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "bad Jelly message size",
        ));
    }
    let mut body = vec![0; n];
    stream.read_exact(&mut body)?;
    Ok(Some(body))
}
fn write_message(stream: &Arc<Mutex<UnixStream>>, body: &[u8]) -> io::Result<()> {
    let mut s = stream.lock().unwrap();
    s.write_all(&(body.len() as u32).to_le_bytes())?;
    s.write_all(body)
}
fn write_error(stream: &Arc<Mutex<UnixStream>>, error: &str) {
    let mut p = vec![b'E'];
    p.extend_from_slice(error.as_bytes());
    let _ = write_message(stream, &p);
}
fn f64_at(data: &[u8], at: usize) -> io::Result<f64> {
    data.get(at..at + 8)
        .and_then(|v| v.try_into().ok())
        .map(f64::from_le_bytes)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "truncated Jelly message"))
}
fn parse_config(data: &[u8]) -> io::Result<Config> {
    if data.len() != 88 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Jelly config must contain 11 f64 values",
        ));
    }
    let c = Config {
        max_speed: f64_at(data, 0)?,
        turn_scale: f64_at(data, 8)?,
        slew: f64_at(data, 16)?,
        axis_snap_deg: f64_at(data, 24)?,
        yaw_gain: f64_at(data, 32)?,
        yaw_max: f64_at(data, 40)?,
        hold_kp: f64_at(data, 48)?,
        hold_kd: f64_at(data, 56)?,
        frequency: f64_at(data, 64)?,
        timeout: f64_at(data, 72)?,
        can_timeout_ms: f64_at(data, 80)?,
    };
    if !(1.0..=500.0).contains(&c.frequency)
        || !(c.timeout.is_finite() && c.timeout > 0.0)
        || c.max_speed < 0.0
        || c.slew < 0.0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid Jelly config values",
        ));
    }
    // The motor-side timeout is the safety layer; refuse a config that
    // disables it or that the command stream cannot reliably feed.
    let period_ms = 1000.0 / c.frequency;
    if !(c.can_timeout_ms.is_finite() && c.can_timeout_ms > 0.0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Jelly can_timeout_ms must be a positive number — the wheel \
             loss-of-comms alarm is the runaway safety layer and cannot be disabled",
        ));
    }
    if c.can_timeout_ms < 2.0 * period_ms {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Jelly can_timeout_ms ({} ms) must be at least twice the command \
                 period ({period_ms} ms at {} Hz), or a single late tick trips the wheels",
                c.can_timeout_ms, c.frequency
            ),
        ));
    }
    Ok(c)
}

/// Whether a feedback status nibble is one the loop may clear and re-enable.
fn recoverable(status: u8) -> bool {
    status == STATUS_DISABLED || status == STATUS_LOST_COMM
}

fn status_name(status: u8) -> &'static str {
    match status {
        STATUS_DISABLED => "disabled",
        STATUS_ENABLED => "enabled",
        0x8 => "over-voltage",
        0x9 => "under-voltage",
        0xA => "over-current",
        0xB => "mos-over-temp",
        0xC => "rotor-over-temp",
        STATUS_LOST_COMM => "lost-comm",
        0xE => "overload",
        _ => "unknown",
    }
}

pub fn run(socket_path: &str, iface: &str) -> io::Result<()> {
    SIGNAL_STOP.store(false, Ordering::SeqCst);
    unsafe {
        libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t);
    }
    let _ = std::fs::remove_file(socket_path);
    let listener = UnixListener::bind(socket_path)?;
    let (mut stream, _) = listener.accept()?;
    let output = Arc::new(Mutex::new(stream.try_clone()?));
    let Some(first) = read_message(&mut stream)? else {
        return Ok(());
    };
    if first.first() != Some(&b'C') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Jelly expected config first",
        ));
    }
    let cfg = parse_config(&first[1..])?;
    let inputs = Arc::new(Mutex::new(Inputs::default()));
    let stop = Arc::new(AtomicBool::new(false));
    let controller = std::thread::spawn({
        let inputs = Arc::clone(&inputs);
        let stop = Arc::clone(&stop);
        let output = Arc::clone(&output);
        let iface = iface.to_owned();
        move || {
            if let Err(e) = control_loop(&iface, cfg, &inputs, &stop, &output) {
                write_error(&output, &e.to_string());
                let _ = output.lock().unwrap().shutdown(Shutdown::Both);
            } else if SIGNAL_STOP.load(Ordering::SeqCst) {
                let _ = output.lock().unwrap().shutdown(Shutdown::Both);
            }
        }
    });
    let result = (|| -> io::Result<()> {
        while let Some(p) = read_message(&mut stream)? {
            match p.first().copied() {
                Some(b'T') if p.len() == 33 => {
                    let age = f64_at(&p, 25)?.max(0.0);
                    inputs.lock().unwrap().target = Some(Target {
                        vx: f64_at(&p, 1)?.clamp(-1.0, 1.0),
                        vy: f64_at(&p, 9)?.clamp(-1.0, 1.0),
                        wz: f64_at(&p, 17)?.clamp(-1.0, 1.0),
                        at: Instant::now()
                            .checked_sub(Duration::from_secs_f64(age))
                            .unwrap_or_else(Instant::now),
                    });
                }
                Some(b'Y') if p.len() == 17 => {
                    let age = f64_at(&p, 9)?.max(0.0);
                    inputs.lock().unwrap().yaw = Some(Yaw {
                        rate: f64_at(&p, 1)?,
                        at: Instant::now()
                            .checked_sub(Duration::from_secs_f64(age))
                            .unwrap_or_else(Instant::now),
                    });
                }
                Some(b'Q') if p.len() == 1 => break,
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unknown Jelly message",
                    ))
                }
            }
        }
        Ok(())
    })();
    stop.store(true, Ordering::Release);
    let _ = controller.join();
    let _ = std::fs::remove_file(socket_path);
    result
}

fn write_register(sock: &CanSock, id: u8, rid: u8, value: [u8; 4]) -> io::Result<()> {
    let mut p = [0u8; 8];
    p[0] = id;
    p[2] = DM_REG_WRITE;
    p[3] = rid;
    p[4..].copy_from_slice(&value);
    sock.send(proto::DM_REG_ARB, &p)
}
fn set_mode(sock: &CanSock, mode: u32) -> io::Result<()> {
    set_mode_for(sock, &IDS, mode)
}
fn set_mode_for(sock: &CanSock, ids: &[u8], mode: u32) -> io::Result<()> {
    for &id in ids {
        write_register(sock, id, proto::DM_REG_CTRL_MODE, mode.to_le_bytes())?;
    }
    Ok(())
}

/// Write the loss-of-comms alarm to every wheel and verify the readback.
///
/// RAM-only on purpose: re-asserted on every enable, so the safety layer never
/// depends on what a motor happens to have in flash. A wheel that does not
/// read back the requested value fails the enable — driving with an
/// unverified timeout is exactly the runaway case.
fn arm_can_timeout(sock: &CanSock, can_timeout_ms: f64) -> io::Result<()> {
    let ticks = (can_timeout_ms / DM_TIMEOUT_MS_PER_TICK).round() as u32;
    for id in IDS {
        write_register(sock, id, DM_REG_TIMEOUT, ticks.to_le_bytes())?;
    }
    let mut mismatched = Vec::new();
    for id in IDS {
        let readback = bringup::read_dm_register(sock, id as u16, DM_REG_TIMEOUT)?;
        if readback.round() as u32 != ticks {
            mismatched.push(format!(
                "wheel {id}={}ms",
                readback * DM_TIMEOUT_MS_PER_TICK
            ));
        }
    }
    if !mismatched.is_empty() {
        return Err(io::Error::other(format!(
            "Jelly wheel CAN timeout readback mismatch (wanted {can_timeout_ms} ms): {} — \
             refusing to drive without the loss-of-comms safety layer",
            mismatched.join(", ")
        )));
    }
    Ok(())
}

/// Enable `ids` into VELOCITY mode without replaying a stale target.
///
/// A Damiao keeps its last command target across a fault or a torque-off and
/// the enable alone can act on it; only a control-mode *switch* zeroes the
/// command state. So: switch to MIT first (zero gains, torque-free), clear the
/// fault, enable, then switch to VELOCITY — a second zeroing — so the first
/// thing the wheel acts on is the next frame this loop sends.
fn enable_for_velocity(sock: &CanSock, ids: &[u8]) -> io::Result<()> {
    set_mode_for(sock, ids, MODE_MIT)?;
    std::thread::sleep(ENABLE_STEP_SETTLE);
    for &id in ids {
        sock.send(id as u16, &proto::DM_CLEAR_ERRORS)?;
    }
    std::thread::sleep(ENABLE_STEP_SETTLE);
    for &id in ids {
        sock.send(id as u16, &proto::DM_ENABLE)?;
    }
    std::thread::sleep(ENABLE_STEP_SETTLE);
    set_mode_for(sock, ids, MODE_VELOCITY)?;
    std::thread::sleep(ENABLE_STEP_SETTLE);
    Ok(())
}

/// Ask every wheel for a feedback frame (request/response, no command) and
/// record position, velocity, and status for the ones that answer.
///
/// Used while the wheels are *not* being streamed — tripped or torqued off —
/// where the echo-driven `collect_feedback` has nothing to collect. Returns
/// which wheels answered; a silent wheel keeps its previous values.
fn poll_feedback(
    sock: &CanSock,
    ranges: &[MitRanges; 4],
    positions: &mut [f64; 4],
    velocities: &mut [f64; 4],
    statuses: &mut [u8; 4],
) -> io::Result<[bool; 4]> {
    let mut answered = [false; 4];
    for (i, id) in IDS.into_iter().enumerate() {
        if let Some((data, _)) = txn::dm_request_feedback(sock, id as u16, POLL_TIMEOUT)? {
            let d =
                proto::dm_decode_feedback(&data, ranges[i].p_max, ranges[i].v_max, ranges[i].t_max);
            positions[i] = d.position;
            velocities[i] = d.velocity;
            statuses[i] = d.status;
            answered[i] = true;
        }
    }
    Ok(answered)
}
fn velocity_frame(v: f64) -> [u8; 8] {
    let mut p = [0; 8];
    p[..4].copy_from_slice(&(v as f32).to_le_bytes());
    p
}
fn mix(vx: f64, vy: f64, wz: f64, cfg: Config) -> [f64; 4] {
    let w = wz * cfg.turn_scale;
    let mut raw = [vx - vy - w, -(vx + vy + w), vx + vy - w, -(vx - vy + w)];
    let scale = raw.iter().fold(1.0f64, |a, v| a.max(v.abs()));
    for v in &mut raw {
        *v = *v / scale * cfg.max_speed;
    }
    raw
}
fn snap(mut vx: f64, mut vy: f64, deg: f64) -> (f64, f64) {
    if deg > 0.0 && (vx != 0.0 || vy != 0.0) {
        let h = vy.atan2(vx);
        let q = std::f64::consts::FRAC_PI_2;
        let near = (h / q).round() * q;
        if (h - near).abs() <= deg.to_radians() {
            let mag = vx.hypot(vy);
            vx = mag * near.cos();
            vy = mag * near.sin();
        }
    }
    (vx, vy)
}
fn collect_feedback(
    sock: &CanSock,
    ranges: &[MitRanges; 4],
    positions: &mut [f64; 4],
    velocities: &mut [f64; 4],
    statuses: &mut [u8; 4],
    deadline: Instant,
) -> io::Result<usize> {
    let mut seen = [false; 4];
    while seen.iter().any(|value| !value) {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let Some(f) = sock.recv_timeout(deadline - now)? else {
            break;
        };
        if !(0x11..=0x14).contains(&f.id) {
            continue;
        }
        let i = (f.id - 0x11) as usize;
        if f.data[1] <= 0x0f && matches!(f.data[2], 0x33 | 0x55 | 0xaa | 0xcc) {
            continue;
        }
        let d =
            proto::dm_decode_feedback(&f.data, ranges[i].p_max, ranges[i].v_max, ranges[i].t_max);
        seen[i] = true;
        positions[i] = d.position;
        velocities[i] = d.velocity;
        statuses[i] = d.status;
    }
    Ok(seen.into_iter().filter(|value| *value).count())
}

/// Once the first enable may have reached a wheel, every exit must return all
/// wheels to velocity mode, command zero speed, and disable them.  Arming the
/// guard before the send also covers a write whose result is ambiguous.
struct WheelDisableGuard<'a> {
    sock: &'a CanSock,
    iface: &'a str,
    armed: bool,
    bus_dead: bool,
}

impl Drop for WheelDisableGuard<'_> {
    fn drop(&mut self) {
        if !self.armed || self.bus_dead {
            return;
        }
        let mut cleanup_failed = false;
        // A wheel whose CAN timeout tripped while the source was away sits in
        // LOST_COMM; the torque-off below is only confirmable from a clean
        // DISABLED status, so clear the fault first.
        for id in IDS {
            if let Err(err) = self.sock.send(id as u16, &proto::DM_CLEAR_ERRORS) {
                cleanup_failed = true;
                eprintln!(
                    "{}: wheel 0x{id:02X} rollback clear-errors failed: {err}",
                    self.iface
                );
            }
        }
        if let Err(err) = set_mode(self.sock, MODE_VELOCITY) {
            cleanup_failed = true;
            eprintln!("{}: wheel rollback mode switch failed: {err}", self.iface);
        }
        for id in IDS {
            if let Err(err) = self.sock.send(0x200 + id as u16, &velocity_frame(0.0)) {
                cleanup_failed = true;
                eprintln!(
                    "{}: wheel 0x{id:02X} rollback zero failed: {err}",
                    self.iface
                );
            }
            for _ in 0..3 {
                if let Err(err) = self.sock.send(id as u16, &proto::DM_DISABLE) {
                    cleanup_failed = true;
                    eprintln!(
                        "{}: wheel 0x{id:02X} rollback disable failed: {err}",
                        self.iface
                    );
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        if cleanup_failed {
            let purged = purge_tx_queue(self.iface);
            eprintln!(
                "{}: wheel rollback had failed writes{}",
                self.iface,
                if purged {
                    "; stale TX queue purged"
                } else {
                    "; QUEUE PURGE FAILED, flap the interface before re-powering"
                }
            );
        }
    }
}

/// Everything the loop knows about one wheel set between ticks.
struct Wheels {
    ranges: [MitRanges; 4],
    pos: [f64; 4],
    vel: [f64; 4],
    /// Feedback status nibble per wheel, from the last echo or poll.
    status: [u8; 4],
    /// Which wheels answered the last explicit poll (`poll_feedback`).
    answered: [bool; 4],
}

impl Wheels {
    fn any_tripped(&self) -> bool {
        self.status.iter().any(|s| recoverable(*s))
    }
    fn faults(&self) -> Vec<String> {
        IDS.iter()
            .enumerate()
            .filter(|(i, _)| !self.answered[*i] || self.status[*i] != STATUS_ENABLED)
            .map(|(i, id)| {
                if self.answered[i] {
                    format!("wheel {id}={}", status_name(self.status[i]))
                } else {
                    format!("wheel {id}=silent")
                }
            })
            .collect()
    }
}

/// Switch the wheels to the MIT position hold at their current positions.
///
/// Returns the per-wheel anchors, or `None` when parking is not currently
/// safe: a wheel is still measurably moving (retried), or a position is too
/// close to the widened ±PMAX mapping limit (sets `park_failed`, not retried).
///
/// With `reenable` the wheels are expected to be torqued off (their CAN
/// timeout tripped after the source went away): they are polled for fresh
/// state, the mode is switched to MIT *first* — a mode switch zeroes the
/// motor's command state, and MIT with zero gains is torque-free — and only
/// then cleared and re-enabled, so no velocity target can be replayed by the
/// enable. The hold frames that follow are the only thing they then act on.
fn park(
    sock: &CanSock,
    iface: &str,
    wheels: &mut Wheels,
    park_failed: &mut bool,
    reenable: bool,
) -> io::Result<Option<[f64; 4]>> {
    if reenable {
        wheels.answered = poll_feedback(
            sock,
            &wheels.ranges,
            &mut wheels.pos,
            &mut wheels.vel,
            &mut wheels.status,
        )?;
        if wheels.answered.iter().any(|a| !a) {
            return Ok(None);
        }
    }
    if wheels.vel.iter().any(|v| v.abs() > PARK_MAX_SPEED) {
        return Ok(None);
    }
    if wheels.pos.iter().any(|p| p.abs() > 0.9 * PMAX) {
        if !*park_failed {
            eprintln!(
                "{iface}: Jelly wheel position near the ±PMAX mapping limit — parking \
                 disabled. Power-cycle the base to reset wheel positions."
            );
        }
        *park_failed = true;
        return Ok(None);
    }
    set_mode(sock, MODE_MIT)?;
    if reenable {
        std::thread::sleep(ENABLE_STEP_SETTLE);
        for id in IDS {
            sock.send(id as u16, &proto::DM_CLEAR_ERRORS)?;
        }
        std::thread::sleep(ENABLE_STEP_SETTLE);
        for id in IDS {
            sock.send(id as u16, &proto::DM_ENABLE)?;
        }
        std::thread::sleep(ENABLE_STEP_SETTLE);
    }
    Ok(Some(wheels.pos))
}

/// Clear and re-enable wheels the CAN timeout (or a torque-off) left faulted,
/// so a returning command source can drive again.
///
/// Polls every wheel and re-runs the enable sequence on those reporting
/// `DISABLED` or `LOST_COMM`, then polls again so the cached status reflects
/// the re-enabled state. Other faults are reported and left alone — an
/// over-current or thermal trip is not ours to clear blindly.
///
/// Returns `(healthy, touched)`: whether every wheel is now enabled and
/// fault-free, and whether any wheel was actually re-enabled (the caller only
/// restarts its slew ramp / park state in that case).
fn recover_wheels(
    sock: &CanSock,
    iface: &str,
    wheels: &mut Wheels,
    last_faults: &mut Vec<String>,
) -> io::Result<(bool, bool)> {
    wheels.answered = poll_feedback(
        sock,
        &wheels.ranges,
        &mut wheels.pos,
        &mut wheels.vel,
        &mut wheels.status,
    )?;
    let tripped: Vec<u8> = IDS
        .iter()
        .enumerate()
        .filter(|(i, _)| wheels.answered[*i] && recoverable(wheels.status[*i]))
        .map(|(_, id)| *id)
        .collect();
    let touched = !tripped.is_empty();
    if touched {
        eprintln!(
            "{iface}: Jelly re-enabling wheels after loss-of-comms trip: {}",
            tripped
                .iter()
                .map(|id| format!("wheel {id}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        enable_for_velocity(sock, &tripped)?;
        wheels.answered = poll_feedback(
            sock,
            &wheels.ranges,
            &mut wheels.pos,
            &mut wheels.vel,
            &mut wheels.status,
        )?;
    }
    let faults = wheels.faults();
    if !faults.is_empty() && faults != *last_faults {
        if wheels.answered.iter().all(|a| !a) {
            eprintln!(
                "{iface}: Jelly wheels not answering — the base stays torqued off until they do"
            );
        } else {
            eprintln!(
                "{iface}: Jelly wheel fault(s) the loop will not clear on its own: {}",
                faults.join(", ")
            );
        }
    }
    let healthy = faults.is_empty();
    *last_faults = faults;
    Ok((healthy, touched))
}

fn control_loop(
    iface: &str,
    cfg: Config,
    inputs: &Mutex<Inputs>,
    stop: &AtomicBool,
    output: &Arc<Mutex<UnixStream>>,
) -> io::Result<()> {
    let sock = CanSock::open(iface)?;
    sock.set_send_timeout(Duration::from_millis(20))?;
    let _ = sock.drain();
    for id in IDS {
        sock.send(id as u16, &proto::DM_CLEAR_ERRORS)?;
    }
    // Arm the loss-of-comms alarm first (RAM only, so it is re-applied every
    // session and never depends on flash state), and prove every wheel took
    // it before any torque is applied.
    arm_can_timeout(&sock, cfg.can_timeout_ms)?;
    // Widen the position-mapping range (RAM only) before the ranges are read
    // back, so multi-turn wheel positions stay valid for the MIT park hold.
    for id in IDS {
        write_register(&sock, id, proto::DM_REG_PMAX, (PMAX as f32).to_le_bytes())?;
    }
    let mut wheels = Wheels {
        ranges: [MitRanges {
            p_max: PMAX,
            v_max: 45.0,
            kp_max: 500.0,
            kd_max: 5.0,
            t_max: 18.0,
        }; 4],
        pos: [0.0; 4],
        vel: [0.0; 4],
        status: [STATUS_DISABLED; 4],
        answered: [false; 4],
    };
    // This guard predates the first enable attempt so any later register,
    // mode, command, feedback, or IPC failure rolls the wheels back.
    let mut disable = WheelDisableGuard {
        sock: &sock,
        iface,
        armed: false,
        bus_dead: false,
    };
    for (i, id) in IDS.into_iter().enumerate() {
        wheels.ranges[i].p_max = bringup::read_dm_register(&sock, id as u16, proto::DM_REG_PMAX)?;
        wheels.ranges[i].v_max = bringup::read_dm_register(&sock, id as u16, proto::DM_REG_VMAX)?;
        wheels.ranges[i].t_max = bringup::read_dm_register(&sock, id as u16, proto::DM_REG_TMAX)?;
        if (wheels.ranges[i].p_max - PMAX).abs() > 1.0 {
            eprintln!(
                "{iface}: Jelly wheel {id} PMAX readback {} != {PMAX} — parking may misbehave",
                wheels.ranges[i].p_max
            );
        }
    }
    // Bracketed enable (MIT → clear → enable → VELOCITY): a wheel left in
    // VELOCITY mode by a crashed session never replays its old target.
    disable.armed = true;
    enable_for_velocity(&sock, &IDS)?;
    let period = Duration::from_secs_f64(1.0 / cfg.frequency);
    let mut cmd = [0.0; 3];
    let mut hold: Option<[f64; 4]> = None;
    let mut park_failed = false;
    let mut send_failed = false;
    let mut yaw_err = 0.0;
    let mut yaw_bias = 0.0;
    let mut next = Instant::now() + period;
    let mut next_status = Instant::now();
    let mut enobufs = None;
    for id in IDS {
        sock.send(0x200 + id as u16, &velocity_frame(0.0))?;
    }
    let initial_feedback = collect_feedback(
        &sock,
        &wheels.ranges,
        &mut wheels.pos,
        &mut wheels.vel,
        &mut wheels.status,
        Instant::now() + Duration::from_millis(200),
    )?;
    if initial_feedback != IDS.len() {
        return Err(io::Error::other(format!(
            "Jelly startup received {initial_feedback}/{} wheel replies",
            IDS.len()
        )));
    }
    eprintln!(
        "{iface}: Jelly wheels enabled (loss-of-comms alarm {} ms: the wheels torque off \
         on their own when the command stream stops)",
        cfg.can_timeout_ms
    );
    write_message(output, b"R")?;

    // Source-link state (see the module docs).
    let mut linked = false; // the target was fresh on the previous tick
    let mut wheels_ok = true; // every wheel enabled and healthy
    let mut last_faults: Vec<String> = Vec::new();
    let mut next_recover = Instant::now();
    // The trip is certain this long after the last velocity frame; then the
    // wheels are torque-off and merely need to stop rolling before being
    // anchored into the park hold.
    let trip_settle = Duration::from_secs_f64(cfg.can_timeout_ms / 1e3) + 2 * period;
    // Unlinked and not holding: when the last velocity frame went out, and
    // when to next try anchoring. Freshly enabled wheels have never been given
    // a velocity, so with no source yet they are anchored on the very first
    // tick rather than left to trip first.
    let t0 = Instant::now();
    let mut silent_since = t0.checked_sub(trip_settle).unwrap_or(t0);
    let mut next_park = t0;

    // One guarded frame per wheel; a stall is the e-stop path.
    let send_all = |frames: [(u16, [u8; 8]); 4], enobufs: &mut Option<Instant>| -> io::Result<()> {
        for (arb, frame) in frames {
            if let SendOutcome::Stalled = guarded_send(&sock, arb, &frame, enobufs)? {
                return Err(io::Error::other("Jelly CAN TX stalled"));
            }
        }
        Ok(())
    };
    let hold_frames = |anchor: &[f64; 4], ranges: &[MitRanges; 4]| -> [(u16, [u8; 8]); 4] {
        let mut out = [(0u16, [0u8; 8]); 4];
        for i in 0..4 {
            out[i] = (
                IDS[i] as u16,
                proto::mit_encode(anchor[i], 0.0, cfg.hold_kp, cfg.hold_kd, 0.0, &ranges[i]),
            );
        }
        out
    };
    let velocity_frames = |speeds: &[f64; 4]| -> [(u16, [u8; 8]); 4] {
        let mut out = [(0u16, [0u8; 8]); 4];
        for i in 0..4 {
            out[i] = (0x200 + IDS[i] as u16, velocity_frame(speeds[i]));
        }
        out
    };

    let result = (|| -> io::Result<()> {
        while !stop.load(Ordering::Acquire) && !SIGNAL_STOP.load(Ordering::SeqCst) {
            super_sleep_until(next);
            let now = Instant::now();
            next += period;
            let input = *inputs.lock().unwrap();
            let fresh_target = input
                .target
                .filter(|t| now.duration_since(t.at).as_secs_f64() <= cfg.timeout);

            let mut speeds = [0.0; 4];
            let mut yaw_corr = 0.0;
            let mut sent = false;
            let mut poll_failed = false;
            let send_result: io::Result<()> = (|| {
                if fresh_target.is_none() {
                    // ---- no live source: motion is off; only the hold streams
                    if linked {
                        linked = false;
                        if hold.is_none() {
                            // Driving: one stop, then silence. From here the
                            // CAN timeout is what stops the wheels — never a
                            // re-sent stale command.
                            eprintln!(
                                "{iface}: Jelly command source silent for {:.0} ms while \
                                 driving — stopping and going quiet on CAN (wheels torque \
                                 off after {:.0} ms, then re-anchor)",
                                cfg.timeout * 1e3,
                                cfg.can_timeout_ms
                            );
                            send_all(velocity_frames(&[0.0; 4]), &mut enobufs)?;
                            sent = true;
                            silent_since = now;
                            next_park = now + trip_settle;
                            wheels_ok = false; // the silence is about to trip them
                            next_recover = now;
                        } else {
                            eprintln!(
                                "{iface}: Jelly command source silent for {:.0} ms while \
                                 parked — keeping the park hold",
                                cfg.timeout * 1e3
                            );
                        }
                    }
                    cmd = [0.0; 3];
                    yaw_err = 0.0;
                    if hold.is_some() && wheels.any_tripped() {
                        // The hold lapsed anyway (a stall of this loop past the
                        // CAN timeout): the wheels are torque-off, so treat
                        // them like a tripped drive and re-anchor.
                        eprintln!(
                            "{iface}: Jelly park hold lapsed (wheel loss-of-comms) — re-anchoring"
                        );
                        hold = None;
                        wheels_ok = false;
                        silent_since = now;
                        next_park = now + trip_settle;
                    }
                    if let Some(anchor) = hold {
                        send_all(hold_frames(&anchor, &wheels.ranges), &mut enobufs)?;
                        sent = true;
                    } else if cfg.hold_kp > 0.0
                        && !park_failed
                        && now >= next_park
                        && now.duration_since(silent_since) >= trip_settle
                    {
                        next_park = now + PARK_RETRY;
                        // A wheel that is unpowered or behind a dead bus is
                        // retried next window (the base is torque-off, which is
                        // the safe state); only the streaming path below treats
                        // a TX stall as the e-stop.
                        match park(&sock, iface, &mut wheels, &mut park_failed, true) {
                            Ok(Some(anchor)) => {
                                hold = Some(anchor);
                                wheels_ok = true;
                                last_faults.clear();
                                send_all(hold_frames(&anchor, &wheels.ranges), &mut enobufs)?;
                                sent = true;
                                eprintln!(
                                    "{iface}: Jelly wheels re-enabled into the park hold \
                                     (no command source)"
                                );
                            }
                            Ok(None) => {}
                            Err(err) => {
                                let report = vec![format!("re-anchor failed: {err}")];
                                if report != last_faults {
                                    eprintln!(
                                        "{iface}: Jelly wheels not answering or refusing to \
                                         re-enable ({err}) — the base stays torqued off until \
                                         they do"
                                    );
                                }
                                last_faults = report;
                                poll_failed = true;
                            }
                        }
                    }
                    return Ok(());
                }

                // ---- live source
                let mut target = fresh_target.unwrap();
                if !linked {
                    linked = true;
                    eprintln!("{iface}: Jelly command source live — resuming control");
                }
                // A wheel the CAN timeout tripped (a source outage, or a stall
                // of this very loop) reports LOST_COMM on the feedback it echoes
                // for each command; re-enable it before commanding it, and ramp
                // from rest since it has stopped.
                if wheels_ok && wheels.any_tripped() {
                    wheels_ok = false;
                    next_recover = now;
                    eprintln!(
                        "{iface}: Jelly wheel dropped out of enable mid-stream (loss-of-comms \
                         or torque-off) — re-enabling"
                    );
                }
                if !wheels_ok && now >= next_recover {
                    next_recover = now + WHEEL_RECOVER_MIN;
                    let (healthy, reenabled) =
                        match recover_wheels(&sock, iface, &mut wheels, &mut last_faults) {
                            Ok(outcome) => outcome,
                            Err(err) => {
                                let report = vec![format!("re-enable failed: {err}")];
                                if report != last_faults {
                                    eprintln!(
                                        "{iface}: Jelly wheels not answering or refusing to \
                                         re-enable ({err}) — the base stays torqued off until \
                                         they do"
                                    );
                                }
                                last_faults = report;
                                poll_failed = true;
                                (false, false)
                            }
                        };
                    wheels_ok = healthy;
                    if reenabled {
                        // Those wheels stopped and are back in VELOCITY mode:
                        // ramp from rest and let the park re-anchor them all.
                        cmd = [0.0; 3];
                        hold = None;
                    }
                }

                (target.vx, target.vy) = snap(target.vx, target.vy, cfg.axis_snap_deg);
                let delta = [target.vx - cmd[0], target.vy - cmd[1], target.wz - cmd[2]];
                let norm = delta.iter().map(|v| v * v).sum::<f64>().sqrt();
                let max_delta = cfg.slew * period.as_secs_f64();
                let k = if norm > max_delta && norm > 0.0 {
                    max_delta / norm
                } else {
                    1.0
                };
                for i in 0..3 {
                    cmd[i] += delta[i] * k;
                }
                let moving = cmd.iter().any(|v| v.abs() >= 1e-3);
                let driving = moving
                    || [target.vx, target.vy, target.wz]
                        .iter()
                        .any(|v| v.abs() >= 1e-3);
                let translating = cmd[0].hypot(cmd[1]) > 0.1;
                let turning = cmd[2].abs() > 0.05;
                if cfg.yaw_gain != 0.0 {
                    if let Some(y) = input.yaw {
                        if now.duration_since(y.at) <= Duration::from_millis(300) {
                            if translating && !turning {
                                yaw_err += (y.rate - yaw_bias) * period.as_secs_f64();
                                yaw_corr =
                                    (-cfg.yaw_gain * yaw_err).clamp(-cfg.yaw_max, cfg.yaw_max);
                            } else {
                                yaw_err = 0.0;
                                if !driving {
                                    yaw_bias += 0.02 * (y.rate - yaw_bias);
                                }
                            }
                        } else {
                            yaw_err = 0.0;
                        }
                    }
                }
                speeds = mix(cmd[0], cmd[1], cmd[2] + yaw_corr, cfg);
                if driving && hold.is_some() {
                    set_mode(&sock, MODE_VELOCITY)?;
                    hold = None;
                    park_failed = false;
                }
                if !driving && cfg.hold_kp > 0.0 && !park_failed && hold.is_none() {
                    hold = park(&sock, iface, &mut wheels, &mut park_failed, false)?;
                }
                match hold {
                    Some(anchor) => send_all(hold_frames(&anchor, &wheels.ranges), &mut enobufs)?,
                    None => send_all(velocity_frames(&speeds), &mut enobufs)?,
                }
                sent = true;
                Ok(())
            })();
            send_failed = send_result.is_err() || poll_failed;
            if let Err(e) = send_result {
                disable.bus_dead = true;
                let _ = purge_tx_queue(iface);
                return Err(e);
            }
            if sent {
                // Every command frame is echoed with a feedback frame; that
                // echo is also where a mid-stream LOST_COMM shows up.
                let _ = collect_feedback(
                    &sock,
                    &wheels.ranges,
                    &mut wheels.pos,
                    &mut wheels.vel,
                    &mut wheels.status,
                    now + period.mul_f64(0.8),
                )?;
            }
            if now >= next_status {
                next_status = now + Duration::from_millis(50);
                let mut p = vec![b'U'];
                for v in cmd
                    .into_iter()
                    .chain(speeds)
                    .chain([yaw_corr, yaw_err, yaw_bias])
                {
                    p.extend_from_slice(&v.to_le_bytes());
                }
                p.push(
                    (hold.is_some() as u8)
                        | ((park_failed as u8) << 1)
                        | ((send_failed as u8) << 2)
                        | ((linked as u8) << 3)
                        | ((!wheels_ok as u8) << 4),
                );
                write_message(output, &p)?;
            }
        }
        Ok(())
    })();
    result
}

fn super_sleep_until(deadline: Instant) {
    loop {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let left = deadline - now;
        if left > Duration::from_micros(200) {
            std::thread::sleep(left - Duration::from_micros(100));
        } else {
            std::hint::spin_loop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forward_mix_has_calibrated_signs() {
        let c = Config {
            max_speed: 20.0,
            turn_scale: 1.0,
            slew: 0.5,
            axis_snap_deg: 15.0,
            yaw_gain: 2.0,
            yaw_max: 0.3,
            hold_kp: 60.0,
            hold_kd: 1.5,
            frequency: 50.0,
            timeout: 0.3,
            can_timeout_ms: 200.0,
        };
        assert_eq!(mix(1.0, 0.0, 0.0, c), [20.0, -20.0, 20.0, -20.0]);
    }

    /// Mirrors `struct.pack("<11d", ...)` in `almond_axol/robot/jelly.py`.
    fn wire(values: [f64; 11]) -> Vec<u8> {
        values.into_iter().flat_map(f64::to_le_bytes).collect()
    }

    const PY_CONFIG: [f64; 11] = [20.0, 1.0, 0.5, 15.0, 2.0, 0.3, 60.0, 1.5, 50.0, 0.2, 200.0];

    #[test]
    fn python_config_layout_roundtrips() {
        let cfg = parse_config(&wire(PY_CONFIG)).unwrap();
        assert_eq!(cfg.frequency, 50.0);
        assert_eq!(cfg.hold_kp, 60.0);
        assert_eq!(cfg.timeout, 0.2);
        assert_eq!(cfg.can_timeout_ms, 200.0);
    }

    #[test]
    fn old_ten_value_config_is_rejected() {
        // A Python side that predates the CAN-timeout field must not be able
        // to start the core with an implicit (zero = disabled) alarm.
        let old: Vec<u8> = PY_CONFIG[..10]
            .iter()
            .copied()
            .flat_map(f64::to_le_bytes)
            .collect();
        assert!(parse_config(&old).is_err());
    }

    #[test]
    fn can_timeout_cannot_be_disabled() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut values = PY_CONFIG;
            values[10] = bad;
            let err = parse_config(&wire(values)).unwrap_err();
            assert!(err.to_string().contains("can_timeout_ms"), "{bad}: {err}");
        }
    }

    #[test]
    fn can_timeout_must_be_at_least_two_command_periods() {
        // 50 Hz → 20 ms period; 39 ms would trip on a single late tick.
        let mut values = PY_CONFIG;
        values[10] = 39.0;
        assert!(parse_config(&wire(values)).is_err());
        values[10] = 40.0;
        assert!(parse_config(&wire(values)).is_ok());
    }

    #[test]
    fn timeout_register_ticks_match_python_scale() {
        // 200 ms at 50 µs per tick = 4000, the value the readback must echo.
        let ticks = (200.0 / DM_TIMEOUT_MS_PER_TICK).round() as u32;
        assert_eq!(ticks, 4000);
        assert!(proto::DM_UINT32_REGS.contains(&DM_REG_TIMEOUT));
    }

    #[test]
    fn only_timeout_trips_and_torque_off_are_recoverable() {
        assert!(recoverable(STATUS_DISABLED));
        assert!(recoverable(STATUS_LOST_COMM));
        assert!(!recoverable(STATUS_ENABLED));
        for fault in [0x8u8, 0x9, 0xA, 0xB, 0xC, 0xE] {
            assert!(!recoverable(fault), "{}", status_name(fault));
        }
    }

    #[test]
    fn feedback_status_nibble_is_recorded() {
        // A LOST_COMM echo (status nibble 0xD, motor id 1 in the low nibble).
        let ranges = MitRanges {
            p_max: PMAX,
            v_max: 45.0,
            kp_max: 500.0,
            kd_max: 5.0,
            t_max: 18.0,
        };
        let data = [0xD1u8, 0x80, 0x00, 0x80, 0x08, 0x00, 30, 30];
        let d = proto::dm_decode_feedback(&data, ranges.p_max, ranges.v_max, ranges.t_max);
        assert_eq!(d.status, STATUS_LOST_COMM);
        assert!(recoverable(d.status));
    }

    #[test]
    fn wheel_faults_name_silent_and_faulted_wheels() {
        let mut w = Wheels {
            ranges: [MitRanges {
                p_max: PMAX,
                v_max: 45.0,
                kp_max: 500.0,
                kd_max: 5.0,
                t_max: 18.0,
            }; 4],
            pos: [0.0; 4],
            vel: [0.0; 4],
            status: [STATUS_ENABLED, 0xA, STATUS_ENABLED, STATUS_LOST_COMM],
            answered: [true, true, false, true],
        };
        assert_eq!(
            w.faults(),
            vec![
                "wheel 2=over-current",
                "wheel 3=silent",
                "wheel 4=lost-comm"
            ]
        );
        assert!(w.any_tripped());
        w.status = [STATUS_ENABLED; 4];
        w.answered = [true; 4];
        assert!(w.faults().is_empty());
        assert!(!w.any_tripped());
    }
}
