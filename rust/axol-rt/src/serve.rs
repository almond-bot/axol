//! The realtime core: own the CAN buses and run the control loop, driven by
//! impedance targets streamed from Python over a Unix socket.
//!
//! Python keeps everything smart — VR, IK, MuJoCo gravity/inertia, host
//! damping, friction feedforward (all of `AxolArm.motion_control`) — and
//! ships the *final* per-joint MIT tuples `(p_des, v_des, kp, kd, t_ff)` at
//! its own rate (~120 Hz). This loop owns the wire: it paces a hard
//! `loop_hz` tick, linearly interpolates `p_des`/`t_ff` between successive
//! targets (one sender period of added latency, in exchange for no steps on
//! the bus), and holds the last played position when targets stop arriving.
//!
//! ## Protocol (length-prefixed messages: u32 LE size, then payload)
//!
//! Python -> Rust:
//! - `C` + text        config: `loop_hz`/`watchdog_ms`/`max_step_rad`/
//!                     `abort_deg` keys, one `joint <side> <iface> <name>
//!                     <motor_id> <kp> <kd>` line per arm joint, and an
//!                     optional `gripper <side> <iface> <motor_id>` line
//! - `P`               prep: MyActuator 0x76 reset + settle, Damiao
//!                     clear-errors (torque-neutral; run *before* Python
//!                     resolves joint offsets, so the wrap state it verifies
//!                     is the post-reset one; the gripper is never touched)
//! - `A`               arm: bring-up, enable, hold current pose (the
//!                     gripper must already be enabled + calibrated in
//!                     POSITION_FORCE mode by the Python side)
//! - `T` + binary      target: side u8, seq u32 LE, 8 x 5 f64 LE — slots
//!                     0-6 are arm-joint MIT tuples (p_des, v_des, kp, kd,
//!                     t_ff); slot 7 is the gripper (p_des motor-frame,
//!                     max_speed rad/s, max_torque Nm, 0, 0)
//! - `D`               disarm: disable motors, threads exit
//!
//! Rust -> Python (text): `S` + state/fault message, `L` + log line.
//!
//! ## Safety
//! - Targets stepping more than `max_step_rad` from the currently played
//!   position are rejected (counted, reported) — corruption defense; the
//!   Python side has its own max-step gate. The gripper slot is exempt
//!   (its targets legitimately jump, matching the Python gate).
//! - A joint deviating more than `abort_deg` from its played target disables
//!   both buses (e.g. a collision or a runaway). The gripper is exempt —
//!   stalling against an object is its normal operation.
//! - The gripper is not commanded at all until the first target arrives
//!   (matching classic mode, where it sits idle until motion_control).
//! - Watchdog: no target for `watchdog_ms` freezes the played position
//!   (finishing the in-flight interpolation segment). The arms keep holding
//!   — matching what the firmware itself does if the host dies — until a
//!   disarm or an operator e-stop.
//! - SIGINT/SIGTERM disable everything before exit.

use std::io::{self, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::bringup::{self, MotorSpec, Vendor};
use crate::can::CanSock;
use crate::hold::sleep_until;
use crate::proto;

/// Target-tuple slots per arm: 7 arm joints + the gripper.
const N_SLOTS: usize = 8;
const GRIPPER_SLOT: usize = 7;

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    SHUTDOWN.store(true, Ordering::SeqCst);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct JointCmd {
    pub p_des: f64,
    pub v_des: f64,
    pub kp: f64,
    pub kd: f64,
    pub t_ff: f64,
}

#[derive(Clone, Copy)]
struct Target {
    cmds: [JointCmd; N_SLOTS],
    seq: u32,
    arrival: Instant,
}

/// Latest target per arm plus arrival bookkeeping, written by the socket
/// reader, consumed by the bus thread.
#[derive(Default)]
struct TargetSlot {
    target: Option<Target>,
}

struct Config {
    loop_hz: f64,
    watchdog_ms: f64,
    max_step_rad: f64,
    abort_deg: f64,
    /// (side, iface, specs) — side 0 = left, 1 = right.
    buses: Vec<(u8, String, Vec<MotorSpec>)>,
}

fn parse_config(text: &str) -> io::Result<Config> {
    let mut loop_hz = 240.0;
    let mut watchdog_ms = 150.0;
    let mut max_step_rad = 0.35;
    let mut abort_deg = 25.0;
    let mut buses: Vec<(u8, String, Vec<MotorSpec>)> = Vec::new();

    let bad = |line: &str| {
        io::Error::new(io::ErrorKind::InvalidData, format!("config: bad line: {line}"))
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        match f[0] {
            "loop_hz" => loop_hz = f.get(1).and_then(|v| v.parse().ok()).ok_or_else(|| bad(line))?,
            "watchdog_ms" => {
                watchdog_ms = f.get(1).and_then(|v| v.parse().ok()).ok_or_else(|| bad(line))?
            }
            "max_step_rad" => {
                max_step_rad = f.get(1).and_then(|v| v.parse().ok()).ok_or_else(|| bad(line))?
            }
            "abort_deg" => {
                abort_deg = f.get(1).and_then(|v| v.parse().ok()).ok_or_else(|| bad(line))?
            }
            "joint" | "gripper" => {
                // joint <side 0|1> <iface> <name> <motor_id> <kp> <kd>
                // gripper <side 0|1> <iface> <motor_id>
                let gripper = f[0] == "gripper";
                let side: u8 = f.get(1).and_then(|v| v.parse().ok()).ok_or_else(|| bad(line))?;
                let iface = f.get(2).ok_or_else(|| bad(line))?.to_string();
                let bus = match buses.iter_mut().find(|(s, i, _)| *s == side && *i == iface) {
                    Some(bus) => bus,
                    None => {
                        buses.push((side, iface.clone(), Vec::new()));
                        buses.last_mut().unwrap()
                    }
                };
                let spec = if gripper {
                    MotorSpec {
                        joint: "gripper".to_string(),
                        motor_id: f.get(3).and_then(|v| v.parse().ok()).ok_or_else(|| bad(line))?,
                        kp: 0.0,
                        kd: 0.0,
                        gripper: true,
                        slot: GRIPPER_SLOT,
                    }
                } else {
                    MotorSpec {
                        joint: f.get(3).ok_or_else(|| bad(line))?.to_string(),
                        motor_id: f.get(4).and_then(|v| v.parse().ok()).ok_or_else(|| bad(line))?,
                        kp: f.get(5).and_then(|v| v.parse().ok()).ok_or_else(|| bad(line))?,
                        kd: f.get(6).and_then(|v| v.parse().ok()).ok_or_else(|| bad(line))?,
                        gripper: false,
                        // Arm joints arrive in Joint enum order per bus.
                        slot: bus.2.iter().filter(|s| !s.gripper).count(),
                    }
                };
                if spec.slot >= N_SLOTS {
                    return Err(bad(line));
                }
                bus.2.push(spec);
            }
            _ => return Err(bad(line)),
        }
    }
    if buses.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "config: no joints"));
    }
    Ok(Config {
        loop_hz,
        watchdog_ms,
        max_step_rad,
        abort_deg,
        buses,
    })
}

fn parse_target(payload: &[u8]) -> io::Result<(u8, Target)> {
    // side u8, seq u32, 7 x 5 f64
    let expected = 1 + 4 + N_SLOTS * 5 * 8;
    if payload.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("target: {} bytes, expected {expected}", payload.len()),
        ));
    }
    let side = payload[0];
    let seq = u32::from_le_bytes(payload[1..5].try_into().unwrap());
    let mut cmds = [JointCmd::default(); N_SLOTS];
    let mut off = 5;
    for cmd in &mut cmds {
        let mut vals = [0.0f64; 5];
        for v in &mut vals {
            *v = f64::from_le_bytes(payload[off..off + 8].try_into().unwrap());
            off += 8;
        }
        *cmd = JointCmd {
            p_des: vals[0],
            v_des: vals[1],
            kp: vals[2],
            kd: vals[3],
            t_ff: vals[4],
        };
    }
    Ok((
        side,
        Target {
            cmds,
            seq,
            arrival: Instant::now(),
        },
    ))
}

fn read_msg(stream: &mut UnixStream) -> io::Result<Option<Vec<u8>>> {
    let mut len_buf = [0u8; 4];
    match stream.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    if len == 0 || len > 1 << 20 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("message size {len}"),
        ));
    }
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload)?;
    Ok(Some(payload))
}

fn writer_thread(mut stream: UnixStream, rx: mpsc::Receiver<Vec<u8>>) {
    for msg in rx {
        let len = (msg.len() as u32).to_le_bytes();
        if stream.write_all(&len).and_then(|_| stream.write_all(&msg)).is_err() {
            return; // peer gone; reader side handles shutdown semantics
        }
    }
}

fn send_text(tx: &mpsc::Sender<Vec<u8>>, tag: u8, text: &str) {
    let mut msg = Vec::with_capacity(1 + text.len());
    msg.push(tag);
    msg.extend_from_slice(text.as_bytes());
    let _ = tx.send(msg);
}

pub fn run(socket_path: &str) -> io::Result<()> {
    unsafe {
        libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_signal as *const () as libc::sighandler_t);
    }
    let _ = std::fs::remove_file(socket_path);
    let listener = UnixListener::bind(socket_path)?;
    println!("axol-rt serve: listening on {socket_path}");
    let (mut stream, _) = listener.accept()?;
    println!("axol-rt serve: client connected");

    let (out_tx, out_rx) = mpsc::channel::<Vec<u8>>();
    let writer = std::thread::spawn({
        let stream = stream.try_clone()?;
        move || writer_thread(stream, out_rx)
    });

    let mut config: Option<Arc<Config>> = None;
    // Index 0 = left, 1 = right.
    let targets: Arc<[Mutex<TargetSlot>; 2]> =
        Arc::new([Mutex::new(TargetSlot::default()), Mutex::new(TargetSlot::default())]);
    let stop = Arc::new(AtomicBool::new(false));
    // 0 = running, 1 = fault (set by a bus thread on abort).
    let fault = Arc::new(AtomicU8::new(0));
    let mut bus_threads: Vec<std::thread::JoinHandle<io::Result<()>>> = Vec::new();

    loop {
        if SHUTDOWN.load(Ordering::SeqCst) {
            break;
        }
        let Some(payload) = read_msg(&mut stream)? else {
            if bus_threads.is_empty() {
                // Clean exit: disarmed (or never armed) before disconnecting.
                println!("axol-rt serve: client disconnected");
                break;
            }
            // Peer died while armed. Hold for a grace period (a crashed
            // Python side can't reconnect, but a signal may still arrive
            // first), then disable and exit so an orphaned core never keeps
            // the arms energized indefinitely.
            println!(
                "axol-rt serve: client disconnected while armed — holding 10 s, then disabling"
            );
            let grace_end = Instant::now() + Duration::from_secs(10);
            while !SHUTDOWN.load(Ordering::SeqCst)
                && fault.load(Ordering::SeqCst) == 0
                && Instant::now() < grace_end
            {
                std::thread::sleep(Duration::from_millis(100));
            }
            break;
        };
        let (tag, body) = (payload[0], &payload[1..]);
        match tag {
            b'C' => {
                let text = std::str::from_utf8(body)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                config = Some(Arc::new(parse_config(text)?));
                send_text(&out_tx, b'S', "config-ok");
            }
            b'P' => {
                let Some(cfg) = &config else {
                    send_text(&out_tx, b'S', "fault: prep before config");
                    continue;
                };
                let mut ok = true;
                for (_, iface, specs) in &cfg.buses {
                    let sock = CanSock::open(iface)?;
                    if let Err(err) = bringup::prep(&sock, specs) {
                        send_text(&out_tx, b'S', &format!("fault: prep {iface}: {err}"));
                        ok = false;
                        break;
                    }
                }
                if ok {
                    send_text(&out_tx, b'S', "prepped");
                }
            }
            b'A' => {
                let Some(cfg) = &config else {
                    send_text(&out_tx, b'S', "fault: arm before config");
                    continue;
                };
                stop.store(false, Ordering::SeqCst);
                fault.store(0, Ordering::SeqCst);
                let (ready_tx, ready_rx) = mpsc::channel::<io::Result<()>>();
                for (side, iface, specs) in cfg.buses.clone() {
                    let cfg = Arc::clone(cfg);
                    let targets = Arc::clone(&targets);
                    let stop = Arc::clone(&stop);
                    let fault = Arc::clone(&fault);
                    let out_tx = out_tx.clone();
                    let ready_tx = ready_tx.clone();
                    bus_threads.push(std::thread::spawn(move || {
                        bus_loop(
                            &iface, side, &specs, &cfg, &targets, &stop, &fault, &out_tx,
                            &ready_tx,
                        )
                    }));
                }
                drop(ready_tx);
                let mut ok = true;
                for _ in 0..cfg.buses.len() {
                    match ready_rx.recv() {
                        Ok(Ok(())) => {}
                        Ok(Err(err)) => {
                            send_text(&out_tx, b'S', &format!("fault: arm: {err}"));
                            ok = false;
                            stop.store(true, Ordering::SeqCst);
                            break;
                        }
                        Err(_) => {
                            send_text(&out_tx, b'S', "fault: arm: bus thread died");
                            ok = false;
                            stop.store(true, Ordering::SeqCst);
                            break;
                        }
                    }
                }
                if ok {
                    send_text(&out_tx, b'S', "armed");
                }
            }
            b'T' => {
                let (side, target) = parse_target(body)?;
                if side <= 1 {
                    targets[side as usize].lock().unwrap().target = Some(target);
                }
            }
            b'D' => {
                stop.store(true, Ordering::SeqCst);
                for handle in bus_threads.drain(..) {
                    if let Err(err) = handle.join().expect("bus thread panicked") {
                        send_text(&out_tx, b'L', &format!("bus loop: {err}"));
                    }
                }
                send_text(&out_tx, b'S', "disarmed");
            }
            other => {
                send_text(&out_tx, b'S', &format!("fault: unknown tag {other}"));
            }
        }
    }

    // Signal or peer loss: stop and disable everything before exit.
    stop.store(true, Ordering::SeqCst);
    for handle in bus_threads {
        let _ = handle.join();
    }
    drop(out_tx);
    let _ = writer.join();
    let _ = std::fs::remove_file(socket_path);
    Ok(())
}

/// One bus: bring-up, hold, then play streamed targets at `loop_hz`.
#[allow(clippy::too_many_arguments)]
fn bus_loop(
    iface: &str,
    side: u8,
    specs: &[MotorSpec],
    cfg: &Config,
    targets: &[Mutex<TargetSlot>; 2],
    stop: &AtomicBool,
    fault: &AtomicU8,
    out_tx: &mpsc::Sender<Vec<u8>>,
    ready_tx: &mpsc::Sender<io::Result<()>>,
) -> io::Result<()> {
    let sock = match CanSock::open(iface) {
        Ok(s) => s,
        Err(err) => {
            let _ = ready_tx.send(Err(io::Error::other(format!("{iface}: {err}"))));
            return Ok(());
        }
    };
    let _ = sock.drain();
    let motors = match bringup::prepare(&sock, iface, specs) {
        Ok(m) => m,
        Err(err) => {
            let _ = ready_tx.send(Err(err));
            return Ok(());
        }
    };
    if let Err(err) = bringup::enable(&sock, &motors) {
        let _ = ready_tx.send(Err(err));
        return Ok(());
    }
    let _ = ready_tx.send(Ok(()));

    // Play state, indexed by target slot: start by holding the measured
    // pose with config gains. The gripper slot's hold values are never sent
    // (it isn't commanded until the first target arrives).
    let mut play: [JointCmd; N_SLOTS] = [JointCmd::default(); N_SLOTS];
    for m in &motors {
        play[m.slot] = JointCmd {
            p_des: m.hold_pos,
            v_des: 0.0,
            kp: m.kp,
            kd: m.kd,
            t_ff: 0.0,
        };
    }
    // Interpolation segment: from -> the pending target over `dur`.
    let mut seg_from: [JointCmd; N_SLOTS] = play;
    let mut seg_target: Option<Target> = None;
    let mut seg_started = Instant::now();
    let mut last_seq: Option<u32> = None;
    let mut last_arrival: Option<Instant> = None;
    let mut period_est = Duration::from_secs_f64(1.0 / 120.0);

    let period = Duration::from_secs_f64(1.0 / cfg.loop_hz);
    let abort_rad = cfg.abort_deg.to_radians();
    let watchdog = Duration::from_secs_f64(cfg.watchdog_ms / 1e3);
    let mut rejected: u64 = 0;
    let mut late: u64 = 0;
    let mut ticks: u64 = 0;
    let mut watchdog_frozen = false;
    let mut next_stats = Instant::now() + Duration::from_secs(5);

    let start = Instant::now() + period;
    let result = (|| -> io::Result<()> {
        loop {
            if stop.load(Ordering::SeqCst) || SHUTDOWN.load(Ordering::SeqCst) {
                return Ok(());
            }
            let deadline = start + period * ticks as u32;
            sleep_until(deadline);
            let began = Instant::now();
            if began - deadline > Duration::from_micros(500) {
                late += 1;
            }
            ticks += 1;

            // Adopt a newly arrived target.
            {
                let slot = targets[side as usize].lock().unwrap();
                if let Some(t) = slot.target {
                    if last_seq != Some(t.seq) {
                        if let Some(prev) = last_arrival {
                            let dt = t.arrival.duration_since(prev);
                            if dt > Duration::from_millis(1) && dt < Duration::from_millis(100) {
                                period_est = Duration::from_secs_f64(
                                    0.9 * period_est.as_secs_f64() + 0.1 * dt.as_secs_f64(),
                                );
                            }
                        }
                        // Gripper slot exempt: its targets legitimately jump
                        // (the Python max-step gate excludes it too).
                        let step_ok = t.cmds[..GRIPPER_SLOT]
                            .iter()
                            .zip(play.iter())
                            .all(|(c, p)| (c.p_des - p.p_des).abs() <= cfg.max_step_rad);
                        if step_ok {
                            seg_from = play;
                            seg_target = Some(t);
                            seg_started = t.arrival;
                        } else {
                            rejected += 1;
                        }
                        last_seq = Some(t.seq);
                        last_arrival = Some(t.arrival);
                    }
                }
            }

            // Watchdog: no fresh target — the segment finishes on its own
            // (alpha reaches 1) and the play state freezes there.
            let starved =
                last_arrival.is_some_and(|a| began.duration_since(a) > watchdog);
            if starved && !watchdog_frozen {
                watchdog_frozen = true;
                send_text(
                    out_tx,
                    b'L',
                    &format!("{iface}: target stream stalled — holding position"),
                );
            } else if !starved && watchdog_frozen {
                watchdog_frozen = false;
                send_text(out_tx, b'L', &format!("{iface}: target stream resumed"));
            }

            // Interpolate toward the current target over one sender period.
            if let Some(t) = &seg_target {
                let dur = period_est.clamp(Duration::from_millis(2), Duration::from_millis(50));
                let alpha =
                    (began.duration_since(seg_started).as_secs_f64() / dur.as_secs_f64()).min(1.0);
                for i in 0..N_SLOTS {
                    let (from, to) = (&seg_from[i], &t.cmds[i]);
                    play[i] = JointCmd {
                        p_des: from.p_des + (to.p_des - from.p_des) * alpha,
                        t_ff: from.t_ff + (to.t_ff - from.t_ff) * alpha,
                        // Already-smooth / slow-varying: step to the new value.
                        v_des: to.v_des,
                        kp: to.kp,
                        kd: to.kd,
                    };
                }
            }

            // Send all commands back-to-back.
            let mut sent = 0usize;
            for m in motors.iter() {
                let c = &play[m.slot];
                if m.gripper {
                    // Idle until the first target (classic mode leaves the
                    // gripper uncommanded until motion_control too). Slot
                    // layout: v_des carries max_speed, kp carries max_torque;
                    // the wire wants current as a fraction of rated (t_max).
                    if seg_target.is_none() {
                        continue;
                    }
                    let frame =
                        proto::dm_pos_force_encode(c.p_des, c.v_des, c.kp / m.ranges.t_max);
                    sock.send(proto::DM_POS_FORCE_ARB_BASE + m.id as u16, &frame)?;
                } else {
                    let frame =
                        proto::mit_encode(c.p_des, c.v_des, c.kp, c.kd, c.t_ff, &m.ranges);
                    match m.vendor {
                        Vendor::MyActuator => sock.send(proto::MA_MC_REQ + m.id as u16, &frame)?,
                        Vendor::Damiao => sock.send(m.id as u16, &frame)?,
                    }
                }
                sent += 1;
            }

            // Collect replies; deviation abort against the played target.
            let reply_deadline = deadline + Duration::from_secs_f64(period.as_secs_f64() * 0.8);
            let mut pending = sent;
            while pending > 0 {
                let now = Instant::now();
                if now >= reply_deadline {
                    break;
                }
                sock.set_recv_timeout(reply_deadline - now)?;
                let Some(frame) = sock.recv()? else { break };
                let (idx, pos) = match frame.id {
                    id if (0x501..=0x505).contains(&id) => {
                        let motor_id = (id - 0x500) as u8;
                        let Some(idx) = motors.iter().position(|m| m.id == motor_id) else {
                            continue;
                        };
                        let (pos, _, _) = proto::ma_decode_mit_feedback(
                            &frame.data,
                            motors[idx].ranges.p_max,
                            motors[idx].ranges.t_max,
                        );
                        (idx, pos)
                    }
                    id if (0x16..=0x18).contains(&id) => {
                        let motor_id = (id - 0x10) as u8;
                        let Some(idx) = motors.iter().position(|m| m.id == motor_id) else {
                            continue;
                        };
                        let m = &motors[idx];
                        let fb = proto::dm_decode_feedback(
                            &frame.data,
                            m.ranges.p_max,
                            m.ranges.v_max,
                            m.ranges.t_max,
                        );
                        (idx, fb.position)
                    }
                    _ => continue,
                };
                pending = pending.saturating_sub(1);
                // No deviation abort for the gripper: stalling against an
                // object (or a jaw span the target overshoots) is normal.
                if motors[idx].gripper {
                    continue;
                }
                let e = (pos - play[motors[idx].slot].p_des).abs();
                if e > abort_rad {
                    fault.store(1, Ordering::SeqCst);
                    stop.store(true, Ordering::SeqCst);
                    return Err(io::Error::other(format!(
                        "{iface}: {} deviated {:.2}° from target (abort at {:.0}°)",
                        motors[idx].joint,
                        e.to_degrees(),
                        cfg.abort_deg,
                    )));
                }
            }

            if began >= next_stats {
                next_stats = began + Duration::from_secs(5);
                send_text(
                    out_tx,
                    b'L',
                    &format!(
                        "{iface}: {ticks} ticks, {late} late ({:.2}%), {rejected} rejected targets, seq {:?}",
                        late as f64 / ticks as f64 * 100.0,
                        last_seq,
                    ),
                );
            }
        }
    })();

    bringup::disable(&sock, &motors);
    if let Err(err) = &result {
        send_text(out_tx, b'S', &format!("fault: {err}"));
    }
    result
}
