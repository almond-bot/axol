//! First motion command path: enable the seven arm joints and actively hold
//! the current pose with an MIT impedance stream, then disable.
//!
//! The bring-up mirrors the Python driver's cold-enable sequence exactly
//! (MyActuator: 0x76 reset + settle, capability detection, 0x77 brake
//! release; Damiao: clear errors, range-register reads, 0xFC enable). Gains
//! and gravity feedforward come from a params file written by
//! `tools/gen_hold_params.py` — Python owns calibration offsets and the
//! MuJoCo gravity model, this loop consumes plain motor-space numbers.
//!
//! Safety:
//! - Without `--yes` this is a dry run: full bring-up prep and the plan are
//!   printed, but nothing is enabled and no motion command is sent.
//! - Any joint deviating more than `--abort-deg` from its hold target stops
//!   both buses and disables everything.
//! - Every exit path (completion, abort, Ctrl-C, errors) runs the disable
//!   sequence; the gripper (ID 0x08) is never touched.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::can::CanSock;
use crate::proto;
use crate::txn;

const TIMEOUT: Duration = Duration::from_millis(100);
/// Post-0x76 reboot settle; the Python driver measures ~1.12 s and waits 2.
const RESET_SETTLE: Duration = Duration::from_millis(2200);

/// Parameter sanity limits — a params file outside these is rejected whole.
const KP_LIMIT: f64 = 250.0;
const KD_LIMIT: f64 = 5.0;
const T_FF_LIMIT: f64 = 30.0;

static SIGINT: AtomicBool = AtomicBool::new(false);

extern "C" fn on_sigint(_: libc::c_int) {
    SIGINT.store(true, Ordering::SeqCst);
}

#[derive(Clone, Debug)]
pub struct JointParams {
    pub iface: String,
    pub joint: String,
    pub motor_id: u8,
    pub kp: f64,
    pub kd: f64,
    pub t_ff: f64,
}

enum Vendor {
    MyActuator,
    Damiao,
}

struct MotorCtx {
    id: u8,
    joint: String,
    vendor: Vendor,
    ranges: proto::MitRanges,
    hold_pos: f64, // motor frame, rad
    kp: f64,
    kd: f64,
    t_ff: f64,
    max_err: f64, // rad
    replies: u64,
}

pub fn parse_params(path: &str) -> io::Result<Vec<JointParams>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        let parsed = (|| -> Option<JointParams> {
            Some(JointParams {
                iface: fields.first()?.to_string(),
                joint: fields.get(1)?.to_string(),
                motor_id: fields.get(2)?.parse().ok()?,
                kp: fields.get(3)?.parse().ok()?,
                kd: fields.get(4)?.parse().ok()?,
                t_ff: fields.get(5)?.parse().ok()?,
            })
        })();
        let p = parsed.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{path}:{}: malformed line: {line}", lineno + 1),
            )
        })?;
        if !(0.0..=KP_LIMIT).contains(&p.kp)
            || !(0.0..=KD_LIMIT).contains(&p.kd)
            || p.t_ff.abs() > T_FF_LIMIT
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{path}:{}: {} gains out of sanity range \
                     (kp {} <= {KP_LIMIT}, kd {} <= {KD_LIMIT}, |t_ff| {} <= {T_FF_LIMIT})",
                    lineno + 1,
                    p.joint,
                    p.kp,
                    p.kd,
                    p.t_ff
                ),
            ));
        }
        out.push(p);
    }
    if out.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path}: no joint parameters found"),
        ));
    }
    Ok(out)
}

pub fn run(
    params_path: &str,
    secs: f64,
    hz: f64,
    abort_deg: f64,
    yes: bool,
) -> io::Result<()> {
    let params = parse_params(params_path)?;
    let mut ifaces: Vec<String> = Vec::new();
    for p in &params {
        if !ifaces.contains(&p.iface) {
            ifaces.push(p.iface.clone());
        }
    }
    unsafe { libc::signal(libc::SIGINT, on_sigint as libc::sighandler_t) };

    println!(
        "hold: {} joints on {} bus(es), {hz} Hz for {secs}s, abort at {abort_deg}° deviation{}",
        params.len(),
        ifaces.len(),
        if yes { "" } else { "  [DRY RUN — pass --yes to actuate]" },
    );

    let stop = Arc::new(AtomicBool::new(false));
    let handles: Vec<_> = ifaces
        .iter()
        .map(|iface| {
            let joints: Vec<JointParams> =
                params.iter().filter(|p| &p.iface == iface).cloned().collect();
            let iface = iface.clone();
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || -> io::Result<()> {
                let result = bus_hold(&iface, &joints, secs, hz, abort_deg, yes, &stop);
                // Any bus failing (or finishing early on abort) stops the other.
                stop.store(true, Ordering::SeqCst);
                result
            })
        })
        .collect();

    let mut failed = false;
    for handle in handles {
        if let Err(err) = handle.join().expect("hold thread panicked") {
            eprintln!("error: {err}");
            failed = true;
        }
    }
    if failed {
        return Err(io::Error::other("hold failed on at least one bus"));
    }
    Ok(())
}

fn bus_hold(
    iface: &str,
    joints: &[JointParams],
    secs: f64,
    hz: f64,
    abort_deg: f64,
    yes: bool,
    stop: &AtomicBool,
) -> io::Result<()> {
    let sock = CanSock::open(iface)?;
    sock.drain()?;
    let mut motors = prepare(&sock, iface, joints)?;

    for m in &motors {
        println!(
            "  {iface} 0x{:02X} {:<10} hold {:+8.2}°  kp {:5.1}  kd {:.2}  t_ff {:+6.2} Nm  \
             (p_max {:.3}, t_max {:.0})",
            m.id,
            m.joint,
            m.hold_pos.to_degrees(),
            m.kp,
            m.kd,
            m.t_ff,
            m.ranges.p_max,
            m.ranges.t_max,
        );
    }
    if !yes {
        return Ok(());
    }

    enable(&sock, &motors)?;
    let result = stream(&sock, &mut motors, secs, hz, abort_deg, stop);
    disable(&sock, &motors)?;

    println!("-- {iface} hold report --");
    for m in &motors {
        println!(
            "  0x{:02X} {:<10} max deviation {:6.3}°  ({} feedback frames)",
            m.id,
            m.joint,
            m.max_err.to_degrees(),
            m.replies,
        );
    }
    result
}

/// Cold bring-up prep, mirroring the Python drivers. Read/reset only — the
/// motors are left torque-off (enable happens after the operator gate).
fn prepare(sock: &CanSock, iface: &str, joints: &[JointParams]) -> io::Result<Vec<MotorCtx>> {
    let err = |msg: String| io::Error::other(format!("{iface}: {msg}"));

    // MyActuator system reset (no reply), all motors at once, one settle.
    let ma: Vec<&JointParams> = joints.iter().filter(|p| p.motor_id <= 5).collect();
    let dm: Vec<&JointParams> = joints.iter().filter(|p| p.motor_id >= 6).collect();
    for p in &ma {
        sock.send(proto::MA_REQ + p.motor_id as u16, &proto::ma_cmd(proto::MA_RESET))?;
    }
    if !ma.is_empty() {
        std::thread::sleep(RESET_SETTLE);
        sock.drain()?;
    }

    let mut motors = Vec::new();
    for p in &ma {
        let version = txn::ma_request(sock, p.motor_id, proto::ma_cmd(proto::MA_READ_VERSION), TIMEOUT)?
            .map(|(d, _)| proto::ma_decode_version(&d));
        let model = read_ma_model(sock, p.motor_id)?;
        let (p_max, t_max) = proto::ma_mit_ranges(version, model.as_deref());

        let Some((s1, _)) =
            txn::ma_request(sock, p.motor_id, proto::ma_cmd(proto::MA_READ_STATUS1), TIMEOUT)?
        else {
            return Err(err(format!("{} (0x{:02X}): no status reply", p.joint, p.motor_id)));
        };
        let (_, errors) = proto::ma_decode_status1(&s1);
        if errors != 0 {
            return Err(err(format!(
                "{} (0x{:02X}): latched fault 0x{errors:04X} — not enabling",
                p.joint, p.motor_id
            )));
        }
        let Some((pos_frame, _)) =
            txn::ma_request(sock, p.motor_id, proto::ma_cmd(proto::MA_MULTI_TURN_ANGLE), TIMEOUT)?
        else {
            return Err(err(format!("{} (0x{:02X}): no position reply", p.joint, p.motor_id)));
        };
        motors.push(MotorCtx {
            id: p.motor_id,
            joint: p.joint.clone(),
            vendor: Vendor::MyActuator,
            ranges: proto::MitRanges {
                p_max,
                v_max: proto::MA_V_MAX,
                kp_max: proto::MA_KP_MAX,
                kd_max: proto::MA_KD_MAX,
                t_max,
            },
            hold_pos: proto::ma_decode_position(&pos_frame),
            kp: p.kp,
            kd: p.kd,
            t_ff: p.t_ff,
            max_err: 0.0,
            replies: 0,
        });
    }

    for p in &dm {
        let id = p.motor_id as u16;
        sock.send(id, &proto::DM_CLEAR_ERRORS)?;
        std::thread::sleep(Duration::from_millis(10));

        let mode = read_dm_register(sock, id, proto::DM_REG_CTRL_MODE)?;
        if mode != 1.0 {
            return Err(err(format!(
                "{} (0x{:02X}): control mode {mode} is not MIT — not enabling",
                p.joint, p.motor_id
            )));
        }
        let p_max = read_dm_register(sock, id, proto::DM_REG_PMAX)?;
        let v_max = read_dm_register(sock, id, proto::DM_REG_VMAX)?;
        let t_max = read_dm_register(sock, id, proto::DM_REG_TMAX)?;

        let Some((fb, _)) = txn::dm_request_feedback(sock, id, TIMEOUT)? else {
            return Err(err(format!("{} (0x{:02X}): no feedback reply", p.joint, p.motor_id)));
        };
        let decoded = proto::dm_decode_feedback(&fb, p_max, v_max, t_max);
        motors.push(MotorCtx {
            id: p.motor_id,
            joint: p.joint.clone(),
            vendor: Vendor::Damiao,
            ranges: proto::MitRanges {
                p_max,
                v_max,
                kp_max: 500.0,
                kd_max: 5.0,
                t_max,
            },
            hold_pos: decoded.position,
            kp: p.kp,
            kd: p.kd,
            t_ff: p.t_ff,
            max_err: 0.0,
            replies: 0,
        });
    }
    Ok(motors)
}

fn read_ma_model(sock: &CanSock, motor_id: u8) -> io::Result<Option<String>> {
    let mut raw = Vec::new();
    for block in [0x01u8, 0x02] {
        let req = [proto::MA_READ_MODEL, 0x01, block, 0, 0, 0, 0, 0];
        match txn::ma_request(sock, motor_id, req, TIMEOUT)? {
            Some((d, _)) => raw.extend_from_slice(&proto::ma_decode_model_block(&d)),
            None => return Ok(None),
        }
    }
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    Ok(Some(String::from_utf8_lossy(&raw[..end]).into_owned()))
}

/// Register read with the Python driver's retry (single unacked reply frame).
fn read_dm_register(sock: &CanSock, motor_id: u16, rid: u8) -> io::Result<f64> {
    for _ in 0..5 {
        if let Some((value, _)) = txn::dm_read_register(sock, motor_id, rid, TIMEOUT)? {
            return Ok(value);
        }
    }
    Err(io::Error::other(format!(
        "damiao 0x{motor_id:02X}: register {rid} read timed out"
    )))
}

fn enable(sock: &CanSock, motors: &[MotorCtx]) -> io::Result<()> {
    for m in motors {
        match m.vendor {
            Vendor::MyActuator => {
                if txn::ma_request(sock, m.id, proto::ma_cmd(proto::MA_RELEASE_BRAKE), TIMEOUT)?
                    .is_none()
                {
                    return Err(io::Error::other(format!(
                        "{}: brake release not acknowledged",
                        m.joint
                    )));
                }
            }
            Vendor::Damiao => sock.send(m.id as u16, &proto::DM_ENABLE)?,
        }
    }
    Ok(())
}

fn disable(sock: &CanSock, motors: &[MotorCtx]) -> io::Result<()> {
    // Best-effort on every motor even if one errors; Damiao gets the
    // repeated-send treatment the Python driver uses.
    for m in motors {
        for _ in 0..3 {
            let sent = match m.vendor {
                Vendor::MyActuator => {
                    sock.send(proto::MA_REQ + m.id as u16, &proto::ma_cmd(proto::MA_SHUTDOWN))
                }
                Vendor::Damiao => sock.send(m.id as u16, &proto::DM_DISABLE),
            };
            if let Err(err) = sent {
                eprintln!("disable {}: {err}", m.joint);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    Ok(())
}

/// The MIT command stream: paced ticks, pipelined send + reply collection,
/// deviation tracking with abort.
fn stream(
    sock: &CanSock,
    motors: &mut [MotorCtx],
    secs: f64,
    hz: f64,
    abort_deg: f64,
    stop: &AtomicBool,
) -> io::Result<()> {
    let period = Duration::from_secs_f64(1.0 / hz);
    let abort_rad = abort_deg.to_radians();
    let ticks = (secs * hz) as u64;
    let start = Instant::now() + period;

    for k in 0..ticks {
        if stop.load(Ordering::SeqCst) || SIGINT.load(Ordering::SeqCst) {
            break;
        }
        let deadline = start + period * k as u32;
        sleep_until(deadline);

        // Send all commands back-to-back.
        for m in motors.iter() {
            let frame = proto::mit_encode(m.hold_pos, 0.0, m.kp, m.kd, m.t_ff, &m.ranges);
            match m.vendor {
                Vendor::MyActuator => sock.send(proto::MA_MC_REQ + m.id as u16, &frame)?,
                Vendor::Damiao => sock.send(m.id as u16, &frame)?,
            }
        }

        // Collect replies until the tick budget runs out.
        let reply_deadline = deadline + period.mul_add_safe(0.8);
        let mut pending = motors.len();
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
                id if (0x16..=0x17).contains(&id) => {
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
            let m = &mut motors[idx];
            m.replies += 1;
            pending -= 1;
            let e = (pos - m.hold_pos).abs();
            if e > m.max_err {
                m.max_err = e;
            }
            if e > abort_rad {
                stop.store(true, Ordering::SeqCst);
                return Err(io::Error::other(format!(
                    "{} deviated {:.2}° from hold target (abort at {abort_deg}°)",
                    m.joint,
                    e.to_degrees()
                )));
            }
        }
    }
    Ok(())
}

trait DurationExt {
    fn mul_add_safe(&self, f: f64) -> Duration;
}

impl DurationExt for Duration {
    fn mul_add_safe(&self, f: f64) -> Duration {
        Duration::from_secs_f64(self.as_secs_f64() * f)
    }
}

fn sleep_until(deadline: Instant) {
    const SPIN: Duration = Duration::from_micros(300);
    loop {
        let now = Instant::now();
        if now >= deadline {
            return;
        }
        let remaining = deadline - now;
        if remaining > SPIN {
            std::thread::sleep(remaining - SPIN);
        } else {
            std::hint::spin_loop();
        }
    }
}
