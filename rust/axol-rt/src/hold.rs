//! Standalone motion test: enable the seven arm joints and actively hold the
//! current pose with an MIT impedance stream, then disable.
//!
//! Bring-up is shared with `serve` (see `bringup`). Gains and gravity
//! feedforward come from a params file written by `tools/gen_hold_params.py`.
//!
//! Safety:
//! - Without `--yes` this is a dry run: bring-up prep and the plan are
//!   printed, but nothing is enabled and no motion command is sent.
//! - Any joint deviating more than `--abort-deg` from its hold target stops
//!   both buses and disables everything.
//! - Every exit path (completion, abort, Ctrl-C, errors) runs the disable
//!   sequence; the gripper (ID 0x08) is never touched.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::bringup::{self, MotorSpec, ReadyMotor, Vendor};
use crate::can::CanSock;
use crate::proto;

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
    pub spec: MotorSpec,
    pub t_ff: f64,
}

struct HeldMotor {
    ready: ReadyMotor,
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
                spec: MotorSpec {
                    joint: fields.get(1)?.to_string(),
                    motor_id: fields.get(2)?.parse().ok()?,
                    kp: fields.get(3)?.parse().ok()?,
                    kd: fields.get(4)?.parse().ok()?,
                    gripper: false, // hold never touches the gripper
                    slot: 0,        // unused: hold has no target stream
                    // Tracker/friction params are serve-only.
                    max_vel: 0.0,
                    max_accel: 0.0,
                    fc: 0.0,
                    k: 0.0,
                    fv: 0.0,
                    fo: 0.0,
                },
                t_ff: fields.get(5)?.parse().ok()?,
            })
        })();
        let p = parsed.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{path}:{}: malformed line: {line}", lineno + 1),
            )
        })?;
        if !(0.0..=KP_LIMIT).contains(&p.spec.kp)
            || !(0.0..=KD_LIMIT).contains(&p.spec.kd)
            || p.t_ff.abs() > T_FF_LIMIT
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{path}:{}: {} gains out of sanity range \
                     (kp {} <= {KP_LIMIT}, kd {} <= {KD_LIMIT}, |t_ff| {} <= {T_FF_LIMIT})",
                    lineno + 1,
                    p.spec.joint,
                    p.spec.kp,
                    p.spec.kd,
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

pub fn run(params_path: &str, secs: f64, hz: f64, abort_deg: f64, yes: bool) -> io::Result<()> {
    let params = parse_params(params_path)?;
    let mut ifaces: Vec<String> = Vec::new();
    for p in &params {
        if !ifaces.contains(&p.iface) {
            ifaces.push(p.iface.clone());
        }
    }
    unsafe { libc::signal(libc::SIGINT, on_sigint as *const () as libc::sighandler_t) };

    println!(
        "hold: {} joints on {} bus(es), {hz} Hz for {secs}s, abort at {abort_deg}° deviation{}",
        params.len(),
        ifaces.len(),
        if yes {
            ""
        } else {
            "  [DRY RUN — pass --yes to actuate]"
        },
    );

    let stop = Arc::new(AtomicBool::new(false));
    let handles: Vec<_> = ifaces
        .iter()
        .map(|iface| {
            let joints: Vec<JointParams> = params
                .iter()
                .filter(|p| &p.iface == iface)
                .cloned()
                .collect();
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
    let specs: Vec<MotorSpec> = joints.iter().map(|p| p.spec.clone()).collect();
    bringup::prep(&sock, &specs)?;
    let ready = bringup::prepare(&sock, iface, &specs)?;
    let mut motors: Vec<HeldMotor> = ready
        .into_iter()
        .map(|ready| {
            let t_ff = joints
                .iter()
                .find(|p| p.spec.motor_id == ready.id)
                .map_or(0.0, |p| p.t_ff);
            HeldMotor {
                ready,
                t_ff,
                max_err: 0.0,
                replies: 0,
            }
        })
        .collect();

    for m in &motors {
        println!(
            "  {iface} 0x{:02X} {:<10} hold {:+8.2}°  kp {:5.1}  kd {:.2}  t_ff {:+6.2} Nm  \
             (p_max {:.3}, t_max {:.0})",
            m.ready.id,
            m.ready.joint,
            m.ready.hold_pos.to_degrees(),
            m.ready.kp,
            m.ready.kd,
            m.t_ff,
            m.ready.ranges.p_max,
            m.ready.ranges.t_max,
        );
    }
    if !yes {
        return Ok(());
    }

    let readies: Vec<ReadyMotor> = motors.iter().map(|m| m.ready.clone()).collect();
    bringup::enable(&sock, &readies)?;
    let result = stream(&sock, &mut motors, secs, hz, abort_deg, stop);
    bringup::disable(&sock, &readies);

    println!("-- {iface} hold report --");
    for m in &motors {
        println!(
            "  0x{:02X} {:<10} max deviation {:6.3}°  ({} feedback frames)",
            m.ready.id,
            m.ready.joint,
            m.max_err.to_degrees(),
            m.replies,
        );
    }
    result
}

/// The MIT command stream: paced ticks, pipelined send + reply collection,
/// deviation tracking with abort.
fn stream(
    sock: &CanSock,
    motors: &mut [HeldMotor],
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

        // Do not attribute a reply that missed the previous window to this
        // command batch (the motor protocols carry no sequence number).
        sock.drain_nonblocking()?;

        for m in motors.iter() {
            let frame = proto::mit_encode(
                m.ready.hold_pos,
                0.0,
                m.ready.kp,
                m.ready.kd,
                m.t_ff,
                &m.ready.ranges,
            );
            match m.ready.vendor {
                Vendor::MyActuator => sock.send(proto::MA_MC_REQ + m.ready.id as u16, &frame)?,
                Vendor::Damiao => sock.send(m.ready.id as u16, &frame)?,
            }
        }

        let reply_deadline = deadline + Duration::from_secs_f64(period.as_secs_f64() * 0.8);
        let mut pending = motors.len();
        let mut seen = vec![false; motors.len()];
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
                    let Some(idx) = motors.iter().position(|m| m.ready.id == motor_id) else {
                        continue;
                    };
                    let (pos, _, _) = proto::ma_decode_mit_feedback(
                        &frame.data,
                        motors[idx].ready.ranges.p_max,
                        motors[idx].ready.ranges.t_max,
                    );
                    (idx, pos)
                }
                id if (0x16..=0x17).contains(&id) => {
                    let motor_id = (id - 0x10) as u8;
                    let Some(idx) = motors.iter().position(|m| m.ready.id == motor_id) else {
                        continue;
                    };
                    let m = &motors[idx];
                    let fb = proto::dm_decode_feedback(
                        &frame.data,
                        m.ready.ranges.p_max,
                        m.ready.ranges.v_max,
                        m.ready.ranges.t_max,
                    );
                    (idx, fb.position)
                }
                _ => continue,
            };
            if seen[idx] {
                continue;
            }
            seen[idx] = true;
            let m = &mut motors[idx];
            m.replies += 1;
            pending -= 1;
            let e = (pos - m.ready.hold_pos).abs();
            if e > m.max_err {
                m.max_err = e;
            }
            if e > abort_rad {
                stop.store(true, Ordering::SeqCst);
                return Err(io::Error::other(format!(
                    "{} deviated {:.2}° from hold target (abort at {abort_deg}°)",
                    m.ready.joint,
                    e.to_degrees()
                )));
            }
        }
    }
    Ok(())
}

pub fn sleep_until(deadline: Instant) {
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
