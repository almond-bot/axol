//! Realtime wheel controller for the Jelly mobile base.
//!
//! Python owns VR mapping and the optional lift. This service owns the wheel
//! motor lifecycle, command watchdog, axis snap, vector slew, x-drive mix,
//! gyro heading hold, park/unpark state machine, and every wheel CAN frame.

use crate::bringup;
use crate::can::CanSock;
use crate::proto::{self, MitRanges};
use crate::safety::{guarded_send, purge_tx_queue, SendOutcome};
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
static SIGNAL_STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    SIGNAL_STOP.store(true, Ordering::SeqCst);
}

#[derive(Clone, Copy)]
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
    if data.len() != 80 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Jelly config must contain 10 f64 values",
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
    };
    if !(1.0..=500.0).contains(&c.frequency)
        || c.timeout <= 0.0
        || c.max_speed < 0.0
        || c.slew < 0.0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid Jelly config values",
        ));
    }
    Ok(c)
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
    for id in IDS {
        write_register(sock, id, proto::DM_REG_CTRL_MODE, mode.to_le_bytes())?;
    }
    Ok(())
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
        if let Err(err) = set_mode(self.sock, 3) {
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
        write_register(&sock, id, proto::DM_REG_PMAX, (PMAX as f32).to_le_bytes())?;
    }
    let mut ranges = [MitRanges {
        p_max: PMAX,
        v_max: 45.0,
        kp_max: 500.0,
        kd_max: 5.0,
        t_max: 18.0,
    }; 4];
    // This guard predates the first enable attempt so any later register,
    // mode, command, feedback, or IPC failure rolls the wheels back.
    let mut disable = WheelDisableGuard {
        sock: &sock,
        iface,
        armed: false,
        bus_dead: false,
    };
    for (i, id) in IDS.into_iter().enumerate() {
        ranges[i].p_max = bringup::read_dm_register(&sock, id as u16, proto::DM_REG_PMAX)?;
        ranges[i].v_max = bringup::read_dm_register(&sock, id as u16, proto::DM_REG_VMAX)?;
        ranges[i].t_max = bringup::read_dm_register(&sock, id as u16, proto::DM_REG_TMAX)?;
        disable.armed = true;
        sock.send(id as u16, &proto::DM_ENABLE)?;
    }
    set_mode(&sock, 3)?;
    let period = Duration::from_secs_f64(1.0 / cfg.frequency);
    let mut cmd = [0.0; 3];
    let mut pos = [0.0f64; 4];
    let mut vel = [0.0f64; 4];
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
        &ranges,
        &mut pos,
        &mut vel,
        Instant::now() + Duration::from_millis(200),
    )?;
    if initial_feedback != IDS.len() {
        return Err(io::Error::other(format!(
            "Jelly startup received {initial_feedback}/{} wheel replies",
            IDS.len()
        )));
    }
    write_message(output, b"R")?;
    let result = (|| -> io::Result<()> {
        while !stop.load(Ordering::Acquire) && !SIGNAL_STOP.load(Ordering::SeqCst) {
            super_sleep_until(next);
            let now = Instant::now();
            next += period;
            let input = *inputs.lock().unwrap();
            let mut target = input
                .target
                .filter(|t| now.duration_since(t.at).as_secs_f64() <= cfg.timeout)
                .unwrap_or(Target {
                    vx: 0.0,
                    vy: 0.0,
                    wz: 0.0,
                    at: now,
                });
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
            let mut yaw_corr = 0.0;
            if cfg.yaw_gain != 0.0 {
                if let Some(y) = input.yaw {
                    if now.duration_since(y.at) <= Duration::from_millis(300) {
                        if translating && !turning {
                            yaw_err += (y.rate - yaw_bias) * period.as_secs_f64();
                            yaw_corr = (-cfg.yaw_gain * yaw_err).clamp(-cfg.yaw_max, cfg.yaw_max);
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
            let speeds = mix(cmd[0], cmd[1], cmd[2] + yaw_corr, cfg);
            if driving && hold.is_some() {
                set_mode(&sock, 3)?;
                hold = None;
                park_failed = false;
            }
            if !driving
                && cfg.hold_kp > 0.0
                && !park_failed
                && hold.is_none()
                && vel.iter().all(|v| (*v).abs() <= PARK_MAX_SPEED)
            {
                if pos.iter().any(|p| (*p).abs() > 0.9 * PMAX) {
                    park_failed = true;
                } else {
                    set_mode(&sock, 1)?;
                    hold = Some(pos);
                }
            }
            let send_result: io::Result<()> = (|| {
                if let Some(anchor) = hold {
                    for i in 0..4 {
                        let f = proto::mit_encode(
                            anchor[i],
                            0.0,
                            cfg.hold_kp,
                            cfg.hold_kd,
                            0.0,
                            &ranges[i],
                        );
                        if let SendOutcome::Stalled =
                            guarded_send(&sock, IDS[i] as u16, &f, &mut enobufs)?
                        {
                            return Err(io::Error::other("Jelly CAN TX stalled"));
                        }
                    }
                } else {
                    for i in 0..4 {
                        if let SendOutcome::Stalled = guarded_send(
                            &sock,
                            0x200 + IDS[i] as u16,
                            &velocity_frame(speeds[i]),
                            &mut enobufs,
                        )? {
                            return Err(io::Error::other("Jelly CAN TX stalled"));
                        }
                    }
                }
                Ok(())
            })();
            send_failed = send_result.is_err();
            if let Err(e) = send_result {
                disable.bus_dead = true;
                let _ = purge_tx_queue(iface);
                return Err(e);
            }
            let _ = collect_feedback(
                &sock,
                &ranges,
                &mut pos,
                &mut vel,
                now + period.mul_f64(0.8),
            )?;
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
                        | ((send_failed as u8) << 2),
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
        };
        assert_eq!(mix(1.0, 0.0, 0.0, c), [20.0, -20.0, 20.0, -20.0]);
    }

    #[test]
    fn python_config_layout_roundtrips() {
        let values = [20.0f64, 1.0, 0.5, 15.0, 2.0, 0.3, 60.0, 1.5, 50.0, 0.3];
        let wire: Vec<u8> = values.into_iter().flat_map(f64::to_le_bytes).collect();
        let cfg = parse_config(&wire).unwrap();
        assert_eq!(cfg.frequency, 50.0);
        assert_eq!(cfg.hold_kp, 60.0);
    }
}
