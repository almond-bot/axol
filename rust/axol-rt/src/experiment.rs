//! Precisely paced single-joint tuning programs for the maintenance proxy.
//!
//! Python plans the slow reference/gravity samples and analyzes the returned
//! log. Rust owns every timed tick, command derivative, friction/inertia term,
//! host damping update, CAN send, and feedback sample.

use crate::can::CanSock;
use crate::filter::{self, BandPass, LpDiff};
use crate::proto::{self, MitRanges};
use crate::safety::{guarded_send, purge_tx_queue, SendOutcome};
use std::collections::HashMap;
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub struct RawSample {
    pub data: [u8; 8],
    pub at: Instant,
    pub generation: u64,
}

pub type FeedbackStore = Arc<(Mutex<HashMap<u32, RawSample>>, Condvar)>;

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}
impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn take<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        let end = self
            .pos
            .checked_add(N)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "experiment overflow"))?;
        let bytes = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "truncated experiment"))?;
        self.pos = end;
        Ok(bytes.try_into().unwrap())
    }
    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take::<1>()?[0])
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take()?))
    }
    fn f64(&mut self) -> io::Result<f64> {
        Ok(f64::from_le_bytes(self.take()?))
    }
}

struct Config {
    vendor: u8,
    id: u8,
    differentiate: bool,
    rate: f64,
    offset: f64,
    kp: f64,
    kd: f64,
    ranges: MitRanges,
    fc: f64,
    k: f64,
    fv: f64,
    fo: f64,
    j_eff: f64,
    host_kd: f64,
    damp_w0: f64,
    damp_q: f64,
    samples: Vec<(f64, f64, f64, f64)>, // command, clean reference, gravity, velocity override
}

fn parse(payload: &[u8]) -> io::Result<Config> {
    let mut c = Cursor::new(payload);
    let vendor = c.u8()?;
    let id = c.u8()?;
    let differentiate = c.u8()? != 0;
    let _reserved = c.u8()?;
    let rate = c.f64()?;
    let offset = c.f64()?;
    let kp = c.f64()?;
    let kd = c.f64()?;
    let ranges = MitRanges {
        p_max: c.f64()?,
        v_max: c.f64()?,
        kp_max: c.f64()?,
        kd_max: c.f64()?,
        t_max: c.f64()?,
    };
    let fc = c.f64()?;
    let k = c.f64()?;
    let fv = c.f64()?;
    let fo = c.f64()?;
    let j_eff = c.f64()?;
    let host_kd = c.f64()?;
    let damp_w0 = c.f64()?;
    let damp_q = c.f64()?;
    let count = c.u32()? as usize;
    if vendor > 1 || id == 0 || !(1.0..=1000.0).contains(&rate) || count > 1_000_000 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid experiment configuration",
        ));
    }
    let mut samples = Vec::with_capacity(count);
    for _ in 0..count {
        samples.push((c.f64()?, c.f64()?, c.f64()?, c.f64()?));
    }
    if c.pos != payload.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "trailing experiment bytes",
        ));
    }
    Ok(Config {
        vendor,
        id,
        differentiate,
        rate,
        offset,
        kp,
        kd,
        ranges,
        fc,
        k,
        fv,
        fo,
        j_eff,
        host_kd,
        damp_w0,
        damp_q,
        samples,
    })
}

fn sleep_until(deadline: Instant) {
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

/// Execute an `X` payload and return an `X` response containing
/// `(t, clean_target, actual, torque)` f64 rows.
pub fn run(
    payload: &[u8],
    iface: &str,
    socket: &CanSock,
    feedback: &FeedbackStore,
    enobufs_since: &mut Option<Instant>,
    control: &UnixStream,
) -> io::Result<Vec<u8>> {
    let cfg = parse(payload)?;
    let command_id = if cfg.vendor == 0 {
        proto::MA_MC_REQ + cfg.id as u16
    } else {
        cfg.id as u16
    };
    let feedback_id = if cfg.vendor == 0 {
        (proto::MA_MC_RESP + cfg.id as u16) as u32
    } else {
        (proto::DM_FEEDBACK_BASE + cfg.id as u16) as u32
    };
    let period = Duration::from_secs_f64(1.0 / cfg.rate);
    let mut v_cmd = LpDiff::new(20.0);
    let mut a_cmd = LpDiff::new(20.0);
    let mut v_fast = LpDiff::new(80.0);
    let mut v_meas = LpDiff::new(80.0);
    let mut bp = BandPass::new();
    let mut previous_tick: Option<Instant> = None;
    let mut previous_feedback: Option<Instant> = None;
    let start = Instant::now() + period;
    let mut output = Vec::with_capacity(1 + 4 + cfg.samples.len() * 32);
    output.push(b'X');
    output.extend_from_slice(&(cfg.samples.len() as u32).to_le_bytes());

    for (index, &(target_joint, reference_joint, gravity, velocity_override)) in
        cfg.samples.iter().enumerate()
    {
        if cancel_requested(control)? {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "tuning experiment cancelled",
            ));
        }
        let deadline = start + period * index as u32;
        sleep_until(deadline);
        let now = Instant::now();
        let dt = previous_tick.map_or(period.as_secs_f64(), |p| {
            now.duration_since(p).as_secs_f64()
        });
        previous_tick = Some(now);
        let target = target_joint - cfg.offset;
        let (v_des, accel, v_des_fast) = if velocity_override.is_finite() {
            v_cmd.seed(target);
            a_cmd.seed(velocity_override);
            v_fast.seed(target);
            bp.reset();
            (velocity_override, 0.0, velocity_override)
        } else if cfg.differentiate {
            let v = v_cmd.update(target, dt);
            let a = a_cmd.update(v, dt);
            let vf = v_fast.update(target, dt);
            (v, a, vf)
        } else {
            v_cmd.seed(target);
            a_cmd.seed(0.0);
            v_fast.seed(target);
            bp.reset();
            (0.0, 0.0, 0.0)
        };
        let before = feedback
            .0
            .lock()
            .unwrap()
            .get(&feedback_id)
            .map_or(0, |s| s.generation);
        let measured_velocity = {
            let guard = feedback.0.lock().unwrap();
            guard.get(&feedback_id).map_or(0.0, |s| {
                let pos = decode(&cfg, &s.data).0;
                let fdt = previous_feedback
                    .map_or(0.0, |p| s.at.saturating_duration_since(p).as_secs_f64());
                previous_feedback = Some(s.at);
                v_meas.update(pos, fdt)
            })
        };
        let friction = filter::friction(v_des, cfg.fc, cfg.k, cfg.fv, cfg.fo);
        let damping =
            cfg.host_kd * bp.update(v_des_fast - measured_velocity, cfg.damp_w0, cfg.damp_q, dt);
        let torque = gravity + friction + cfg.j_eff * accel + damping;
        let frame = proto::mit_encode(target, v_des, cfg.kp, cfg.kd, torque, &cfg.ranges);
        match guarded_send(socket, command_id, &frame, enobufs_since)? {
            SendOutcome::Sent => {}
            SendOutcome::Dropped => {}
            SendOutcome::Stalled => {
                let purged = purge_tx_queue(iface);
                return Err(io::Error::other(format!(
                    "CAN {iface} TX stalled during tuning experiment{}",
                    if purged {
                        "; stale queue purged"
                    } else {
                        "; QUEUE PURGE FAILED, flap the interface before re-powering"
                    }
                )));
            }
        }

        let wait = period.mul_f64(0.8);
        let guard = feedback.0.lock().unwrap();
        let (guard, _) = feedback
            .1
            .wait_timeout_while(guard, wait, |m| {
                m.get(&feedback_id).map_or(0, |s| s.generation) <= before
            })
            .unwrap();
        let (actual, measured_torque, sample_time) = guard
            .get(&feedback_id)
            .filter(|s| s.generation > before)
            .map(|s| {
                let (p, t) = decode(&cfg, &s.data);
                (
                    p + cfg.offset,
                    t,
                    s.at.saturating_duration_since(start).as_secs_f64(),
                )
            })
            .unwrap_or((
                f64::NAN,
                f64::NAN,
                now.saturating_duration_since(start).as_secs_f64(),
            ));
        output.extend_from_slice(&sample_time.to_le_bytes());
        let reference_at_feedback = if cfg.differentiate && index + 1 < cfg.samples.len() {
            let phase = (sample_time / period.as_secs_f64() - index as f64).clamp(0.0, 1.0);
            reference_joint + (cfg.samples[index + 1].1 - reference_joint) * phase
        } else {
            reference_joint
        };
        output.extend_from_slice(&reference_at_feedback.to_le_bytes());
        output.extend_from_slice(&actual.to_le_bytes());
        output.extend_from_slice(&measured_torque.to_le_bytes());
    }
    Ok(output)
}

/// Consume the one-byte `K` control packet without moving Unix-socket reading
/// onto the timed loop. Local writes this small are atomic; a partial header
/// simply waits until the following tick.
fn cancel_requested(stream: &UnixStream) -> io::Result<bool> {
    let mut bytes = [0u8; 5];
    let n = unsafe {
        libc::recv(
            stream.as_raw_fd(),
            bytes.as_mut_ptr().cast(),
            bytes.len(),
            libc::MSG_PEEK | libc::MSG_DONTWAIT,
        )
    };
    if n == 0 {
        return Err(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "tuning client disconnected",
        ));
    }
    if n < 0 {
        let err = io::Error::last_os_error();
        return if matches!(err.kind(), io::ErrorKind::WouldBlock) {
            Ok(false)
        } else {
            Err(err)
        };
    }
    if n < 5 || bytes[..4] != [1, 0, 0, 0] || !matches!(bytes[4], b'K' | b'Q') {
        return Ok(false);
    }
    let consumed = unsafe {
        libc::recv(
            stream.as_raw_fd(),
            bytes.as_mut_ptr().cast(),
            bytes.len(),
            0,
        )
    };
    if consumed == 5 {
        Ok(true)
    } else {
        Err(io::Error::last_os_error())
    }
}

fn decode(cfg: &Config, data: &[u8; 8]) -> (f64, f64) {
    if cfg.vendor == 0 {
        let (p, _, t) = proto::ma_decode_mit_feedback(data, cfg.ranges.p_max, cfg.ranges.t_max);
        (p, t)
    } else {
        let f =
            proto::dm_decode_feedback(data, cfg.ranges.p_max, cfg.ranges.v_max, cfg.ranges.t_max);
        (f.position, f.torque)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_truncated_program() {
        assert!(parse(&[0; 12]).is_err());
    }

    #[test]
    fn python_wire_layout_roundtrips() {
        let mut wire = vec![1, 6, 1, 0];
        for value in [
            240.0f64, 1.25, 100.0, 2.0, 12.5, 45.0, 500.0, 5.0, 18.0, 0.6, 20.0, 0.1, 0.0, 0.02,
            30.0, 25.0, 0.8,
        ] {
            wire.extend_from_slice(&value.to_le_bytes());
        }
        wire.extend_from_slice(&1u32.to_le_bytes());
        for value in [0.3f64, 0.2, 1.1, f64::NAN] {
            wire.extend_from_slice(&value.to_le_bytes());
        }
        let cfg = parse(&wire).unwrap();
        assert_eq!(cfg.id, 6);
        assert_eq!(cfg.rate, 240.0);
        assert_eq!(cfg.samples[0].0, 0.3);
        assert_eq!(cfg.samples[0].1, 0.2);
        assert_eq!(cfg.samples[0].2, 1.1);
        assert!(cfg.samples[0].3.is_nan());
    }
}
