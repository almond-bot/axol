//! Read-only control-loop timing benchmark.
//!
//! Paces a loop at the requested rate and, each tick, queries every motor on
//! the bus (MyActuator 0x92 position reads, Damiao 0xCC feedback requests) —
//! the same request/reply bus load as a telemetry cycle, with no enable or
//! motion commands. Both arm buses run in parallel on their own threads,
//! mirroring how the realtime core would drive them.

use std::io;
use std::time::{Duration, Instant};

use crate::can::CanSock;
use crate::proto;
use crate::txn;

#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    /// Fire every request back-to-back, then collect all replies.
    Pipelined,
    /// One request/response transaction per motor, sequentially.
    Serial,
}

pub struct TickStats {
    /// How late the tick started vs its deadline.
    pub lateness: Vec<f64>,
    /// Time from first send to last reply (or timeout) within the tick.
    pub cycle: Vec<f64>,
    pub missing: u64,
    pub ticks: u64,
}

pub fn run(ifaces: &[String], hz: f64, secs: f64, mode: Mode) -> io::Result<()> {
    let mode_name = match mode {
        Mode::Pipelined => "pipelined",
        Mode::Serial => "serial",
    };
    println!(
        "bench: {hz} Hz for {secs}s, {mode_name}, buses: {}",
        ifaces.join(", ")
    );

    let handles: Vec<_> = ifaces
        .iter()
        .map(|iface| {
            let iface = iface.clone();
            std::thread::spawn(move || -> io::Result<(String, TickStats)> {
                let sock = CanSock::open(&iface)?;
                sock.drain()?;
                let stats = bench_bus(&sock, hz, secs, mode)?;
                Ok((iface, stats))
            })
        })
        .collect();

    for handle in handles {
        let (iface, stats) = handle.join().expect("bench thread panicked")?;
        report(&iface, hz, &stats);
    }
    Ok(())
}

fn bench_bus(sock: &CanSock, hz: f64, secs: f64, mode: Mode) -> io::Result<TickStats> {
    let period = Duration::from_secs_f64(1.0 / hz);
    let ticks = (secs * hz) as u64;
    // Per-motor reply budget within a tick; the whole cycle must fit a period.
    let reply_timeout = period.min(Duration::from_millis(4));

    let mut stats = TickStats {
        lateness: Vec::with_capacity(ticks as usize),
        cycle: Vec::with_capacity(ticks as usize),
        missing: 0,
        ticks,
    };

    let start = Instant::now() + period;
    for k in 0..ticks {
        let deadline = start + period * k as u32;
        sleep_until(deadline);
        let began = Instant::now();
        stats.lateness.push((began - deadline).as_secs_f64() * 1e3);

        let missing = match mode {
            Mode::Pipelined => tick_pipelined(sock, reply_timeout)?,
            Mode::Serial => tick_serial(sock, reply_timeout)?,
        };
        stats.missing += missing as u64;
        stats.cycle.push(began.elapsed().as_secs_f64() * 1e3);
    }
    Ok(stats)
}

/// Fire all 8 requests, then collect replies until complete or timeout.
/// Returns the number of motors that did not answer.
fn tick_pipelined(sock: &CanSock, timeout: Duration) -> io::Result<usize> {
    let began = Instant::now();
    for &id in &proto::MA_IDS {
        sock.send(
            proto::MA_REQ + id as u16,
            &proto::ma_cmd(proto::MA_MULTI_TURN_ANGLE),
        )?;
    }
    for &id in &proto::DM_IDS {
        sock.send(proto::DM_REG_ARB, &proto::dm_request_feedback(id as u16))?;
    }

    let mut pending: u32 = (proto::MA_IDS.len() + proto::DM_IDS.len()) as u32;
    let mut seen = [false; 9]; // indexed by motor id 1..=8
    let deadline = began + timeout;
    while pending > 0 {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let Some(frame) = sock.recv_timeout(deadline - now)? else {
            break;
        };
        let motor_id = match frame.id {
            id if (0x241..=0x245).contains(&id) && frame.data[0] == proto::MA_MULTI_TURN_ANGLE => {
                (id - 0x240) as usize
            }
            id if (0x16..=0x18).contains(&id) => (id - 0x10) as usize,
            _ => continue,
        };
        if !seen[motor_id] {
            seen[motor_id] = true;
            pending -= 1;
        }
    }
    Ok(pending as usize)
}

/// Sequential request/response per motor. Returns the number of timeouts.
fn tick_serial(sock: &CanSock, timeout: Duration) -> io::Result<usize> {
    let mut missing = 0;
    for &id in &proto::MA_IDS {
        let resp = txn::ma_request(sock, id, proto::ma_cmd(proto::MA_MULTI_TURN_ANGLE), timeout)?;
        if resp.is_none() {
            missing += 1;
        }
    }
    for &id in &proto::DM_IDS {
        if txn::dm_request_feedback(sock, id as u16, timeout)?.is_none() {
            missing += 1;
        }
    }
    Ok(missing)
}

/// Sleep until `deadline`, finishing with a short spin for precision.
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

fn report(iface: &str, hz: f64, stats: &TickStats) {
    let pct = |sorted: &[f64], p: f64| -> f64 {
        if sorted.is_empty() {
            return f64::NAN;
        }
        let idx = ((sorted.len() as f64 - 1.0) * p / 100.0).round() as usize;
        sorted[idx]
    };
    let mut lateness = stats.lateness.clone();
    lateness.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut cycle = stats.cycle.clone();
    cycle.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let total_queries = stats.ticks * 8;
    println!("\n-- {iface} @ {hz} Hz ({} ticks) --", stats.ticks);
    println!(
        "  tick lateness ms: p50 {:.3}  p90 {:.3}  p99 {:.3}  max {:.3}",
        pct(&lateness, 50.0),
        pct(&lateness, 90.0),
        pct(&lateness, 99.0),
        pct(&lateness, 100.0),
    );
    println!(
        "  cycle time    ms: p50 {:.3}  p90 {:.3}  p99 {:.3}  max {:.3}  (budget {:.3})",
        pct(&cycle, 50.0),
        pct(&cycle, 90.0),
        pct(&cycle, 99.0),
        pct(&cycle, 100.0),
        1e3 / hz,
    );
    println!(
        "  replies missing: {} / {} ({:.3}%)",
        stats.missing,
        total_queries,
        stats.missing as f64 / total_queries as f64 * 100.0,
    );
}
