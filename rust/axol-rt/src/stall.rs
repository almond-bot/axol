//! Attribute a late bus tick to what kept the thread off the CPU.
//!
//! A `SCHED_FIFO` loop on a dedicated core that wakes tens of milliseconds
//! late was stopped by one of three things, and the fault line alone cannot
//! say which: it was *runnable but not scheduled* (preempted by a
//! higher-priority task, or by IRQ / softirq work that landed on its core),
//! it took a *page fault* that went into reclaim or compaction (nothing
//! locked its memory), or it was *blocked inside the kernel* — a
//! non-preemptible section, a timer that fired late, a deep idle exit. The
//! kernel already keeps the counters that tell these apart, per thread:
//!
//! - `/proc/thread-self/schedstat` — CPU time run, time spent *runnable and
//!   waiting* for the CPU, and the number of timeslices. The wait figure is
//!   the same one the camera relay reads to attribute skipped exposures.
//! - `getrusage(RUSAGE_THREAD)` — minor / major page faults and voluntary /
//!   involuntary context switches.
//!
//! [`StallProbe`] samples both at the top of every tick (one `pread` on an
//! already-open fd plus one syscall — a few microseconds, off the reply
//! window) so that when a tick does come in late, the deltas across *that*
//! sleep are already in hand and the log line can say what they read as.
//! Sampling on demand would be too late: the counters must bracket the
//! stall, and the stall is only known once the thread is running again.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::time::Duration;

/// Per-thread scheduler and memory counters at one instant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sample {
    /// Nanoseconds spent running on a CPU.
    pub run_ns: u64,
    /// Nanoseconds spent runnable but waiting for a CPU.
    pub wait_ns: u64,
    pub minflt: u64,
    pub majflt: u64,
    /// Voluntary context switches (the thread blocked or slept).
    pub nvcsw: u64,
    /// Involuntary context switches (the thread was preempted).
    pub nivcsw: u64,
}

/// Counter deltas across one tick's sleep-and-wake.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Delta {
    pub run: Duration,
    pub wait: Duration,
    pub minflt: u64,
    pub majflt: u64,
    pub nvcsw: u64,
    pub nivcsw: u64,
}

/// What a late tick's counters read as. Coarse by design: the point is to
/// tell a field log's readers which subsystem to look at next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reading {
    /// Runnable for most of the lateness: something else had the CPU.
    Preempted,
    /// The thread faulted memory in during the stall.
    PageFault,
    /// Neither waiting nor faulting: blocked inside the kernel, or woken late.
    KernelStall,
}

impl Delta {
    /// Classify the stall behind `lateness`. Faults win over waiting — a
    /// fault that goes into reclaim shows up as both — and waiting must
    /// account for at least half the lateness to count as preemption.
    pub fn reading(&self, lateness: Duration) -> Reading {
        if self.majflt > 0 || self.minflt > 0 {
            Reading::PageFault
        } else if lateness > Duration::ZERO && self.wait * 2 >= lateness {
            Reading::Preempted
        } else {
            Reading::KernelStall
        }
    }

    /// One clause for the fault / degraded log line, e.g.
    /// `runnable-waiting 58.9 ms, 0 minor / 0 major page faults, 1
    /// involuntary switch — reads as preempted`.
    pub fn describe(&self, lateness: Duration) -> String {
        let reading = match self.reading(lateness) {
            Reading::Preempted => "preempted (runnable, not scheduled — IRQ/softirq or higher-priority work on this CPU)",
            Reading::PageFault => "page fault (memory reclaim/compaction; is the process mlocked?)",
            Reading::KernelStall => "kernel stall (non-preemptible section, late timer, or idle-state exit)",
        };
        format!(
            "runnable-waiting {:.1} ms, {} minor / {} major page faults, {} involuntary switch{} — reads as {reading}",
            self.wait.as_secs_f64() * 1e3,
            self.minflt,
            self.majflt,
            self.nivcsw,
            if self.nivcsw == 1 { "" } else { "es" },
        )
    }
}

/// Per-thread stall counters, sampled once per tick.
pub struct StallProbe {
    schedstat: Option<File>,
    prev: Option<Sample>,
    // Reused read buffer: schedstat is three decimal u64s and two spaces.
    buf: [u8; 96],
}

impl StallProbe {
    /// Open the calling thread's counters. Must be called *on* the bus
    /// thread: `/proc/thread-self` resolves to the opener, and
    /// `RUSAGE_THREAD` reads the caller. A missing or unreadable schedstat
    /// (kernel built without `CONFIG_SCHEDSTATS`) leaves the wait figure at
    /// zero rather than failing the loop.
    pub fn open() -> Self {
        Self {
            schedstat: File::open("/proc/thread-self/schedstat").ok(),
            prev: None,
            buf: [0; 96],
        }
    }

    /// Read the counters now and return the deltas since the previous call
    /// (all zero on the first call, or if a read fails).
    pub fn sample(&mut self) -> Delta {
        let now = self.read();
        let delta = match (self.prev, now) {
            (Some(prev), Some(now)) => Delta {
                run: Duration::from_nanos(now.run_ns.saturating_sub(prev.run_ns)),
                wait: Duration::from_nanos(now.wait_ns.saturating_sub(prev.wait_ns)),
                minflt: now.minflt.saturating_sub(prev.minflt),
                majflt: now.majflt.saturating_sub(prev.majflt),
                nvcsw: now.nvcsw.saturating_sub(prev.nvcsw),
                nivcsw: now.nivcsw.saturating_sub(prev.nivcsw),
            },
            _ => Delta::default(),
        };
        if now.is_some() {
            self.prev = now;
        }
        delta
    }

    fn read(&mut self) -> Option<Sample> {
        let mut sample = Sample::default();
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut usage) };
        if rc != 0 {
            return None;
        }
        sample.minflt = usage.ru_minflt.max(0) as u64;
        sample.majflt = usage.ru_majflt.max(0) as u64;
        sample.nvcsw = usage.ru_nvcsw.max(0) as u64;
        sample.nivcsw = usage.ru_nivcsw.max(0) as u64;
        if let Some(file) = &self.schedstat {
            if let Ok(n) = file.read_at(&mut self.buf, 0) {
                if let Some((run, wait)) = parse_schedstat(&self.buf[..n]) {
                    sample.run_ns = run;
                    sample.wait_ns = wait;
                }
            }
        }
        Some(sample)
    }
}

/// `schedstat` is `<run_ns> <wait_ns> <timeslices>\n`.
fn parse_schedstat(text: &[u8]) -> Option<(u64, u64)> {
    let text = std::str::from_utf8(text).ok()?;
    let mut fields = text.split_ascii_whitespace();
    let run = fields.next()?.parse().ok()?;
    let wait = fields.next()?.parse().ok()?;
    Some((run, wait))
}

/// The smallest `RLIMIT_MEMLOCK` under which locking is attempted without
/// `CAP_IPC_LOCK`: the binary plus four 2 MiB thread stacks plus heap and
/// socket buffers, with headroom. The common unprivileged default (8 MiB)
/// is below it — see `lock_memory`.
const MIN_MEMLOCK_LIMIT: u64 = 64 << 20;

/// Lock every current and future page of the process into RAM.
///
/// Standard real-time hygiene the core lacked: with nothing locked, a page
/// the bus thread touches after memory pressure has reclaimed it — or a
/// fresh heap page under compaction — is a fault that can hold a
/// `SCHED_FIFO` thread for tens of milliseconds, exactly while the dataset
/// writer is flushing gigabytes of video. `MCL_FUTURE` also locks the bus
/// threads' stacks as they are created, so they are faulted in at spawn.
///
/// Without `CAP_IPC_LOCK` (the production service runs as root; `axol
/// rt.install` grants the capability to dev builds) every locked mapping
/// counts against `RLIMIT_MEMLOCK`, and with `MCL_FUTURE` set a later
/// `mmap` past the limit *fails* — a thread spawn would abort the session
/// long after the lock call itself succeeded. So under a small limit this
/// declines to lock at all rather than lock partially. Failure is reported,
/// not fatal: an unlocked core is what every session ran with before.
pub fn lock_memory() -> io::Result<()> {
    let privileged = unsafe { libc::geteuid() } == 0 || has_cap_ipc_lock();
    if !privileged {
        let mut limit: libc::rlimit = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut limit) };
        if rc == 0 && limit.rlim_cur != libc::RLIM_INFINITY && limit.rlim_cur < MIN_MEMLOCK_LIMIT {
            return Err(io::Error::other(format!(
                "RLIMIT_MEMLOCK is {} MiB (need {} MiB or CAP_IPC_LOCK)",
                limit.rlim_cur >> 20,
                MIN_MEMLOCK_LIMIT >> 20,
            )));
        }
    }
    let rc = unsafe { libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Whether the effective capability set holds `CAP_IPC_LOCK` (bit 14), read
/// from `/proc/thread-self/status` (`CapEff:`) — no libcap dependency.
fn has_cap_ipc_lock() -> bool {
    const CAP_IPC_LOCK: u32 = 14;
    let Ok(status) = std::fs::read_to_string("/proc/thread-self/status") else {
        return false;
    };
    status
        .lines()
        .find_map(|line| line.strip_prefix("CapEff:"))
        .and_then(|hex| u64::from_str_radix(hex.trim(), 16).ok())
        .is_some_and(|caps| caps & (1 << CAP_IPC_LOCK) != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_schedstat_fields() {
        assert_eq!(
            parse_schedstat(b"123456789 987654321 42\n"),
            Some((123_456_789, 987_654_321))
        );
        assert_eq!(parse_schedstat(b"garbage"), None);
        assert_eq!(parse_schedstat(b""), None);
    }

    #[test]
    fn readings_follow_the_counters() {
        let late = Duration::from_millis(60);
        let preempted = Delta {
            wait: Duration::from_millis(58),
            nivcsw: 1,
            ..Delta::default()
        };
        assert_eq!(preempted.reading(late), Reading::Preempted);
        assert!(preempted.describe(late).contains("reads as preempted"));
        assert!(preempted.describe(late).contains("1 involuntary switch —"));

        let faulted = Delta {
            wait: Duration::from_millis(58),
            minflt: 3,
            ..Delta::default()
        };
        assert_eq!(faulted.reading(late), Reading::PageFault);

        let blocked = Delta {
            wait: Duration::from_micros(200),
            nvcsw: 1,
            ..Delta::default()
        };
        assert_eq!(blocked.reading(late), Reading::KernelStall);
        assert!(blocked.describe(late).contains("0 involuntary switches"));
    }

    #[test]
    fn probe_samples_this_thread_and_reports_deltas() {
        let mut probe = StallProbe::open();
        // First sample has no baseline.
        assert_eq!(probe.sample(), Delta::default());
        // Burn a little CPU and fault a page so the counters move.
        let mut v = vec![0u8; 1 << 20];
        for (i, b) in v.iter_mut().enumerate().step_by(4096) {
            *b = i as u8;
        }
        std::hint::black_box(&v);
        let delta = probe.sample();
        // A fresh 1 MiB mapping faults in as it is touched (THP or not).
        assert!(delta.minflt > 0, "{delta:?}");
        assert_eq!(delta.reading(Duration::from_millis(5)), Reading::PageFault);
    }

    /// Per-tick cost of the probe (one `pread` + one `getrusage`), for the
    /// budget note in `serve.rs`: `cargo test --release stall -- --ignored
    /// --nocapture`. A few µs on an x86 dev box; the 240 Hz tick has 4.17 ms.
    #[test]
    #[ignore]
    fn probe_cost_per_sample() {
        let mut probe = StallProbe::open();
        probe.sample();
        let n = 20_000;
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(probe.sample());
        }
        let per = t0.elapsed() / n;
        println!("stall probe: {per:?} per sample");
        assert!(per < Duration::from_micros(200), "{per:?}");
    }
}
