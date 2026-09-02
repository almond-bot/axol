//! The realtime core: own the CAN buses and run the control loop, driven by
//! impedance targets streamed from Python over a Unix socket.
//!
//! Python keeps the slow model math — VR, IK, target shaping, MuJoCo
//! gravity/inertia, and the pose *scheduling* of the fast terms (all of
//! `AxolArm.motion_control`) — and ships per-joint tuples at its own rate
//! (~120 Hz). This loop owns the wire and the *fast* physics, all computed
//! per tick from its own trajectory and feedback states:
//!
//! - a velocity/acceleration-limited tracker (`filter::Trapezoid`, the
//!   golden-ported `TrapezoidalFilter`) chases the latest streamed target
//!   at `loop_hz`, replacing linear segment interpolation — its position
//!   renders the wire trajectory;
//! - friction (`filter::friction`, per-joint params from the config) and
//!   inertia (`j_eff` streamed pose-scaled per target) feedforwards from
//!   low-pass derivatives of that trajectory. This preserves the classic
//!   Python command chain and prevents the 120 Hz target staircase from
//!   becoming an alternating 240 Hz acceleration torque;
//! - the host-damping torque — band-passed velocity damping from the latest
//!   feedback, using the pose-scheduled coefficients streamed with each
//!   target and reaching the wire within one core tick;
//! - and the last target is held (tracker converges and stays, damping
//!   live) when targets stop arriving.
//!
//! On robot-sized hosts the launcher reserves two CPUs for CAN and exports
//! one assignment per side. Each bus thread pins itself before opening its
//! interface and requests SCHED_FIFO priority when permitted, keeping camera,
//! IK, and dataset load from stretching the damping loop's phase delay.
//!
//! Targets carry a mode flag: gravity-comp / hold flows stream
//! *passthrough* targets (`mode 0`) that bypass the tracker and the
//! friction/inertia terms — a hand-guided limp arm needs `v_des = 0` and
//! model gravity only.
//!
//! Damping lives here, not in Python, because damping is a phase race: the
//! remote chain (Python's 120 Hz sample → socket → adoption wait → a
//! stretched interpolation segment) added ~14 ms between measuring a
//! velocity and the counter-torque reaching the wire. On top of the loop's
//! intrinsic lags that pushed the shoulder burst band (4-9 Hz) past 90° —
//! where a damper stops damping and *pumps* the mode. That was measured on
//! hardware as violent shaking in rt teleop (2026-08-27; see the
//! dissipated-power test in `filter.rs`). In-core the torque applies within
//! one 240 Hz tick of the feedback it acts on. It also keeps damping active
//! through every core-owned hold (watchdog starvation, orphaned client) —
//! frozen-`t_ff` holds used to leave the shoulders with firmware kd only,
//! which is a 62%-overshoot ring at their tuned kp.
//!
//! ## Protocol (length-prefixed messages: u32 LE size, then payload)
//!
//! Python -> Rust:
//! - `C` + text        config: `loop_hz`/`watchdog_ms`/`max_step_rad`/
//!                     `abort_deg` keys, one `joint <side> <iface> <name>
//!                     <motor_id> <kp> <kd> <max_vel> <max_accel> <fc> <k>
//!                     <fv> <fo> <tau_cap>` line per arm joint (tracker
//!                     limits, friction params, spring-torque cap in Nm or
//!                     `inf`), and an optional `gripper <side> <iface>
//!                     <motor_id>` line
//! - `P`               prep: MyActuator 0x76 reset + settle, Damiao
//!                     clear-errors (torque-neutral; run *before* Python
//!                     resolves joint offsets, so the wrap state it verifies
//!                     is the post-reset one; the gripper is never touched)
//! - `A`               arm: bring-up, enable, hold current pose (the
//!                     gripper must already be enabled + calibrated in
//!                     POSITION_FORCE mode by the Python side)
//! - `T` + binary      target: side u8, seq u32 LE, 8 x 9 f64 LE — slots
//!                     0-6 are arm-joint tuples (p_des, mode, kp, kd,
//!                     t_ff, kd_host, damp_w0, damp_q, j_eff) where mode
//!                     ≥ 0.5 runs the tracker + friction/inertia terms
//!                     (teleop) and mode 0 is passthrough (gravity comp);
//!                     t_ff carries the *slow* model feedforward (gravity
//!                     only in tracked mode — friction/inertia/damping are
//!                     computed in-core), kd_host/damp_w0/damp_q are the
//!                     pose-scheduled damping coefficients, and j_eff is
//!                     the pose-scaled inertia feedforward gain (Nm·s²/rad);
//!                     slot 7 is the gripper (p_des motor-frame, max_speed
//!                     rad/s, max_torque Nm, then six zeros)
//! - `R` + binary      flight-recorder gate: enabled u8 and, on enable, the
//!                     Python monotonic timestamp f64 LE. A rising edge
//!                     truncates the previous segment and starts a gated
//!                     `AXOL_RT_TRACE`; disable stops adding rows.
//! - `D`               disarm: disable motors, threads exit
//!
//! Rust -> Python:
//! - `S` + text        state/fault message
//! - `L` + text        log line
//! - `F` + binary      telemetry, one per bus per tick while armed: side
//!                     u8, valid-mask u8, then 8 x (pos f64, vel f64, tau
//!                     f64, age_us u32) — the latest decoded feedback per
//!                     slot. Python fills its Motor caches from these (see
//!                     `build_feedback`); it does not read CAN while the
//!                     core is armed.
//!
//! ## Safety
//! - Targets stepping more than `max_step_rad` from the previous target
//!   are rejected (counted, reported) — corruption defense; the Python
//!   side has its own max-step gate. The gripper slot is exempt (its
//!   targets legitimately jump, matching the Python gate). Whatever gets
//!   through, the tracker's velocity/acceleration limits bound what the
//!   wire can ever see.
//! - A joint deviating more than `abort_deg` from its *commanded* position
//!   (the tracker output) disables both buses (e.g. a collision or a
//!   runaway). The gripper is exempt — stalling against an object is its
//!   normal operation.
//! - Joints with a finite `tau_cap` (the wrists) never send a position
//!   more than `tau_cap / kp` from their last measured one
//!   (`filter::cap_spring`), bounding the impedance spring torque a blocked
//!   joint develops. Being blocked is therefore *allowed* for them: the
//!   commanded position stays within that window of the measured one, so
//!   the deviation abort only ever sees the window, and a runaway is
//!   bounded by `tau_cap` rather than caught.
//! - Every command batch accepts exactly one fresh reply per motor. A missed
//!   sample suppresses host damping for that tick; bursty loss (4 of the last
//!   32 ticks) marks the joint *degraded* — host damping stays off until a
//!   clean 32-tick window, the transition is logged, and the loop keeps
//!   running on firmware kd. Only a motor silent for a full second faults
//!   both buses. Clustered late ticks still fail closed, and a full-cycle
//!   overrun is an immediate fault.
//! - The gripper is not commanded at all until the first target arrives
//!   (matching classic mode, where it sits idle until motion_control).
//! - Watchdog: no target for `watchdog_ms` holds the last target (the
//!   tracker converges and stays, damping live). The arms keep holding —
//!   matching what the firmware itself does if the host dies — until a
//!   disarm or an operator e-stop.
//! - SIGINT/SIGTERM disable everything before exit.

use std::io::{self, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::bringup::{self, MotorSpec, Vendor};
use crate::can::CanSock;
use crate::filter::{self, BandPass, LpDiff, Trapezoid};
use crate::hold::sleep_until;
use crate::proto;
use crate::safety::{guarded_send, purge_tx_queue, SendOutcome, STALL_DETECT};

/// Pole (rad/s) of the motor-facing command derivatives — `CUTOFF_FREQ` in
/// `almond_axol.robot.control`.  The slow pole keeps target-rate steps out of
/// the friction and especially the inertia feedforward torque.
const CONTROL_CUTOFF: f64 = 20.0;

/// Pole (rad/s) of the damping chain's differentiators — `VEL_CUTOFF_FREQ`
/// in `almond_axol.robot.control`: well above the shoulder resonance so the
/// damping arrives in phase; the band-pass supplies the high-side rolloff.
const VEL_CUTOFF: f64 = 80.0;

/// Target-tuple slots per arm: 7 arm joints + the gripper.
const N_SLOTS: usize = 8;
const GRIPPER_SLOT: usize = 7;
/// Rolling feedback loss at or above this many misses in the last 32 ticks
/// (12.5% over 133 ms at 240 Hz) marks a joint *degraded*: its host damping
/// stays off until a full clean window has passed, and the transition is
/// logged. It is not a fault. Host damping is already suppressed on every
/// tick without a fresh sample, so a missed frame can never feed stale
/// velocity into the damping term; the arm simply runs on firmware kd for
/// the lossy stretch. Bursty loss is routine while cameras, IK compilation,
/// and the dataset writer boot on the same host and USB fabric as the CAN
/// adapters — disabling both arms for it ended otherwise healthy sessions.
const DEGRADED_RECENT_MISSED_FEEDBACK: u32 = 4;
/// A motor that has not replied for this long is treated as gone rather than
/// lossy. Commanding it blind leaves the deviation abort inactive for that
/// joint, so past this point the core fails closed. Matches the TX-stall
/// e-stop detection window rather than the old 12.5 ms.
const SILENT_FEEDBACK_FAULT: Duration = Duration::from_secs(1);
/// Degraded/recovered transitions are logged at most this often per bus so a
/// joint flapping across the threshold during boot cannot flood the log; the
/// five-second stats line carries the cumulative count regardless.
const DEGRADED_LOG_INTERVAL: Duration = Duration::from_secs(5);
/// Leave a small slice of each cycle for telemetry handoff and the next
/// absolute sleep. The rest is valid reply time; the old 80% window discarded
/// delayed USB-CAN replies despite there still being cycle headroom.
const REPLY_GUARD: Duration = Duration::from_micros(150);
/// A tick starting more than this far past its deadline is phase-degraded. Its
/// host damping is suppressed even when feedback itself is fresh.
const LATE_TICK: Duration = Duration::from_micros(500);
const MAX_RECENT_LATE_TICKS: u32 = 8;

static SHUTDOWN: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    SHUTDOWN.store(true, Ordering::SeqCst);
}

/// Put one CAN loop on its reserved CPU and request a real-time scheduler.
///
/// CPU isolation is mandatory when the launcher supplied an assignment: a
/// failure means the process topology is not the one Python planned, so it is
/// safer to refuse to arm than to silently recreate collection-time jitter.
/// SCHED_FIFO is mandatory when the launcher requests it: a normal Linux
/// timeslice can exceed the entire 240 Hz period even on a dedicated CPU.
/// Production runs as a privileged service; development builds receive only
/// `CAP_SYS_NICE` via `axol rt.install` / the documented `setcap` command.
fn configure_bus_scheduling(
    iface: &str,
    side: u8,
    out_tx: &mpsc::Sender<Vec<u8>>,
) -> io::Result<()> {
    let cpu_key = if side == 0 {
        "AXOL_RT_CPU_LEFT"
    } else {
        "AXOL_RT_CPU_RIGHT"
    };
    if let Ok(raw) = std::env::var(cpu_key) {
        let cpu: usize = raw.parse().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{iface}: invalid {cpu_key}={raw:?}"),
            )
        })?;
        let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
        unsafe {
            libc::CPU_ZERO(&mut set);
            libc::CPU_SET(cpu, &mut set);
        }
        let rc = unsafe {
            libc::sched_setaffinity(
                0,
                std::mem::size_of::<libc::cpu_set_t>(),
                &set as *const libc::cpu_set_t,
            )
        };
        if rc != 0 {
            return Err(io::Error::other(format!(
                "{iface}: could not pin CAN loop to CPU {cpu}: {}",
                io::Error::last_os_error()
            )));
        }
        send_text(
            out_tx,
            b'L',
            &format!("{iface}: CAN loop isolated on CPU {cpu}"),
        );
    }

    if let Ok(raw) = std::env::var("AXOL_RT_FIFO_PRIORITY") {
        let priority: libc::c_int = raw.parse().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{iface}: invalid AXOL_RT_FIFO_PRIORITY={raw:?}"),
            )
        })?;
        let min = unsafe { libc::sched_get_priority_min(libc::SCHED_FIFO) };
        let max = unsafe { libc::sched_get_priority_max(libc::SCHED_FIFO) };
        if priority < min || priority > max {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{iface}: SCHED_FIFO priority {priority} outside {min}..={max}"),
            ));
        }
        let param = libc::sched_param {
            sched_priority: priority,
        };
        let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) };
        if rc == 0 {
            send_text(
                out_tx,
                b'L',
                &format!("{iface}: CAN loop SCHED_FIFO priority {priority}"),
            );
        } else {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "{iface}: cannot enter SCHED_FIFO priority {priority}: {}; refusing to arm without real-time scheduling (run `axol rt.install` or grant CAP_SYS_NICE to axol-rt)",
                    io::Error::last_os_error()
                ),
            ));
        }
    }
    Ok(())
}

/// Schedule the next batch from the instant this batch actually began.
///
/// A deadline based on the old absolute grid compresses the interval following
/// any late wake: 1.5 ms late at 240 Hz would make the next command gap only
/// 2.67 ms. Motors cannot recover elapsed control time, and that shortened
/// feedback/command phase can turn host damping into excitation. A relative
/// start-to-start period gives up an unobservable amount of wall-clock phase
/// instead: lateness can lower the average rate briefly, but can never produce
/// a catch-up command faster than the configured rate.
fn next_bus_deadline(began: Instant, period: Duration) -> Instant {
    began + period
}

/// Accept at most one reply from each motor commanded in this tick.
///
/// CAN frames carry no command sequence number. The bus loop therefore drains
/// late frames before sending and uses this per-batch set to prevent duplicate
/// or unsolicited feedback from satisfying another motor's reply budget.
fn mark_unique_expected_reply(expected: &[bool], seen: &mut [bool], idx: usize) -> bool {
    if idx >= expected.len() || idx >= seen.len() || !expected[idx] || seen[idx] {
        return false;
    }
    seen[idx] = true;
    true
}

/// Outcome of one feedback opportunity for one arm joint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FeedbackVerdict {
    /// Nothing to report: healthy, an isolated miss, or an ongoing degraded
    /// stretch that has neither cleared nor gone silent.
    Steady,
    /// Rolling loss just crossed the degraded threshold.
    Degraded,
    /// A full clean window just closed out a degraded stretch.
    Recovered,
    /// The motor has not replied for the silent-fault interval: treat it as
    /// gone and fail closed.
    Silent,
}

#[derive(Clone, Copy, Default)]
struct FeedbackHealth {
    consecutive_misses: u32,
    recent_misses: u32,
    degraded: bool,
}

impl FeedbackHealth {
    /// Record one 240 Hz feedback opportunity. `silent_limit` is the
    /// consecutive-miss count that turns loss into a fault.
    ///
    /// Degradation has hysteresis: it starts at
    /// `DEGRADED_RECENT_MISSED_FEEDBACK` misses in the 32-tick window and only
    /// clears once that window is entirely clean, so a joint flapping around
    /// the threshold does not toggle host damping every few ticks.
    fn record(&mut self, received: bool, silent_limit: u32) -> FeedbackVerdict {
        self.recent_misses = (self.recent_misses << 1) | u32::from(!received);
        if received {
            self.consecutive_misses = 0;
        } else {
            self.consecutive_misses = self.consecutive_misses.saturating_add(1);
        }
        if self.consecutive_misses >= silent_limit {
            return FeedbackVerdict::Silent;
        }
        let lossy = self.recent_misses.count_ones() >= DEGRADED_RECENT_MISSED_FEEDBACK;
        match (self.degraded, lossy) {
            (false, true) => {
                self.degraded = true;
                FeedbackVerdict::Degraded
            }
            (true, false) if self.recent_misses == 0 => {
                self.degraded = false;
                FeedbackVerdict::Recovered
            }
            _ => FeedbackVerdict::Steady,
        }
    }
}

/// Consecutive missed replies that constitute a silent motor at `loop_hz`.
fn silent_feedback_limit(loop_hz: f64) -> u32 {
    (loop_hz * SILENT_FEEDBACK_FAULT.as_secs_f64())
        .ceil()
        .max(1.0) as u32
}

#[derive(Clone, Copy, Default)]
struct TimingHealth {
    recent_late: u32,
    consecutive_late: u8,
}

impl TimingHealth {
    fn record(&mut self, on_time: bool) -> bool {
        self.recent_late = (self.recent_late << 1) | u32::from(!on_time);
        if on_time {
            self.consecutive_late = 0;
        } else {
            self.consecutive_late = self.consecutive_late.saturating_add(1);
        }
        self.consecutive_late >= 3 || self.recent_late.count_ones() >= MAX_RECENT_LATE_TICKS
    }

    /// Record this tick before applying the immediate full-cycle limit.  The
    /// ordering matters: the fault diagnostic must include the tick that
    /// triggered it, even when `lateness >= period` is already sufficient to
    /// stop the loop.
    fn record_lateness(&mut self, lateness: Duration, period: Duration) -> bool {
        let clustered_lateness = self.record(lateness <= LATE_TICK);
        lateness >= period || clustered_lateness
    }
}

type BusStartGate = (Mutex<Option<Instant>>, Condvar);

#[derive(Clone, Copy, Debug, Default)]
pub struct JointCmd {
    pub p_des: f64,
    /// ≥ 0.5: tracked mode — the in-core trapezoid chases `p_des` and the
    /// friction/inertia feedforwards apply. 0: passthrough — `p_des` goes
    /// to the wire as-is with `v_des = 0` (gravity comp, bring-up hold).
    /// The gripper slot repurposes this field as its max_speed (rad/s).
    pub mode: f64,
    /// The gripper slot repurposes `kp` as its max torque (Nm).
    pub kp: f64,
    pub kd: f64,
    /// Slow model feedforward (gravity only in tracked mode) — friction,
    /// inertia, and damping are computed in-core each tick and added.
    pub t_ff: f64,
    /// Effective (pose-scheduled) host damping gain, Nm·s/rad.
    pub kd_host: f64,
    /// Damping band-pass centre (rad/s) and quality factor.
    pub damp_w0: f64,
    pub damp_q: f64,
    /// Pose-scaled inertia feedforward gain (Nm·s²/rad), applied to the
    /// low-pass acceleration derivative of tracker position in tracked mode.
    pub j_eff: f64,
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

/// One fixed-size diagnostic sample passed from the realtime bus loop to a
/// background CSV writer. Enabled only when `AXOL_RT_TRACE` is set; keeping
/// formatting and disk I/O off the bus thread makes tracing safe to leave on
/// while reproducing a timing-sensitive vibration.
#[derive(Clone, Copy, Default)]
struct TraceRow {
    tick: u64,
    time_s: f64,
    seq: u32,
    slot: usize,
    motor_id: u8,
    mode: f64,
    target_p: f64,
    cmd_p: f64,
    cmd_v: f64,
    cmd_a: f64,
    cmd_v_fast: f64,
    meas_p: f64,
    motor_v: f64,
    meas_v: f64,
    meas_tau: f64,
    gravity_ff: f64,
    friction_ff: f64,
    inertia_ff: f64,
    damping_ff: f64,
    total_ff: f64,
    kd_host: f64,
    damp_w0: f64,
    damp_q: f64,
    tick_dt: f64,
    fb_dt: f64,
}

type TraceHandle = JoinHandle<io::Result<()>>;

enum TraceMsg {
    /// Discard the previous engage segment and start the file over.  This
    /// mirrors the Python flight recorder's latest-segment semantics.
    Reset,
    Row(TraceRow),
}

fn trace_file(path: &PathBuf) -> io::Result<io::BufWriter<std::fs::File>> {
    let mut out = io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(
        out,
        "tick,time_s,seq,slot,motor_id,mode,target_p,cmd_p,cmd_v,cmd_a,cmd_v_fast,meas_p,motor_v,meas_v,meas_tau,gravity_ff,friction_ff,inertia_ff,damping_ff,total_ff,kd_host,damp_w0,damp_q,tick_dt,fb_dt"
    )?;
    Ok(out)
}

fn write_trace_row(out: &mut io::BufWriter<std::fs::File>, r: TraceRow) -> io::Result<()> {
    writeln!(
        out,
        "{},{:.9},{},{},{},{:.1},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.9},{:.9}",
        r.tick,
        r.time_s,
        r.seq,
        r.slot,
        r.motor_id,
        r.mode,
        r.target_p,
        r.cmd_p,
        r.cmd_v,
        r.cmd_a,
        r.cmd_v_fast,
        r.meas_p,
        r.motor_v,
        r.meas_v,
        r.meas_tau,
        r.gravity_ff,
        r.friction_ff,
        r.inertia_ff,
        r.damping_ff,
        r.total_ff,
        r.kd_host,
        r.damp_w0,
        r.damp_q,
        r.tick_dt,
        r.fb_dt,
    )
}

fn parse_cpu_set(raw: &str, source: &str) -> io::Result<libc::cpu_set_t> {
    if raw.trim().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid {source}: CPU set is empty"),
        ));
    }
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    unsafe { libc::CPU_ZERO(&mut set) };
    for item in raw.split(',') {
        let item = item.trim();
        let cpu: usize = item.parse().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid {source}={raw:?}: expected comma-separated CPU numbers"),
            )
        })?;
        if cpu >= libc::CPU_SETSIZE as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid {source}={raw:?}: CPU {cpu} is out of range"),
            ));
        }
        unsafe { libc::CPU_SET(cpu, &mut set) };
    }
    Ok(set)
}

/// A trace writer is throughput work, never realtime work. Writer threads
/// are spawned before their parent bus thread changes its affinity, and this
/// is a second fail-safe against a future call-site move: shed any inherited
/// realtime policy and move to the launcher's background cores before opening
/// or writing the trace file.
fn configure_trace_writer_scheduling(affinity: Option<&libc::cpu_set_t>) -> io::Result<()> {
    let param = libc::sched_param { sched_priority: 0 };
    let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_OTHER, &param) };
    if rc != 0 {
        return Err(io::Error::other(format!(
            "could not put trace writer under SCHED_OTHER: {}",
            io::Error::last_os_error()
        )));
    }
    if let Some(set) = affinity {
        let rc = unsafe {
            libc::sched_setaffinity(
                0,
                std::mem::size_of::<libc::cpu_set_t>(),
                set as *const libc::cpu_set_t,
            )
        };
        if rc != 0 {
            return Err(io::Error::other(format!(
                "could not pin trace writer to background CPUs: {}",
                io::Error::last_os_error()
            )));
        }
    }
    Ok(())
}

fn start_trace_writer(
    side: u8,
) -> io::Result<Option<(mpsc::SyncSender<TraceMsg>, TraceHandle, PathBuf)>> {
    let Ok(prefix) = std::env::var("AXOL_RT_TRACE") else {
        return Ok(None);
    };
    if prefix.trim().is_empty() {
        return Ok(None);
    }
    let side_name = if side == 0 { "left" } else { "right" };
    let path = PathBuf::from(format!("{prefix}-{side_name}.csv"));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let affinity = match std::env::var("AXOL_RT_BACKGROUND_CPUS") {
        Ok(raw) => Some(parse_cpu_set(&raw, "AXOL_RT_BACKGROUND_CPUS")?),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "AXOL_RT_BACKGROUND_CPUS is not valid UTF-8",
            ));
        }
    };
    // About 17 seconds of headroom per arm at 240 Hz x 7 joints. A full
    // channel never blocks the control loop: samples are dropped and counted.
    let (tx, rx) = mpsc::sync_channel::<TraceMsg>(28_000);
    // Do not report tracing as active until the writer has shed any inherited
    // realtime policy and moved off the control CPUs. Failure disables the
    // optional trace rather than weakening motor-loop isolation.
    let (setup_tx, setup_rx) = mpsc::sync_channel::<io::Result<()>>(0);
    let writer_path = path.clone();
    let handle = std::thread::spawn(move || -> io::Result<()> {
        if let Err(err) = configure_trace_writer_scheduling(affinity.as_ref()) {
            let _ = setup_tx.send(Err(err));
            return Ok(());
        }
        if setup_tx.send(Ok(())).is_err() {
            return Ok(());
        }
        let mut out = trace_file(&writer_path)?;
        for msg in rx {
            match msg {
                TraceMsg::Reset => {
                    out.flush()?;
                    out = trace_file(&writer_path)?;
                }
                TraceMsg::Row(row) => write_trace_row(&mut out, row)?,
            }
        }
        out.flush()
    });
    match setup_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            let _ = handle.join();
            return Err(err);
        }
        Err(_) => {
            return match handle.join() {
                Ok(Err(err)) => Err(err),
                Ok(Ok(())) => Err(io::Error::other(
                    "trace writer exited before configuring its scheduler",
                )),
                Err(_) => Err(io::Error::other(
                    "trace writer panicked while configuring its scheduler",
                )),
            };
        }
    }
    Ok(Some((tx, handle, path)))
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
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("config: bad line: {line}"),
        )
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        match f[0] {
            "loop_hz" => {
                loop_hz = f
                    .get(1)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| bad(line))?
            }
            "watchdog_ms" => {
                watchdog_ms = f
                    .get(1)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| bad(line))?
            }
            "max_step_rad" => {
                max_step_rad = f
                    .get(1)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| bad(line))?
            }
            "abort_deg" => {
                abort_deg = f
                    .get(1)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| bad(line))?
            }
            "joint" | "gripper" => {
                // joint <side 0|1> <iface> <name> <motor_id> <kp> <kd>
                //       <max_vel> <max_accel> <fc> <k> <fv> <fo> <tau_cap>
                // gripper <side 0|1> <iface> <motor_id>
                let gripper = f[0] == "gripper";
                let side: u8 = f
                    .get(1)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| bad(line))?;
                let iface = f.get(2).ok_or_else(|| bad(line))?.to_string();
                let bus = match buses.iter_mut().find(|(s, i, _)| *s == side && *i == iface) {
                    Some(bus) => bus,
                    None => {
                        buses.push((side, iface.clone(), Vec::new()));
                        buses.last_mut().unwrap()
                    }
                };
                let num = |i: usize| -> io::Result<f64> {
                    f.get(i)
                        .and_then(|v| v.parse().ok())
                        .ok_or_else(|| bad(line))
                };
                let spec = if gripper {
                    MotorSpec {
                        joint: "gripper".to_string(),
                        motor_id: f
                            .get(3)
                            .and_then(|v| v.parse().ok())
                            .ok_or_else(|| bad(line))?,
                        kp: 0.0,
                        kd: 0.0,
                        gripper: true,
                        slot: GRIPPER_SLOT,
                        max_vel: 0.0,
                        max_accel: 0.0,
                        fc: 0.0,
                        k: 0.0,
                        fv: 0.0,
                        fo: 0.0,
                        tau_cap: f64::INFINITY,
                    }
                } else {
                    MotorSpec {
                        joint: f.get(3).ok_or_else(|| bad(line))?.to_string(),
                        motor_id: f
                            .get(4)
                            .and_then(|v| v.parse().ok())
                            .ok_or_else(|| bad(line))?,
                        kp: num(5)?,
                        kd: num(6)?,
                        gripper: false,
                        // Arm joints arrive in Joint enum order per bus.
                        slot: bus.2.iter().filter(|s| !s.gripper).count(),
                        max_vel: num(7)?,
                        max_accel: num(8)?,
                        fc: num(9)?,
                        k: num(10)?,
                        fv: num(11)?,
                        fo: num(12)?,
                        // "inf" parses to +infinity = uncapped. A NaN or
                        // non-positive cap would clamp every command onto
                        // the measured position (a joint that cannot move).
                        tau_cap: match num(13)? {
                            cap if cap > 0.0 => cap,
                            _ => return Err(bad(line)),
                        },
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
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "config: no joints",
        ));
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
    // side u8, seq u32, 8 slots x 9 f64
    let expected = 1 + 4 + N_SLOTS * 9 * 8;
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
        let mut vals = [0.0f64; 9];
        for v in &mut vals {
            *v = f64::from_le_bytes(payload[off..off + 8].try_into().unwrap());
            off += 8;
        }
        *cmd = JointCmd {
            p_des: vals[0],
            mode: vals[1],
            kp: vals[2],
            kd: vals[3],
            t_ff: vals[4],
            kd_host: vals[5],
            damp_w0: vals[6],
            damp_q: vals[7],
            j_eff: vals[8],
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

fn parse_record_gate(payload: &[u8]) -> io::Result<Option<f64>> {
    match payload {
        [0] => Ok(None),
        [1, timestamp @ ..] if timestamp.len() == 8 => {
            let mut raw = [0u8; 8];
            raw.copy_from_slice(timestamp);
            Ok(Some(f64::from_le_bytes(raw)))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "record gate: expected enabled byte plus optional f64 timestamp",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors `RtLink.send_target`'s packing: side u8, seq u32 LE, then
    /// 8 slots x 9 f64 LE.
    #[test]
    fn parse_target_roundtrip() {
        let mut payload = vec![1u8];
        payload.extend_from_slice(&0xDEADBEEFu32.to_le_bytes());
        for slot in 0..N_SLOTS {
            for field in 0..9 {
                let v = slot as f64 * 10.0 + field as f64;
                payload.extend_from_slice(&v.to_le_bytes());
            }
        }
        let (side, t) = parse_target(&payload).unwrap();
        assert_eq!(side, 1);
        assert_eq!(t.seq, 0xDEADBEEF);
        let c = &t.cmds[2];
        assert_eq!(
            (c.p_des, c.mode, c.kp, c.kd, c.t_ff, c.kd_host, c.damp_w0, c.damp_q, c.j_eff),
            (20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0)
        );
        // Wrong size (the previous 8-field layout) must be rejected, not
        // misparsed — a version-skewed client fails loudly.
        assert!(parse_target(&payload[..1 + 4 + N_SLOTS * 8 * 8]).is_err());
    }

    #[test]
    fn parse_record_gate_roundtrip() {
        let timestamp = 87_419.125f64;
        let mut enabled = vec![1];
        enabled.extend_from_slice(&timestamp.to_le_bytes());
        assert_eq!(parse_record_gate(&enabled).unwrap(), Some(timestamp));
        assert_eq!(parse_record_gate(&[0]).unwrap(), None);
        assert!(parse_record_gate(&[1]).is_err());
        assert!(parse_record_gate(&[0, 0]).is_err());
    }

    #[test]
    fn bus_deadline_never_catches_up_after_overrun() {
        let base = Instant::now();
        let period = Duration::from_millis(4);
        assert_eq!(
            next_bus_deadline(base + Duration::from_millis(2), period),
            base + Duration::from_millis(6)
        );
    }

    #[test]
    fn replies_must_be_expected_and_unique() {
        let expected = [true, true, false];
        let mut seen = [false; 3];
        assert!(mark_unique_expected_reply(&expected, &mut seen, 0));
        assert!(!mark_unique_expected_reply(&expected, &mut seen, 0));
        assert!(!mark_unique_expected_reply(&expected, &mut seen, 2));
        assert!(mark_unique_expected_reply(&expected, &mut seen, 1));
        assert_eq!(seen, [true, true, false]);
    }

    #[test]
    fn feedback_health_degrades_on_bursty_loss_and_recovers_after_clean_window() {
        let limit = silent_feedback_limit(240.0);
        assert_eq!(limit, 240);

        // The startup pattern from the field: isolated single-tick misses.
        // Three of them stay quiet; the fourth degrades the joint, and it
        // remains degraded (Steady, not re-announced) while lossy.
        let mut bursty = FeedbackHealth::default();
        for _ in 0..3 {
            assert_eq!(bursty.record(false, limit), FeedbackVerdict::Steady);
            assert_eq!(bursty.record(true, limit), FeedbackVerdict::Steady);
        }
        assert_eq!(bursty.record(false, limit), FeedbackVerdict::Degraded);
        assert!(bursty.degraded);
        assert_eq!(bursty.record(false, limit), FeedbackVerdict::Steady);
        assert!(bursty.degraded);

        // Hysteresis: dropping below four misses is not enough; the window
        // must be entirely clean before damping is allowed back on.
        for _ in 0..31 {
            assert_eq!(bursty.record(true, limit), FeedbackVerdict::Steady);
            assert!(bursty.degraded);
        }
        assert_eq!(bursty.record(true, limit), FeedbackVerdict::Recovered);
        assert!(!bursty.degraded);

        // The old 3-consecutive trip is now just a degraded stretch...
        let mut consecutive = FeedbackHealth::default();
        for _ in 0..3 {
            assert_ne!(consecutive.record(false, limit), FeedbackVerdict::Silent);
        }
        // ...and only a motor silent for the whole interval faults.
        for _ in 3..limit - 1 {
            assert_ne!(consecutive.record(false, limit), FeedbackVerdict::Silent);
        }
        assert_eq!(consecutive.record(false, limit), FeedbackVerdict::Silent);

        let mut healthy = FeedbackHealth::default();
        for tick in 0..128 {
            assert_eq!(
                healthy.record(tick % 32 != 0, limit),
                FeedbackVerdict::Steady
            );
            assert!(!healthy.degraded);
        }
    }

    #[test]
    fn timing_health_catches_clustered_late_ticks() {
        let mut health = TimingHealth::default();
        for _ in 0..7 {
            assert!(!health.record(false));
            assert!(!health.record(true));
        }
        assert!(health.record(false));
    }

    #[test]
    fn timing_health_counts_immediate_full_cycle_fault() {
        let mut health = TimingHealth::default();
        let period = Duration::from_micros(4_167);
        assert!(health.record_lateness(period, period));
        assert_eq!(health.recent_late.count_ones(), 1);
        assert_eq!(health.consecutive_late, 1);
    }

    #[test]
    fn trace_writer_forces_normal_scheduling_and_requested_affinity() {
        let (policy, affinity_ok) = std::thread::spawn(|| {
            let mut available = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
            let rc = unsafe {
                libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut available)
            };
            assert_eq!(rc, 0, "{}", io::Error::last_os_error());
            let cpu = (0..libc::CPU_SETSIZE as usize)
                .find(|&cpu| unsafe { libc::CPU_ISSET(cpu, &available) })
                .expect("test thread has no available CPU");
            let requested = parse_cpu_set(&cpu.to_string(), "test CPU set").unwrap();

            configure_trace_writer_scheduling(Some(&requested)).unwrap();

            let mut applied = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
            let rc = unsafe {
                libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut applied)
            };
            assert_eq!(rc, 0, "{}", io::Error::last_os_error());
            let only_requested_cpu = (0..libc::CPU_SETSIZE as usize)
                .filter(|&candidate| unsafe { libc::CPU_ISSET(candidate, &applied) })
                .eq(std::iter::once(cpu));
            (unsafe { libc::sched_getscheduler(0) }, only_requested_cpu)
        })
        .join()
        .unwrap();
        assert_eq!(policy, libc::SCHED_OTHER);
        assert!(affinity_ok);
        assert!(parse_cpu_set("", "test CPU set").is_err());
        assert!(parse_cpu_set("0,nope", "test CPU set").is_err());
    }

    /// Live stall-detection check against a real interface whose bus has no
    /// powered nodes (motors off = the e-stop condition). Uses ID 0x7F0 —
    /// unused by both motor protocols — so the frames left in the TX queue
    /// are ignored by every motor if they ever transmit. Run explicitly:
    /// `cargo test stall_detection_live -- --ignored`.
    #[test]
    #[ignore = "needs a live CAN interface with unpowered motors"]
    fn stall_detection_live() {
        let sock = CanSock::open("can_alm_axol_l").expect("open can_alm_axol_l");
        sock.set_send_timeout(Duration::from_millis(20)).unwrap();
        let mut since: Option<Instant> = None;
        let start = Instant::now();
        let mut dropped = 0u32;
        let mut sent = 0u32;
        loop {
            match guarded_send(&sock, 0x7F0, &[0u8; 8], &mut since).unwrap() {
                SendOutcome::Sent => sent += 1,
                SendOutcome::Dropped => dropped += 1,
                SendOutcome::Stalled => break,
            }
            assert!(
                start.elapsed() < Duration::from_secs(15),
                "never stalled (sent {sent}, dropped {dropped})"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
        println!(
            "stalled after {:.2}s: {sent} queued, {dropped} dropped",
            start.elapsed().as_secs_f64()
        );
        assert!(dropped > 0, "expected a Dropped phase before the stall");
        assert!(start.elapsed() >= STALL_DETECT);
    }

    /// Layout contract with `RtLink._parse_feedback`: F-packets are
    /// side u8, mask u8, then 8 x (pos f64, vel f64, tau f64, age_us u32),
    /// all little-endian.
    #[test]
    fn feedback_packet_layout() {
        let now = Instant::now();
        let mut latest: [SlotFeedback; N_SLOTS] = [None; N_SLOTS];
        latest[0] = Some((1.5, -0.25, 3.0, now - Duration::from_micros(1200)));
        latest[7] = Some((0.5, 0.0, 0.1, now));
        let msg = build_feedback(1, &latest, now);
        assert_eq!(msg.len(), 3 + N_SLOTS * 28);
        assert_eq!(msg[0], b'F');
        assert_eq!(msg[1], 1);
        assert_eq!(msg[2], 0b1000_0001);
        let pos0 = f64::from_le_bytes(msg[3..11].try_into().unwrap());
        let vel0 = f64::from_le_bytes(msg[11..19].try_into().unwrap());
        let tau0 = f64::from_le_bytes(msg[19..27].try_into().unwrap());
        let age0 = u32::from_le_bytes(msg[27..31].try_into().unwrap());
        assert_eq!((pos0, vel0, tau0, age0), (1.5, -0.25, 3.0, 1200));
        let slot7 = 3 + 7 * 28;
        let pos7 = f64::from_le_bytes(msg[slot7..slot7 + 8].try_into().unwrap());
        assert_eq!(pos7, 0.5);
    }

    #[test]
    fn parse_config_assigns_slots() {
        let cfg = parse_config(
            "loop_hz 240\n\
             joint 0 canL shoulder_1 1 250 3.5 9.4 33.0 0.6 250 0.15 0.02 inf\n\
             joint 0 canL shoulder_2 2 250 3.5 9.4 33.0 0.5 250 0.10 0.0 inf\n\
             gripper 0 canL 8\n\
             joint 0 canL wrist_2 6 130 3.5 9.4 33.0 0.4 250 0.08 0.0 3.0\n",
        )
        .unwrap();
        let specs = &cfg.buses[0].2;
        assert_eq!(
            specs.iter().map(|s| s.slot).collect::<Vec<_>>(),
            vec![0, 1, GRIPPER_SLOT, 2]
        );
        assert!(specs[2].gripper);
        assert_eq!(specs[0].max_vel, 9.4);
        assert_eq!(specs[0].max_accel, 33.0);
        assert_eq!(
            (specs[0].fc, specs[0].k, specs[0].fv, specs[0].fo),
            (0.6, 250.0, 0.15, 0.02)
        );
        // Python formats an unset torque_limit as "inf" — uncapped.
        assert_eq!(specs[0].tau_cap, f64::INFINITY);
        assert_eq!(specs[3].tau_cap, 3.0);
        assert_eq!(specs[2].tau_cap, f64::INFINITY);
        // A joint line missing the tracker/friction params (the previous
        // 7-field layout) must be rejected, not defaulted.
        assert!(parse_config("joint 0 canL shoulder_1 1 250 3.5\n").is_err());
        // ... as must the 12-field layout without the torque cap, and a cap
        // that would pin the joint to its measured position.
        assert!(
            parse_config("joint 0 canL shoulder_1 1 250 3.5 9.4 33.0 0.6 250 0.15 0.02\n")
                .is_err()
        );
        assert!(parse_config(
            "joint 0 canL shoulder_1 1 250 3.5 9.4 33.0 0.6 250 0.15 0.02 0\n"
        )
        .is_err());
        assert!(parse_config(
            "joint 0 canL shoulder_1 1 250 3.5 9.4 33.0 0.6 250 0.15 0.02 nan\n"
        )
        .is_err());
    }
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
        if stream
            .write_all(&len)
            .and_then(|_| stream.write_all(&msg))
            .is_err()
        {
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

/// Latest decoded feedback for one slot: (position, velocity, torque,
/// receive time).
type SlotFeedback = Option<(f64, f64, f64, Instant)>;

/// Build one telemetry packet: `F`, side u8, valid-mask u8 (bit i = slot i
/// has been seen), then per slot: pos f64, vel f64, tau f64 (all LE) and
/// age_us u32 — microseconds between the frame's CAN receive and this
/// packet, so the Python side can reconstruct per-slot receive timestamps
/// on its own clock. Mirrored by `RtLink._parse_feedback`.
fn build_feedback(side: u8, latest: &[SlotFeedback; N_SLOTS], now: Instant) -> Vec<u8> {
    let mut msg = Vec::with_capacity(3 + N_SLOTS * 28);
    msg.push(b'F');
    msg.push(side);
    let mut mask = 0u8;
    for (i, slot) in latest.iter().enumerate() {
        if slot.is_some() {
            mask |= 1 << i;
        }
    }
    msg.push(mask);
    for slot in latest {
        let (pos, vel, tau, ts) = slot.unwrap_or((0.0, 0.0, 0.0, now));
        msg.extend_from_slice(&pos.to_le_bytes());
        msg.extend_from_slice(&vel.to_le_bytes());
        msg.extend_from_slice(&tau.to_le_bytes());
        let age_us = now.duration_since(ts).as_micros().min(u32::MAX as u128) as u32;
        msg.extend_from_slice(&age_us.to_le_bytes());
    }
    msg
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
    let targets: Arc<[Mutex<TargetSlot>; 2]> = Arc::new([
        Mutex::new(TargetSlot::default()),
        Mutex::new(TargetSlot::default()),
    ]);
    // A normal `--teleop.record` launch sets AXOL_RT_TRACE_GATED and uses
    // `R` messages to retain only the latest engaged segment. A manually set
    // AXOL_RT_TRACE remains the low-level always-on escape hatch.
    let trace_gated = std::env::var_os("AXOL_RT_TRACE_GATED").is_some();
    let trace_enabled = Arc::new(AtomicBool::new(!trace_gated));
    let trace_generation = Arc::new(AtomicU64::new(u64::from(!trace_gated)));
    let trace_origin_bits = Arc::new(AtomicU64::new(0.0f64.to_bits()));
    let stop = Arc::new(AtomicBool::new(false));
    // 0 = running, 1 = fault (set by a bus thread on abort).
    let fault = Arc::new(AtomicU8::new(0));
    let mut bus_threads: Vec<std::thread::JoinHandle<io::Result<()>>> = Vec::new();

    // Errors below `break` out (never early-return): the cleanup after the
    // loop must always run so the bus threads stop and disable the motors —
    // an early `?` here would leave them energized with no owner.
    let mut loop_err: Option<io::Error> = None;
    loop {
        if SHUTDOWN.load(Ordering::SeqCst) {
            break;
        }
        let payload = match read_msg(&mut stream) {
            Ok(p) => p,
            Err(err) => {
                loop_err = Some(err);
                break;
            }
        };
        let Some(payload) = payload else {
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
                let parsed = std::str::from_utf8(body)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
                    .and_then(parse_config);
                match parsed {
                    Ok(cfg) => {
                        config = Some(Arc::new(cfg));
                        send_text(&out_tx, b'S', "config-ok");
                    }
                    Err(err) => {
                        loop_err = Some(err);
                        break;
                    }
                }
            }
            b'P' => {
                let Some(cfg) = &config else {
                    send_text(&out_tx, b'S', "fault: prep before config");
                    continue;
                };
                let mut ok = true;
                for (_, iface, specs) in &cfg.buses {
                    let step = CanSock::open(iface).and_then(|sock| bringup::prep(&sock, specs));
                    if let Err(err) = step {
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
                // Release both buses onto one epoch only after bring-up has
                // completed. Each side then gets half a period of phase
                // separation, preventing both USB-CAN adapters from bursting
                // commands and replies through the same xHCI interrupt at once.
                let start_gate = Arc::new((Mutex::new(None), Condvar::new()));
                for (side, iface, specs) in cfg.buses.clone() {
                    let cfg = Arc::clone(cfg);
                    let targets = Arc::clone(&targets);
                    let stop = Arc::clone(&stop);
                    let fault = Arc::clone(&fault);
                    let trace_enabled = Arc::clone(&trace_enabled);
                    let trace_generation = Arc::clone(&trace_generation);
                    let trace_origin_bits = Arc::clone(&trace_origin_bits);
                    let out_tx = out_tx.clone();
                    let ready_tx = ready_tx.clone();
                    let start_gate = Arc::clone(&start_gate);
                    bus_threads.push(std::thread::spawn(move || {
                        bus_loop(
                            &iface,
                            side,
                            &specs,
                            &cfg,
                            &targets,
                            &stop,
                            &fault,
                            &trace_enabled,
                            &trace_generation,
                            &trace_origin_bits,
                            &out_tx,
                            &ready_tx,
                            &start_gate,
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
                {
                    let (lock, wake) = &*start_gate;
                    *lock.lock().unwrap() = Some(Instant::now() + Duration::from_millis(20));
                    wake.notify_all();
                }
                if ok {
                    send_text(&out_tx, b'S', "armed");
                }
            }
            b'T' => match parse_target(body) {
                Ok((side, target)) => {
                    if side <= 1 {
                        targets[side as usize].lock().unwrap().target = Some(target);
                    }
                }
                Err(err) => {
                    loop_err = Some(err);
                    break;
                }
            },
            b'R' => match parse_record_gate(body) {
                Ok(None) => trace_enabled.store(false, Ordering::Release),
                Ok(Some(timestamp)) => {
                    trace_origin_bits.store(timestamp.to_bits(), Ordering::Release);
                    trace_generation.fetch_add(1, Ordering::AcqRel);
                    trace_enabled.store(true, Ordering::Release);
                }
                Err(err) => {
                    loop_err = Some(err);
                    break;
                }
            },
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

    // Signal, peer loss, or protocol error: stop and disable everything
    // before exit.
    stop.store(true, Ordering::SeqCst);
    for handle in bus_threads {
        let _ = handle.join();
    }
    drop(out_tx);
    let _ = writer.join();
    let _ = std::fs::remove_file(socket_path);
    match loop_err {
        Some(err) => Err(err),
        None => Ok(()),
    }
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
    trace_enabled: &AtomicBool,
    trace_generation: &AtomicU64,
    trace_origin_bits: &AtomicU64,
    out_tx: &mpsc::Sender<Vec<u8>>,
    ready_tx: &mpsc::Sender<io::Result<()>>,
    start_gate: &BusStartGate,
) -> io::Result<()> {
    // Spawn throughput helpers before this thread pins itself to the CAN CPU
    // and enters SCHED_FIFO. Linux threads inherit their creator's scheduler
    // and affinity, so moving this below `configure_bus_scheduling` would let
    // trace flush/truncate work contend directly with the control loop.
    let (trace_tx, trace_handle) = match start_trace_writer(side) {
        Ok(Some((tx, handle, path))) => {
            send_text(
                out_tx,
                b'L',
                &format!("{iface}: RT control trace -> {}", path.display()),
            );
            (Some(tx), Some(handle))
        }
        Ok(None) => (None, None),
        Err(err) => {
            send_text(
                out_tx,
                b'L',
                &format!("{iface}: RT control trace disabled: {err}"),
            );
            (None, None)
        }
    };
    if let Err(err) = configure_bus_scheduling(iface, side, out_tx) {
        let _ = ready_tx.send(Err(err));
        return Ok(());
    }
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
    if let Err(err) = bringup::enable(&sock, iface, &motors) {
        let _ = ready_tx.send(Err(err));
        return Ok(());
    }
    let _ = ready_tx.send(Ok(()));

    // Play state, indexed by target slot: the latest adopted command per
    // slot, starting as a passthrough hold of the measured pose with config
    // gains. The gripper slot's hold values are never sent (it isn't
    // commanded until the first target arrives). kd_host starts at 0 —
    // matching classic mode, where enable() holds on firmware gains until
    // the first motion_control; the first streamed target brings the
    // pose-scheduled coefficients, and from then on damping stays live
    // through every hold (watchdog, orphaned client).
    let mut play: [JointCmd; N_SLOTS] = [JointCmd::default(); N_SLOTS];
    for m in &motors {
        play[m.slot] = JointCmd {
            p_des: m.hold_pos,
            mode: 0.0,
            kp: m.kp,
            kd: m.kd,
            t_ff: 0.0,
            kd_host: 0.0,
            damp_w0: 20.0,
            damp_q: 0.8,
            j_eff: 0.0,
        };
    }
    // The in-core target tracker, per slot: chases the latest streamed
    // target at loop rate under the config vel/accel limits. Its position
    // drives the classic command-derivative chains below; its internal
    // velocity remains the integration state that bounds the trajectory.
    // Seeded at the bring-up hold pose so the first tracked target starts
    // transient-free.
    let mut trk: Vec<Trapezoid> = (0..N_SLOTS).map(|_| Trapezoid::new(0.0, 0.0)).collect();
    for m in &motors {
        trk[m.slot] = Trapezoid::new(m.max_vel, m.max_accel);
        trk[m.slot].seed(m.hold_pos);
    }
    // In-core command derivatives and host damping, per slot.  The tracker
    // position is differentiated through the same slow chains as classic
    // Python before it drives friction/inertia feedforward.  Host damping
    // gets separate fast desired/measured velocity derivatives followed by
    // the resonance band-pass.
    struct Damp {
        v_cmd: LpDiff,
        a_cmd: LpDiff,
        v_cmd_fast: LpDiff,
        v_meas: LpDiff,
        bp: BandPass,
        vel_meas: f64,
        last_fb: Option<Instant>,
    }
    let mut damp: Vec<Damp> = (0..N_SLOTS)
        .map(|_| Damp {
            v_cmd: LpDiff::new(CONTROL_CUTOFF),
            a_cmd: LpDiff::new(CONTROL_CUTOFF),
            v_cmd_fast: LpDiff::new(VEL_CUTOFF),
            v_meas: LpDiff::new(VEL_CUTOFF),
            bp: BandPass::new(),
            vel_meas: 0.0,
            last_fb: None,
        })
        .collect();
    let mut prev_tick: Option<Instant> = None;
    // Latest decoded feedback per slot, shipped to Python once per tick as
    // an `F` packet — the core is the only CAN consumer; Python fills its
    // Motor caches from these instead of passively reading the bus.
    let mut latest: [SlotFeedback; N_SLOTS] = [None; N_SLOTS];
    // Whether the immediately preceding tick produced a fresh sample for
    // each slot. Host damping is suppressed for one tick after a miss; the
    // firmware's local kd remains active without relying on stale host state.
    let mut feedback_fresh = [false; N_SLOTS];
    // Wire command per slot this tick (tracker output or passthrough) —
    // the reference the deviation abort measures against.
    let mut cmd_pos: [f64; N_SLOTS] = [0.0; N_SLOTS];
    for m in &motors {
        cmd_pos[m.slot] = m.hold_pos;
    }
    let mut have_target = false;
    let mut last_seq: Option<u32> = None;
    let mut last_arrival: Option<Instant> = None;

    let period = Duration::from_secs_f64(1.0 / cfg.loop_hz);
    let abort_rad = cfg.abort_deg.to_radians();
    let watchdog = Duration::from_secs_f64(cfg.watchdog_ms / 1e3);
    let mut rejected: u64 = 0;
    let mut late: u64 = 0;
    let mut missed: u64 = 0;
    let mut trace_dropped: u64 = 0;
    let mut ticks: u64 = 0;
    let mut watchdog_frozen = false;
    let mut next_stats = Instant::now() + Duration::from_secs(5);
    // TX-stall (e-stop) tracking — see `guarded_send`. A dead bus skips the
    // motor disable on the way out (nothing is powered to hear it, and the
    // freshly purged queue should stay empty).
    let mut enobufs_since: Option<Instant> = None;
    let mut bus_dead = false;
    let mut feedback_health = [FeedbackHealth::default(); N_SLOTS];
    let silent_limit = silent_feedback_limit(cfg.loop_hz);
    let mut degraded_ticks: u64 = 0;
    let mut degraded_episodes: u64 = 0;
    let mut degraded_announced = [false; N_SLOTS];
    let mut next_degraded_log = Instant::now();
    let mut timing_health = TimingHealth::default();
    // Belt-and-braces: sends on a dead bus normally fail fast with ENOBUFS,
    // but if the socket sndbuf fills first a blocking write would hang the
    // loop; the timeout turns that into EAGAIN (treated as TX-full).
    let _ = sock.set_send_timeout(Duration::from_millis(20));

    let start_at = {
        let (lock, wake) = start_gate;
        let mut value = lock.lock().unwrap();
        while value.is_none() {
            value = wake.wait(value).unwrap();
        }
        value.unwrap()
    };
    let phase = if side == 0 {
        Duration::ZERO
    } else {
        period.mul_f64(0.5)
    };
    let mut deadline = start_at + phase;
    let mut trace_epoch = deadline;
    let mut trace_origin_s = f64::from_bits(trace_origin_bits.load(Ordering::Acquire));
    let mut trace_generation_seen = if trace_enabled.load(Ordering::Acquire) {
        trace_generation.load(Ordering::Acquire)
    } else {
        0
    };
    let result = (|| -> io::Result<()> {
        loop {
            if stop.load(Ordering::SeqCst) || SHUTDOWN.load(Ordering::SeqCst) {
                return Ok(());
            }
            sleep_until(deadline);
            let began = Instant::now();
            let lateness = began.saturating_duration_since(deadline);
            let timing_on_time = lateness <= LATE_TICK;
            if !timing_on_time {
                late += 1;
            }
            // A whole-cycle overrun means the causal sample/command ordering
            // has been lost. Stop before issuing another impedance command.
            // Smaller isolated late ticks are tolerated with host damping
            // suppressed below; clustered lateness also fails closed.
            if timing_health.record_lateness(lateness, period) {
                fault.store(1, Ordering::SeqCst);
                stop.store(true, Ordering::SeqCst);
                return Err(io::Error::other(format!(
                    "{iface}: control timing unhealthy ({:.3} ms late, {} of the last 32 ticks late); stopping before phase-sensitive damping",
                    lateness.as_secs_f64() * 1e3,
                    timing_health.recent_late.count_ones(),
                )));
            }
            ticks += 1;

            // Adopt a newly arrived target: latest-wins — the tracker
            // renders the trajectory toward it at loop rate, so no
            // interpolation segment is needed.
            {
                let slot = targets[side as usize].lock().unwrap();
                if let Some(t) = slot.target {
                    if last_seq != Some(t.seq) {
                        // Corruption defense on the raw target step; the
                        // gripper slot is exempt (its targets legitimately
                        // jump — the Python max-step gate excludes it too).
                        let step_ok = t.cmds[..GRIPPER_SLOT]
                            .iter()
                            .zip(play.iter())
                            .all(|(c, p)| (c.p_des - p.p_des).abs() <= cfg.max_step_rad);
                        if step_ok {
                            play = t.cmds;
                            have_target = true;
                        } else {
                            rejected += 1;
                        }
                        last_seq = Some(t.seq);
                        last_arrival = Some(t.arrival);
                    }
                }
            }

            // Watchdog: no fresh target — the tracker converges on the last
            // target and holds there (damping stays live).
            let starved = last_arrival.is_some_and(|a| began.duration_since(a) > watchdog);
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

            // Tick spacing for the damping chain (measured, not nominal —
            // a late tick then damps over the interval it actually covers).
            let tick_dt = prev_tick.map_or(0.0, |p| began.duration_since(p).as_secs_f64());
            prev_tick = Some(began);

            // The user-facing flight recorder gates the verbose core trace to
            // the same engage segment as IK/cmd/meas. On each new segment the
            // writer truncates its prior file, preserving latest-only
            // semantics without any formatting or disk I/O on this thread.
            let mut trace_this_tick = trace_enabled.load(Ordering::Acquire);
            if trace_this_tick {
                let generation = trace_generation.load(Ordering::Acquire);
                if generation != trace_generation_seen {
                    let reset = trace_tx
                        .as_ref()
                        .is_some_and(|tx| tx.try_send(TraceMsg::Reset).is_ok());
                    if reset {
                        trace_generation_seen = generation;
                        trace_epoch = began;
                        trace_origin_s = f64::from_bits(trace_origin_bits.load(Ordering::Acquire));
                    } else {
                        // Never write a new segment behind stale rows. Retry
                        // the reset next tick if the background queue is full.
                        trace_this_tick = false;
                        trace_dropped += 1;
                    }
                }
            }

            // Discard replies that missed the preceding tick's window before
            // issuing this batch. The protocols carry no sequence number, so
            // this boundary is what makes every accepted sample current.
            sock.drain_nonblocking()?;

            // Send all commands back-to-back and remember exactly which
            // motors were successfully queued in this tick.
            let mut expected = vec![false; motors.len()];
            let mut trace_pending: [Option<TraceRow>; N_SLOTS] = [None; N_SLOTS];
            for (motor_index, m) in motors.iter().enumerate() {
                let c = &play[m.slot];
                let (arb, frame) = if m.gripper {
                    // Idle until the first target (classic mode leaves the
                    // gripper uncommanded until motion_control too). Slot
                    // layout: mode carries max_speed, kp carries max_torque;
                    // the wire wants current as a fraction of rated (t_max).
                    if !have_target {
                        continue;
                    }
                    (
                        proto::DM_POS_FORCE_ARB_BASE + m.id as u16,
                        proto::dm_pos_force_encode(c.p_des, c.mode, c.kp / m.ranges.t_max),
                    )
                } else {
                    // Tracked mode: the trapezoid renders this tick's
                    // position toward the latest target, and the wire
                    // velocity plus fast feedforwards come from low-pass
                    // derivatives of the trajectory the wire actually
                    // carries.  Do not use the tracker's raw acceleration:
                    // its 240 Hz loop sees Python's 120 Hz targets as a
                    // two-tick staircase and turns that into alternating
                    // inertia torque (the motion vibration fixed here).
                    // Passthrough (gravity comp / bring-up hold): p_des
                    // as-is, v_des = 0, slow t_ff only; the tracker re-seeds
                    // so a later mode switch starts transient-free.
                    let tracked = c.mode >= 0.5;
                    let p_track = if tracked {
                        let (p, _, _) = trk[m.slot].update(c.p_des, tick_dt);
                        p
                    } else {
                        trk[m.slot].seed(c.p_des);
                        c.p_des
                    };
                    // Spring-torque cap (wrists): keep the wire position
                    // within tau_cap / kp of the last measured position so a
                    // joint blocked by an object leans on it with at most
                    // tau_cap instead of kp times the operator's run-ahead.
                    // The tracker keeps rendering the real trajectory
                    // underneath (no re-seed: a transient clip during a fast
                    // move must not zero its velocity), and everything
                    // downstream — the derivative chain feeding friction /
                    // inertia / host damping, the deviation abort reference,
                    // the trace — sees the clamped command, so those
                    // feedforwards fall to zero while the joint is held
                    // rather than adding to the press. The measured position
                    // is the previous tick's reply (one 240 Hz period old).
                    let p_cmd = filter::cap_spring(
                        p_track,
                        latest[m.slot].map(|(pos, _, _, _)| pos),
                        c.kp,
                        m.tau_cap,
                    );
                    cmd_pos[m.slot] = p_cmd;
                    let d = &mut damp[m.slot];
                    let (v_wire, a_cmd, v_cmd_fast, friction_ff, inertia_ff, v_damp) = if tracked {
                        // Match classic AxolArm.motion_control: friction uses
                        // the 20 rad/s low-pass position derivative, inertia
                        // uses a second identical derivative, and damping
                        // uses its independent 80 rad/s desired-velocity
                        // derivative.  Only the source position/rate differ:
                        // the core can use the trajectory it really sends.
                        let v_cmd = d.v_cmd.update(p_cmd, tick_dt);
                        let a_cmd = d.a_cmd.update(v_cmd, tick_dt);
                        let v_cmd_fast = d.v_cmd_fast.update(p_cmd, tick_dt);
                        let friction_ff = filter::friction(v_cmd, m.fc, m.k, m.fv, m.fo);
                        let inertia_ff = c.j_eff * a_cmd;
                        let damp_ok = feedback_fresh[m.slot]
                            && timing_on_time
                            && !feedback_health[m.slot].degraded;
                        let v_damp = if damp_ok {
                            d.bp.update(v_cmd_fast - d.vel_meas, c.damp_w0, c.damp_q, tick_dt)
                        } else {
                            // A missing frame makes measured velocity stale.
                            // Reset rather than carrying band-pass energy into
                            // the first tick after feedback recovers. While the
                            // joint is degraded, damping stays off for the whole
                            // stretch: re-engaging a freshly reset band-pass
                            // every few ticks is a torque transient, not damping.
                            d.bp.reset();
                            0.0
                        };
                        (v_cmd, a_cmd, v_cmd_fast, friction_ff, inertia_ff, v_damp)
                    } else {
                        d.v_cmd.seed(p_cmd);
                        d.a_cmd.seed(0.0);
                        d.v_cmd_fast.seed(p_cmd);
                        d.bp.reset();
                        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    };
                    let damping_ff = c.kd_host * v_damp;
                    let t_ff = c.t_ff + friction_ff + inertia_ff + damping_ff;
                    if trace_this_tick && trace_tx.is_some() {
                        trace_pending[m.slot] = Some(TraceRow {
                            tick: ticks,
                            time_s: trace_origin_s
                                + began.saturating_duration_since(trace_epoch).as_secs_f64(),
                            seq: last_seq.unwrap_or(0),
                            slot: m.slot,
                            motor_id: m.id,
                            mode: c.mode,
                            target_p: c.p_des,
                            cmd_p: p_cmd,
                            cmd_v: v_wire,
                            cmd_a: a_cmd,
                            cmd_v_fast: v_cmd_fast,
                            meas_p: f64::NAN,
                            motor_v: f64::NAN,
                            meas_v: f64::NAN,
                            meas_tau: f64::NAN,
                            gravity_ff: c.t_ff,
                            friction_ff,
                            inertia_ff,
                            damping_ff,
                            total_ff: t_ff,
                            kd_host: c.kd_host,
                            damp_w0: c.damp_w0,
                            damp_q: c.damp_q,
                            tick_dt,
                            fb_dt: f64::NAN,
                        });
                    }
                    let frame = proto::mit_encode(p_cmd, v_wire, c.kp, c.kd, t_ff, &m.ranges);
                    let arb = match m.vendor {
                        Vendor::MyActuator => proto::MA_MC_REQ + m.id as u16,
                        Vendor::Damiao => m.id as u16,
                    };
                    (arb, frame)
                };
                match guarded_send(&sock, arb, &frame, &mut enobufs_since)? {
                    SendOutcome::Sent => expected[motor_index] = true,
                    SendOutcome::Dropped => {}
                    SendOutcome::Stalled => {
                        // The e-stop path: nothing has ACKed for >1 s. Stop
                        // commanding, purge the poisoned TX queue so the
                        // stale commands can't replay on re-power, and take
                        // the whole core down as a fault — re-powered motors
                        // come back disabled, so the session needs a fresh
                        // bring-up anyway.
                        bus_dead = true;
                        fault.store(1, Ordering::SeqCst);
                        stop.store(true, Ordering::SeqCst);
                        let purged = purge_tx_queue(iface);
                        let purge_message = if purged {
                            format!("{iface}: purged the stale TX queue (bus flapped)")
                        } else {
                            format!(
                                "{iface}: could not purge the TX queue — stale motion \\
                                 commands will replay when motors power back on; flap \\
                                 the interface before re-powering"
                            )
                        };
                        send_text(out_tx, b'L', &purge_message);
                        return Err(io::Error::other(format!(
                            "{iface}: TX queue stalled >{}s — no node ACKing \
                             frames (e-stop / motors unpowered?); commands \
                             stopped{}",
                            STALL_DETECT.as_secs(),
                            if purged {
                                ", stale queue purged"
                            } else {
                                " — QUEUE PURGE FAILED, flap the interface \
                                 before re-powering"
                            },
                        )));
                    }
                }
            }

            // Collect replies; deviation abort against the played target. The
            // window begins when this tick actually began, not at its nominal
            // schedule point: a late wake must not discard shoulder feedback
            // simply because the old absolute deadline has already elapsed.
            // The wait is hrtimer-precise (`recv_timeout`): a missing reply
            // must end the window at `reply_deadline`, never a jiffy or two
            // later, or the overrun lands on the next tick as lateness.
            let reply_deadline = began + period.saturating_sub(REPLY_GUARD);
            let mut seen = vec![false; motors.len()];
            let mut pending = expected.iter().filter(|&&value| value).count();
            while pending > 0 {
                let now = Instant::now();
                if now >= reply_deadline {
                    break;
                }
                let Some(frame) = sock.recv_timeout(reply_deadline - now)? else {
                    break;
                };
                let (idx, pos, vel, tau) = match frame.id {
                    id if (0x501..=0x505).contains(&id) => {
                        let motor_id = (id - 0x500) as u8;
                        let Some(idx) = motors.iter().position(|m| m.id == motor_id) else {
                            continue;
                        };
                        let (pos, vel, tau) = proto::ma_decode_mit_feedback(
                            &frame.data,
                            motors[idx].ranges.p_max,
                            motors[idx].ranges.t_max,
                        );
                        (idx, pos, vel, tau)
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
                        (idx, fb.position, fb.velocity, fb.torque)
                    }
                    _ => continue,
                };
                if !mark_unique_expected_reply(&expected, &mut seen, idx) {
                    continue;
                }
                pending -= 1;
                let recv_time = Instant::now();
                latest[motors[idx].slot] = Some((pos, vel, tau, recv_time));
                // No deviation abort for the gripper: stalling against an
                // object (or a jaw span the target overshoots) is normal.
                if motors[idx].gripper {
                    continue;
                }
                // Feed the damping chain's measured velocity from the
                // frame's own receive spacing (the CAN reply cadence is the
                // loop cadence; arrival jitter within a tick is µs-scale).
                let (meas_v, fb_dt) = {
                    let d = &mut damp[motors[idx].slot];
                    let dt = d
                        .last_fb
                        .map_or(0.0, |p| recv_time.duration_since(p).as_secs_f64());
                    d.last_fb = Some(recv_time);
                    d.vel_meas = d.v_meas.update(pos, dt);
                    (d.vel_meas, dt)
                };
                if let (Some(tx), Some(mut row)) =
                    (trace_tx.as_ref(), trace_pending[motors[idx].slot].take())
                {
                    row.meas_p = pos;
                    row.motor_v = vel;
                    row.meas_v = meas_v;
                    row.meas_tau = tau;
                    row.fb_dt = fb_dt;
                    match tx.try_send(TraceMsg::Row(row)) {
                        Ok(()) => {}
                        Err(mpsc::TrySendError::Full(_)) => trace_dropped += 1,
                        Err(mpsc::TrySendError::Disconnected(_)) => trace_dropped += 1,
                    }
                }
                // Deviation abort against the *commanded* position (the
                // tracker output), not the raw target — during a legitimate
                // catch-up move the target may briefly lead the arm by more
                // than abort_deg while the command never does.
                let e = (pos - cmd_pos[motors[idx].slot]).abs();
                if e > abort_rad {
                    fault.store(1, Ordering::SeqCst);
                    stop.store(true, Ordering::SeqCst);
                    return Err(io::Error::other(format!(
                        "{iface}: {} deviated {:.2}° from command (abort at {:.0}°)",
                        motors[idx].joint,
                        e.to_degrees(),
                        cfg.abort_deg,
                    )));
                }
            }

            // Replies still outstanding at the window's end. Counted into the
            // periodic stats so a rare miss is visible in the log even when it
            // stays far below the fail-closed thresholds.
            missed += pending as u64;

            // Never run phase-sensitive host damping on stale feedback. A
            // single miss suppresses host damping on the next tick; bursty
            // loss marks the joint degraded (damping off until a clean
            // window, logged) and the loop keeps running on firmware kd. Only
            // a motor that falls silent for `SILENT_FEEDBACK_FAULT` faults
            // both buses — the deviation abort has been blind on it for too
            // long to keep commanding it. The gripper is intentionally
            // excluded.
            let mut any_degraded = false;
            for (idx, motor) in motors.iter().enumerate() {
                if motor.gripper {
                    continue;
                }
                feedback_fresh[motor.slot] = seen[idx];
                let health = &mut feedback_health[motor.slot];
                match health.record(seen[idx], silent_limit) {
                    FeedbackVerdict::Steady => {}
                    FeedbackVerdict::Degraded => {
                        degraded_episodes += 1;
                        degraded_announced[motor.slot] = began >= next_degraded_log;
                        if degraded_announced[motor.slot] {
                            next_degraded_log = began + DEGRADED_LOG_INTERVAL;
                            send_text(
                                out_tx,
                                b'L',
                                &format!(
                                    "{iface}: {} feedback degraded ({} of the last 32 ticks missing) — host damping off on this joint until a clean window; firmware kd holds",
                                    motor.joint,
                                    health.recent_misses.count_ones(),
                                ),
                            );
                        }
                    }
                    FeedbackVerdict::Recovered => {
                        if std::mem::take(&mut degraded_announced[motor.slot]) {
                            send_text(
                                out_tx,
                                b'L',
                                &format!(
                                    "{iface}: {} feedback recovered — host damping resumed",
                                    motor.joint,
                                ),
                            );
                        }
                    }
                    FeedbackVerdict::Silent => {
                        fault.store(1, Ordering::SeqCst);
                        stop.store(true, Ordering::SeqCst);
                        return Err(io::Error::other(format!(
                            "{iface}: {} silent for {} consecutive ticks ({:.1} s) — motor unreachable; stopping rather than commanding it blind",
                            motor.joint,
                            health.consecutive_misses,
                            health.consecutive_misses as f64 / cfg.loop_hz,
                        )));
                    }
                }
                any_degraded |= health.degraded;
            }
            if any_degraded {
                degraded_ticks += 1;
            }

            // Ship this tick's telemetry to Python (non-blocking mpsc; the
            // writer thread does the socket I/O). Skipped until the first
            // reply so an all-empty packet never races the bring-up reads.
            if latest.iter().any(|s| s.is_some()) {
                let _ = out_tx.send(build_feedback(side, &latest, Instant::now()));
            }

            if began >= next_stats {
                next_stats = began + Duration::from_secs(5);
                send_text(
                    out_tx,
                    b'L',
                    &format!(
                        "{iface}: {ticks} ticks, {late} late ({:.2}%), {missed} missed replies, {degraded_ticks} degraded ticks in {degraded_episodes} episodes, {rejected} rejected targets, {trace_dropped} trace drops, seq {:?}",
                        late as f64 / ticks as f64 * 100.0,
                        last_seq,
                    ),
                );
            }
            deadline = next_bus_deadline(began, period);
        }
    })();

    if !bus_dead {
        bringup::disable(&sock, &motors);
    }
    drop(trace_tx);
    if let Some(handle) = trace_handle {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => send_text(out_tx, b'L', &format!("{iface}: trace writer: {err}")),
            Err(_) => send_text(out_tx, b'L', &format!("{iface}: trace writer panicked")),
        }
    }
    if let Err(err) = &result {
        send_text(out_tx, b'S', &format!("fault: {err}"));
    }
    result
}
