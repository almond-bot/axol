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
//! - a late target is carried forward along the stream's own velocity
//!   (`filter::Holdover`, ≤ 80 ms, gliding to rest) so a Python tick that
//!   lost the CPU renders as smooth motion instead of a stop-then-lunge —
//!   smoothness is not left to the host's scheduler;
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
//! - `C` + text        config: a mandatory `proto <n>` line (must equal
//!                     `CONFIG_PROTO` — the client and this binary must
//!                     agree on the slot layout below, and a stale build
//!                     of either fails here, loudly, instead of silently
//!                     rejecting every target), `loop_hz`/`watchdog_ms`/
//!                     `max_step_rad` keys, one `joint <side> <iface> <name>
//!                     <motor_id> <kp> <kd> <max_vel> <max_accel> <fc> <k>
//!                     <fv> <fo>` line per arm joint (tracker limits +
//!                     friction params; the motor id 1..=7 fixes the
//!                     joint's target slot, so a bus may carry any subset
//!                     of the arm), and an optional `gripper <side>
//!                     <iface> <motor_id>` line
//! - `P`               prep: MyActuator 0x76 reset + settle, Damiao
//!                     clear-errors on every *cold* arm joint (torque-neutral
//!                     on a disabled motor; run *before* Python resolves
//!                     joint offsets, so the wrap state it verifies is the
//!                     post-reset one). Joints found already enabled and
//!                     holding are skipped — the reset would reboot them and
//!                     drop the arm — and named in an `L` line; the gripper
//!                     is never touched
//! - `A`               arm: bring-up, enable the cold joints, hold current
//!                     pose (holding joints are attached to without a brake
//!                     release / enable frame; the gripper must already be
//!                     enabled + calibrated in POSITION_FORCE mode by the
//!                     Python side)
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
//! - `D`               disarm: disable motors, threads exit. The *only*
//!                     path that ever disables the motors (and only when no
//!                     fault is latched); see Safety.
//!
//! Rust -> Python:
//! - `S` + text        state message. `fault: ...` — the core stopped
//!                     streaming (bus dead / arm failed); `limp: ...` — the
//!                     core is still serving but at kp = 0 with the streamed
//!                     gravity t_ff on every arm joint (see Safety); the
//!                     client should keep streaming gravity-comp targets.
//! - `L` + text        log line
//! - `W` + text        warning line: a degraded transition worth the
//!                     operator's attention (the client logs it at WARNING)
//! - `F` + binary      telemetry, one per bus per tick while armed: side
//!                     u8, valid-mask u8, then 8 x (pos f64, vel f64, tau
//!                     f64, age_us u32) — the latest decoded feedback per
//!                     slot. Python fills its Motor caches from these (see
//!                     `build_feedback`); it does not read CAN while the
//!                     core is armed.
//!
//! ## Safety
//! - **A fault never disables the motors.** Cutting torque drops the arms,
//!   which is worse than any condition the checks detect. Torque comes off
//!   only on an explicit `D` disarm of a healthy session (or an e-stop,
//!   which removes motor power itself).
//! - **A loss-of-trust fault goes *limp*, not dead.** A motor silent for a
//!   second means the core should no longer be applying stiffness or
//!   phase-sensitive damping to a joint it cannot see — so it stops doing
//!   exactly that and nothing more: every arm joint on both buses goes to
//!   kp = 0, firmware kd `LIMP_KD`, no host damping or inertia term, with
//!   the streamed gravity `t_ff` still applied (Python keeps evaluating it
//!   at the measured pose). The loop keeps running, so the arms are
//!   weightless and hand-guidable — the operator moves them to rest and
//!   restarts. This is the classic contact-hold gravity comp, entered from
//!   the core side. Limp is enforced in-core regardless of what targets
//!   arrive, and is never cleared within a session. A disarm while limp
//!   leaves the motors limp rather than disabling.
//! - **Hard faults stop the stream and leave the last command in place.**
//!   A dead bus (TX stall — the e-stop cut motor power), a bring-up
//!   failure, a protocol error, a signal, or a lost client: the core
//!   reports `fault: ...` (where it can), stops streaming, and exits with
//!   each motor holding its last MIT command on firmware gains — the same
//!   outcome as a classic Python session dying mid-command.
//! - Targets stepping more than `max_step_rad` from the previous target
//!   are rejected (counted, and warned about at most every
//!   `DEGRADED_LOG_INTERVAL` naming the joint and step, since a rejected
//!   stream leaves the arm parked while the client believes it is moving)
//!   — corruption defense; the Python side has its own max-step gate. The
//!   gripper slot is exempt (its targets legitimately jump, matching the
//!   Python gate). Whatever gets through, the tracker's
//!   velocity/acceleration limits bound what the wire can ever see.
//! - There is deliberately **no position-deviation abort**, matching the
//!   classic Python controller. Position error on a compliant impedance
//!   controller is not a safety signal: a hand on the arm and a joint that
//!   lost torque (overtemp self-disable) both look like "deviation", and
//!   the old 25° abort took down healthy arms for both. Contact is handled
//!   by the Python torque-residual `ContactWatchdog` (limp gravity-comp
//!   hold, operator resets); a self-disabled motor just stops contributing
//!   while the rest of the arm keeps working, as in classic mode.
//! - Every command batch accepts exactly one fresh reply per motor. A missed
//!   sample suppresses host damping for that tick; bursty loss (4 of the last
//!   32 ticks) marks the joint *degraded* — host damping stays off until a
//!   clean 32-tick window, the transition is logged, and the loop keeps
//!   running on firmware kd. Only a motor silent for a full second takes
//!   the session limp.
//! - **Late ticks degrade, never limp.** Timing gets feedback's degraded
//!   tier and nothing above it. A whole-cycle overrun (a wake a full
//!   period or more late), three late ticks in a row, or 8 of the last 32
//!   late marks the *bus* timing-degraded: host damping off on every joint
//!   of that bus until a clean 32-tick window, and the overrun tick's
//!   tracker advances one nominal period with its derivative chains
//!   re-seeded at rest — the motors held the previous command across the
//!   gap, so there is no trajectory to differentiate. Firmware kp/kd and
//!   the streamed gravity `t_ff` are untouched, so the arm keeps holding —
//!   a late host is what the firmware is built to ride out, which is why
//!   no amount of lateness is a loss of trust. The transition (and each
//!   further overrun inside a degraded stretch, rate-limited) is logged as
//!   a warning with the thread's own scheduler/memory counters across that
//!   wake (`stall.rs`: runnable-wait, page faults, involuntary switches) so
//!   the line says whether the loop was preempted, faulting, or blocked in
//!   the kernel.
//! - The process locks its memory (`mlockall`, `stall::lock_memory`) before
//!   accepting a client, so a page reclaimed under the dataset writer's I/O
//!   pressure can never fault a bus thread mid-tick. A failed lock (no
//!   `CAP_IPC_LOCK`, low `RLIMIT_MEMLOCK` on a dev build) is logged and
//!   the core runs unlocked as it always did.
//! - The gripper is not commanded at all until the first target arrives
//!   (matching classic mode, where it sits idle until motion_control).
//! - Watchdog: no target for `watchdog_ms` holds the last target (the
//!   holdover has glided to rest well before then; the tracker converges
//!   and stays, damping live). The arms keep holding —
//!   matching what the firmware itself does if the host dies — until a
//!   disarm or an operator e-stop.
//! - Client disconnect while armed, SIGINT/SIGTERM, and protocol errors
//!   stop the stream and exit with the motors holding (not disabled).

use std::io::{self, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::bringup::{self, MotorSpec, ReadyMotor, Vendor, WireMode};
use crate::can::CanSock;
use crate::filter::{self, BandPass, Cadence, Holdover, LpDiff, Trapezoid};
use crate::hold::sleep_until;
use crate::proto;
use crate::safety::{guarded_send, purge_tx_queue, SendOutcome, STALL_DETECT};
use crate::stall;

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

/// How long a late target is carried forward along the stream's velocity
/// (`filter::Holdover`) before the carried target has glided to rest. Covers
/// the 15-65 ms Python-tick stalls measured on a loaded host (2026-09-15)
/// with margin, and sits well inside the 150 ms watchdog: a stream that
/// really stopped is at rest long before the watchdog names it stalled.
const HOLDOVER_MAX: f64 = 0.080;

/// Config/target protocol generation the client must declare (`proto <n>`).
/// Bumped whenever the meaning of the config or target layout changes, so a
/// Python package and an `axol-rt` binary built from different checkouts
/// refuse each other at configure time. Silent skew is the failure mode
/// this guards against: before it existed, a client that slotted joints by
/// motor id against a core that slotted them by list order armed fine and
/// then rejected every target on the max-step gate — the arms just held.
///
/// - 1 (implicit; never declared): arm joints took slots in list order.
/// - 2: slots come from the motor id (`slot = motor_id - 1`), so a bus may
///   carry any subset of the arm.
/// - 3: each `joint` line carries two more fields, the stiction
///   compensation gain and its error scale (`filter::stiction`).
/// - 4: plus the load-proportional stiction gain (`filter::stiction_amplitude`).
/// - 5: plus the torque dither amplitude and frequency (`filter::dither_step`).
/// - 6: plus the wire mode token (`mit` | `a4`, `bringup::WireMode`).
/// - 7: plus the four Stribeck cancellation fields (`filter::stribeck_excess`).
/// - 8: plus the load-proportional Coulomb friction `fl` (Nm per Nm of gravity).
/// - 9: plus the Stribeck term's measured-velocity pole (rad/s).
/// - 10: the wire token gains `pv` (Damiao position-velocity,
///   `bringup::WireMode::Pv`); `loop_hz` above `THIN_ABOVE_HZ` thins the bus
///   schedule (`Thinning`).
/// - 11: impedance joints run at `IMPEDANCE_HZ` only — `loop_hz` 240, or
///   `MIXED_LOOP_HZ` with them on alternate ticks (`Thinning::mit_lane`) —
///   and anything else is refused. A proto-10 core given 480 would command
///   them at 480.
const CONFIG_PROTO: u32 = 11;
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
/// lossy. Past this point the session goes limp (see the Safety notes)
/// rather than streaming stiffness to it blind. Matches the TX-stall e-stop
/// detection window rather than the old 12.5 ms.
const SILENT_FEEDBACK_FAULT: Duration = Duration::from_secs(1);
/// Degraded/recovered transitions are logged at most this often per bus so a
/// joint flapping across the threshold during boot cannot flood the log; the
/// five-second stats line carries the cumulative count regardless.
const DEGRADED_LOG_INTERVAL: Duration = Duration::from_secs(5);
/// Firmware velocity damping (Nm·s/rad) on the arm joints while the core is
/// *limp* — the fallback for a loss-of-trust fault (a silent motor).
/// Matches `VRTeleopConfig.reset_gravity_comp_kd`, the classic contact-hold
/// gravity comp: enough to keep a hand-guided arm from feeling twitchy, far
/// too little to hold it up. Gravity itself comes from the streamed `t_ff`.
const LIMP_KD: f64 = 0.25;
/// Leave a small slice of each cycle for telemetry handoff and the next
/// absolute sleep. The rest is valid reply time; the old 80% window discarded
/// delayed USB-CAN replies despite there still being cycle headroom.
const REPLY_GUARD: Duration = Duration::from_micros(150);
/// A tick starting more than this far past its deadline is phase-degraded. Its
/// host damping is suppressed even when feedback itself is fresh.
const LATE_TICK: Duration = Duration::from_micros(500);
/// Clustered lateness that marks a bus's *timing* degraded (host damping off
/// on every joint of that bus until a clean 32-tick window): three late
/// ticks in a row, or this many in the window. Not a fault — the same
/// treatment bursty feedback loss gets. A whole-cycle overrun (a tick that
/// wakes a full period or more late) degrades on its own: the sample/command
/// ordering that tick was lost, so its damping and inertia terms are
/// re-seeded rather than computed over the gap.
const DEGRADED_RECENT_LATE_TICKS: u32 = 8;
// Bad control timing never takes the session limp — degraded is the whole
// response, however long it lasts. A late tick invalidates exactly the terms
// degraded turns off (phase-sensitive damping, the derivative chains); the
// motors themselves hold the last command on firmware kp/kd across any gap,
// which is the same thing they do if the host dies. Limp is reserved for a
// motor that has gone silent (`SILENT_FEEDBACK_FAULT`), where the core would
// otherwise stream stiffness to a joint it cannot see. The field record
// (2026-09-04) is single 20–60 ms stalls in otherwise perfect ~770k-tick
// sessions, each while the dataset writer flushed a save; the old single-tick
// limp turned each into a session-ending, hand-guide-and-restart fault while
// the arms were holding still. Persistent lateness is still made visible: the
// degraded warning carries the stall attribution, repeated overruns inside a
// degraded stretch are logged (rate-limited), and the stats line counts them.

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
/// `(p50, p95, max)` of the per-tick bus-busy fractions since the last stats
/// line; NaN when no tick had a reply. Sorts in place.
fn bus_busy_percentiles(busy: &mut [f64]) -> (f64, f64, f64) {
    if busy.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    busy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let at = |q: f64| busy[((busy.len() - 1) as f64 * q).round() as usize];
    (at(0.5), at(0.95), busy[busy.len() - 1])
}

fn next_bus_deadline(began: Instant, period: Duration) -> Instant {
    began + period
}

/// Accept at most as many replies from each motor as it was sent commands
/// this tick (one for MIT, two for an 0xA4 joint: the command reply and the
/// 0x92 position read).
///
/// CAN frames carry no command sequence number. The bus loop therefore drains
/// late frames before sending and uses this per-batch budget to prevent
/// duplicate or unsolicited feedback from satisfying another motor's reply
/// budget.
fn mark_unique_expected_reply(expected: &[u8], seen: &mut [u8], idx: usize) -> bool {
    if idx >= expected.len() || idx >= seen.len() || seen[idx] >= expected[idx] {
        return false;
    }
    seen[idx] += 1;
    true
}

/// Every reply the motor was budgeted for this tick arrived.
fn reply_complete(expected: &[u8], seen: &[u8], idx: usize) -> bool {
    expected[idx] > 0 && seen[idx] >= expected[idx]
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
    /// gone and take the session limp (see the Safety notes).
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

/// Outcome of one tick's wake-up timing for the bus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TimingVerdict {
    /// Nothing to report: on time, an isolated late tick, or an ongoing
    /// degraded stretch that has neither cleared nor worsened.
    Steady,
    /// Timing just became untrustworthy: host damping off on this bus until
    /// a clean window (see `DEGRADED_RECENT_LATE_TICKS`).
    Degraded,
    /// A full clean window just closed out a degraded stretch.
    Recovered,
}

#[derive(Clone, Copy, Default)]
struct TimingHealth {
    /// Ticks that woke more than `LATE_TICK` past their deadline.
    recent_late: u32,
    /// Ticks that woke a whole period or more late.
    recent_overruns: u32,
    consecutive_late: u8,
    degraded: bool,
}

impl TimingHealth {
    /// Record one tick's wake-up lateness. The verdict includes the tick
    /// itself: the degraded warning must describe the tick that triggered it.
    ///
    /// Mirrors `FeedbackHealth`'s degraded tier: hysteresis (it starts at an
    /// overrun or a late cluster and only clears once the 32-tick window has
    /// no late tick at all). Unlike feedback there is no fault tier above it
    /// — see the note at `DEGRADED_RECENT_LATE_TICKS`.
    fn record(&mut self, lateness: Duration, period: Duration) -> TimingVerdict {
        let late = lateness > LATE_TICK;
        let overrun = lateness >= period;
        self.recent_late = (self.recent_late << 1) | u32::from(late);
        self.recent_overruns = (self.recent_overruns << 1) | u32::from(overrun);
        if late {
            self.consecutive_late = self.consecutive_late.saturating_add(1);
        } else {
            self.consecutive_late = 0;
        }
        let unhealthy = overrun
            || self.consecutive_late >= 3
            || self.recent_late.count_ones() >= DEGRADED_RECENT_LATE_TICKS;
        match (self.degraded, unhealthy) {
            (false, true) => {
                self.degraded = true;
                TimingVerdict::Degraded
            }
            (true, false) if self.recent_late == 0 => {
                self.degraded = false;
                TimingVerdict::Recovered
            }
            _ => TimingVerdict::Steady,
        }
    }
}

type BusStartGate = (Mutex<Option<Instant>>, Condvar);

/// Take the whole session limp (both buses — the flag is shared) and tell
/// the client why, once. Edge-triggered: a repeating condition (a motor that
/// stays silent) reports on its first tick only. Returns whether this call
/// made the transition.
fn go_limp(limp: &AtomicBool, out_tx: &mpsc::Sender<Vec<u8>>, reason: &str) -> bool {
    if limp
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return false;
    }
    send_text(out_tx, b'S', &format!("limp: {reason}"));
    send_text(
        out_tx,
        b'L',
        "limp: arm joints at kp = 0 with gravity feedforward on both buses — hand-guide the arms to rest, then stop and restart the session",
    );
    true
}

/// Whether this tick's frame for a joint is the 0xA4 position command.
///
/// Only MyActuator joints on `wire_mode a4`, and for them on every tick that
/// commands a position: tracked ticks *and* passthrough holds (bring-up
/// hold, a stalled stream, Python's hold-at-measured-pose), which is any
/// tick with a position gain. The alternative — MIT for the hold, 0xA4 once
/// tracking starts — is what the classic design did, and it is exactly what
/// the X6-P20's 2025070202 firmware refuses: after an MIT frame, 0xA4 is
/// ignored until the motor is reset. The right elbow held its pose through
/// two whole replays that way (2026-09-21) while the X8-P20 shoulders, whose
/// 2026042402 firmware switches freely, tracked. Keeping an a4 joint on 0xA4
/// from its first frame works on both. Limp and gravity comp (`kp == 0`)
/// stay MIT: those exist to make the joint compliant, which the firmware
/// position loop cannot be — so on the older firmware a joint that has been
/// hand-guided needs a re-enable before it will track on a4 again.
fn a4_wire(vendor: Vendor, wire: WireMode, tracked: bool, kp: f64) -> bool {
    vendor == Vendor::MyActuator && wire == WireMode::A4 && (tracked || kp > 0.0)
}

/// Whether this tick's frame for a joint is the Damiao position-velocity
/// command (0x100 + id): Damiao wrists on `wire_mode pv`, on every tick that
/// commands a position — the same rule as [`a4_wire`]. Limp and gravity
/// comp (`kp == 0`) fall back to MIT for compliance; the bus loop switches
/// the wrist's control-mode register along with the frame, because the
/// firmware ignores the frame of the mode it is not in.
fn pv_wire(vendor: Vendor, wire: WireMode, tracked: bool, kp: f64) -> bool {
    vendor == Vendor::Damiao && wire == WireMode::Pv && (tracked || kp > 0.0)
}

/// Above this loop rate the bus cannot carry every motor's frames every
/// tick and the schedule is thinned (`Thinning`). A 1 Mbps bus moves a
/// frame in ~0.13 ms with the USB adapters in the loop (measured: three a4
/// joints at 240 Hz ran the bus 66-73% busy, 22 frames in ~2.9 ms). The
/// full eight-motor tick under the position controller — five 0xA4
/// commands with echoes, five 0x92 reads, two wrist commands, the gripper —
/// is 26 frames, 3.4 ms: it fits a 240 Hz tick (4.17 ms), not a 400 Hz one
/// (2.5 ms, of which `REPLY_GUARD` is reserved).
const THIN_ABOVE_HZ: f64 = 300.0;

/// The only rate an impedance (MIT) arm joint is commanded at. Its gains,
/// feedforward and damping filters were tuned and verified at 240 Hz; on a
/// loop mixing it with firmware-loop joints that run faster, it gets its own
/// 240 Hz cadence inside the faster loop (`Thinning::mit_lane`), not the
/// loop's rate — at 400 Hz with the firmware joints beside it, right
/// shoulder_3 / wrist_1 on impedance shook the arm (2026-09-22).
const IMPEDANCE_HZ: f64 = 240.0;

/// The loop rate of a bus that mixes impedance arm joints with firmware-loop
/// ones: twice `IMPEDANCE_HZ`, so every impedance joint lands on alternate
/// ticks at exactly 240 Hz while the 0xA4 joints get 480 Hz (above the
/// position controller's 400, so no audible 200 Hz staircase either).
const MIXED_LOOP_HZ: f64 = 2.0 * IMPEDANCE_HZ;

/// Refuse a loop rate an impedance arm joint cannot run at: 240 Hz, or
/// `MIXED_LOOP_HZ` with it on alternate ticks. Anything else would command
/// it at a rate its tuning was never verified at.
fn check_impedance_rate(loop_hz: f64, specs: &[MotorSpec]) -> io::Result<()> {
    let ok = |hz: f64| (loop_hz - hz).abs() < 1e-6;
    if ok(IMPEDANCE_HZ) || ok(MIXED_LOOP_HZ) {
        return Ok(());
    }
    if let Some(s) = specs.iter().find(|s| !s.gripper && s.wire == WireMode::Mit) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "config: loop_hz {loop_hz} with {} on the impedance frame — impedance \
                 runs at {IMPEDANCE_HZ} Hz only (loop_hz {IMPEDANCE_HZ}, or {MIXED_LOOP_HZ} \
                 with it on alternate ticks)",
                s.joint
            ),
        ));
    }
    Ok(())
}

/// The bus schedule when the loop runs faster than the bus (`THIN_ABOVE_HZ`).
///
/// Every MyActuator command still goes out every tick — that is the point
/// of the higher rate (the position loop's target staircase is audible at
/// 200 Hz and gone at 400). The rest rides in two round-robin lanes of one
/// request/reply pair per tick each, so a full tick is a fixed 14 frames
/// (~1.8 ms, ~72% of a 2.5 ms tick):
///
/// - the Damiao wrists take turns being commanded (200 Hz each with two
///   wrists; their own profiler shapes the staircase);
/// - the 0xA4 joints take turns having their 0.01° position read (0x92)
///   and the gripper takes a turn in the same lane (about 67 Hz each on
///   the full arm). Between reads an a4 joint's position is carried
///   forward from the last read on the speed its command echo reports
///   every tick (`a4_extrapolate`).
///
/// On a bus that mixes impedance (MIT) arm joints with firmware-loop ones
/// the loop runs at `MIXED_LOOP_HZ` and the impedance joints form their own
/// lane instead: each is commanded every other tick, at exactly 240 Hz, the
/// lane split across both phases so each tick carries half of them. Their
/// whole host pipeline runs on those ticks only (see `bus_loop`), so they
/// behave as the verified 240 Hz loop, interleaved.
///
/// Below the threshold nothing is thinned: every motor is commanded, and
/// every a4 joint read, every tick.
struct Thinning {
    enabled: bool,
    /// Impedance (MIT) arm joints as `(motor index, phase)`: commanded on
    /// ticks where `tick % mit_div == phase`.
    mit_lane: Vec<(usize, u64)>,
    /// Ticks per impedance command: 2 on a mixed bus, 1 otherwise.
    mit_div: u64,
    /// Motor indices of the Damiao wrists not in `mit_lane`, one commanded
    /// per tick.
    dm_lane: Vec<usize>,
    /// Motor indices of the a4 joints and the gripper, one served per tick.
    read_lane: Vec<usize>,
    /// The gripper's motor index, when the bus has one.
    gripper: Option<usize>,
}

impl Thinning {
    fn plan(motors: &[ReadyMotor], loop_hz: f64) -> Self {
        let enabled = loop_hz > THIN_ABOVE_HZ;
        let mit: Vec<usize> = motors
            .iter()
            .enumerate()
            .filter(|(_, m)| !m.gripper && m.wire == WireMode::Mit)
            .map(|(i, _)| i)
            .collect();
        // `check_impedance_rate` has held a bus with impedance joints to 240
        // or 480 Hz, so above the threshold this is 2.
        let mit_div = if enabled && !mit.is_empty() {
            (loop_hz / IMPEDANCE_HZ).round().max(1.0) as u64
        } else {
            1
        };
        let mit_lane: Vec<(usize, u64)> = if mit_div > 1 {
            mit.iter()
                .enumerate()
                .map(|(k, &idx)| (idx, k as u64 % mit_div))
                .collect()
        } else {
            Vec::new()
        };
        let dm_lane = motors
            .iter()
            .enumerate()
            .filter(|(i, m)| {
                m.vendor == Vendor::Damiao && !m.gripper && !mit_lane.iter().any(|(j, _)| j == i)
            })
            .map(|(i, _)| i)
            .collect();
        let read_lane = motors
            .iter()
            .enumerate()
            .filter(|(_, m)| {
                m.gripper || (m.vendor == Vendor::MyActuator && m.wire == WireMode::A4)
            })
            .map(|(i, _)| i)
            .collect();
        let gripper = motors.iter().position(|m| m.gripper);
        Self {
            enabled,
            mit_lane,
            mit_div,
            dm_lane,
            read_lane,
            gripper,
        }
    }

    /// The phase of an impedance joint on its own 240 Hz cadence, or `None`
    /// for a motor commanded at the loop's rate (or its round-robin lane).
    fn mit_phase(&self, idx: usize) -> Option<u64> {
        self.mit_lane
            .iter()
            .find(|(j, _)| *j == idx)
            .map(|(_, phase)| *phase)
    }

    fn turn(lane: &[usize], tick: u64) -> Option<usize> {
        if lane.is_empty() {
            None
        } else {
            Some(lane[(tick % lane.len() as u64) as usize])
        }
    }

    /// Whether motor `idx` is commanded on `tick`.
    fn commanded(&self, idx: usize, tick: u64) -> bool {
        if !self.enabled {
            return true;
        }
        if let Some(phase) = self.mit_phase(idx) {
            return tick % self.mit_div == phase;
        }
        if self.dm_lane.contains(&idx) {
            return Self::turn(&self.dm_lane, tick) == Some(idx);
        }
        if self.gripper == Some(idx) {
            return Self::turn(&self.read_lane, tick) == Some(idx);
        }
        true
    }

    /// Whether an a4 joint's command on `tick` is followed by its 0x92 read.
    fn a4_read(&self, idx: usize, tick: u64) -> bool {
        !self.enabled || Self::turn(&self.read_lane, tick) == Some(idx)
    }
}

/// An a4 joint's position between 0x92 reads: the last read (or the
/// previous carry) advanced along the speed its 0xA4 echo reports this
/// tick. The echo's speed is 1 dps resolution, so over a 15 ms read
/// interval the carry is within ~0.01° of the next read — the read's own
/// resolution — and a joint at rest (speed 0) never drifts.
fn a4_extrapolate(anchor: (f64, Instant), speed: f64, now: Instant) -> f64 {
    anchor.0 + speed * now.saturating_duration_since(anchor.1).as_secs_f64()
}

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
    stiction_ff: f64,
    dither_ff: f64,
    stribeck_ff: f64,
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
        "tick,time_s,seq,slot,motor_id,mode,target_p,cmd_p,cmd_v,cmd_a,cmd_v_fast,meas_p,motor_v,meas_v,meas_tau,gravity_ff,friction_ff,inertia_ff,damping_ff,stiction_ff,dither_ff,stribeck_ff,total_ff,kd_host,damp_w0,damp_q,tick_dt,fb_dt"
    )?;
    Ok(out)
}

fn write_trace_row(out: &mut io::BufWriter<std::fs::File>, r: TraceRow) -> io::Result<()> {
    writeln!(
        out,
        "{},{:.9},{},{},{},{:.1},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.9},{:.9}",
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
        r.stiction_ff,
        r.dither_ff,
        r.stribeck_ff,
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
    /// (side, iface, specs) — side 0 = left, 1 = right.
    buses: Vec<(u8, String, Vec<MotorSpec>)>,
}

fn parse_config(text: &str) -> io::Result<Config> {
    let mut loop_hz = 240.0;
    let mut watchdog_ms = 150.0;
    let mut max_step_rad = 0.35;
    let mut buses: Vec<(u8, String, Vec<MotorSpec>)> = Vec::new();
    let mut proto: Option<u32> = None;

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
            "proto" => {
                let declared: u32 = f
                    .get(1)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| bad(line))?;
                if declared != CONFIG_PROTO {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "config: client speaks proto {declared}, this axol-rt speaks \
                             proto {CONFIG_PROTO} — the almond-axol package and the axol-rt \
                             binary must be built from the same checkout (rebuild with \
                             `axol rt.install`)"
                        ),
                    ));
                }
                proto = Some(declared);
            }
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
            "joint" | "gripper" => {
                // joint <side 0|1> <iface> <name> <motor_id> <kp> <kd>
                //       <max_vel> <max_accel> <fc> <k> <fv> <fo>
                //       <stiction_gain> <stiction_err> <stiction_load_gain>
                //       <dither_nm> <dither_hz> <wire mit|a4|pv>
                //       <stribeck_gain> <stribeck_dfs> <stribeck_load_gain> <stribeck_vs>
                //       <fl> <stribeck_pole>
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
                        stiction_gain: 0.0,
                        stiction_err: 0.0,
                        stiction_load_gain: 0.0,
                        dither_nm: 0.0,
                        dither_hz: 0.0,
                        wire: WireMode::Mit,
                        stribeck_gain: 0.0,
                        stribeck_dfs: 0.0,
                        stribeck_load_gain: 0.0,
                        stribeck_vs: 0.0,
                        fl: 0.0,
                        stribeck_pole: 0.0,
                    }
                } else {
                    let motor_id: u8 = f
                        .get(4)
                        .and_then(|v| v.parse().ok())
                        .ok_or_else(|| bad(line))?;
                    // Arm joint motor ids are 1..=7 in Joint enum order, so the
                    // id fixes the target-tuple slot regardless of which
                    // joints a bus carries: a bench arm with only its wrist
                    // motors keeps them in the wrist slots rather than
                    // sliding down into the shoulders'.
                    if !(1..=GRIPPER_SLOT as u8).contains(&motor_id) {
                        return Err(bad(line));
                    }
                    MotorSpec {
                        joint: f.get(3).ok_or_else(|| bad(line))?.to_string(),
                        motor_id,
                        kp: num(5)?,
                        kd: num(6)?,
                        gripper: false,
                        slot: motor_id as usize - 1,
                        max_vel: num(7)?,
                        max_accel: num(8)?,
                        fc: num(9)?,
                        k: num(10)?,
                        fv: num(11)?,
                        fo: num(12)?,
                        stiction_gain: num(13)?,
                        stiction_err: num(14)?,
                        stiction_load_gain: num(15)?,
                        dither_nm: num(16)?,
                        dither_hz: num(17)?,
                        wire: f
                            .get(18)
                            .and_then(|t| WireMode::parse(t))
                            .ok_or_else(|| bad(line))?,
                        stribeck_gain: num(19)?,
                        stribeck_dfs: num(20)?,
                        stribeck_load_gain: num(21)?,
                        stribeck_vs: num(22)?,
                        fl: num(23)?,
                        stribeck_pole: num(24)?,
                    }
                };
                if spec.slot >= N_SLOTS || bus.2.iter().any(|s| s.slot == spec.slot) {
                    return Err(bad(line));
                }
                bus.2.push(spec);
            }
            _ => return Err(bad(line)),
        }
    }
    if proto.is_none() {
        // A client that predates the `proto` line slots arm joints by list
        // order, which this core no longer does — refuse rather than arm a
        // layout it would then misinterpret.
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "config: no `proto` line — the client predates proto {CONFIG_PROTO}; the \
                 almond-axol package and the axol-rt binary must be built from the same \
                 checkout (rebuild with `axol rt.install`)"
            ),
        ));
    }
    if buses.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "config: no joints",
        ));
    }
    for (_, _, specs) in &buses {
        check_impedance_rate(loop_hz, specs)?;
    }
    Ok(Config {
        loop_hz,
        watchdog_ms,
        max_step_rad,
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
        let expected = [1u8, 2, 0];
        let mut seen = [0u8; 3];
        assert!(mark_unique_expected_reply(&expected, &mut seen, 0));
        assert!(!mark_unique_expected_reply(&expected, &mut seen, 0));
        assert!(!mark_unique_expected_reply(&expected, &mut seen, 2));
        // An 0xA4 joint is budgeted two replies (command echo + 0x92 read)
        // and is only complete once both are in.
        assert!(mark_unique_expected_reply(&expected, &mut seen, 1));
        assert!(!reply_complete(&expected, &seen, 1));
        assert!(mark_unique_expected_reply(&expected, &mut seen, 1));
        assert!(reply_complete(&expected, &seen, 1));
        assert!(!mark_unique_expected_reply(&expected, &mut seen, 1));
        assert!(reply_complete(&expected, &seen, 0));
        assert!(!reply_complete(&expected, &seen, 2));
        assert_eq!(seen, [1, 2, 0]);
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

    const PERIOD: Duration = Duration::from_micros(4_167);
    const ON_TIME: Duration = Duration::from_micros(50);
    const LATE: Duration = Duration::from_micros(800);

    #[test]
    fn timing_health_degrades_on_clustered_late_ticks_and_recovers() {
        // Alternating late/on-time: the eighth late tick in the window
        // degrades the bus (the old limp trip), and it stays degraded
        // (Steady, not re-announced) while the pattern continues.
        let mut health = TimingHealth::default();
        for _ in 0..7 {
            assert_eq!(health.record(LATE, PERIOD), TimingVerdict::Steady);
            assert_eq!(health.record(ON_TIME, PERIOD), TimingVerdict::Steady);
        }
        assert_eq!(health.record(LATE, PERIOD), TimingVerdict::Degraded);
        assert!(health.degraded);

        // Hysteresis: only a fully clean window (32 on-time ticks after the
        // last late one) recovers.
        for _ in 0..31 {
            assert_eq!(health.record(ON_TIME, PERIOD), TimingVerdict::Steady);
            assert!(health.degraded);
        }
        assert_eq!(health.record(ON_TIME, PERIOD), TimingVerdict::Recovered);
        assert!(!health.degraded);

        // Three in a row degrades too.
        let mut consecutive = TimingHealth::default();
        assert_eq!(consecutive.record(LATE, PERIOD), TimingVerdict::Steady);
        assert_eq!(consecutive.record(LATE, PERIOD), TimingVerdict::Steady);
        assert_eq!(consecutive.record(LATE, PERIOD), TimingVerdict::Degraded);
    }

    #[test]
    fn bus_busy_percentiles_report_median_tail_and_peak() {
        let mut busy = vec![0.5, 0.7, 0.6, 0.9, 0.55, 0.65, 0.6, 0.62, 0.58, 0.61, 0.95];
        let (p50, p95, max) = bus_busy_percentiles(&mut busy);
        assert_eq!(p50, 0.61);
        assert_eq!(p95, 0.95);
        assert_eq!(max, 0.95);
        let (a, b, c) = bus_busy_percentiles(&mut []);
        assert!(a.is_nan() && b.is_nan() && c.is_nan());
    }

    #[test]
    fn a4_joints_stay_on_the_position_frame_through_holds_but_not_limp() {
        use crate::bringup::{Vendor, WireMode};
        // Tracked and holding (kp > 0) both take 0xA4 on an a4 MyActuator …
        assert!(a4_wire(Vendor::MyActuator, WireMode::A4, true, 250.0));
        assert!(a4_wire(Vendor::MyActuator, WireMode::A4, false, 250.0));
        // … limp / gravity comp (kp = 0) fall back to MIT for compliance …
        assert!(!a4_wire(Vendor::MyActuator, WireMode::A4, false, 0.0));
        // … and nothing else ever does, whatever the tick.
        assert!(!a4_wire(Vendor::MyActuator, WireMode::Mit, true, 250.0));
        assert!(!a4_wire(Vendor::Damiao, WireMode::A4, true, 250.0));
    }

    #[test]
    fn pv_joints_follow_the_same_hold_rule_on_damiao_only() {
        use crate::bringup::{Vendor, WireMode};
        assert!(pv_wire(Vendor::Damiao, WireMode::Pv, true, 130.0));
        assert!(pv_wire(Vendor::Damiao, WireMode::Pv, false, 130.0));
        assert!(!pv_wire(Vendor::Damiao, WireMode::Pv, false, 0.0));
        assert!(!pv_wire(Vendor::Damiao, WireMode::Mit, true, 130.0));
        assert!(!pv_wire(Vendor::MyActuator, WireMode::Pv, true, 130.0));
        assert_eq!(WireMode::Pv.dm_mode(), proto::DM_MODE_POS_VEL);
        assert_eq!(WireMode::Mit.dm_mode(), proto::DM_MODE_MIT);
    }

    fn ready(id: u8, vendor: Vendor, wire: WireMode) -> ReadyMotor {
        ReadyMotor {
            id,
            joint: format!("m{id}"),
            vendor,
            ranges: proto::MitRanges {
                p_max: 12.5,
                v_max: 30.0,
                kp_max: 500.0,
                kd_max: 5.0,
                t_max: 10.0,
            },
            hold_pos: 0.0,
            holding: false,
            kp: 100.0,
            kd: 1.0,
            gripper: id == 8,
            slot: id as usize - 1,
            max_vel: 9.4,
            max_accel: 33.0,
            fc: 0.0,
            k: 0.0,
            fv: 0.0,
            fo: 0.0,
            stiction_gain: 0.0,
            stiction_err: 0.0,
            stiction_load_gain: 0.0,
            dither_nm: 0.0,
            dither_hz: 0.0,
            wire,
            stribeck_gain: 0.0,
            stribeck_dfs: 0.0,
            stribeck_load_gain: 0.0,
            stribeck_vs: 0.0,
            fl: 0.0,
        }
    }

    fn full_arm_position_controller() -> Vec<ReadyMotor> {
        let mut v: Vec<ReadyMotor> = (1..=5)
            .map(|id| ready(id, Vendor::MyActuator, WireMode::A4))
            .collect();
        v.push(ready(6, Vendor::Damiao, WireMode::Pv));
        v.push(ready(7, Vendor::Damiao, WireMode::Pv));
        v.push(ready(8, Vendor::Damiao, WireMode::Mit));
        v
    }

    #[test]
    fn thinning_is_off_at_240_hz() {
        let motors = full_arm_position_controller();
        let sched = Thinning::plan(&motors, 240.0);
        assert!(!sched.enabled);
        for tick in 0..20 {
            for idx in 0..motors.len() {
                assert!(sched.commanded(idx, tick));
                assert!(sched.a4_read(idx, tick));
            }
        }
    }

    #[test]
    fn thinning_at_400_hz_is_a_fixed_fourteen_frame_tick() {
        let motors = full_arm_position_controller();
        let sched = Thinning::plan(&motors, 400.0);
        assert!(sched.enabled);
        let mut wrist_turns = [0u32; 2];
        let mut gripper_turns = 0u32;
        let mut reads = [0u32; 5];
        for tick in 0..60u64 {
            // Frames this tick: 2 per MyActuator command (echo), 2 per 0x92
            // read, 2 per Damiao command.
            let mut frames = 0;
            for idx in 0..motors.len() {
                if !sched.commanded(idx, tick) {
                    continue;
                }
                frames += 2;
                match idx {
                    0..=4 => {
                        if sched.a4_read(idx, tick) {
                            frames += 2;
                            reads[idx] += 1;
                        }
                    }
                    5 | 6 => wrist_turns[idx - 5] += 1,
                    _ => gripper_turns += 1,
                }
            }
            assert_eq!(frames, 14, "tick {tick}");
            // MyActuator joints are never thinned.
            for idx in 0..5 {
                assert!(sched.commanded(idx, tick));
            }
        }
        // Two wrists alternate: 200 Hz each. Five reads and the gripper share
        // the other lane: 400/6 Hz each.
        assert_eq!(wrist_turns, [30, 30]);
        assert_eq!(gripper_turns, 10);
        assert_eq!(reads, [10; 5]);
    }

    #[test]
    fn thinning_only_reads_a4_joints_and_puts_mit_joints_on_their_own_lane() {
        let motors = vec![
            ready(1, Vendor::MyActuator, WireMode::Mit),
            ready(2, Vendor::MyActuator, WireMode::A4),
            ready(6, Vendor::Damiao, WireMode::Pv),
        ];
        let sched = Thinning::plan(&motors, MIXED_LOOP_HZ);
        assert_eq!(sched.read_lane, vec![1]);
        assert_eq!(sched.dm_lane, vec![2]);
        assert_eq!(sched.mit_lane, vec![(0, 0)]);
        for tick in 0..8 {
            // The impedance joint alternates (240 Hz); the a4 joint and the
            // only pv wrist go every tick, and the a4 joint is read every tick.
            assert_eq!(sched.commanded(0, tick), tick % 2 == 0);
            assert!(sched.commanded(1, tick));
            assert!(sched.commanded(2, tick));
            assert!(sched.a4_read(1, tick));
        }
    }

    /// shoulder_1 + elbow on 0xA4, the rest on impedance: the split that
    /// shook the arm when it all ran at 400 Hz.
    fn mixed_arm() -> Vec<ReadyMotor> {
        vec![
            ready(1, Vendor::MyActuator, WireMode::A4),
            ready(2, Vendor::MyActuator, WireMode::Mit),
            ready(3, Vendor::MyActuator, WireMode::Mit),
            ready(4, Vendor::MyActuator, WireMode::A4),
            ready(5, Vendor::MyActuator, WireMode::Mit),
            ready(6, Vendor::Damiao, WireMode::Mit),
            ready(7, Vendor::Damiao, WireMode::Mit),
            ready(8, Vendor::Damiao, WireMode::Mit),
        ]
    }

    #[test]
    fn a_mixed_bus_commands_every_impedance_joint_at_exactly_240_hz() {
        let motors = mixed_arm();
        let sched = Thinning::plan(&motors, MIXED_LOOP_HZ);
        assert!(sched.enabled);
        assert_eq!(sched.mit_div, 2);
        // The wrists are impedance joints here, so they ride the impedance
        // lane rather than taking turns with each other.
        assert!(sched.dm_lane.is_empty());
        let mit: Vec<usize> = sched.mit_lane.iter().map(|(i, _)| *i).collect();
        assert_eq!(mit, vec![1, 2, 4, 5, 6]);
        let mut last: [Option<u64>; 8] = [None; 8];
        let mut worst_frames = 0;
        for tick in 0..240u64 {
            let mut frames = 0;
            let mut mit_this_tick = 0;
            for idx in 0..motors.len() {
                if !sched.commanded(idx, tick) {
                    continue;
                }
                frames += 2;
                if let Some(prev) = last[idx] {
                    if sched.mit_phase(idx).is_some() {
                        // Evenly spaced: exactly every other tick, never 1 or 3.
                        assert_eq!(tick - prev, 2, "motor {idx}");
                    }
                }
                last[idx] = Some(tick);
                if sched.mit_phase(idx).is_some() {
                    mit_this_tick += 1;
                }
                if motors[idx].wire == WireMode::A4 {
                    assert_eq!(tick - last[idx].unwrap(), 0);
                    if sched.a4_read(idx, tick) {
                        frames += 2;
                    }
                }
            }
            // Five impedance joints split 3 / 2 across the two phases.
            assert!(mit_this_tick == 2 || mit_this_tick == 3);
            // The 0xA4 joints go every tick.
            assert!(sched.commanded(0, tick) && sched.commanded(3, tick));
            worst_frames = worst_frames.max(frames);
        }
        // Fits the 480 Hz tick: 2.08 ms less the reply guard at ~0.13 ms a
        // frame is ~14 frames.
        assert!(worst_frames <= 14, "{worst_frames} frames");
    }

    #[test]
    fn the_position_controller_has_no_impedance_lane() {
        let sched = Thinning::plan(&full_arm_position_controller(), 400.0);
        assert!(sched.mit_lane.is_empty());
        assert_eq!(sched.mit_div, 1);
        // An all-impedance arm at 240 Hz is not thinned at all.
        let all_mit: Vec<ReadyMotor> = (1..=7)
            .map(|id| {
                ready(
                    id,
                    if id <= 5 {
                        Vendor::MyActuator
                    } else {
                        Vendor::Damiao
                    },
                    WireMode::Mit,
                )
            })
            .collect();
        let sched = Thinning::plan(&all_mit, IMPEDANCE_HZ);
        assert!(!sched.enabled && sched.mit_lane.is_empty());
        assert!((0..7).all(|i| sched.commanded(i, 1)));
    }

    #[test]
    fn impedance_joints_run_at_240_hz_only() {
        let spec = |wire: &str, gripper: bool| {
            let text = if gripper {
                "proto 11\ngripper 0 canL 8\n".to_string()
            } else {
                format!(
                    "proto 11\njoint 0 canL shoulder_1 1 250 3.5 9.4 33.0 0.6 250 0.15 0.02 0 0 0 0 60 {wire} 0 0.3 0.1 0.1 0 20\n"
                )
            };
            text
        };
        for hz in [240.0, 480.0] {
            assert!(parse_config(&format!("loop_hz {hz}\n{}", spec("mit", false))).is_ok());
        }
        for hz in [200.0, 300.0, 400.0, 960.0] {
            let err = parse_config(&format!("loop_hz {hz}\n{}", spec("mit", false)))
                .err()
                .expect("refused")
                .to_string();
            assert!(err.contains("240 Hz only"), "{err}");
        }
        // Firmware-loop joints, and the always-MIT gripper, are not held to it.
        assert!(parse_config(&format!("loop_hz 400\n{}", spec("a4", false))).is_ok());
        assert!(parse_config(&format!("loop_hz 400\n{}", spec("", true))).is_ok());
    }

    #[test]
    fn a4_carry_follows_the_echo_speed_and_holds_at_rest() {
        let t0 = Instant::now();
        let t1 = t0 + Duration::from_millis(10);
        let p = a4_extrapolate((1.0, t0), 0.5, t1);
        assert!((p - 1.005).abs() < 1e-9);
        assert_eq!(a4_extrapolate((1.0, t0), 0.0, t1), 1.0);
        // A clock that has not advanced (or ran backwards) adds nothing.
        assert_eq!(a4_extrapolate((1.0, t1), 3.0, t0), 1.0);
    }

    #[test]
    fn timing_health_isolated_overrun_degrades_not_limps() {
        // The field record: one 60 ms stall in an otherwise perfect stream.
        let mut health = TimingHealth::default();
        for _ in 0..1000 {
            assert_eq!(health.record(ON_TIME, PERIOD), TimingVerdict::Steady);
        }
        assert_eq!(
            health.record(Duration::from_millis(60), PERIOD),
            TimingVerdict::Degraded
        );
        assert!(health.degraded);
        assert_eq!(health.recent_overruns.count_ones(), 1);
        assert_eq!(health.recent_late.count_ones(), 1);
        // The next tick is on time again; damping stays off for the window...
        for _ in 0..31 {
            assert_eq!(health.record(ON_TIME, PERIOD), TimingVerdict::Steady);
        }
        // ...then comes back.
        assert_eq!(health.record(ON_TIME, PERIOD), TimingVerdict::Recovered);
        assert_eq!(health.recent_overruns, 0);
    }

    #[test]
    fn timing_health_never_escalates_past_degraded() {
        // Repeated whole-cycle overruns inside one window: still degraded,
        // never a fault — the loop stays in the degraded stretch (Steady,
        // damping off) and the stretch is what the log reports.
        let mut health = TimingHealth::default();
        assert_eq!(health.record(PERIOD, PERIOD), TimingVerdict::Degraded);
        for _ in 0..20 {
            assert_eq!(health.record(ON_TIME, PERIOD), TimingVerdict::Steady);
        }
        assert_eq!(health.record(PERIOD * 3, PERIOD), TimingVerdict::Steady);
        assert!(health.degraded);
        assert_eq!(health.recent_overruns.count_ones(), 2);
        // The window must be clean again before it recovers.
        for _ in 0..31 {
            assert_eq!(health.record(ON_TIME, PERIOD), TimingVerdict::Steady);
        }
        assert_eq!(health.record(ON_TIME, PERIOD), TimingVerdict::Recovered);

        // Two overruns further apart than the window are two degraded
        // episodes.
        let mut spaced = TimingHealth::default();
        assert_eq!(spaced.record(PERIOD, PERIOD), TimingVerdict::Degraded);
        for _ in 0..31 {
            assert_eq!(spaced.record(ON_TIME, PERIOD), TimingVerdict::Steady);
        }
        assert_eq!(spaced.record(ON_TIME, PERIOD), TimingVerdict::Recovered);
        assert_eq!(spaced.record(PERIOD, PERIOD), TimingVerdict::Degraded);

        // Late on every other tick for a long stretch: one Degraded
        // transition, then Steady for as long as it lasts.
        let mut persistent = TimingHealth::default();
        let mut verdicts = Vec::new();
        for _ in 0..500 {
            verdicts.push(persistent.record(LATE, PERIOD));
            verdicts.push(persistent.record(ON_TIME, PERIOD));
        }
        assert_eq!(
            verdicts
                .iter()
                .filter(|v| **v == TimingVerdict::Degraded)
                .count(),
            1
        );
        assert!(!verdicts.contains(&TimingVerdict::Recovered));
        assert!(persistent.degraded);
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
            "proto 11\n\
             loop_hz 240\n\
             joint 0 canL shoulder_1 1 250 3.5 9.4 33.0 0.6 250 0.15 0.02 0 0 0 0 60 mit 0 0.3 0.1 0.1 0 20\n\
             joint 0 canL shoulder_2 2 250 3.5 9.4 33.0 0.5 250 0.10 0.0 0 0 0 0 60 mit 0 0.3 0.1 0.1 0 20\n\
             gripper 0 canL 8\n\
             joint 0 canL shoulder_3 3 180 2.0 9.4 33.0 0.4 250 0.08 0.0 0.6 0.0017 0.2 1.5 60 a4 0.7 0.3 0.1 0.1 0.08 40\n",
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
        assert_eq!((specs[0].stiction_gain, specs[0].stiction_err), (0.0, 0.0));
        assert_eq!(
            (specs[3].stiction_gain, specs[3].stiction_err),
            (0.6, 0.0017)
        );
        assert_eq!(
            (specs[0].stiction_load_gain, specs[3].stiction_load_gain),
            (0.0, 0.2)
        );
        assert_eq!((specs[0].dither_nm, specs[0].dither_hz), (0.0, 60.0));
        assert_eq!((specs[3].dither_nm, specs[3].dither_hz), (1.5, 60.0));
        assert_eq!(
            (specs[0].wire, specs[3].wire),
            (WireMode::Mit, WireMode::A4)
        );
        assert_eq!(specs[0].stribeck_gain, 0.0);
        assert_eq!(
            (
                specs[3].stribeck_gain,
                specs[3].stribeck_dfs,
                specs[3].stribeck_load_gain,
                specs[3].stribeck_vs
            ),
            (0.7, 0.3, 0.1, 0.1)
        );
        assert_eq!((specs[0].fl, specs[3].fl), (0.0, 0.08));
        assert_eq!(
            (specs[0].stribeck_pole, specs[3].stribeck_pole),
            (20.0, 40.0)
        );
        // An unknown wire token is a bad line, not a silent MIT.
        assert!(parse_config(
            "proto 11\njoint 0 canL shoulder_1 1 250 3.5 9.4 33.0 0.6 250 0.15 0.02 0 0 0 0 60 a9 0 0.3 0.1 0.1 0 20\n"
        )
        .is_err());
        // A joint line missing the tracker/friction params (the previous
        // 7-field layout) must be rejected, not defaulted.
        assert!(parse_config("proto 11\njoint 0 canL shoulder_1 1 250 3.5\n").is_err());
        // ... and so must the proto-2 … 8 layouts (13 … 24 fields).
        assert!(parse_config(
            "proto 11\njoint 0 canL shoulder_1 1 250 3.5 9.4 33.0 0.6 250 0.15 0.02\n"
        )
        .is_err());
        assert!(parse_config(
            "proto 11\njoint 0 canL shoulder_1 1 250 3.5 9.4 33.0 0.6 250 0.15 0.02 0 0\n"
        )
        .is_err());
        assert!(parse_config(
            "proto 11\njoint 0 canL shoulder_1 1 250 3.5 9.4 33.0 0.6 250 0.15 0.02 0 0 0\n"
        )
        .is_err());
        assert!(parse_config(
            "proto 11\njoint 0 canL shoulder_1 1 250 3.5 9.4 33.0 0.6 250 0.15 0.02 0 0 0 0 60\n"
        )
        .is_err());
        assert!(parse_config(
            "proto 11\njoint 0 canL shoulder_1 1 250 3.5 9.4 33.0 0.6 250 0.15 0.02 0 0 0 0 60 mit\n"
        )
        .is_err());
        assert!(parse_config(
            "proto 11\njoint 0 canL shoulder_1 1 250 3.5 9.4 33.0 0.6 250 0.15 0.02 0 0 0 0 60 mit 0 0.3 0.1 0.1\n"
        )
        .is_err());
        assert!(parse_config(
            "proto 11\njoint 0 canL shoulder_1 1 250 3.5 9.4 33.0 0.6 250 0.15 0.02 0 0 0 0 60 mit 0 0.3 0.1 0.1 0\n"
        )
        .is_err());
    }

    /// A bus carrying only some of the arm joints (a bench wrist assembly)
    /// keeps each motor in the slot Python's Joint enum assigns it — the
    /// slot comes from the motor id, not from the order joints are listed.
    #[test]
    fn parse_config_subset_keeps_joint_slots() {
        let cfg = parse_config(
            "proto 11\n\
             joint 0 can0 wrist_2 6 40 1.0 9.4 33.0 0.0 0.0 0.0 0.0 0 0 0 0 60 mit 0 0.3 0.1 0.1 0 20\n\
             joint 0 can0 wrist_3 7 40 1.0 9.4 33.0 0.0 0.0 0.0 0.0 0 0 0 0 60 mit 0 0.3 0.1 0.1 0 20\n\
             gripper 0 can0 8\n",
        )
        .unwrap();
        let specs = &cfg.buses[0].2;
        assert_eq!(
            specs.iter().map(|s| s.slot).collect::<Vec<_>>(),
            vec![5, 6, GRIPPER_SLOT]
        );
        // Arm joint ids outside 1..=7 have no slot; a repeated id would
        // double-book one.
        assert!(parse_config(
            "proto 11\njoint 0 can0 wrist_3 8 40 1.0 9.4 33.0 0 0 0 0 0 0 0 0 60 mit 0 0.3 0.1 0.1 0 20\n"
        )
        .is_err());
        assert!(parse_config(
            "proto 11\njoint 0 can0 bogus 0 40 1.0 9.4 33.0 0 0 0 0 0 0 0 0 60 mit 0 0.3 0.1 0.1 0 20\n"
        )
        .is_err());
        assert!(parse_config(
            "proto 11\n\
             joint 0 can0 wrist_2 6 40 1.0 9.4 33.0 0 0 0 0 0 0 0 0 60 mit 0 0.3 0.1 0.1 0 20\n\
             joint 0 can0 wrist_2 6 40 1.0 9.4 33.0 0 0 0 0 0 0 0 0 60 mit 0 0.3 0.1 0.1 0 20\n"
        )
        .is_err());
    }

    /// A client and a core built from different checkouts must fail at
    /// configure time. Without the guard, a proto-1 core (list-order slots)
    /// armed against a proto-2 client's subset config and then rejected every
    /// target on the max-step gate — the arms enabled and never moved.
    #[test]
    fn parse_config_requires_matching_proto() {
        let joint =
            "joint 0 can0 wrist_2 6 40 1.0 9.4 33.0 0 0 0 0 0 0 0 0 60 mit 0 0.3 0.1 0.1 0 20\n";
        let error_of = |text: &str| match parse_config(text) {
            Ok(_) => panic!("accepted a skewed config: {text:?}"),
            Err(err) => err.to_string(),
        };
        // No `proto` line: a client that predates the slot-by-id layout.
        let err = error_of(joint);
        assert!(err.contains("no `proto` line"), "{err}");
        assert!(err.contains("axol rt.install"), "{err}");
        // A future client generation this core does not understand.
        let err = error_of(&format!("proto 99\n{joint}"));
        assert!(err.contains("proto 99"), "{err}");
        assert!(err.contains("proto 11"), "{err}");
        // Malformed declarations are bad lines, not silently accepted.
        assert!(parse_config(&format!("proto\n{joint}")).is_err());
        assert!(parse_config(&format!("proto two\n{joint}")).is_err());
        // Order does not matter; the line just has to be there.
        assert!(parse_config(&format!("{joint}proto 11\n")).is_ok());
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
    // Lock memory before anything the bus threads will touch is mapped:
    // MCL_FUTURE then covers the socket buffers, the channel, and the
    // threads' stacks as they are created (see the Safety notes).
    let memory_locked = match stall::lock_memory() {
        Ok(()) => {
            println!("axol-rt serve: memory locked (mlockall)");
            true
        }
        Err(err) => {
            println!(
                "axol-rt serve: could not lock memory ({err}); running unlocked — a page fault can stall a bus thread (grant CAP_IPC_LOCK / raise RLIMIT_MEMLOCK)"
            );
            false
        }
    };
    let listener = UnixListener::bind(socket_path)?;
    println!("axol-rt serve: listening on {socket_path}");
    let (mut stream, _) = listener.accept()?;
    println!("axol-rt serve: client connected");

    let (out_tx, out_rx) = mpsc::channel::<Vec<u8>>();
    if !memory_locked {
        // Repeat over the link so the warning lands in the session log, not
        // only on the core's stdout.
        send_text(
            &out_tx,
            b'W',
            "memory not locked (mlockall failed) — a page fault can stall a bus thread; grant CAP_IPC_LOCK or raise RLIMIT_MEMLOCK",
        );
    }
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
    // Set only by an explicit `D` from the client. It is the one exit that
    // disables the motors; every other way out of the bus loops (fault,
    // signal, peer loss, protocol error) stops commanding and leaves the
    // motors holding their last MIT command on firmware gains — dropping
    // the arms is never an acceptable failure response.
    let disarm = Arc::new(AtomicBool::new(false));
    // Set by a bus thread on a loss-of-trust fault (a silent motor):
    // both buses drop to gravity comp (kp = 0, streamed gravity t_ff) and
    // keep serving so the operator can hand-guide the arms to rest. Never
    // cleared within a session; a disarm in this state leaves the motors
    // limp rather than disabling.
    let limp = Arc::new(AtomicBool::new(false));
    let mut bus_threads: Vec<std::thread::JoinHandle<io::Result<()>>> = Vec::new();

    // Errors below `break` out (never early-return): the cleanup after the
    // loop must always run so the bus threads stop streaming and the socket
    // is torn down in order.
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
            // Peer died while armed (or closed after a fault). Stop
            // streaming and exit; the motors keep holding their last
            // command on firmware gains, exactly as when a classic Python
            // session died mid-command.
            println!(
                "axol-rt serve: client disconnected while armed — exiting; motors left holding (not disabled)"
            );
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
                    match step {
                        Ok(held) if !held.is_empty() => send_text(
                            &out_tx,
                            b'L',
                            &format!(
                                "{iface}: already holding, attached without reset: {}",
                                held.join(", ")
                            ),
                        ),
                        Ok(_) => {}
                        Err(err) => {
                            send_text(&out_tx, b'S', &format!("fault: prep {iface}: {err}"));
                            ok = false;
                            break;
                        }
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
                disarm.store(false, Ordering::SeqCst);
                limp.store(false, Ordering::SeqCst);
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
                    let disarm = Arc::clone(&disarm);
                    let limp = Arc::clone(&limp);
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
                            &disarm,
                            &limp,
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
                // The deliberate operator stop: the only path that disables
                // the motors. After a fault the bus loops have already
                // stopped without disabling, and after a limp transition
                // they are serving gravity comp; either way this ack just
                // lets the client tear down in order — torque stays as it is
                // (holding, or limp with gravity feedforward).
                let faulted = fault.load(Ordering::SeqCst) != 0;
                let limped = limp.load(Ordering::SeqCst);
                if !faulted && !limped {
                    disarm.store(true, Ordering::SeqCst);
                } else {
                    send_text(
                        &out_tx,
                        b'L',
                        if limped {
                            "disarm while limp: motors left at kp = 0 with their last gravity feedforward (not disabled)"
                        } else {
                            "disarm after fault: motors left holding their last command (not disabled)"
                        },
                    );
                }
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

    // Signal, peer loss, or protocol error: stop streaming and exit. The
    // motors are *not* disabled here — `disarm` is still false, so the bus
    // loops leave them holding their last command rather than dropping the
    // arms on a host-side failure.
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
    disarm: &AtomicBool,
    limp: &AtomicBool,
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
    // Late-target holdover, per slot: when Python's tick is late the
    // tracker is given the last target carried forward at the stream's own
    // velocity instead of a target that stops dead and then jumps (see
    // filter::Holdover). Reach is bounded by the same corruption limit as a
    // raw target step. The stream's cadence is learned from arrivals
    // (filter::Cadence — follows a faster stream at once, a slower one
    // after a run of long gaps, and ignores isolated late arrivals).
    let mut hold: Vec<Holdover> = (0..N_SLOTS)
        .map(|_| Holdover::new(HOLDOVER_MAX, cfg.max_step_rad))
        .collect();
    let mut cadence = Cadence::new();
    // Ticks whose target was late enough for the holdover to carry it, and
    // the oldest a target has been when a fresh one landed — the host-side
    // stall the core papered over, for the 5 s stats line.
    let mut held_ticks: u64 = 0;
    let mut worst_target_age: f64 = 0.0;
    let mut carrying = false;
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
        /// Slower (20 rad/s) measured-velocity estimate for the Stribeck
        /// term: smooth enough at 0.05 rad/s that the feedforward does not
        /// step with the encoder, ~30° behind at the 2 Hz cycle.
        v_meas_slow: LpDiff,
        vel_meas_slow: f64,
        last_fb: Option<Instant>,
        /// Torque-dither oscillator phase (`filter::dither_step`), started a
        /// golden angle apart per slot.
        dither_phase: f64,
    }
    let mut damp: Vec<Damp> = (0..N_SLOTS)
        .map(|slot| Damp {
            v_cmd: LpDiff::new(CONTROL_CUTOFF),
            a_cmd: LpDiff::new(CONTROL_CUTOFF),
            v_cmd_fast: LpDiff::new(VEL_CUTOFF),
            v_meas: LpDiff::new(VEL_CUTOFF),
            bp: BandPass::new(),
            vel_meas: 0.0,
            v_meas_slow: LpDiff::new(
                specs
                    .iter()
                    .find(|s| s.slot == slot && !s.gripper && s.stribeck_pole > 0.0)
                    .map(|s| s.stribeck_pole)
                    .unwrap_or(CONTROL_CUTOFF),
            ),
            vel_meas_slow: 0.0,
            last_fb: None,
            dither_phase: slot as f64 * filter::DITHER_PHASE_STAGGER,
        })
        .collect();
    let mut prev_tick: Option<Instant> = None;
    // Impedance joints on their own 240 Hz cadence (`Thinning::mit_lane`):
    // when each last ran, for a time step and overrun of its own.
    let mut own_prev: [Option<Instant>; N_SLOTS] = [None; N_SLOTS];
    // Latest decoded feedback per slot, shipped to Python once per tick as
    // an `F` packet — the core is the only CAN consumer; Python fills its
    // Motor caches from these instead of passively reading the bus.
    let mut latest: [SlotFeedback; N_SLOTS] = [None; N_SLOTS];
    // Whether the immediately preceding tick produced a fresh sample for
    // each slot. Host damping is suppressed for one tick after a miss; the
    // firmware's local kd remains active without relying on stale host state.
    let mut feedback_fresh = [false; N_SLOTS];
    let mut have_target = false;
    let mut last_seq: Option<u32> = None;
    let mut last_arrival: Option<Instant> = None;
    // Arrival of the last *accepted* target: the holdover's clock. A
    // rejected packet feeds the watchdog (the client is alive) but must not
    // restart the carry — after a stall the catch-up step can exceed
    // max_step_rad, and resetting the age there would yank the tracker from
    // the carried pose back onto the last accepted target.
    let mut last_accepted: Option<Instant> = None;

    let period = Duration::from_secs_f64(1.0 / cfg.loop_hz);
    let watchdog = Duration::from_secs_f64(cfg.watchdog_ms / 1e3);
    let mut rejected: u64 = 0;
    let mut next_reject_log = Instant::now();
    let mut late: u64 = 0;
    let mut missed: u64 = 0;
    let mut trace_dropped: u64 = 0;
    let mut ticks: u64 = 0;
    let mut watchdog_frozen = false;
    let mut next_stats = Instant::now() + Duration::from_secs(5);
    // Bus occupancy per tick: tick start → last reply received, as a
    // fraction of the period. This is what bounds the loop rate — at 240 Hz
    // with three a4 joints (two request/reply pairs each) the right arm is
    // estimated near three quarters of a 1 Mbps bus — so it is reported in
    // the 5 s stats line to size any rate change against a measurement.
    let mut bus_busy: Vec<f64> = Vec::with_capacity(2048);
    let mut bus_last_reply: Option<Instant> = None;
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
    let mut timing_degraded_ticks: u64 = 0;
    let mut timing_degraded_episodes: u64 = 0;
    let mut timing_announced = false;
    let mut next_timing_log = Instant::now();
    let mut overruns: u64 = 0;
    // Per-thread scheduler/memory counters, sampled at every wake so a late
    // tick can be attributed to what actually kept the thread off the CPU.
    let mut stall_probe = stall::StallProbe::open();
    // Per-tick reply bookkeeping, allocated once: the loop must not grow
    // the heap (a fresh page is a fault, see `stall::lock_memory`).
    let mut expected = vec![0u8; motors.len()];
    let mut seen = vec![0u8; motors.len()];
    // 0xA4 joints: the command reply arrives before the 0x92 position read;
    // its speed and iq are staged here until the fine position completes
    // the sample.
    let mut a4_stage: [(f64, f64); N_SLOTS] = [(0.0, 0.0); N_SLOTS];
    let mut a4_follow = vec![false; motors.len()];
    // Which motors this tick tried to command (a thinned motor's off-tick
    // is not a missed reply; a dropped send still is).
    let mut attempted = vec![false; motors.len()];
    // 0xA4 joints: the last 0.01° position and when it was taken, carried
    // forward on the echo's speed between reads (`a4_extrapolate`). Seeded
    // from bring-up's own 0x92 read so the first echo-only tick has a base.
    let mut a4_anchor: [Option<(f64, Instant)>; N_SLOTS] = [None; N_SLOTS];
    for m in motors.iter() {
        if m.vendor == Vendor::MyActuator && m.wire == WireMode::A4 {
            a4_anchor[m.slot] = Some((m.hold_pos, Instant::now()));
        }
    }
    // Damiao wrists: the control-mode register the motor is in (bring-up
    // put it in the wire's mode). The frame the core wants can change
    // (pv ↔ MIT across limp / gravity comp), and the register must follow
    // it or the firmware ignores the frame.
    let mut dm_mode: Vec<u32> = motors
        .iter()
        .map(|m| {
            if m.vendor == Vendor::Damiao && !m.gripper {
                m.wire.dm_mode()
            } else {
                0
            }
        })
        .collect();
    let sched = Thinning::plan(&motors, cfg.loop_hz);
    if !sched.mit_lane.is_empty() {
        send_text(
            out_tx,
            b'L',
            &format!(
                "{iface}: {:.0} Hz loop — {} impedance joint(s) on alternate ticks at {:.0} Hz each, firmware-loop joints every tick",
                cfg.loop_hz,
                sched.mit_lane.len(),
                cfg.loop_hz / sched.mit_div as f64,
            ),
        );
    }
    if sched.enabled {
        send_text(
            out_tx,
            b'L',
            &format!(
                "{iface}: {:.0} Hz loop — bus schedule thinned: {} wrist(s) take turns, {} read-lane entries (a4 position reads + gripper) take turns",
                cfg.loop_hz,
                sched.dm_lane.len(),
                sched.read_lane.len(),
            ),
        );
    }
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
            let overrun = lateness >= period;
            if overrun {
                overruns += 1;
            }
            // Bracket this wake with the thread's own scheduler/memory
            // counters (µs, before any bus work) so a late tick is attributed
            // — preempted, page fault, kernel stall — not just measured.
            let stall = stall_probe.sample();
            // A whole-cycle overrun means the causal sample/command ordering
            // was lost for this tick; clustered lateness says timing cannot be
            // trusted more slowly. Phase-sensitive host damping must not run
            // on either, so the bus goes *timing-degraded*: damping off on
            // every joint until a clean window, and the overrun tick's command
            // derivatives re-seeded rather than integrated across the gap
            // (firmware kp/kd and the streamed gravity t_ff are unaffected,
            // so the arm keeps holding). That is the whole response — timing
            // never takes the session limp (see `DEGRADED_RECENT_LATE_TICKS`);
            // a stretch that stays bad just stays degraded, with its repeat
            // overruns logged so the persistence is visible.
            let is_limp = limp.load(Ordering::SeqCst);
            match timing_health.record(lateness, period) {
                TimingVerdict::Steady => {
                    if overrun && timing_health.degraded && began >= next_timing_log {
                        next_timing_log = began + DEGRADED_LOG_INTERVAL;
                        timing_announced = true;
                        send_text(
                            out_tx,
                            b'W',
                            &format!(
                                "{iface}: control timing still degraded ({:.3} ms late, {} of the last 32 ticks late, {} whole-cycle overruns; {}) — host damping stays off on this bus; firmware gains hold",
                                lateness.as_secs_f64() * 1e3,
                                timing_health.recent_late.count_ones(),
                                timing_health.recent_overruns.count_ones(),
                                stall.describe(lateness),
                            ),
                        );
                    }
                }
                TimingVerdict::Degraded => {
                    timing_degraded_episodes += 1;
                    // Rate-limited like the feedback transitions; the stats
                    // line carries the cumulative counts regardless.
                    timing_announced = began >= next_timing_log;
                    if timing_announced {
                        next_timing_log = began + DEGRADED_LOG_INTERVAL;
                        send_text(
                            out_tx,
                            b'W',
                            &format!(
                                "{iface}: control timing degraded ({:.3} ms late, {} of the last 32 ticks late, {} whole-cycle overrun{}; {}) — host damping off on this bus until a clean window; firmware gains hold",
                                lateness.as_secs_f64() * 1e3,
                                timing_health.recent_late.count_ones(),
                                timing_health.recent_overruns.count_ones(),
                                if timing_health.recent_overruns.count_ones() == 1 { "" } else { "s" },
                                stall.describe(lateness),
                            ),
                        );
                    }
                }
                TimingVerdict::Recovered => {
                    if std::mem::take(&mut timing_announced) {
                        send_text(
                            out_tx,
                            b'L',
                            &format!("{iface}: control timing recovered — host damping resumed"),
                        );
                    }
                }
            }
            if timing_health.degraded {
                timing_degraded_ticks += 1;
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
                        // Limp: p_des carries no torque (kp = 0) and follows
                        // the hand-guided arm, so a large step is normal and
                        // the fresh gravity t_ff it carries must not be lost.
                        // Only slots with a motor on this bus are gated: an
                        // absent joint's slot never leaves its default hold,
                        // so its (meaningless) target would otherwise reject
                        // every packet. A NaN target is a rejected step too.
                        let worst = if is_limp {
                            None
                        } else {
                            motors
                                .iter()
                                .filter(|m| !m.gripper)
                                .map(|m| (m, (t.cmds[m.slot].p_des - play[m.slot].p_des).abs()))
                                .filter(|(_, step)| step.is_nan() || *step > cfg.max_step_rad)
                                .max_by(|a, b| a.1.total_cmp(&b.1))
                        };
                        // Spacing since the previous *accepted* target — the
                        // holdover's velocity baseline and the cadence sample.
                        let gap = last_accepted
                            .map(|prev| t.arrival.saturating_duration_since(prev).as_secs_f64());
                        match worst {
                            None => {
                                play = t.cmds;
                                have_target = true;
                                last_accepted = Some(t.arrival);
                                if let Some(gap) = gap {
                                    cadence.observe(gap);
                                }
                                for m in &motors {
                                    let h = &mut hold[m.slot];
                                    let c = &play[m.slot];
                                    if m.gripper || is_limp || c.mode < 0.5 {
                                        // Gripper targets legitimately jump;
                                        // passthrough/limp targets follow the
                                        // hand or hold — neither is a
                                        // trajectory to extrapolate.
                                        h.reset();
                                    } else {
                                        h.observe(c.p_des, gap, cadence.get());
                                    }
                                }
                            }
                            Some((m, step)) => {
                                rejected += 1;
                                // The holdover is left alone: its carry is
                                // already gliding to rest within HOLDOVER_MAX
                                // of the last accepted target, and the
                                // velocity estimate only ever takes accepted
                                // targets (see last_accepted).
                                // A rejected target is a hold the client did
                                // not ask for. One corrupt packet is what the
                                // gate is for, but a *stream* of rejections
                                // (a client whose slot layout disagrees with
                                // this core's, a target frame the offsets
                                // never lined up with) leaves the arm parked
                                // while the client believes it is sweeping —
                                // say so, rate-limited like the other
                                // degradations; the stats line has the count.
                                if began >= next_reject_log {
                                    next_reject_log = began + DEGRADED_LOG_INTERVAL;
                                    send_text(
                                        out_tx,
                                        b'W',
                                        &format!(
                                            "{iface}: target rejected — {} steps {:.3} rad from the last accepted target (max_step_rad {:.3}); holding the last accepted target ({rejected} rejected so far)",
                                            m.joint,
                                            step,
                                            cfg.max_step_rad,
                                        ),
                                    );
                                }
                            }
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
            // An overrun is the exception: the motors held the previous
            // command for the whole gap, so the tracker advances one nominal
            // period rather than rendering a max-velocity catch-up step over
            // the wall-clock it lost, and the command derivatives are
            // re-seeded (below) instead of differentiated across it.
            let tick_dt = prev_tick.map_or(0.0, |p| began.duration_since(p).as_secs_f64());
            prev_tick = Some(began);
            let cmd_dt = if overrun {
                period.as_secs_f64()
            } else {
                tick_dt
            };
            // Age of the latest accepted target this tick, for the holdover.
            let target_age =
                last_accepted.map_or(0.0, |a| began.saturating_duration_since(a).as_secs_f64());
            carrying = have_target
                && !watchdog_frozen
                && cadence
                    .get()
                    .is_some_and(|c| target_age > Holdover::SLACK * c);
            if carrying {
                held_ticks += 1;
                worst_target_age = worst_target_age.max(target_age);
            }

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
            expected.fill(0);
            attempted.fill(false);
            let mut trace_pending: [Option<TraceRow>; N_SLOTS] = [None; N_SLOTS];
            for (motor_index, m) in motors.iter().enumerate() {
                // An impedance joint on its own 240 Hz cadence runs nothing on
                // its off-ticks — not the tracker, derivatives, band-pass or
                // dither — and on its own ticks steps them over its own
                // interval, so it is the verified 240 Hz loop, interleaved.
                // Every other motor steps at the loop's rate as before.
                let (tick_dt, cmd_dt, overrun) = if sched.mit_phase(motor_index).is_some() {
                    if !sched.commanded(motor_index, ticks) {
                        continue;
                    }
                    let nominal = period.as_secs_f64() * sched.mit_div as f64;
                    let dt =
                        own_prev[m.slot].map_or(0.0, |p| began.duration_since(p).as_secs_f64());
                    own_prev[m.slot] = Some(began);
                    // Its own cycle slipped a whole period: the same rule the
                    // loop applies to itself, at the joint's own rate.
                    let own_overrun = dt >= 2.0 * nominal;
                    (dt, if own_overrun { nominal } else { dt }, own_overrun)
                } else {
                    (tick_dt, cmd_dt, overrun)
                };
                let c = if is_limp && !m.gripper {
                    // Limp is enforced here, whatever the target says: no
                    // stiffness, firmware damping only, no host damping or
                    // inertia term, passthrough mode. Only the streamed
                    // gravity t_ff (Python evaluates it at the measured pose
                    // while limp) and p_des (informational) pass through.
                    // The gripper keeps its own command — Python's gravity
                    // comp holds it softly, or drives it as asked.
                    JointCmd {
                        mode: 0.0,
                        kp: 0.0,
                        kd: LIMP_KD,
                        kd_host: 0.0,
                        j_eff: 0.0,
                        ..play[m.slot]
                    }
                } else {
                    play[m.slot]
                };
                let c = &c;
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
                    // The target the tracker chases: the latest streamed
                    // one, carried forward along the stream's velocity when
                    // that target is late (identity while the stream is on
                    // time — see filter::Holdover and the adoption above).
                    let p_tgt = if tracked {
                        hold[m.slot].target(c.p_des, target_age, cadence.get())
                    } else {
                        c.p_des
                    };
                    let p_cmd = if tracked {
                        let (p, _, _) = trk[m.slot].update(p_tgt, cmd_dt);
                        p
                    } else {
                        trk[m.slot].seed(c.p_des);
                        c.p_des
                    };
                    let d = &mut damp[m.slot];
                    let (
                        v_wire,
                        a_cmd,
                        v_cmd_fast,
                        friction_ff,
                        inertia_ff,
                        v_damp,
                        stiction_ff,
                        dither_ff,
                        stribeck_ff,
                    ) = if tracked && overrun {
                        // The gap since the last command is not a trajectory
                        // segment the motor followed — it held. Re-prime the
                        // derivative chains at rest here so the first tick
                        // back carries no fictitious velocity, acceleration
                        // (inertia torque), or band-pass energy; they ramp
                        // in again from the next tick as tracking resumes.
                        d.v_cmd.seed(p_cmd);
                        d.a_cmd.seed(0.0);
                        d.v_cmd_fast.seed(p_cmd);
                        d.bp.reset();
                        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    } else if tracked {
                        // Match classic AxolArm.motion_control: friction uses
                        // the 20 rad/s low-pass position derivative, inertia
                        // uses a second identical derivative, and damping
                        // uses its independent 80 rad/s desired-velocity
                        // derivative.  Only the source position/rate differ:
                        // the core can use the trajectory it really sends.
                        let v_cmd = d.v_cmd.update(p_cmd, tick_dt);
                        let a_cmd = d.a_cmd.update(v_cmd, tick_dt);
                        let v_cmd_fast = d.v_cmd_fast.update(p_cmd, tick_dt);
                        // Sliding friction grows with the torque the gear
                        // meshes carry (`fl`, Nm per Nm of gravity).
                        let fc_eff = m.fc + m.fl * c.t_ff.abs();
                        let friction_ff = filter::friction(v_cmd, fc_eff, m.k, m.fv, m.fo);
                        // Stiction compensation acts on the measured error
                        // against the latest accepted feedback. It is a slow
                        // term (it resolves a stuck joint over tens of ms),
                        // so a one-tick-old sample is fine, but a joint with
                        // no fresh reply this tick gets none rather than a
                        // push computed from a stale position.
                        let stiction_ff = match latest[m.slot] {
                            Some((pos, vel, _, _)) if feedback_fresh[m.slot] => filter::stiction(
                                p_cmd - pos,
                                vel,
                                filter::stiction_amplitude(
                                    m.fc,
                                    m.stiction_gain,
                                    m.stiction_load_gain,
                                    c.t_ff,
                                ),
                                m.stiction_err,
                            ),
                            _ => 0.0,
                        };
                        let dither_ff = filter::dither_step(
                            &mut d.dither_phase,
                            m.dither_nm,
                            m.dither_hz,
                            tick_dt,
                        );
                        // Stribeck cancellation on the measured velocity —
                        // only with a fresh sample behind it, like the
                        // stiction push.
                        let stribeck_ff = if feedback_fresh[m.slot] {
                            filter::stribeck_excess(
                                d.vel_meas_slow,
                                filter::stribeck_amplitude(
                                    m.stribeck_gain,
                                    m.stribeck_dfs,
                                    m.stribeck_load_gain,
                                    c.t_ff,
                                ),
                                m.stribeck_vs,
                            )
                        } else {
                            0.0
                        };
                        let inertia_ff = c.j_eff * a_cmd;
                        let damp_ok = feedback_fresh[m.slot]
                            && timing_on_time
                            && !timing_health.degraded
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
                        (
                            v_cmd,
                            a_cmd,
                            v_cmd_fast,
                            friction_ff,
                            inertia_ff,
                            v_damp,
                            stiction_ff,
                            dither_ff,
                            stribeck_ff,
                        )
                    } else {
                        d.v_cmd.seed(p_cmd);
                        d.a_cmd.seed(0.0);
                        d.v_cmd_fast.seed(p_cmd);
                        d.bp.reset();
                        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    };
                    let damping_ff = c.kd_host * v_damp;
                    let t_ff = c.t_ff
                        + friction_ff
                        + stiction_ff
                        + dither_ff
                        + stribeck_ff
                        + inertia_ff
                        + damping_ff;
                    if trace_this_tick && trace_tx.is_some() {
                        trace_pending[m.slot] = Some(TraceRow {
                            tick: ticks,
                            time_s: trace_origin_s
                                + began.saturating_duration_since(trace_epoch).as_secs_f64(),
                            seq: last_seq.unwrap_or(0),
                            slot: m.slot,
                            motor_id: m.id,
                            mode: c.mode,
                            target_p: p_tgt,
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
                            stiction_ff,
                            dither_ff,
                            stribeck_ff,
                            total_ff: t_ff,
                            kd_host: c.kd_host,
                            damp_w0: c.damp_w0,
                            damp_q: c.damp_q,
                            tick_dt,
                            fb_dt: f64::NAN,
                        });
                    }
                    if a4_wire(m.vendor, m.wire, tracked, c.kp) {
                        // Firmware position loop: the streamed trajectory (or
                        // the hold pose) as an absolute 0.01° target under the
                        // tracker's own velocity limit as the speed cap. No
                        // feedforward reaches the wire; the 0x92 read below
                        // restores fine position to the host.
                        a4_follow[motor_index] = true;
                        (
                            proto::MA_REQ + m.id as u16,
                            proto::ma_a4_encode(p_cmd, m.max_vel.to_degrees()),
                        )
                    } else if pv_wire(m.vendor, m.wire, tracked, c.kp) {
                        // Damiao firmware position loop: same target, the
                        // tracker's velocity limit as the speed cap; the
                        // wrist's profiler (ACC/DEC) shapes it further.
                        (
                            proto::DM_POS_VEL_ARB_BASE + m.id as u16,
                            proto::dm_pos_vel_encode(p_cmd, m.max_vel),
                        )
                    } else {
                        let frame = proto::mit_encode(p_cmd, v_wire, c.kp, c.kd, t_ff, &m.ranges);
                        let arb = match m.vendor {
                            Vendor::MyActuator => proto::MA_MC_REQ + m.id as u16,
                            Vendor::Damiao => m.id as u16,
                        };
                        (arb, frame)
                    }
                };
                // The bus cannot carry every motor every tick at the
                // higher loop rate: a thinned motor's off-tick still ran
                // its tracker (above), it just sends nothing.
                if !sched.commanded(motor_index, ticks) {
                    continue;
                }
                // A Damiao wrist acts only on the frame of the control mode
                // it is in. Switch the register (RAM write, immediate) on
                // the tick the wanted frame changes — pv ↔ MIT across limp
                // or gravity comp — before that frame goes out.
                if dm_mode[motor_index] != 0 {
                    let wanted = if arb == proto::DM_POS_VEL_ARB_BASE + m.id as u16 {
                        proto::DM_MODE_POS_VEL
                    } else {
                        proto::DM_MODE_MIT
                    };
                    if wanted != dm_mode[motor_index] {
                        let reg = proto::dm_write_register(
                            m.id as u16,
                            proto::DM_REG_CTRL_MODE,
                            wanted.to_le_bytes(),
                        );
                        if let SendOutcome::Sent =
                            guarded_send(&sock, proto::DM_REG_ARB, &reg, &mut enobufs_since)?
                        {
                            dm_mode[motor_index] = wanted;
                            send_text(
                                out_tx,
                                b'L',
                                &format!(
                                    "{iface}: {} control mode → {} ({})",
                                    m.joint,
                                    wanted,
                                    if wanted == proto::DM_MODE_POS_VEL {
                                        "position-velocity, firmware loop"
                                    } else {
                                        "MIT, compliant"
                                    },
                                ),
                            );
                        }
                    }
                }
                attempted[motor_index] = true;
                match guarded_send(&sock, arb, &frame, &mut enobufs_since)? {
                    SendOutcome::Sent => expected[motor_index] = 1,
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

            // 0xA4 joints: ask for the 0.01° multi-turn angle right behind the
            // command so the reply pair lands inside this tick's window.
            for (motor_index, m) in motors.iter().enumerate() {
                if !a4_follow[motor_index] {
                    continue;
                }
                a4_follow[motor_index] = false;
                if expected[motor_index] == 0 || !sched.a4_read(motor_index, ticks) {
                    continue;
                }
                if let SendOutcome::Sent = guarded_send(
                    &sock,
                    proto::MA_REQ + m.id as u16,
                    &proto::MA_MULTI_TURN_REQUEST,
                    &mut enobufs_since,
                )? {
                    expected[motor_index] = 2;
                }
            }

            // Collect replies. The
            // window begins when this tick actually began, not at its nominal
            // schedule point: a late wake must not discard shoulder feedback
            // simply because the old absolute deadline has already elapsed.
            // The wait is hrtimer-precise (`recv_timeout`): a missing reply
            // must end the window at `reply_deadline`, never a jiffy or two
            // later, or the overrun lands on the next tick as lateness.
            let reply_deadline = began + period.saturating_sub(REPLY_GUARD);
            seen.fill(0);
            bus_last_reply = None;
            let mut pending: usize = expected.iter().map(|&n| n as usize).sum();
            while pending > 0 {
                let now = Instant::now();
                if now >= reply_deadline {
                    break;
                }
                let Some(frame) = sock.recv_timeout(reply_deadline - now)? else {
                    break;
                };
                bus_last_reply = Some(Instant::now());
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
                    id if (0x241..=0x245).contains(&id) => {
                        // 0xA4 joints answer twice: the command echo (iq,
                        // speed, whole-degree angle) is staged; the 0x92
                        // multi-turn read completes the sample. Torque is
                        // not available in Nm on this path (the echo carries
                        // q-axis current), so it is reported as NaN and the
                        // contact watchdog is blind on the joint.
                        let motor_id = (id - 0x240) as u8;
                        let Some(idx) = motors.iter().position(|m| m.id == motor_id) else {
                            continue;
                        };
                        let slot = motors[idx].slot;
                        match frame.data[0] {
                            0xA4 => {
                                let (iq, speed, _) = proto::ma_decode_a4_reply(&frame.data);
                                if expected[idx] >= 2 {
                                    // A 0x92 read follows: stage, let it
                                    // complete the sample.
                                    if mark_unique_expected_reply(&expected, &mut seen, idx) {
                                        pending -= 1;
                                        a4_stage[slot] = (speed, iq);
                                    }
                                    continue;
                                }
                                // No read this tick (thinned schedule): the
                                // echo completes the sample with the last
                                // fine position carried on its speed.
                                let now = Instant::now();
                                let Some(anchor) = a4_anchor[slot] else {
                                    continue;
                                };
                                let pos = a4_extrapolate(anchor, speed, now);
                                (idx, pos, speed, f64::NAN)
                            }
                            proto::MA_MULTI_TURN_ANGLE => {
                                let pos = proto::ma_decode_position(&frame.data);
                                let (vel, _iq) = a4_stage[slot];
                                (idx, pos, vel, f64::NAN)
                            }
                            _ => continue,
                        }
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
                // Every accepted sample re-anchors the a4 carry — the 0x92
                // read, the carried echo itself, and the MIT reply of an a4
                // joint that is limp right now, so the carry resumes from
                // where the joint really is when it goes back on 0xA4.
                a4_anchor[motors[idx].slot] = Some((pos, recv_time));
                // The gripper has no damping chain or trace row: its
                // POSITION_FORCE reply only feeds the telemetry cache.
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
                    d.vel_meas_slow = d.v_meas_slow.update(pos, dt);
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
                // No position-deviation abort here, deliberately. Position
                // error is not a safety signal on a compliant impedance
                // controller: a hand on the arm or a joint that lost torque
                // (overtemp self-disable) both look like "deviation", and
                // cutting torque on every joint for it dropped otherwise
                // healthy arms. Contact is the Python torque-residual
                // watchdog's job (limp gravity-comp hold, as in classic
                // mode); a self-disabled motor simply stops contributing,
                // as it did in classic mode.
            }

            // Replies still outstanding at the window's end. Counted into the
            // periodic stats so a rare miss is visible in the log even when it
            // stays far below the fail-closed thresholds.
            missed += pending as u64;

            // Never run phase-sensitive host damping on stale feedback. A
            // single miss suppresses host damping on the next tick; bursty
            // loss marks the joint degraded (damping off until a clean
            // window, logged) and the loop keeps running on firmware kd. Only
            // a motor that falls silent for `SILENT_FEEDBACK_FAULT` takes the
            // session limp — the core has been streaming stiffness to it
            // blind for too long; gravity comp on every joint is the safe
            // thing it can still do. The gripper is intentionally excluded.
            let mut any_degraded = false;
            for (idx, motor) in motors.iter().enumerate() {
                if motor.gripper {
                    continue;
                }
                if !attempted[idx] {
                    // Not this motor's tick on the thinned schedule: no
                    // reply was owed, so none is missing, and the latest
                    // sample stays "fresh" — it is the newest the schedule
                    // can produce. Clearing it here would leave the
                    // feedforwards that need a sample (stiction, Stribeck,
                    // host damping) permanently off on a thinned MIT joint,
                    // since every commanded tick follows an off-tick.
                    continue;
                }
                let complete = reply_complete(&expected, &seen, idx);
                feedback_fresh[motor.slot] = complete;
                let health = &mut feedback_health[motor.slot];
                // Counted in the joint's own commands, so the silent limit
                // stays `SILENT_FEEDBACK_FAULT` of wall time on the 240 Hz lane.
                let limit = if sched.mit_phase(idx).is_some() {
                    silent_limit.div_ceil(sched.mit_div as u32)
                } else {
                    silent_limit
                };
                match health.record(complete, limit) {
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
                        // Repeats every tick past the limit; go_limp is
                        // edge-triggered so only the first one reports.
                        go_limp(
                            limp,
                            out_tx,
                            &format!(
                                "{iface}: {} silent for {} consecutive ticks ({:.1} s) — motor unreachable; going limp rather than commanding it stiff and blind",
                                motor.joint,
                                health.consecutive_misses,
                                health.consecutive_misses as f64 * sched.mit_phase(idx).map_or(1.0, |_| sched.mit_div as f64) / cfg.loop_hz,
                            ),
                        );
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

            if let Some(t) = bus_last_reply {
                bus_busy.push(t.duration_since(began).as_secs_f64() / period.as_secs_f64());
            }
            if began >= next_stats {
                next_stats = began + Duration::from_secs(5);
                let (busy_p50, busy_p95, busy_max) = bus_busy_percentiles(&mut bus_busy);
                bus_busy.clear();
                send_text(
                    out_tx,
                    b'L',
                    &format!(
                        "{iface}: {ticks} ticks, {late} late ({:.2}%), {overruns} overruns, {timing_degraded_ticks} timing-degraded ticks in {timing_degraded_episodes} episodes, {missed} missed replies, {degraded_ticks} feedback-degraded ticks in {degraded_episodes} episodes, {rejected} rejected targets, {held_ticks} held-over ticks (oldest target {:.1} ms, cadence {:.1} ms), bus busy p50 {:.0}% p95 {:.0}% max {:.0}% of the tick, {trace_dropped} trace drops, seq {:?}",
                        late as f64 / ticks as f64 * 100.0,
                        worst_target_age * 1e3,
                        cadence.get().unwrap_or(f64::NAN) * 1e3,
                        busy_p50 * 100.0,
                        busy_p95 * 100.0,
                        busy_max * 100.0,
                        last_seq,
                    ),
                );
            }
            deadline = next_bus_deadline(began, period);
        }
    })();

    // Torque comes off only for an explicit disarm on a healthy session. A
    // fault on either bus, a signal, a lost client, or a protocol error all
    // stop the stream and leave every motor on its last MIT command — holding
    // on firmware gains, or limp with gravity feedforward if the session had
    // gone limp — the classic-Python outcome when the host process died.
    // A dead bus (e-stop) has nothing powered to hear a disable anyway.
    let deliberate = disarm.load(Ordering::SeqCst)
        && fault.load(Ordering::SeqCst) == 0
        && !limp.load(Ordering::SeqCst);
    if bus_dead {
        // Nothing to say: the TX-stall fault message already covers it.
    } else if deliberate {
        bringup::disable(&sock, &motors);
    } else if limp.load(Ordering::SeqCst) {
        send_text(
            out_tx,
            b'L',
            &format!(
                "{iface}: stopped streaming — motors left limp (kp = 0) with their last gravity feedforward (not disabled)"
            ),
        );
    } else {
        send_text(
            out_tx,
            b'L',
            &format!(
                "{iface}: stopped streaming — motors left holding their last command (not disabled)"
            ),
        );
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
