//! Realtime filters for the in-core control loop, ported exactly from the
//! Python originals: `Differentiator` and `BandPass` from
//! `almond_axol.robot.control` (host damping), `TrapezoidalFilter` from
//! `almond_axol.teleop.filter` (the 240 Hz joint-target tracker), and the
//! tanh friction model. The golden tests at the bottom pin the outputs to
//! the Python implementations sample-for-sample, so the physics a joint
//! feels is identical whichever side computes it — only the rate and
//! freshness differ.

/// First-order low-pass differentiator: `a = 1/(1 + dt·ω)`,
/// `v ← a·v + a·ω·(x − x_prev)`. Unity DC gain as a differentiator, single
/// pole at `cutoff` rad/s. First update primes the state and returns 0.
pub struct LpDiff {
    cutoff: f64,
    vel: f64,
    pos_prev: Option<f64>,
}

impl LpDiff {
    pub fn new(cutoff: f64) -> Self {
        Self {
            cutoff,
            vel: 0.0,
            pos_prev: None,
        }
    }

    pub fn update(&mut self, pos: f64, dt: f64) -> f64 {
        let Some(prev) = self.pos_prev else {
            self.pos_prev = Some(pos);
            return 0.0;
        };
        if dt <= 0.0 {
            return self.vel;
        }
        let a = 1.0 / (1.0 + dt * self.cutoff);
        self.vel = self.vel * a + a * self.cutoff * (pos - prev);
        self.pos_prev = Some(pos);
        self.vel
    }

    /// Re-prime the differentiator at a stationary position.
    ///
    /// Passthrough control can move a joint without advancing the tracked
    /// command trajectory.  Re-seeding prevents that mode switch from
    /// becoming a fictitious velocity/acceleration impulse when tracked
    /// control resumes.
    pub fn seed(&mut self, pos: f64) {
        self.vel = 0.0;
        self.pos_prev = Some(pos);
    }
}

/// Chamberlin state-variable band-pass, unity gain and ~zero phase at the
/// centre frequency, 6 dB/oct rolloff both sides. The centre `w0` (rad/s)
/// and quality `q` are per-update inputs: the Python side streams a
/// pose-tracked centre, and the SVF recomputes its coefficient every step so
/// a slowly-varying centre is well-behaved. First update primes the clock
/// and returns 0 (matching the Python filter's first-call contract).
pub struct BandPass {
    lp: f64,
    bp: f64,
    primed: bool,
}

impl BandPass {
    pub fn new() -> Self {
        Self {
            lp: 0.0,
            bp: 0.0,
            primed: false,
        }
    }

    pub fn update(&mut self, x: f64, w0: f64, q: f64, dt: f64) -> f64 {
        if !self.primed {
            self.primed = true;
            return 0.0;
        }
        let q = q.max(1e-6);
        if dt <= 0.0 {
            return self.bp / q;
        }
        // sin() form keeps the centre accurate at low sample rates; the
        // clamp keeps the filter stable across loop stalls.
        let f = 2.0 * (0.5 * w0 * dt).min(0.7).sin();
        self.lp += f * self.bp;
        let hp = x - self.lp - self.bp / q;
        self.bp += f * hp;
        self.bp / q
    }

    /// Clear stored energy when leaving the tracked control mode.
    pub fn reset(&mut self) {
        self.lp = 0.0;
        self.bp = 0.0;
        self.primed = false;
    }
}

/// Tanh friction feedforward, ported from
/// `almond_axol.robot.control.compute_friction`:
/// `τ = fc·tanh(0.1·min(k, K_MAX)·v) + fv·v + fo`. The cap keeps the
/// Coulomb term ramping smoothly through zero crossings.
pub const FRICTION_FF_K_MAX: f64 = 100.0;

pub fn friction(v: f64, fc: f64, k: f64, fv: f64, fo: f64) -> f64 {
    fc * coulomb_unit(v, k, FRICTION_FF_K_MAX) + fv * v + fo
}

/// Saturation fraction of the Coulomb feedforward,
/// `tanh(0.1·min(k, k_max)·v)` ∈ (−1, 1) — `coulomb_unit` in
/// `almond_axol.robot.control`. `k_max` is the experiment-adjustable cap
/// (`exp friction_k_max`, default `FRICTION_FF_K_MAX`).
pub fn coulomb_unit(v: f64, k: f64, k_max: f64) -> f64 {
    (0.1 * k.min(k_max) * v).tanh()
}

/// Rate limiter, `|y[k] − y[k−1]| ≤ rate·dt` — `SlewLimiter` in
/// `almond_axol.robot.control`. Applied to the Coulomb friction term when
/// its tanh is steepened past the cap (`exp friction_slew`): the ±fc step at
/// every arrival/reversal is what the cap protects the shoulders' 2–3 Hz
/// mode from, and a slew does that job without dulling low-speed
/// compensation. `rate <= 0` passes through. The first update adopts `x`.
#[derive(Default)]
pub struct SlewLimiter {
    y: Option<f64>,
}

impl SlewLimiter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, x: f64, rate: f64, dt: f64) -> f64 {
        let y = match self.y {
            Some(prev) if rate > 0.0 && dt > 0.0 => {
                let step = rate * dt;
                prev + (x - prev).clamp(-step, step)
            }
            _ => x,
        };
        self.y = Some(y);
        y
    }

    /// Forget the held output; the next update adopts its input.
    pub fn reset(&mut self) {
        self.y = None;
    }
}

/// Error-sign Coulomb compensation — `stiction_compensation` in
/// `almond_axol.robot.control`: `gain·fc·tanh(err/err_scale)·(1 − |sat|)`.
/// Pushes the fitted Coulomb torque toward the target (`err = q_des −
/// q_meas`) while the joint is stuck and the velocity feedforward has
/// switched itself off, and fades out as that feedforward saturates (`sat`
/// from [`coulomb_unit`]) so the two never stack past `(1 + gain)·fc`.
pub fn stiction(err: f64, sat: f64, fc: f64, gain: f64, err_scale: f64) -> f64 {
    if gain == 0.0 || fc == 0.0 {
        return 0.0;
    }
    gain * fc * (err / err_scale.max(1e-9)).tanh() * (1.0 - sat.abs())
}

/// Per-channel phase offset of the torque dither: the golden angle,
/// `π(3 − √5)` — `DITHER_PHASE_STAGGER` in `almond_axol.robot.control`.
/// Written out because `sqrt` is not available in a `const`; the test below
/// pins it to the expression.
pub const DITHER_PHASE_STAGGER: f64 = 2.399_963_229_728_653;

/// Single-channel stiction-breaking torque dither — `TorqueDither` in
/// `almond_axol.robot.control`, one instance per slot (the Python class is
/// N-channel; the core keeps its per-joint state in per-joint structs).
///
/// A small oscillation added to the feedforward keeps the joint's friction
/// sliding rather than re-sticking between cycles, which is the classic fix
/// for the stick-slip stairs a slow position command leaves behind. Phase
/// advances on an accumulator, so a late tick shifts phase instead of
/// frequency, and each slot starts [`DITHER_PHASE_STAGGER`] apart so seven
/// joints never push the structure in unison.
pub struct Dither {
    phase0: f64,
    phase: f64,
}

impl Dither {
    pub fn new(channel: usize) -> Self {
        let phase0 = (channel as f64 * DITHER_PHASE_STAGGER).rem_euclid(std::f64::consts::TAU);
        Self {
            phase0,
            phase: phase0,
        }
    }

    /// Advance one step and return the dither torque (Nm). `amplitude <= 0`
    /// or `hz <= 0` is off and leaves the phase where it was (matching the
    /// Python original). `fade_vel > 0` fades the amplitude out with
    /// `1 − tanh(|v| / fade_vel)`, so the dither acts where the joint is
    /// stuck and stops being heat and noise once it is moving.
    pub fn update(
        &mut self,
        velocity: f64,
        amplitude: f64,
        hz: f64,
        dt: f64,
        square: bool,
        fade_vel: f64,
    ) -> f64 {
        if amplitude <= 0.0 || hz <= 0.0 {
            return 0.0;
        }
        let step = if dt > 0.0 {
            std::f64::consts::TAU * hz * dt
        } else {
            0.0
        };
        self.phase = (self.phase + step).rem_euclid(std::f64::consts::TAU);
        let mut wave = self.phase.sin();
        if square {
            wave = if wave >= 0.0 { 1.0 } else { -1.0 };
        }
        let fade = if fade_vel > 0.0 {
            1.0 - (velocity.abs() / fade_vel).tanh()
        } else {
            1.0
        };
        amplitude * fade * wave
    }

    /// Restart at this channel's staggered phase.
    pub fn reset(&mut self) {
        self.phase = self.phase0;
    }
}

/// Clamped, freeze-gated position-error integrator → torque —
/// `ErrorIntegrator` in `almond_axol.robot.control`. `ki` is Nm/(rad·s),
/// `clamp` the anti-windup bound on the output (Nm); `|err| > freeze` (rad)
/// or `dt <= 0` holds the state instead of winding up (a large error is a
/// move or a contact, not a residual; no fresh feedback means no fresh
/// error). `ki <= 0` or `clamp <= 0` zeroes it.
#[derive(Default)]
pub struct Integrator {
    i: f64,
}

impl Integrator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, err: f64, ki: f64, clamp: f64, freeze: f64, dt: f64) -> f64 {
        if ki <= 0.0 || clamp <= 0.0 {
            self.i = 0.0;
        } else if dt > 0.0 && err.abs() <= freeze {
            self.i = (self.i + ki * err * dt).clamp(-clamp, clamp);
        }
        self.i
    }

    pub fn reset(&mut self) {
        self.i = 0.0;
    }
}

/// First-order low-pass, `a = 1/(1 + dt·ω)`, `y ← a·y + (1 − a)·x`. The first
/// update adopts `x`. Used on the tracker's own velocity/acceleration states
/// when they replace the position-derivative chains (`exp tracker_wire_vel`
/// / `exp tracker_accel_ff`).
#[derive(Default)]
pub struct LowPass {
    y: Option<f64>,
}

impl LowPass {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, x: f64, pole: f64, dt: f64) -> f64 {
        let y = match self.y {
            Some(prev) if pole > 0.0 && dt > 0.0 => {
                let a = 1.0 / (1.0 + dt * pole);
                a * prev + (1.0 - a) * x
            }
            _ => x,
        };
        self.y = Some(y);
        y
    }

    /// Adopt `x` as the current output (re-priming at rest).
    pub fn seed(&mut self, x: f64) {
        self.y = Some(x);
    }
}

/// Velocity/acceleration-limited target tracker — the per-joint
/// `TrapezoidalFilter` from `almond_axol.teleop.filter`, ported per-scalar
/// with a per-step `dt` (the Python original fixes dt at construction).
///
/// A critically damped second-order linear loop (position error → velocity
/// command → acceleration, ζ = 1) under hard velocity and acceleration
/// clamps, with the time-optimal sqrt braking rule kept only as a velocity
/// *ceiling* for large catch-up moves. See the Python docstring for the
/// designs this replaced and why (bang-bang saturation chatter; velocity
/// feedforward peaking at the arm's structural resonance).
///
/// In the core this runs at the full loop rate against the latest streamed
/// target, replacing linear segment interpolation: its `(pos, vel, accel)`
/// states drive the MIT command and the friction/inertia feedforwards, so
/// the wire physics are coherent with the trajectory actually executed.
pub struct Trapezoid {
    pub max_vel: f64,
    pub max_accel: f64,
    /// Position- and velocity-tracking gains (1/s). Defaults are
    /// [`Trapezoid::POS_TRACK_GAIN`] / [`Trapezoid::VEL_TRACK_GAIN`]; the
    /// `tracker_pos_gain` / `tracker_vel_gain` experiments override them.
    ///
    /// The position gain *is* the tracker's lag: a first-order pole at
    /// `pos_gain` rad/s, so 15.7 is a 64 ms time constant. That lag shows up
    /// at the end effector as path deviation proportional to commanded speed
    /// — measured at 58 ms on hardware, which is what makes a curved path
    /// cut its corner. Raising it trades that accuracy against the reason it
    /// is slow: velocity feedforward peaking at the arm's structural
    /// resonance (see the type docs).
    pos_gain: f64,
    vel_gain: f64,
    pos: f64,
    vel: f64,
    seeded: bool,
}

impl Trapezoid {
    pub const POS_TRACK_GAIN: f64 = 15.7; // 1/s = ωn/2 with ωn = 2π·5 Hz
    pub const VEL_TRACK_GAIN: f64 = 62.8; // 1/s = 2·ωn
    const BRAKE_MARGIN: f64 = 0.8;

    /// Unseeded, matching the Python original: the first `update` adopts
    /// the target as the output (no transient).
    pub fn new(max_vel: f64, max_accel: f64) -> Self {
        Self::with_gains(
            max_vel,
            max_accel,
            Self::POS_TRACK_GAIN,
            Self::VEL_TRACK_GAIN,
        )
    }

    /// As [`Trapezoid::new`], with the tracking gains overridden. A
    /// non-positive gain falls back to its default, so a zeroed experiment
    /// value cannot stall the tracker.
    pub fn with_gains(max_vel: f64, max_accel: f64, pos_gain: f64, vel_gain: f64) -> Self {
        Self {
            max_vel,
            max_accel,
            pos_gain: if pos_gain > 0.0 {
                pos_gain
            } else {
                Self::POS_TRACK_GAIN
            },
            vel_gain: if vel_gain > 0.0 {
                vel_gain
            } else {
                Self::VEL_TRACK_GAIN
            },
            pos: 0.0,
            vel: 0.0,
            seeded: false,
        }
    }

    /// Adopt `pos` as the current output with zero velocity — used at arm
    /// (hold pose) and each passthrough-mode tick, so a later switch to
    /// tracked mode starts from the last commanded position transient-free.
    pub fn seed(&mut self, pos: f64) {
        self.pos = pos;
        self.vel = 0.0;
        self.seeded = true;
    }

    /// Advance one step toward `target`; returns `(pos, vel, accel)`.
    pub fn update(&mut self, target: f64, dt: f64) -> (f64, f64, f64) {
        if !self.seeded {
            self.seed(target);
            return (target, 0.0, 0.0);
        }
        if dt <= 0.0 {
            return (self.pos, self.vel, 0.0);
        }
        let err = target - self.pos;
        let dist = err.abs();
        let adt = self.max_accel * dt;

        // Discrete-time stopping speed (margined): the ceiling for
        // overshoot-free catch-up on large distances.
        let a_brake = Self::BRAKE_MARGIN * self.max_accel;
        let bdt = 0.5 * a_brake * dt;
        let v_stop = -bdt + (bdt * bdt + 2.0 * a_brake * dist).sqrt();

        let ceiling = self.max_vel.min(v_stop);
        let desired = (self.pos_gain * err).clamp(-ceiling, ceiling);

        let vel_prev = self.vel;
        let mut vel = vel_prev + (self.vel_gain * (desired - vel_prev) * dt).clamp(-adt, adt);

        // Acceleration-gated arrival (see the Python docstring: an
        // unconditional snap degenerates into a pass-through).
        let step = vel * dt;
        let snap_vel = err / dt;
        if step.abs() > dist && (snap_vel - vel_prev).abs() <= adt * (1.0 + 1e-6) {
            self.pos = target;
            vel = snap_vel;
        } else {
            self.pos += step;
        }
        let accel = (vel - vel_prev) / dt;
        self.vel = vel;
        (self.pos, vel, accel)
    }
}

/// Running estimate of a target stream's spacing, learned from arrivals.
///
/// Python streams at ~120 Hz in teleop; run-policy / collect-dagger's policy
/// state at the dataset rate (30-60 Hz), and dagger switches between the
/// two mid-session. Learned rather than configured so no client, old or
/// new, has to declare its rate. Gaps shorter than the estimate are always
/// taken (a faster stream is followed at once). A single long gap is a late
/// arrival — exactly what `Holdover` bridges — and must not stretch the
/// estimate, or a stalling host would teach the core that stalls are
/// normal; but `RELEARN` long gaps *in a row* are a slower stream, and the
/// estimate re-seeds from their mean (Bugbot on #306: the first version
/// only ever rejected long gaps, so after a dagger intervention every
/// on-time 33 ms policy frame counted as late).
#[derive(Clone, Copy, Debug, Default)]
pub struct Cadence {
    value: Option<f64>,
    slow_run: u32,
    slow_sum: f64,
}

impl Cadence {
    /// Clamp on any single gap sample (seconds).
    pub const MIN: f64 = 0.002;
    pub const MAX: f64 = 0.100;
    /// Weight of each in-cadence gap in the running estimate.
    const SMOOTHING: f64 = 0.1;
    /// A gap longer than this many cadences is an outlier (late arrival).
    pub const OUTLIER: f64 = 1.75;
    /// This many consecutive outlier gaps are a slower stream, not lateness:
    /// at 30 Hz that is 200 ms of the new rate before the estimate follows.
    pub const RELEARN: u32 = 6;

    pub fn new() -> Self {
        Self::default()
    }

    /// Current estimate (seconds), once two targets have arrived.
    pub fn get(&self) -> Option<f64> {
        self.value
    }

    /// Record the gap (seconds) between two consecutive adopted targets.
    pub fn observe(&mut self, gap: f64) {
        if !gap.is_finite() || gap <= 0.0 {
            return;
        }
        let sample = gap.clamp(Self::MIN, Self::MAX);
        match self.value {
            None => self.value = Some(sample),
            Some(c) if gap > Self::OUTLIER * c => {
                self.slow_run += 1;
                self.slow_sum += sample;
                if self.slow_run >= Self::RELEARN {
                    self.value = Some(self.slow_sum / self.slow_run as f64);
                    self.slow_run = 0;
                    self.slow_sum = 0.0;
                }
            }
            Some(c) => {
                self.slow_run = 0;
                self.slow_sum = 0.0;
                self.value = Some(c + Self::SMOOTHING * (sample - c));
            }
        }
    }
}

/// Carries a *late* streamed target forward along the stream's own velocity.
///
/// The core tracks the latest target Python streamed. When Python's tick is
/// late — the control thread lost the CPU to a camera relay, a desktop, a
/// stats daemon, whatever else the host runs — the tracker's target stops
/// dead for the gap and then jumps by everything the hand moved meanwhile:
/// at 120 Hz a 40 ms stall is a five-target notch, and the 5 Hz tracker
/// renders that as a visible decelerate-then-lunge hitch (2026-09-15, every
/// stalled tick in the flight recorder lined up with an arm "jump").
///
/// Instead of holding, the target the tracker is given keeps moving at the
/// velocity the stream itself had — estimated from consecutive adopted
/// targets, smoothed — starting once the newest target is older than the
/// stream's cadence plus slack, with the velocity ramping linearly to zero
/// over `max_hold`. The reach is clamped to `max_reach`, and the tracker's
/// own vel/accel limits still bound the wire. An on-time stream is
/// untouched (`target` returns `p` unchanged), a stalled one (operator
/// stopped, watchdog) glides to rest within `max_hold`, and when the late
/// target finally lands the step the tracker sees is the extrapolation
/// error, not the whole gap. This is what makes smoothness independent of
/// host scheduling rather than a property of a particular box's load.
#[derive(Clone, Copy, Debug)]
pub struct Holdover {
    /// Seconds a late target is carried forward before the extrapolation
    /// has glided to rest and the tracker simply holds.
    pub max_hold: f64,
    /// Furthest (rad) the carried target may travel from the last real one.
    pub max_reach: f64,
    vel: f64,
    last: Option<f64>,
}

impl Holdover {
    /// Weight of each fresh velocity sample in the running estimate. The
    /// stream is Python's own trapezoid output, so consecutive samples are
    /// already smooth; the light filter only takes socket jitter out of the
    /// finite difference.
    const SMOOTHING: f64 = 0.5;
    /// A target is *late* once its age exceeds this many cadences; inside
    /// that window it is the stream's normal spacing plus transport jitter.
    pub const SLACK: f64 = 1.25;
    /// A gap is a *stall* (the finite difference across it is the stream's
    /// mean velocity — Python's trapezoid kept moving, we heard late) up to
    /// `max_hold` past this many cadences; beyond that the stream *resumed*
    /// after a hold and the estimate restarts from rest. Tied to `max_hold`
    /// so the cutoff sits above every stall the carry bridges: a cadence
    /// multiple alone (4 × 8.3 ms = 33 ms at 120 Hz) fell inside the 15-65 ms
    /// stalls this exists for, and the late target that ended one stall
    /// wiped the velocity the next one needed (Bugbot on #306).
    const RESUME: f64 = 4.0;

    pub fn new(max_hold: f64, max_reach: f64) -> Self {
        Self {
            max_hold,
            max_reach,
            vel: 0.0,
            last: None,
        }
    }

    /// Forget the stream: the next target starts a fresh estimate from rest.
    /// Called for passthrough (gravity comp / limp) and gripper targets. A
    /// *rejected* target does not reset: the carry keeps gliding to rest
    /// from the last accepted target, and only accepted targets are ever
    /// observed.
    pub fn reset(&mut self) {
        self.vel = 0.0;
        self.last = None;
    }

    /// Current velocity estimate, rad/s.
    #[cfg(test)]
    pub fn vel(&self) -> f64 {
        self.vel
    }

    /// Record a freshly adopted tracked target `p`, `dt` seconds after the
    /// previous adopted one (`None` when there is no usable previous
    /// arrival), for a stream whose nominal spacing is `cadence` seconds.
    pub fn observe(&mut self, p: f64, dt: Option<f64>, cadence: Option<f64>) {
        match (self.last, dt, cadence) {
            (Some(prev), Some(dt), Some(cadence))
                if dt > 0.0
                    && dt.is_finite()
                    && dt <= self.max_hold.max(0.0) + Self::RESUME * cadence =>
            {
                let raw = (p - prev) / dt;
                if raw.is_finite() {
                    self.vel += Self::SMOOTHING * (raw - self.vel);
                } else {
                    self.vel = 0.0;
                }
            }
            _ => self.vel = 0.0,
        }
        self.last = Some(p);
    }

    /// The target to track `age` seconds after `p` (the last real target)
    /// arrived. Identity while the stream is on time or its cadence is not
    /// yet known.
    pub fn target(&self, p: f64, age: f64, cadence: Option<f64>) -> f64 {
        let Some(cadence) = cadence else {
            return p;
        };
        let late = age - Self::SLACK * cadence;
        if late <= 0.0 || self.vel == 0.0 || self.max_hold <= 0.0 {
            return p;
        }
        // Velocity ramps linearly from `vel` to zero over `max_hold`, so the
        // carried target decelerates to rest instead of stopping short:
        // reach(t) = ∫₀ᵗ vel·(1 − τ/T) dτ = vel·(t − t²/2T), t ≤ T.
        let t = late.min(self.max_hold);
        let reach = self.vel * (t - t * t / (2.0 * self.max_hold));
        p + reach.clamp(-self.max_reach, self.max_reach)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 1.0 / 240.0;

    /// Golden vectors generated from the Python originals
    /// (`almond_axol.robot.control`) at a fixed 240 Hz step with
    /// `x[k] = sin(2π·4·k·dt)`; see the module docstring.
    #[test]
    fn bandpass_matches_python() {
        let golden = [
            0.000000000000000e+00,
            1.360432435891207e-02,
            3.874584267603020e-02,
            7.335396828981698e-02,
            1.153808032974739e-01,
            1.628252144108736e-01,
            2.137547950518698e-01,
            2.663256810828089e-01,
            3.188001991162444e-01,
            3.695623375589990e-01,
            4.171310415067576e-01,
            4.601713433826090e-01,
        ];
        let mut bp = BandPass::new();
        for (k, want) in golden.iter().enumerate() {
            let x = (2.0 * std::f64::consts::PI * 4.0 * k as f64 * DT).sin();
            let got = bp.update(x, 25.0, 0.8, DT);
            assert!(
                (got - want).abs() < 1e-12,
                "bandpass sample {k}: got {got:e}, want {want:e}"
            );
        }
    }

    /// Golden vectors from `almond_axol.robot.control.TorqueDither`: channel
    /// 3 (the elbow's stagger offset) of a 40 Hz sine dither at 0.25 Nm,
    /// faded against a 1 Hz 0.02 rad/s velocity with `fade_vel = 0.05`.
    #[test]
    fn dither_matches_python() {
        let golden = [
            2.309309714758029e-01,
            3.219017344380157e-02,
            -1.942474090353159e-01,
            -2.236858958780299e-01,
            -3.117141857461361e-02,
            1.880510214080443e-01,
            2.164995766404254e-01,
            3.016366899355213e-02,
            -1.819382352135718e-01,
            -2.094295231161269e-01,
            -2.917492986113358e-02,
            1.759571782350752e-01,
        ];
        let mut d = Dither::new(3);
        for (k, want) in golden.iter().enumerate() {
            let v = 0.02 * (2.0 * std::f64::consts::PI * k as f64 * DT).sin();
            let got = d.update(v, 0.25, 40.0, DT, false, 0.05);
            assert!(
                (got - want).abs() < 1e-12,
                "dither sample {k}: got {got:e}, want {want:e}"
            );
        }
    }

    /// The stagger is the golden angle, and a 40 Hz square on the 240 Hz loop
    /// is three ticks each way — the shape that actually reaches the motor.
    #[test]
    fn dither_square_and_stagger() {
        assert!(
            (DITHER_PHASE_STAGGER - std::f64::consts::PI * (3.0 - 5.0_f64.sqrt())).abs() < 1e-15
        );
        let mut d = Dither::new(0);
        let got: Vec<f64> = (0..12)
            .map(|_| d.update(0.0, 0.25, 40.0, DT, true, 0.0))
            .collect();
        assert_eq!(
            got,
            vec![0.25, 0.25, 0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25]
        );
        // Off is off, and leaves the phase untouched.
        assert_eq!(d.update(0.0, 0.0, 40.0, DT, true, 0.0), 0.0);
        assert_eq!(d.update(0.0, 0.25, 0.0, DT, true, 0.0), 0.0);
        assert_eq!(d.update(0.0, 0.25, 40.0, DT, true, 0.0), 0.25);
    }

    /// Steady-state lag against a constant-velocity target, as a function of
    /// the position-tracking gain. This is the number that reaches the end
    /// effector as speed-proportional path deviation, and the reason
    /// `tracker_pos_gain` is an experiment at all.
    ///
    /// At the shipped gain it is ~55 ms here, against 58 ms measured on
    /// hardware from TCP path deviation vs commanded speed (correlation
    /// 0.98) and 59 ms from the joint traces — three independent reads of
    /// the same constant. Doubling the gain more than halves it: the loop is
    /// second order, so the lag falls faster than `1/gain`.
    fn steady_lag_ms(gain: f64) -> f64 {
        let v = 0.30; // rad/s, ~normal teleop joint speed
        let mut trk = Trapezoid::with_gains(10.0, 200.0, gain, 4.0 * gain);
        let mut target = 0.0;
        // Long enough to leave the acceleration transient behind.
        for _ in 0..4000 {
            target += v * DT;
            trk.update(target, DT);
        }
        1e3 * (target - trk.update(target + v * DT, DT).0) / v
    }

    #[test]
    fn tracker_lag_falls_with_the_position_gain() {
        let shipped = steady_lag_ms(Trapezoid::POS_TRACK_GAIN);
        assert!(
            (50.0..60.0).contains(&shipped),
            "shipped tracker lag {shipped:.1} ms - hardware measures 58"
        );
        let doubled = steady_lag_ms(2.0 * Trapezoid::POS_TRACK_GAIN);
        assert!(
            doubled < shipped / 2.0,
            "doubling the gain should more than halve the lag: {shipped:.1} -> {doubled:.1} ms"
        );
        let mut prev = f64::INFINITY;
        for mult in [0.5, 1.0, 2.0, 4.0] {
            let lag = steady_lag_ms(Trapezoid::POS_TRACK_GAIN * mult);
            assert!(lag < prev, "lag must fall monotonically with gain");
            prev = lag;
        }
    }

    /// A zeroed or negative gain must not stall the tracker — it falls back
    /// to the shipped value rather than freezing the arm.
    #[test]
    fn tracker_rejects_non_positive_gains() {
        let mut bad = Trapezoid::with_gains(10.0, 200.0, 0.0, -5.0);
        let mut good = Trapezoid::new(10.0, 200.0);
        for k in 1..200 {
            let target = 0.001 * k as f64;
            assert_eq!(bad.update(target, DT).0, good.update(target, DT).0);
        }
    }

    #[test]
    fn lpdiff_matches_python() {
        let golden = [
            0.000000000000000e+00,
            6.271707796059207e+00,
            1.090677450005076e+01,
            1.424639908846935e+01,
            1.654797823840318e+01,
            1.800678509425437e+01,
            1.877220395823917e+01,
            1.895987421266248e+01,
            1.866075880660901e+01,
            1.794789923880996e+01,
            1.688142899367694e+01,
            1.551227497674744e+01,
        ];
        let mut d = LpDiff::new(80.0);
        for (k, want) in golden.iter().enumerate() {
            let x = (2.0 * std::f64::consts::PI * 4.0 * k as f64 * DT).sin();
            let got = d.update(x, DT);
            assert!(
                (got - want).abs() < 1e-12,
                "lpdiff sample {k}: got {got:e}, want {want:e}"
            );
        }
    }

    /// Golden vectors from `almond_axol.teleop.filter.TrapezoidalFilter`
    /// (max_vel 2π, max_accel 7π, dt 1/240; the Python filter computes in
    /// float32, hence the 1e-4 tolerance). Two regimes: a sine target the
    /// tracker chases through its acceleration clamp, and a 0.5 rad step
    /// covering the full profile — accel ramp, braking-ceiling saturation,
    /// arrival snap, settle.
    #[test]
    fn trapezoid_matches_python() {
        let sine: [(f64, f64); 16] = [
            (0.000000000e+00, 0.000000000e+00),
            (3.817908000e-04, 9.162978828e-02),
            (1.145372400e-03, 1.832595766e-01),
            (2.290744800e-03, 2.748893499e-01),
            (3.817908000e-03, 3.665191531e-01),
            (5.726862233e-03, 4.581489563e-01),
            (8.017607033e-03, 5.497787595e-01),
            (1.069014333e-02, 6.414085627e-01),
            (1.374446973e-02, 7.330383658e-01),
            (1.718058810e-02, 8.246681690e-01),
            (2.099849656e-02, 9.162979722e-01),
            (2.519819513e-02, 1.007927775e+00),
            (2.977968566e-02, 1.099557519e+00),
            (3.474296629e-02, 1.191187263e+00),
            (4.008803889e-02, 1.282817006e+00),
            (4.581490159e-02, 1.374446750e+00),
        ];
        let (max_vel, max_accel) = (2.0 * std::f64::consts::PI, 7.0 * std::f64::consts::PI);
        let mut trk = Trapezoid::new(max_vel, max_accel);
        for (k, (want_pos, want_vel)) in sine.iter().enumerate() {
            let tgt = 0.8 * (2.0 * std::f64::consts::PI * 2.0 * k as f64 * DT).sin();
            let (pos, vel, _) = trk.update(tgt, DT);
            assert!(
                (pos - want_pos).abs() < 1e-4 && (vel - want_vel).abs() < 1e-4,
                "trapezoid sine sample {k}: got ({pos:e}, {vel:e}), want ({want_pos:e}, {want_vel:e})"
            );
        }

        // (pos, vel) at k = 0, 8, 16, ..., 120 after a 0 -> 0.5 step.
        let step: [(f64, f64); 16] = [
            (3.817908000e-04, 9.162978828e-02),
            (1.718058810e-02, 8.246681690e-01),
            (5.841399729e-02, 1.557706237e+00),
            (1.240820140e-01, 2.290744305e+00),
            (2.140991390e-01, 3.003265142e+00),
            (3.125206530e-01, 2.787965536e+00),
            (3.943324685e-01, 2.159180403e+00),
            (4.525606930e-01, 1.426142454e+00),
            (4.863543212e-01, 6.931042671e-01),
            (4.975558519e-01, 1.449353695e-01),
            (4.996199608e-01, 2.549982630e-02),
            (4.999670088e-01, 3.898998722e-03),
            (5.000000000e-01, 0.000000000e+00),
            (5.000000000e-01, 0.000000000e+00),
            (5.000000000e-01, 0.000000000e+00),
            (5.000000000e-01, 0.000000000e+00),
        ];
        let mut trk = Trapezoid::new(max_vel, max_accel);
        trk.update(0.0, DT);
        for k in 0..=120usize {
            let (pos, vel, _) = trk.update(0.5, DT);
            if k % 8 == 0 {
                let (want_pos, want_vel) = step[k / 8];
                assert!(
                    (pos - want_pos).abs() < 1e-4 && (vel - want_vel).abs() < 1e-4,
                    "trapezoid step sample {k}: got ({pos:e}, {vel:e}), want ({want_pos:e}, {want_vel:e})"
                );
            }
        }
    }

    /// Regression for motion vibration in the split 120/240 Hz pipeline.
    ///
    /// A smooth target arrives twice slower than the wire loop.  The inner
    /// tracker removes position steps, but its raw acceleration still reacts
    /// differently on the adoption tick and the repeated-target tick.  Using
    /// that raw acceleration for `j_eff` created a large alternating torque.
    /// The classic two-differentiator command chain must remove essentially
    /// all of that target-rate component before it reaches the motor.
    #[test]
    fn command_derivatives_reject_target_rate_acceleration() {
        let mut trk = Trapezoid::new(
            1.5 * 2.0 * std::f64::consts::PI,
            1.5 * 7.0 * std::f64::consts::PI,
        );
        trk.seed(0.0);
        let mut vel = LpDiff::new(20.0);
        let mut accel = LpDiff::new(20.0);
        vel.seed(0.0);
        accel.seed(0.0);

        let mut raw_pair_delta_sq = 0.0;
        let mut filtered_pair_delta_sq = 0.0;
        let mut pairs = 0usize;
        let mut raw_first = 0.0;
        let mut filtered_first = 0.0;
        for k in 0..1200usize {
            // 0.5 Hz smooth motion sampled at 120 Hz, each sample held for
            // two 240 Hz wire ticks.
            let target_t = (k / 2) as f64 / 120.0;
            let target = 0.6 * (2.0 * std::f64::consts::PI * 0.5 * target_t).sin();
            let (pos, _, raw_accel) = trk.update(target, DT);
            let v = vel.update(pos, DT);
            let filtered_accel = accel.update(v, DT);

            if k >= 240 {
                if k % 2 == 0 {
                    raw_first = raw_accel;
                    filtered_first = filtered_accel;
                } else {
                    raw_pair_delta_sq += (raw_accel - raw_first).powi(2);
                    filtered_pair_delta_sq += (filtered_accel - filtered_first).powi(2);
                    pairs += 1;
                }
            }
        }

        let raw_rms = (raw_pair_delta_sq / pairs as f64).sqrt();
        let filtered_rms = (filtered_pair_delta_sq / pairs as f64).sqrt();
        assert!(
            raw_rms > 1.0,
            "fixture must expose the raw acceleration ripple"
        );
        assert!(
            filtered_rms < 0.05 * raw_rms,
            "command derivative chain must reject target-rate ripple: raw {raw_rms:e}, filtered {filtered_rms:e}"
        );
    }

    /// Golden values from `almond_axol.robot.control.compute_friction`
    /// with fc=0.6, k=250 (above the cap), fv=0.15, fo=0.02.
    #[test]
    fn friction_matches_python() {
        let golden = [
            (-1.5, -8.049999999998876e-01),
            (-0.2, -5.884165480454902e-01),
            (-0.01, -4.130079677497349e-02),
            (0.0, 2.000000000000000e-02),
            (0.01, 8.130079677497350e-02),
            (0.2, 6.284165480454902e-01),
            (1.5, 8.449999999998876e-01),
        ];
        for (v, want) in golden {
            let got = friction(v, 0.6, 250.0, 0.15, 0.02);
            assert!(
                (got - want).abs() < 1e-12,
                "friction({v}): got {got:e}, want {want:e}"
            );
        }
    }

    /// Golden values from `almond_axol.robot.control.compute_friction` with
    /// the cap raised to 400 (`exp friction_k_max`): fc=0.6, k=800, fv=0.15,
    /// fo=0.02 — the Coulomb term now reaches ~66 % of fc at 0.02 rad/s
    /// (vs ~20 % under the production cap).
    #[test]
    fn friction_k_max_matches_python() {
        let golden = [
            (-0.05, -5.659165480454901e-01),
            (-0.01, -2.094693773531350e-01),
            (0.0, 2.000000000000000e-02),
            (0.005, 1.391751921349424e-01),
            (0.02, 4.214220621607095e-01),
        ];
        for (v, want) in golden {
            let got = 0.6 * coulomb_unit(v, 800.0, 400.0) + 0.15 * v + 0.02;
            assert!(
                (got - want).abs() < 1e-12,
                "friction_k_max({v}): got {got:e}, want {want:e}"
            );
        }
        // The production path is the cap at FRICTION_FF_K_MAX.
        assert_eq!(
            friction(0.02, 0.6, 800.0, 0.15, 0.02),
            0.6 * coulomb_unit(0.02, 800.0, FRICTION_FF_K_MAX) + 0.15 * 0.02 + 0.02
        );
    }

    /// Golden values from `almond_axol.robot.control.SlewLimiter`: a 1.3 Nm
    /// Coulomb term (k=800 under a 400 cap) through a fast reversal at
    /// 30 Nm/s — the output ramps at exactly rate·dt = 0.125 Nm per tick.
    #[test]
    fn slew_limiter_matches_python() {
        let golden = [
            0.000000000000000e+00,
            1.250000000000000e-01,
            2.500000000000000e-01,
            3.750000000000000e-01,
            5.000000000000000e-01,
            6.250000000000000e-01,
            5.000000000000000e-01,
            3.750000000000000e-01,
            2.500000000000000e-01,
            1.250000000000000e-01,
            0.000000000000000e+00,
            -1.250000000000000e-01,
        ];
        let mut sl = SlewLimiter::new();
        for (k, want) in golden.iter().enumerate() {
            let v = 0.05 * (2.0 * std::f64::consts::PI * 20.0 * k as f64 * DT).sin();
            let x = 1.3 * coulomb_unit(v, 800.0, 400.0);
            let got = sl.update(x, 30.0, DT);
            assert!(
                (got - want).abs() < 1e-12,
                "slew sample {k}: got {got:e}, want {want:e}"
            );
        }
        // rate <= 0 passes through; reset re-adopts the input.
        let mut sl = SlewLimiter::new();
        sl.update(0.0, 0.0, DT);
        assert_eq!(sl.update(5.0, 0.0, DT), 5.0);
        sl.update(0.0, 30.0, DT);
        sl.reset();
        assert_eq!(sl.update(-5.0, 30.0, DT), -5.0);
    }

    /// Golden values from `almond_axol.robot.control.stiction_compensation`
    /// (fc=1.3, gain=0.6, scale=0.1°, `sat` from k=800 under the 100 cap):
    /// saturates toward 0.78 Nm within a few mrad of error, and fades to
    /// nothing once the velocity FF is delivering fc itself.
    #[test]
    fn stiction_matches_python() {
        let scale = 0.1_f64.to_radians();
        let golden = [
            (0.0, 0.0, 0.000000000000000e+00),
            (0.0005, 0.0, 2.175348090835001e-01),
            (0.002, 0.0, 6.367892504091777e-01),
            (-0.002, 0.0, -6.367892504091777e-01),
            (0.01, 0.0, 7.799835384077637e-01),
            (0.002, 0.05, 3.425180122363560e-01),
            (0.002, 0.5, 5.781774956240957e-05),
            (-0.01, -0.02, -6.260340377443776e-01),
        ];
        for (err, v, want) in golden {
            let sat = coulomb_unit(v, 800.0, FRICTION_FF_K_MAX);
            let got = stiction(err, sat, 1.3, 0.6, scale);
            assert!(
                (got - want).abs() < 1e-12,
                "stiction({err}, v={v}): got {got:e}, want {want:e}"
            );
        }
        assert_eq!(stiction(0.01, 0.0, 1.3, 0.0, scale), 0.0);
    }

    /// Golden values from `almond_axol.robot.control.ErrorIntegrator`
    /// (kp=250 at a 0.3 Hz crossover → ki = 471.24, clamp 2.6 Nm, freeze
    /// 2°): winds on a 4 mrad error, holds through two 60 mrad (frozen)
    /// samples, unwinds on the reversed error.
    #[test]
    fn integrator_matches_python() {
        let ki = 471.2388980384689;
        let freeze = 2.0_f64.to_radians();
        let golden = [
            7.853981633974482e-03,
            1.570796326794896e-02,
            2.356194490192345e-02,
            3.141592653589793e-02,
            3.926990816987241e-02,
            4.712388980384689e-02,
            4.712388980384689e-02,
            4.712388980384689e-02,
            3.926990816987241e-02,
            3.141592653589793e-02,
            2.356194490192345e-02,
            1.570796326794897e-02,
        ];
        let errs = [0.004; 6]
            .iter()
            .chain([0.06; 2].iter())
            .chain([-0.004; 4].iter())
            .copied()
            .collect::<Vec<_>>();
        let mut it = Integrator::new();
        for (k, (e, want)) in errs.iter().zip(golden.iter()).enumerate() {
            let got = it.update(*e, ki, 2.6, freeze, DT);
            assert!(
                (got - want).abs() < 1e-12,
                "integrator sample {k}: got {got:e}, want {want:e}"
            );
        }
        // Clamp, dt=0 hold, and ki=0 zeroing.
        let mut it = Integrator::new();
        for _ in 0..10_000 {
            it.update(0.01, ki, 0.5, freeze, DT);
        }
        assert_eq!(it.update(0.01, ki, 0.5, freeze, DT), 0.5);
        assert_eq!(it.update(-0.01, ki, 0.5, freeze, 0.0), 0.5);
        assert_eq!(it.update(0.01, 0.0, 0.5, freeze, DT), 0.0);
    }

    /// The tracker-acceleration inertia path (`exp tracker_accel_ff`) must
    /// also keep the 120 Hz target staircase's adoption/repeat-tick ripple
    /// out of the torque. A single 60 rad/s pole attenuates the Nyquist
    /// comb ~10×; this pins that it is at least 5× below the raw ripple so a
    /// future pole change cannot quietly re-introduce the vibration the
    /// classic double-pole chain was built to remove (see the test above).
    #[test]
    fn tracker_accel_low_pass_rejects_target_rate_ripple() {
        let mut trk = Trapezoid::new(
            1.5 * 2.0 * std::f64::consts::PI,
            1.5 * 7.0 * std::f64::consts::PI,
        );
        trk.seed(0.0);
        let mut lp = LowPass::new();
        lp.seed(0.0);
        let mut raw_sq = 0.0;
        let mut lp_sq = 0.0;
        let mut pairs = 0usize;
        let (mut raw_first, mut lp_first) = (0.0, 0.0);
        for k in 0..1200usize {
            let target_t = (k / 2) as f64 / 120.0;
            let target = 0.6 * (2.0 * std::f64::consts::PI * 0.5 * target_t).sin();
            let (_, _, raw_accel) = trk.update(target, DT);
            let a = lp.update(raw_accel, 60.0, DT);
            if k >= 240 {
                if k % 2 == 0 {
                    raw_first = raw_accel;
                    lp_first = a;
                } else {
                    raw_sq += (raw_accel - raw_first).powi(2);
                    lp_sq += (a - lp_first).powi(2);
                    pairs += 1;
                }
            }
        }
        let raw_rms = (raw_sq / pairs as f64).sqrt();
        let lp_rms = (lp_sq / pairs as f64).sqrt();
        assert!(raw_rms > 1.0, "fixture must expose the raw ripple");
        assert!(
            lp_rms < 0.2 * raw_rms,
            "60 rad/s pole must cut target-rate ripple: raw {raw_rms:e}, filtered {lp_rms:e}"
        );
    }

    /// The reason host damping moved into the core: dissipated power vs the
    /// classic remote-damping chain.
    ///
    /// Both chains compute `τ = kd_host · BP(v_des − v_meas)` with identical
    /// filters, against a joint oscillating at 6.5 Hz (the top of the
    /// shoulder burst band, jit14/15) with the pose-tracked centre on the
    /// mode. The in-core chain samples at 240 Hz and applies the torque the
    /// same tick. The classic chain samples at 120 Hz and its torque rides
    /// the transport delay measured on the rt link: ~4 ms adoption wait plus
    /// a stretched interpolation segment (~10 ms), emulated as a 14 ms
    /// output delay. Damping must dissipate (mean τ·v < 0); measured here,
    /// the delayed chain loses ~half its dissipation to phase lag (ratio
    /// 1.9×). On hardware the chain was worse still — feedback-cache
    /// staleness, asyncio scheduling jitter, and the pose-tracked centre
    /// being computed against a stale pose all add lag on top — which
    /// pushed the loop past 90° into *pumping* the mode: the heavy shaking
    /// observed in rt teleop (2026-08-27).
    #[test]
    fn in_core_damping_dissipates_more_than_delayed_chain() {
        let f_hz = 6.5;
        let w = 2.0 * std::f64::consts::PI * f_hz;
        let w0 = w; // pose-tracked centre sitting on the mode
        let kd_host = 35.0;

        // In-core chain: 240 Hz, applied same tick.
        let dt = 1.0 / 240.0;
        let mut diff = LpDiff::new(80.0);
        let mut bp = BandPass::new();
        let mut power_core = 0.0;
        let n = (2.0 / dt) as usize; // 2 s
        for k in 0..n {
            let t = k as f64 * dt;
            let q_meas = 0.01 * (w * t).sin();
            let v_true = 0.01 * w * (w * t).cos();
            let v_meas = diff.update(q_meas, dt);
            let tau = kd_host * bp.update(0.0 - v_meas, w0, 0.8, dt);
            power_core += tau * v_true;
        }
        power_core /= n as f64;

        // Classic chain: 120 Hz sample, output delayed 14 ms (ZOH between
        // samples), power evaluated on the 240 Hz grid it acts on.
        let dt_s = 1.0 / 120.0;
        let delay = 0.014;
        let mut diff = LpDiff::new(80.0);
        let mut bp = BandPass::new();
        let mut sched: Vec<(f64, f64)> = Vec::new(); // (t_apply, tau)
        let ns = (2.0 / dt_s) as usize;
        for k in 0..ns {
            let t = k as f64 * dt_s;
            let q_meas = 0.01 * (w * t).sin();
            let v_meas = diff.update(q_meas, dt_s);
            let tau = kd_host * bp.update(0.0 - v_meas, w0, 0.8, dt_s);
            sched.push((t + delay, tau));
        }
        let mut power_classic = 0.0;
        let mut applied = 0usize;
        for k in 0..n {
            let t = k as f64 * dt;
            while applied + 1 < sched.len() && sched[applied + 1].0 <= t {
                applied += 1;
            }
            let tau = if sched[applied].0 <= t {
                sched[applied].1
            } else {
                0.0
            };
            let v_true = 0.01 * w * (w * t).cos();
            power_classic += tau * v_true;
        }
        power_classic /= n as f64;

        assert!(
            power_core < 0.0,
            "in-core damping must dissipate (got {power_core:e} W)"
        );
        assert!(
            power_core < 1.5 * power_classic,
            "in-core damping should dissipate at least 1.5x the delayed \
             chain's power (core {power_core:e} W vs classic {power_classic:e} W)"
        );
    }

    /// The Python target stream: 120 Hz, constant velocity.
    const CADENCE: f64 = 1.0 / 120.0;
    const STREAM_VEL: f64 = 1.2; // rad/s

    fn streamed(hold: &mut Holdover, n: usize) -> f64 {
        let mut p = 0.0;
        hold.observe(p, None, Some(CADENCE));
        for _ in 0..n {
            p += STREAM_VEL * CADENCE;
            hold.observe(p, Some(CADENCE), Some(CADENCE));
        }
        p
    }

    #[test]
    fn cadence_learns_the_stream_and_ignores_a_single_late_gap() {
        let mut c = Cadence::new();
        assert_eq!(c.get(), None);
        for _ in 0..50 {
            c.observe(CADENCE);
        }
        let learned = c.get().unwrap();
        assert!((learned - CADENCE).abs() < 1e-9);
        // One 40 ms stall (a late arrival) leaves the estimate alone.
        c.observe(0.040);
        assert_eq!(c.get().unwrap(), learned);
        // As do a few scattered ones with on-time gaps between.
        for _ in 0..3 {
            c.observe(0.030);
            c.observe(CADENCE);
        }
        assert!((c.get().unwrap() - CADENCE).abs() < 1e-3);
        // A resume after a long hold (watchdog territory) is not a cadence.
        c.observe(0.5);
        assert!((c.get().unwrap() - CADENCE).abs() < 1e-3);
    }

    #[test]
    fn cadence_relearns_a_slower_stream_after_a_run_of_long_gaps() {
        // Dagger: 120 Hz teleop intervention, then the policy state at 30 Hz.
        let mut c = Cadence::new();
        for _ in 0..50 {
            c.observe(CADENCE);
        }
        let slow = 1.0 / 30.0;
        for k in 1..Cadence::RELEARN {
            c.observe(slow);
            assert!(
                (c.get().unwrap() - CADENCE).abs() < 1e-9,
                "gap {k}: still the old cadence while the run is short"
            );
        }
        c.observe(slow);
        assert!(
            (c.get().unwrap() - slow).abs() < 1e-9,
            "re-seeded from the run: {:?}",
            c.get()
        );
        // Now an on-time 30 Hz frame is not late for the holdover.
        let hold = Holdover::new(0.08, 0.35);
        let p = 1.0;
        assert_eq!(hold.target(p, slow, c.get()), p);
        // And a faster stream is followed at once (short gaps always count).
        for _ in 0..40 {
            c.observe(CADENCE);
        }
        assert!((c.get().unwrap() - CADENCE).abs() < 1e-3);
    }

    #[test]
    fn cadence_clamps_and_ignores_garbage() {
        let mut c = Cadence::new();
        c.observe(0.0);
        c.observe(-1.0);
        c.observe(f64::NAN);
        assert_eq!(c.get(), None);
        c.observe(10.0);
        assert_eq!(c.get(), Some(Cadence::MAX));
        let mut fast = Cadence::new();
        fast.observe(1e-6);
        assert_eq!(fast.get(), Some(Cadence::MIN));
    }

    #[test]
    fn holdover_is_identity_for_an_on_time_stream() {
        let mut hold = Holdover::new(0.08, 0.35);
        let p = streamed(&mut hold, 60);
        assert!((hold.vel() - STREAM_VEL).abs() < 1e-9);
        // Inside cadence + slack nothing is carried, whatever the velocity.
        for age in [0.0, 0.5 * CADENCE, CADENCE, 1.2 * CADENCE] {
            assert_eq!(hold.target(p, age, Some(CADENCE)), p, "age {age}");
        }
        // Unknown cadence (first target of a session) is identity too.
        assert_eq!(hold.target(p, 0.05, None), p);
    }

    #[test]
    fn holdover_carries_a_late_target_along_the_stream_velocity() {
        let mut hold = Holdover::new(0.08, 0.35);
        let p = streamed(&mut hold, 20);
        // 40 ms after the last target the hand has moved STREAM_VEL·40 ms.
        // The carried target covers most of that (minus the slack window
        // and the deceleration ramp), and the step the tracker sees when
        // the late target lands shrinks from the whole gap to the residual.
        let age = 0.040;
        let truth = p + STREAM_VEL * age;
        let carried = hold.target(p, age, Some(CADENCE));
        let gap = truth - p;
        let residual = truth - carried;
        assert!(carried > p, "must move in the stream's direction");
        assert!(
            residual < 0.5 * gap,
            "residual {residual:.4} should be well under the raw gap {gap:.4}"
        );
        // Monotone in age and never past the true line.
        let mut prev = p;
        for k in 1..=20 {
            let a = k as f64 * 0.004;
            let c = hold.target(p, a, Some(CADENCE));
            assert!(c >= prev, "carried target must not retreat (age {a})");
            assert!(
                c <= p + STREAM_VEL * a + 1e-12,
                "must not lead the stream (age {a})"
            );
            prev = c;
        }
    }

    #[test]
    fn holdover_glides_to_rest_within_max_hold() {
        let max_hold = 0.08;
        let mut hold = Holdover::new(max_hold, 0.35);
        let p = streamed(&mut hold, 60);
        let late_start = Holdover::SLACK * CADENCE;
        let at_rest = hold.target(p, late_start + max_hold, Some(CADENCE));
        // Ramp integral: vel·T/2.
        let want = p + STREAM_VEL * max_hold / 2.0;
        assert!((at_rest - want).abs() < 1e-9, "got {at_rest}, want {want}");
        // Past max_hold nothing more is added: the tracker just holds there.
        assert_eq!(
            hold.target(p, late_start + 2.0 * max_hold, Some(CADENCE)),
            at_rest
        );
        assert_eq!(hold.target(p, 10.0, Some(CADENCE)), at_rest);
        // The velocity at the end of the ramp is zero: reach is flat there.
        let just_before = hold.target(p, late_start + max_hold - 1e-4, Some(CADENCE));
        assert!((at_rest - just_before).abs() < STREAM_VEL * 1e-4 * 0.01);
    }

    #[test]
    fn holdover_reach_is_clamped() {
        let mut hold = Holdover::new(1.0, 0.02);
        let p = streamed(&mut hold, 20);
        let far = hold.target(p, 1.0, Some(CADENCE));
        assert!((far - (p + 0.02)).abs() < 1e-12);
        // Negative direction clamps symmetrically.
        let mut back = Holdover::new(1.0, 0.02);
        back.observe(1.0, None, Some(CADENCE));
        back.observe(1.0 - STREAM_VEL * CADENCE, Some(CADENCE), Some(CADENCE));
        let far_back = back.target(1.0 - STREAM_VEL * CADENCE, 1.0, Some(CADENCE));
        assert!((far_back - (1.0 - STREAM_VEL * CADENCE - 0.02)).abs() < 1e-12);
    }

    #[test]
    fn holdover_restarts_from_rest_after_a_resume_or_reset() {
        let mut hold = Holdover::new(0.08, 0.35);
        let p = streamed(&mut hold, 20);
        assert!(hold.vel() > 0.0);
        // A target landing after a long silence (stream resumed) carries no
        // velocity information: the arm was held, not moving.
        hold.observe(p + 0.3, Some(0.5), Some(CADENCE));
        assert_eq!(hold.vel(), 0.0);
        assert_eq!(hold.target(p + 0.3, 0.05, Some(CADENCE)), p + 0.3);
        // Two more on-time targets and the estimate is live again.
        hold.observe(p + 0.3 + STREAM_VEL * CADENCE, Some(CADENCE), Some(CADENCE));
        assert!(hold.vel() > 0.0);
        // Explicit reset (passthrough target) drops it as well, and
        // the first target after a reset has no previous to difference.
        hold.reset();
        assert_eq!(hold.vel(), 0.0);
        hold.observe(0.0, Some(CADENCE), Some(CADENCE));
        assert_eq!(hold.vel(), 0.0);
        // A missing cadence never produces a velocity.
        hold.observe(0.01, Some(CADENCE), None);
        assert_eq!(hold.vel(), 0.0);
    }

    #[test]
    fn holdover_velocity_survives_the_stall_it_bridged() {
        // Bugbot on #306: the late target that ends a 40 ms stall must not
        // zero the estimate, or a second stall right after (common under the
        // same load) gets no carry.
        let mut hold = Holdover::new(0.08, 0.35);
        let p = streamed(&mut hold, 60);
        let before = hold.vel();
        let stall = 0.040;
        hold.observe(p + STREAM_VEL * stall, Some(stall), Some(CADENCE));
        assert!(
            (hold.vel() - before).abs() < 1e-6,
            "vel {} -> {}",
            before,
            hold.vel()
        );
        // Even the longest stall the carry covers keeps the estimate ...
        let p2 = p + STREAM_VEL * stall;
        let long = 0.08 + 2.0 * CADENCE;
        hold.observe(p2 + STREAM_VEL * long, Some(long), Some(CADENCE));
        assert!((hold.vel() - before).abs() < 1e-6);
        // ... and the next stall is carried, not held flat.
        let p3 = p2 + STREAM_VEL * long;
        assert!(hold.target(p3, 0.030, Some(CADENCE)) > p3);
    }

    #[test]
    fn holdover_shrinks_the_tracker_hitch_across_a_stalled_tick() {
        // The whole point: run the 240 Hz trapezoid against a 120 Hz
        // constant-velocity stream with one 40 ms stall in it, with and
        // without holdover, and compare the worst deviation of the rendered
        // trajectory from the ideal constant-velocity line.
        fn worst_deviation(with_holdover: bool) -> f64 {
            let dt = 1.0 / 240.0;
            let mut trk = Trapezoid::new(6.0, 40.0);
            trk.seed(0.0);
            let mut hold = Holdover::new(0.08, 0.35);
            let stall_start = 0.5;
            let stall_len = 0.040;
            let mut next_target_t = 0.0;
            let mut last_p = 0.0;
            let mut last_arrival = 0.0;
            let mut have_prev = false;
            let mut worst = 0.0;
            let n = (1.5 / dt) as usize;
            for k in 0..n {
                let t = k as f64 * dt;
                // Python's tick: on time, except it sleeps through the stall.
                while next_target_t <= t {
                    let arrives = next_target_t;
                    let in_stall = arrives > stall_start && arrives < stall_start + stall_len;
                    if !in_stall {
                        let p = STREAM_VEL * arrives;
                        let gap = if have_prev {
                            Some(arrives - last_arrival)
                        } else {
                            None
                        };
                        hold.observe(p, gap, Some(CADENCE));
                        last_p = p;
                        last_arrival = arrives;
                        have_prev = true;
                    }
                    next_target_t += CADENCE;
                }
                let target = if with_holdover {
                    hold.target(last_p, t - last_arrival, Some(CADENCE))
                } else {
                    last_p
                };
                let (pos, _, _) = trk.update(target, dt);
                // Only judge the stall and its recovery (steady tracking lag
                // is identical in both runs); subtract that lag out.
                if t > stall_start && t < stall_start + 0.4 {
                    let ideal = STREAM_VEL * t - STREAM_VEL / Trapezoid::POS_TRACK_GAIN;
                    worst = f64::max(worst, (pos - ideal).abs());
                }
            }
            worst
        }
        let plain = worst_deviation(false);
        let held = worst_deviation(true);
        assert!(
            plain > 0.01,
            "the stall must produce a visible hitch to fix ({plain:.4})"
        );
        eprintln!(
            "worst deviation across a 40 ms stall: plain {plain:.4} rad, holdover {held:.4} rad"
        );
        assert!(
            held < 0.5 * plain,
            "holdover should at least halve the hitch (plain {plain:.4}, held {held:.4})"
        );
    }
}
