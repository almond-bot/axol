//! Standalone scalar reference equations for the Mink tracking fixture.
//!
//! Adapted from Axol's public `almond_axol.robot.control` differentiator,
//! band-pass and friction equations and `almond_axol.teleop.filter` target
//! tracker. Keep this reference fixed when changing runtime filters: its role
//! is to produce an independent numerical oracle, not mirror such changes.
//! Compiled only by gen_mink_trace.py; never linked into the realtime core.

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
    fc * (0.1 * k.min(FRICTION_FF_K_MAX) * v).tanh() + fv * v + fo
}

/// Velocity/acceleration-limited target tracker — the per-joint
/// `TrapezoidalFilter` from `almond_axol.teleop.filter`, ported per-scalar
/// with a per-step `dt` (the Python original fixes dt at construction).
///
/// A critically damped second-order linear loop (position error → velocity
/// command → acceleration, ζ = 1) under hard velocity and acceleration
/// clamps, with the time-optimal sqrt braking rule kept only as a velocity
/// *ceiling* for large catch-up moves. See the Python docstring for the
/// braking and arrival behavior.
///
/// In the core this runs at the full loop rate against the latest streamed
/// target, replacing linear segment interpolation: its `(pos, vel, accel)`
/// states drive the MIT command and the friction/inertia feedforwards, so
/// the wire physics are coherent with the trajectory actually executed.
pub struct Trapezoid {
    pub max_vel: f64,
    pub max_accel: f64,
    pos: f64,
    vel: f64,
    seeded: bool,
}

impl Trapezoid {
    const POS_TRACK_GAIN: f64 = 15.7; // 1/s = ωn/2 with ωn = 2π·5 Hz
    const VEL_TRACK_GAIN: f64 = 62.8; // 1/s = 2·ωn
    const BRAKE_MARGIN: f64 = 0.8;

    /// Unseeded, matching the Python original: the first `update` adopts
    /// the target as the output (no transient).
    pub fn new(max_vel: f64, max_accel: f64) -> Self {
        Self {
            max_vel,
            max_accel,
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
        let desired = (Self::POS_TRACK_GAIN * err).clamp(-ceiling, ceiling);

        let vel_prev = self.vel;
        let mut vel =
            vel_prev + (Self::VEL_TRACK_GAIN * (desired - vel_prev) * dt).clamp(-adt, adt);

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
