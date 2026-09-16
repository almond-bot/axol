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

    /// Move the reference position without discarding the velocity state.
    ///
    /// For a gap the filter must not differentiate across (a whole-cycle
    /// overrun) but during which the joint kept moving: the next update
    /// then continues from the pre-gap velocity scaled by `vel_keep`
    /// (0..=1, how much of it the gap is assumed to have preserved) instead
    /// of restarting from zero — which would zero the wire velocity and
    /// friction feedforward for a moving joint and re-ramp them over the
    /// filter's time constant, a speed-proportional brake pulse.
    pub fn rebase(&mut self, pos: f64, vel_keep: f64) {
        self.vel *= vel_keep.clamp(0.0, 1.0);
        self.pos_prev = Some(pos);
    }

    /// The current filtered derivative (what the last `update` returned).
    pub fn value(&self) -> f64 {
        self.vel
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
        // clamp keeps the one-step update stable across a long `dt` (a
        // damping hold folding skipped ticks into one update, a bench tool
        // at a low rate) for *this* q, not just for q ≳ 1.1.
        let f = (2.0 * (0.5 * w0 * dt).min(0.7).sin()).min(Self::max_coefficient(q));
        self.lp += f * self.bp;
        let hp = x - self.lp - self.bp / q;
        self.bp += f * hp;
        self.bp / q
    }

    /// Largest coefficient that keeps the one-step update stable at `q`.
    ///
    /// The Chamberlin state update is `[lp, bp] ← [[1, f], [−f, 1 − f² −
    /// f/q]]·[lp, bp]`, whose eigenvalues stay inside the unit circle only
    /// for `f < −1/q + √(1/q² + 4)` — 1.108 at q = 0.8, 1.694 at q = 3. The
    /// plain `2·sin(0.7) = 1.288` clamp is past that limit for q below ~1.1:
    /// one step there multiplies the stored state by −1.64. Mirrors the
    /// Python `BandPass._max_coefficient`; a 10 % margin keeps the
    /// eigenvalues comfortably inside.
    pub fn max_coefficient(q: f64) -> f64 {
        let limit = -1.0 / q + (1.0 / (q * q) + 4.0).sqrt();
        (2.0 * 0.7_f64.sin()).min(0.9 * limit)
    }

    /// Clear stored energy when leaving the tracked control mode.
    pub fn reset(&mut self) {
        self.lp = 0.0;
        self.bp = 0.0;
        self.primed = false;
    }
}

/// Continuity of the host damping output across ticks whose sample cannot
/// be used (a missed reply, a late wake, a degraded stretch).
///
/// The damping torque is the only damper the shoulders have at their tuned
/// stiffness, and it is largest exactly while a mode is ringing. Switching
/// it to zero for one tick and restarting the band-pass from empty state —
/// the previous behaviour on *every* isolated miss or >0.5 ms-late tick —
/// was itself a 1-7 Nm torque step into that mode, repeated silently below
/// the degraded threshold. This gate replaces the on/off with two rules:
///
/// - An isolated gap (up to `HOLD_TICKS` consecutive bad ticks) *holds* the
///   previous output, decaying it by `HOLD_DECAY`, and keeps the band-pass
///   state. The skipped time is folded into the next update so the filter
///   does not lose phase.
/// - A longer gap lets go: output zero, band-pass reset, and when samples
///   return the output *ramps* back in over `RAMP_S` instead of stepping.
///
/// Neither rule ever uses a stale sample as if it were fresh: a held value
/// is the last torque the loop *did* trust, briefly continued, which is a
/// far smaller discontinuity than a step to zero.
pub struct DampGate {
    /// Last valid band-pass output (before the ramp gain).
    last: f64,
    /// Consecutive ticks without a valid sample.
    bad_ticks: u32,
    /// Ramp-in gain after a reset, 0..=1.
    gain: f64,
    /// Time spanned by held ticks, folded into the next real update.
    skipped_dt: f64,
    /// Ticks the output was held through an isolated gap (for stats).
    pub holds: u64,
    /// Times the gate let go and will ramp back in (for stats).
    pub releases: u64,
}

impl DampGate {
    /// Consecutive bad ticks the previous output is held through.
    pub const HOLD_TICKS: u32 = 3;
    /// Per-held-tick factor applied to the held output, cumulatively
    /// (index = bad tick − 1): the hold runs 1.0, 0.5, 0.125 of the last
    /// trusted value, then lets go.
    pub const HOLD_DECAY: [f64; 3] = [1.0, 0.5, 0.25];
    /// Ramp-in time (s) from zero to full damping after a release.
    pub const RAMP_S: f64 = 0.05;

    pub fn new() -> Self {
        Self {
            last: 0.0,
            bad_ticks: 0,
            gain: 0.0,
            skipped_dt: 0.0,
            holds: 0,
            releases: 0,
        }
    }

    /// A tick with a trusted sample: advance the band-pass and return the
    /// damping input to apply (ramp gain included).
    pub fn good(&mut self, bp: &mut BandPass, x: f64, w0: f64, q: f64, dt: f64) -> f64 {
        let dt_total = dt + self.skipped_dt;
        self.skipped_dt = 0.0;
        self.bad_ticks = 0;
        let raw = bp.update(x, w0, q, dt_total);
        self.gain = (self.gain + dt / Self::RAMP_S).min(1.0);
        self.last = raw;
        raw * self.gain
    }

    /// A tick without a trusted sample: hold briefly, then let go.
    pub fn bad(&mut self, bp: &mut BandPass, dt: f64) -> f64 {
        self.bad_ticks += 1;
        if self.bad_ticks <= Self::HOLD_TICKS {
            self.holds += 1;
            self.skipped_dt += dt;
            self.last *= Self::HOLD_DECAY[(self.bad_ticks - 1) as usize];
            return self.last * self.gain;
        }
        if self.bad_ticks == Self::HOLD_TICKS + 1 {
            self.releases += 1;
        }
        self.let_go(bp);
        0.0
    }

    /// Leaving tracked mode (passthrough hold, limp): drop everything so the
    /// next tracked tick ramps in from zero on a fresh band-pass.
    pub fn reset(&mut self, bp: &mut BandPass) {
        self.bad_ticks = 0;
        self.let_go(bp);
    }

    fn let_go(&mut self, bp: &mut BandPass) {
        bp.reset();
        self.last = 0.0;
        self.gain = 0.0;
        self.skipped_dt = 0.0;
    }
}

impl Default for DampGate {
    fn default() -> Self {
        Self::new()
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

    /// `rebase` keeps the velocity state (scaled) and only moves the
    /// reference, so the next update continues from it; `seed` restarts
    /// from zero. This is the overrun re-prime: a moving joint must not
    /// see its wire velocity and friction feedforward zeroed.
    #[test]
    fn lpdiff_rebase_keeps_velocity_seed_does_not() {
        let mut a = LpDiff::new(20.0);
        let mut b = LpDiff::new(20.0);
        // Ramp both to a steady 1 rad/s.
        for k in 0..400 {
            let x = k as f64 * DT;
            a.update(x, DT);
            b.update(x, DT);
        }
        assert!((a.value() - 1.0).abs() < 1e-3, "{}", a.value());
        // A 30 ms gap the joint kept moving through.
        let gap = 0.03;
        let x_after = 400.0 * DT + gap;
        a.rebase(x_after, 0.8);
        b.seed(x_after);
        assert!((a.value() - 0.8).abs() < 1e-9);
        assert_eq!(b.value(), 0.0);
        // Next real step: continuity from the kept state vs. a restart.
        let x_next = x_after + DT;
        let va = a.update(x_next, DT);
        let vb = b.update(x_next, DT);
        assert!(va > 0.75, "rebased chain continued at {va}");
        assert!(vb < 0.1, "seeded chain restarted at {vb}");
        // Scaling is clamped to [0, 1].
        a.rebase(x_next, 7.0);
        assert!((a.value() - va).abs() < 1e-12);
        a.rebase(x_next, -1.0);
        assert_eq!(a.value(), 0.0);
    }

    /// A long step never amplifies the band-pass state: the coefficient is
    /// clamped inside the one-step stability limit for the channel's q.
    #[test]
    fn bandpass_long_step_decays_at_low_q() {
        for q in [0.5, 0.8, 1.0, 3.0] {
            let f = BandPass::max_coefficient(q);
            let limit = -1.0 / q + (1.0 / (q * q) + 4.0).sqrt();
            assert!(f < limit, "q={q}: f {f} vs limit {limit}");
            let trace = 2.0 - f * f - f / q;
            let det = 1.0 - f / q;
            let disc = trace * trace - 4.0 * det;
            let radius = if disc >= 0.0 {
                ((trace + disc.sqrt()) / 2.0)
                    .abs()
                    .max(((trace - disc.sqrt()) / 2.0).abs())
            } else {
                det.abs().sqrt()
            };
            assert!(radius < 1.0, "q={q}: spectral radius {radius}");
        }
        // Inject energy, then feed zeros at a rate that saturates the sin()
        // clamp (0.5·w0·dt > 0.7): the state must decay, where the old
        // fixed clamp at q = 0.8 grew it by 1.64x per step, sign-flipping.
        let mut bp = BandPass::new();
        bp.update(0.0, 50.0, 0.8, 0.03);
        bp.update(1.0, 50.0, 0.8, 0.03);
        let first = bp.lp.hypot(bp.bp);
        let mut worst = first;
        for _ in 0..30 {
            bp.update(0.0, 50.0, 0.8, 0.03);
            worst = worst.max(bp.lp.hypot(bp.bp));
        }
        let last = bp.lp.hypot(bp.bp);
        assert!(last < 1e-3 * first, "state {first} -> {last}");
        assert!(worst < 2.0 * first, "state peaked at {worst} from {first}");
    }

    /// Drive the gate with a steady band-pass input and check each rule:
    /// an isolated bad tick holds (not zero) with the band-pass state
    /// intact, a long gap lets go and ramps back in, a mode reset ramps in.
    #[test]
    fn damp_gate_holds_through_isolated_gaps_and_ramps_after_long_ones() {
        let w0 = 20.0;
        let q = 0.8;
        let amp = 0.2;
        // A 3.2 Hz velocity error — right at the band-pass centre, so the
        // steady-state output is essentially the input.
        let input = |k: usize| (w0 * k as f64 * DT).sin() * amp;
        // The gated chain under test, and a reference chain that never
        // misses a sample, driven with the same input.
        let mut bp = BandPass::new();
        let mut gate = DampGate::new();
        let mut bp_ref = BandPass::new();
        let ramp_ticks = (DampGate::RAMP_S / DT).ceil() as usize;

        // Warm up: full gain is reached within RAMP_S and, once there, the
        // gated chain is the reference chain.
        let mut k = 0;
        while k < 600 {
            let out = gate.good(&mut bp, input(k), w0, q, DT);
            let reference = bp_ref.update(input(k), w0, q, DT);
            if k == 2 {
                assert!(out.abs() < reference.abs(), "still ramping at tick 2");
            }
            if k > 2 * ramp_ticks {
                assert!(
                    (out - reference).abs() < 1e-12,
                    "tick {k}: {out} vs {reference}"
                );
            }
            k += 1;
        }
        assert!((gate.gain - 1.0).abs() < 1e-12);

        // One isolated bad tick: the previous output is held flat (not
        // zeroed) and the band-pass state survives, so the next good tick
        // lands within a few percent of the never-missed reference.
        let before = gate.good(&mut bp, input(k), w0, q, DT);
        bp_ref.update(input(k), w0, q, DT);
        k += 1;
        let held = gate.bad(&mut bp, DT);
        bp_ref.update(input(k), w0, q, DT);
        k += 1;
        assert_eq!(held, before, "first bad tick holds the last trusted output");
        let resumed = gate.good(&mut bp, input(k), w0, q, DT);
        let reference = bp_ref.update(input(k), w0, q, DT);
        k += 1;
        assert!(
            (resumed - reference).abs() < 0.05 * amp,
            "resumed at {resumed}, never-missed chain at {reference}"
        );
        assert_eq!(gate.holds, 1);
        assert_eq!(gate.releases, 0);

        // Three bad ticks fade 1.0, 0.5, 0.125 of the last value; the fourth
        // lets go: zero, band-pass reset, one release counted.
        let base = gate.good(&mut bp, input(k), w0, q, DT);
        k += 1;
        let h1 = gate.bad(&mut bp, DT);
        let h2 = gate.bad(&mut bp, DT);
        let h3 = gate.bad(&mut bp, DT);
        assert_eq!(h1, base);
        assert_eq!(h2, 0.5 * base);
        assert_eq!(h3, 0.125 * base);
        assert_eq!(gate.bad(&mut bp, DT), 0.0);
        assert_eq!(gate.releases, 1);
        assert_eq!(gate.gain, 0.0);
        // Further bad ticks stay at zero without counting more releases.
        assert_eq!(gate.bad(&mut bp, DT), 0.0);
        assert_eq!(gate.releases, 1);
        assert_eq!(gate.holds, 4);

        // Recovery ramps in over RAMP_S rather than stepping. Start where the
        // input sits near its crest so the sine itself barely moves across
        // the ramp window and the growth seen is the gate's.
        let crest = (std::f64::consts::FRAC_PI_2 / (w0 * DT)).round() as usize;
        let cycle = (2.0 * std::f64::consts::PI / (w0 * DT)).round() as usize;
        k = crest + cycle * (k / cycle + 1);
        let mut outs = Vec::new();
        for i in 0..ramp_ticks {
            outs.push(gate.good(&mut bp, input(k + i), w0, q, DT).abs());
        }
        assert!(
            (gate.gain - 1.0).abs() < 1e-9,
            "ramp complete: {}",
            gate.gain
        );
        assert_eq!(outs[0], 0.0, "first tick back only primes the band-pass");
        assert!(
            outs[1] < 0.1 * amp,
            "second tick back is small: {}",
            outs[1]
        );
        assert!(
            outs[1] < outs[ramp_ticks / 2] && outs[ramp_ticks / 2] < outs[ramp_ticks - 1],
            "gain grows through the ramp: {outs:?}"
        );

        // A mode reset drops to zero and ramps in again the same way.
        gate.reset(&mut bp);
        assert_eq!(gate.gain, 0.0);
        assert_eq!(gate.good(&mut bp, input(k), w0, q, DT), 0.0);
        let v = gate.good(&mut bp, input(k + 1), w0, q, DT);
        assert!(v.abs() < 0.1 * amp, "ramping after reset: {v}");
    }
}
