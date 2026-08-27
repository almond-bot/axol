//! Realtime filters for the in-core host-damping loop, ported exactly from
//! `almond_axol.robot.control` (`Differentiator` and `BandPass`). The golden
//! tests at the bottom pin the outputs to the Python implementations
//! sample-for-sample, so the damping a joint feels is identical whichever
//! side computes it — only the phase (rate + freshness) differs.

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
        Self { cutoff, vel: 0.0, pos_prev: None }
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
        Self { lp: 0.0, bp: 0.0, primed: false }
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
            let tau = if sched[applied].0 <= t { sched[applied].1 } else { 0.0 };
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
}
