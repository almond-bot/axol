//! Command ramp and traction guard for the Jelly wheel loop.
//!
//! Both operate on the normalized body command `(vx, vy, wz)`, where a full
//! stick is 1, before it is mixed into wheel speeds, and neither touches CAN.
//! They are plain state machines so they can be unit-tested against the
//! numbers the Python side documents (`JellyConfig` in
//! `almond_axol/robot/jelly.py`).

/// Rate- and jerk-limited ramp of the normalized `(vx, vy, wz)` command.
///
/// The command is slewed as a single vector — the step's magnitude is capped
/// but its direction kept — so a mostly-forward command with a small lateral
/// part doesn't finish its lateral ramp first and veer before straightening.
///
/// Two rate limits apply: `accel` while the step moves the command away from
/// zero (its projection onto the current command is non-negative) and `decel`
/// while it moves toward zero, so stops and speed reductions can be brisker
/// than launches. A reversal uses `decel` down to zero, then `accel` out the
/// other side.
///
/// With `jerk` > 0 the command's rate of change is itself a state — a
/// velocity vector in command space — that may only change by `jerk` per
/// second. It is steered toward the limit speed along the remaining delta,
/// capped at √(2·jerk·remaining) so it reaches zero exactly as the command
/// reaches its target: an S-shaped profile with no acceleration step at
/// either end of a launch or stop, including a stick released mid-launch (the
/// rate swings smoothly through zero instead of flipping sign). `jerk` = 0 is
/// the plain trapezoid.
///
/// All quantities are in normalized command units per second / second².
#[derive(Clone, Debug)]
pub struct VectorRamp {
    accel: f64,
    decel: f64,
    jerk: f64,
    dt: f64,
    /// Command rate of change (normalized/s).
    vel: [f64; 3],
    /// Rate limit in force on the last step (after scaling); 0 when idle.
    limit: f64,
}

impl VectorRamp {
    pub fn new(accel: f64, decel: f64, jerk: f64, dt: f64) -> Self {
        assert!(
            accel > 0.0 && decel > 0.0 && jerk >= 0.0 && dt > 0.0,
            "ramp needs accel > 0, decel > 0, jerk >= 0, dt > 0"
        );
        Self {
            accel,
            decel,
            jerk,
            dt,
            vel: [0.0; 3],
            limit: 0.0,
        }
    }

    /// Magnitude of the command's current rate of change (normalized/s).
    pub fn rate(&self) -> f64 {
        self.vel.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Rate limit the last step ran under (after any scaling); 0 when idle.
    pub fn limit(&self) -> f64 {
        self.limit
    }

    /// Forget the rate state: the next step starts from rest.
    pub fn reset(&mut self) {
        self.vel = [0.0; 3];
        self.limit = 0.0;
    }

    /// Advance `cmd` one interval toward `target` (in place).
    ///
    /// `accel_scale` / `decel_scale` (in (0, 1]) ease the respective rate
    /// limit for this step — the traction guard's lever. The jerk limit is
    /// untouched, so an easing that arrives mid-ramp is itself felt as a
    /// smooth change of acceleration rather than a step.
    pub fn step(
        &mut self,
        cmd: &mut [f64; 3],
        target: [f64; 3],
        accel_scale: f64,
        decel_scale: f64,
    ) {
        let deltas = [target[0] - cmd[0], target[1] - cmd[1], target[2] - cmd[2]];
        let norm = deltas.iter().map(|d| d * d).sum::<f64>().sqrt();
        if norm <= 0.0 {
            self.reset();
            return;
        }
        let toward_zero = deltas
            .iter()
            .zip(cmd.iter())
            .map(|(d, c)| d * c)
            .sum::<f64>()
            < 0.0;
        let limit = if toward_zero {
            self.decel * decel_scale
        } else {
            self.accel * accel_scale
        };
        self.limit = limit;
        let vel = if self.jerk > 0.0 {
            let speed = limit.min((2.0 * self.jerk * norm).sqrt());
            let want = deltas.map(|d| d / norm * speed);
            let mut dv = [
                want[0] - self.vel[0],
                want[1] - self.vel[1],
                want[2] - self.vel[2],
            ];
            let dv_norm = dv.iter().map(|x| x * x).sum::<f64>().sqrt();
            let max_dv = self.jerk * self.dt;
            if dv_norm > max_dv {
                for x in &mut dv {
                    *x *= max_dv / dv_norm;
                }
            }
            [
                self.vel[0] + dv[0],
                self.vel[1] + dv[1],
                self.vel[2] + dv[2],
            ]
        } else {
            deltas.map(|d| d / norm * limit)
        };
        let step = vel.map(|v| v * self.dt);
        let advance = step
            .iter()
            .zip(deltas.iter())
            .map(|(s, d)| s * d)
            .sum::<f64>()
            / norm;
        if advance >= norm {
            // Would reach or pass the target this interval: land on it. The
            // velocity state is zeroed so a fresh target starts from rest.
            *cmd = target;
            self.vel = [0.0; 3];
            return;
        }
        self.vel = vel;
        for i in 0..3 {
            cmd[i] += step[i];
        }
    }
}

/// The guard never eases braking below this fraction of `decel`: a stop that
/// slides a little beats one that takes twice as long, and the command-timeout
/// safety path (dead operator → decay to zero) rides on decel too.
pub const TRACTION_DECEL_FLOOR: f64 = 0.5;
/// The guard judges only while the ramp is moving the command at no less than
/// this fraction of its (scaled) rate limit — the heart of a launch or stop,
/// where the wheels share a large net force. In the S-curve's tails the net
/// force fades and individual torques cross zero, which is not a lift.
pub const TRACTION_MIN_RAMP_FRAC: f64 = 0.3;
/// Recovery time constant of the guard's scale while the ramp is idle.
pub const TRACTION_IDLE_RECOVER_S: f64 = 0.3;
/// Time constant of the drop toward the floor once a wheel is confirmed light.
const TRACTION_DROP_S: f64 = 0.1;
/// Recovery time constant while the ramp is still moving.
const TRACTION_RECOVER_S: f64 = 1.5;
/// Consecutive light cycles before the guard acts.
const TRACTION_CONFIRM: u32 = 3;

/// Eases the command ramp while a wheel has lost the floor.
///
/// A four-wheel x-drive on a rigid frame is statically indeterminate, and a
/// tall base with its mass off the wheelbase centre transfers a good share of
/// its weight front↔back whenever it accelerates or brakes along x. The wheel
/// that goes light spins without pushing, and the two wheels that share a
/// drive diagonal with it and each other become the only ones pushing: they
/// can only push along their diagonal, so the base veers toward it until the
/// lifted wheel lands. No wheel command can push through a wheel that isn't
/// touching the floor; what does work is asking for less acceleration while
/// it's light (measured on Jelly: a 0.5/s ramp veered 6–28°, 0.15/s under 2°).
/// This guard does that automatically, so the ramp can stay brisk whenever the
/// floor takes it.
///
/// Detection is from the motors' torque feedback, which every velocity
/// command's reply carries at no bus cost. A wheel carrying its share of the
/// load shows a torque in proportion; one in the air shows only its own
/// inertia and bearing drag. So the wheel with the smallest |τ| is judged
/// light when it falls under `light` × the mean |τ| of the four, provided that
/// mean exceeds `torque_min` (below it the torques say nothing and the guard
/// stands down), the ramp is moving at a substantial rate (`ramping`; at
/// cruise the torques are dominated by the wheels working against each other,
/// which says nothing about the floor), and the condition has held for
/// `TRACTION_CONFIRM` cycles (every wheel's torque passes through zero when a
/// ramp turns around; a real lift lasts the whole ramp).
///
/// The output is a scale on the ramp's rate limits: it drops toward `floor`
/// within `TRACTION_DROP_S` of a confirmed light wheel and recovers toward 1
/// over `TRACTION_RECOVER_S` while the ramp is still moving (slow, so a launch
/// hunts at most once) and within `TRACTION_IDLE_RECOVER_S` once it isn't, so
/// an easing picked up in the last moments of a stop doesn't soften the next
/// launch. Braking is eased by the same scale but never below
/// `TRACTION_DECEL_FLOOR`.
#[derive(Clone, Debug)]
pub struct TractionGuard {
    light: f64,
    floor: f64,
    torque_min: f64,
    dt: f64,
    streak: u32,
    scale: f64,
    light_wheel: Option<usize>,
}

impl TractionGuard {
    pub fn new(light: f64, floor: f64, torque_min: f64, dt: f64) -> Self {
        assert!(
            light > 0.0
                && light < 1.0
                && floor > 0.0
                && floor <= 1.0
                && torque_min >= 0.0
                && dt > 0.0,
            "traction guard needs 0 < light < 1, 0 < floor <= 1, torque_min >= 0, dt > 0"
        );
        Self {
            light,
            floor,
            torque_min,
            dt,
            streak: 0,
            scale: 1.0,
            light_wheel: None,
        }
    }

    /// Scale on the decel rate limit: the same, floored.
    pub fn decel_scale(&self) -> f64 {
        self.scale.max(TRACTION_DECEL_FLOOR)
    }

    /// Wheel index judged (and confirmed) light on the last update.
    pub fn light_wheel(&self) -> Option<usize> {
        self.light_wheel
    }

    /// Feed this cycle's per-wheel torques (Nm; `None` when there is no fresh
    /// reading) and whether the ramp is moving; returns the accel scale.
    pub fn update(&mut self, torques: Option<&[f64; 4]>, ramping: bool) -> f64 {
        let mut light = None;
        if ramping {
            if let Some(t) = torques {
                let mags = t.map(f64::abs);
                let mean = mags.iter().sum::<f64>() / 4.0;
                let (i, min) =
                    mags.iter()
                        .copied()
                        .enumerate()
                        .fold(
                            (0, f64::INFINITY),
                            |a, (i, v)| if v < a.1 { (i, v) } else { a },
                        );
                if mean >= self.torque_min && min < self.light * mean {
                    light = Some(i);
                }
            }
        }
        self.streak = if light.is_some() { self.streak + 1 } else { 0 };
        if self.streak < TRACTION_CONFIRM {
            light = None;
        }
        self.light_wheel = light;
        if light.is_some() {
            self.scale += (self.floor - self.scale) * (self.dt / TRACTION_DROP_S).min(1.0);
        } else {
            let recover = if ramping {
                TRACTION_RECOVER_S
            } else {
                TRACTION_IDLE_RECOVER_S
            };
            self.scale += (1.0 - self.scale) * (self.dt / recover).min(1.0);
        }
        self.scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 0.02;

    fn drive(
        ramp: &mut VectorRamp,
        cmd: &mut [f64; 3],
        target: [f64; 3],
        cycles: usize,
    ) -> Vec<f64> {
        (0..cycles)
            .map(|_| {
                ramp.step(cmd, target, 1.0, 1.0);
                cmd.iter().map(|c| c * c).sum::<f64>().sqrt()
            })
            .collect()
    }

    #[test]
    fn trapezoid_reaches_target_exactly_at_the_accel_rate() {
        let mut ramp = VectorRamp::new(0.5, 1.0, 0.0, DT);
        let mut cmd = [0.0; 3];
        let trace = drive(&mut ramp, &mut cmd, [1.0, 0.0, 0.0], 110);
        // 0.5/s → 2 s = 100 cycles to full stick, then pinned there.
        assert!((trace[49] - 0.5).abs() < 1e-9);
        assert_eq!(cmd, [1.0, 0.0, 0.0]);
        assert_eq!(trace[100], 1.0);
        assert_eq!(ramp.rate(), 0.0);
    }

    #[test]
    fn release_uses_the_faster_decel_and_lands_on_exact_zero() {
        let mut ramp = VectorRamp::new(0.5, 1.0, 0.0, DT);
        let mut cmd = [1.0, 0.0, 0.0];
        let trace = drive(&mut ramp, &mut cmd, [0.0; 3], 60);
        assert!((trace[24] - 0.5).abs() < 1e-9);
        assert_eq!(cmd, [0.0; 3]);
        assert_eq!(trace[50], 0.0);
    }

    #[test]
    fn reversal_decelerates_through_zero_then_accelerates() {
        let mut ramp = VectorRamp::new(0.5, 1.0, 0.0, DT);
        let mut cmd = [1.0, 0.0, 0.0];
        for _ in 0..50 {
            ramp.step(&mut cmd, [-1.0, 0.0, 0.0], 1.0, 1.0);
        }
        assert!(
            cmd[0].abs() < 1e-9,
            "1 s of decel brings +1 to 0: {}",
            cmd[0]
        );
        for _ in 0..50 {
            ramp.step(&mut cmd, [-1.0, 0.0, 0.0], 1.0, 1.0);
        }
        assert!(
            (cmd[0] + 0.5).abs() < 1e-9,
            "then accel at 0.5/s: {}",
            cmd[0]
        );
    }

    #[test]
    fn direction_is_preserved_while_ramping() {
        let mut ramp = VectorRamp::new(0.5, 1.0, 0.0, DT);
        let mut cmd = [0.0; 3];
        for _ in 0..20 {
            ramp.step(&mut cmd, [0.8, 0.2, 0.0], 1.0, 1.0);
            if cmd[0] > 0.0 {
                assert!((cmd[1] / cmd[0] - 0.25).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn jerk_limit_gives_an_s_curve_that_still_settles_exactly() {
        let mut ramp = VectorRamp::new(0.5, 1.0, 2.0, DT);
        let mut cmd = [0.0; 3];
        let trace = drive(&mut ramp, &mut cmd, [1.0, 0.0, 0.0], 200);
        // Rate builds at 2/s²: after one cycle the command moved by only
        // jerk·dt² = 0.0008 (a trapezoid would have jumped 0.01).
        assert!((trace[0] - 2.0 * DT * DT).abs() < 1e-9, "{}", trace[0]);
        // Monotonic, reaches exactly 1 and stays.
        assert!(trace.windows(2).all(|w| w[1] >= w[0] - 1e-12));
        assert_eq!(cmd, [1.0, 0.0, 0.0]);
        assert_eq!(ramp.rate(), 0.0);
        // Slower than the trapezoid (which takes 100 cycles) but not by much.
        let done = trace.iter().position(|v| *v >= 1.0).unwrap();
        assert!((100..130).contains(&done), "{done}");
    }

    #[test]
    fn scaled_ramp_is_slower() {
        let mut full = VectorRamp::new(0.5, 1.0, 0.0, DT);
        let mut eased = VectorRamp::new(0.5, 1.0, 0.0, DT);
        let (mut a, mut b) = ([0.0; 3], [0.0; 3]);
        for _ in 0..10 {
            full.step(&mut a, [1.0, 0.0, 0.0], 1.0, 1.0);
            eased.step(&mut b, [1.0, 0.0, 0.0], 0.2, 1.0);
        }
        assert!((b[0] / a[0] - 0.2).abs() < 1e-9);
        assert!((eased.limit() - 0.1).abs() < 1e-12);
    }

    const LOADED: [f64; 4] = [1.0, -1.0, 1.0, -1.0];
    const LIFTED: [f64; 4] = [0.06, -1.3, 1.3, -1.3];

    #[test]
    fn loaded_wheels_leave_the_ramp_alone() {
        let mut g = TractionGuard::new(0.35, 0.2, 0.3, DT);
        for _ in 0..50 {
            assert_eq!(g.update(Some(&LOADED), true), 1.0);
        }
        assert_eq!(g.light_wheel(), None);
    }

    #[test]
    fn lifted_wheel_is_confirmed_then_eases_to_the_floor() {
        let mut g = TractionGuard::new(0.35, 0.2, 0.3, DT);
        let scales: Vec<f64> = (0..30).map(|_| g.update(Some(&LIFTED), true)).collect();
        assert_eq!(&scales[..2], &[1.0, 1.0]);
        assert!(scales[2] < 1.0);
        assert_eq!(g.light_wheel(), Some(0));
        assert!((scales[29] - 0.2).abs() < 0.01);
        assert!((g.decel_scale() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn zero_crossing_does_not_trip_it() {
        let mut g = TractionGuard::new(0.35, 0.2, 0.3, DT);
        g.update(Some(&LIFTED), true);
        g.update(Some(&LIFTED), true);
        assert_eq!(g.update(Some(&LOADED), true), 1.0);
        assert_eq!(g.light_wheel(), None);
    }

    #[test]
    fn recovers_slowly_while_ramping_and_fast_when_idle() {
        let mut eased = TractionGuard::new(0.35, 0.2, 0.3, DT);
        for _ in 0..30 {
            eased.update(Some(&LIFTED), true);
        }
        let mut ramping = eased.clone();
        let mut idle = eased.clone();
        for _ in 0..25 {
            ramping.update(Some(&LOADED), true);
            idle.update(Some(&LOADED), false);
        }
        assert!(ramping.update(Some(&LOADED), true) < 0.6);
        assert!(idle.update(Some(&LOADED), false) > 0.8);
    }

    #[test]
    fn stands_down_below_torque_min_and_off_the_ramp() {
        let mut g = TractionGuard::new(0.35, 0.2, 0.3, DT);
        let quiet = [0.01, -0.1, 0.1, -0.1];
        for _ in 0..10 {
            assert_eq!(g.update(Some(&quiet), true), 1.0);
        }
        for _ in 0..10 {
            assert_eq!(g.update(Some(&LIFTED), false), 1.0);
        }
        for _ in 0..10 {
            assert_eq!(g.update(None, true), 1.0);
        }
    }
}
