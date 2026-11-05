//! Thermal limit for the Damiao wrists (`wrist_2`, `wrist_3`).
//!
//! The DM-J4310 is rated for 3.5 Nm continuous (12.5 Nm peak). Held past
//! that — a wrist leaning on a box, or pressing on something — its coil
//! heats until the firmware's over-temperature protection (`OT_Value`, set
//! to 100 °C) or the driver's own 120 °C cutoff takes it out of Enable Mode
//! and the wrist goes limp mid-task.
//!
//! Every Damiao feedback frame carries the coil (`T_Rotor`) and driver
//! (`T_MOS`) temperatures. [`ThermalGuard`] watches them per joint:
//!
//! - Once either reaches its limit, the joint's spring-torque cap ramps down
//!   over `ramp_s` to `cool_cap_nm`, a torque low enough for the motor to
//!   shed heat, and stays there.
//! - Only once both have cooled to their resume temperature does the cap
//!   ramp back up over `ramp_s` and lift. The gap between the two (the
//!   hysteresis) is what lets the motor actually cool before it is loaded
//!   again, instead of flickering at the limit.
//!
//! The cap is a spring cap like box mode's (`filter::cap_spring`): gravity
//! feedforward is outside it, so the wrist still carries its own weight and
//! the gripper's. MIT joints only; a wrist on its firmware position loop
//! (`wire pv`) has no spring to cap.

/// Thresholds and behaviour, from the config's `thermal` line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThermalLimit {
    /// Whether the guard runs at all (`thermal off` turns it off).
    pub enabled: bool,
    /// Coil temperature (°C) that starts the cool-down.
    pub coil_limit_c: f64,
    /// Coil temperature (°C) the cool-down waits for.
    pub coil_resume_c: f64,
    /// Driver (MOSFET) temperature (°C) that starts the cool-down.
    pub driver_limit_c: f64,
    /// Driver temperature (°C) the cool-down waits for.
    pub driver_resume_c: f64,
    /// Spring-torque cap (Nm) while cooling.
    pub cool_cap_nm: f64,
    /// Seconds over which the cap ramps down at the limit, and back up
    /// after the cool-down.
    pub ramp_s: f64,
}

impl Default for ThermalLimit {
    fn default() -> Self {
        // Mirrors `WristThermalConfig` in almond_axol/robot/config.py.
        Self {
            enabled: true,
            coil_limit_c: 85.0,
            coil_resume_c: 75.0,
            driver_limit_c: 105.0,
            driver_resume_c: 95.0,
            cool_cap_nm: 1.5,
            ramp_s: 2.0,
        }
    }
}

/// How far past its limit (°C) a joint that is already cooling may heat
/// before the guard warns that the reduced torque isn't enough.
pub const STILL_HEATING_MARGIN_C: f64 = 10.0;

/// The cap (Nm) the ramp starts from going down and ends at coming up:
/// past the DM-J4310's 12.5 Nm peak, so the ramp's top is no cap at all in
/// practice.
const FULL_CAP_NM: f64 = 12.5;

impl ThermalLimit {
    /// Parse the config line's fields after `thermal`: `off`, or
    /// `<coil_limit> <coil_resume> <driver_limit> <driver_resume>
    /// <cool_cap_nm> <ramp_s>`. `None` on anything else, including resume
    /// temperatures that aren't below their limits.
    pub fn parse(fields: &[&str]) -> Option<Self> {
        if fields == ["off"] {
            return Some(Self {
                enabled: false,
                ..Self::default()
            });
        }
        let v: Vec<f64> = fields
            .iter()
            .map(|f| f.parse().ok())
            .collect::<Option<_>>()?;
        let [coil_limit_c, coil_resume_c, driver_limit_c, driver_resume_c, cool_cap_nm, ramp_s] =
            v[..]
        else {
            return None;
        };
        let limit = Self {
            enabled: true,
            coil_limit_c,
            coil_resume_c,
            driver_limit_c,
            driver_resume_c,
            cool_cap_nm,
            ramp_s,
        };
        let ok = v.iter().all(|x| x.is_finite())
            && coil_resume_c < coil_limit_c
            && driver_resume_c < driver_limit_c
            && cool_cap_nm > 0.0
            && ramp_s >= 0.0;
        ok.then_some(limit)
    }
}

/// A transition worth telling the operator about.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ThermalEvent {
    /// A limit was reached: the torque starts coming down.
    Cooling { coil_c: f64, driver_c: f64 },
    /// Cooling, and still heating well past the limit.
    StillHeating { coil_c: f64, driver_c: f64 },
    /// Cooled to the resume temperatures: the torque comes back.
    Restored { coil_c: f64, driver_c: f64 },
}

/// One joint's thermal state.
#[derive(Clone, Copy, Debug)]
pub struct ThermalGuard {
    cooling: bool,
    warned_still_heating: bool,
    /// 1 = no cap, 0 = `cool_cap_nm`; ramps between the two.
    level: f64,
}

impl Default for ThermalGuard {
    fn default() -> Self {
        Self {
            cooling: false,
            warned_still_heating: false,
            level: 1.0,
        }
    }
}

impl ThermalGuard {
    /// Whether the joint is cooling down (its torque reduced or reducing).
    #[cfg(test)]
    pub fn cooling(&self) -> bool {
        self.cooling
    }

    /// Feed one feedback frame's temperatures (°C).
    pub fn observe(
        &mut self,
        lim: &ThermalLimit,
        coil_c: f64,
        driver_c: f64,
    ) -> Option<ThermalEvent> {
        if !lim.enabled || !coil_c.is_finite() || !driver_c.is_finite() {
            return None;
        }
        if !self.cooling {
            if coil_c >= lim.coil_limit_c || driver_c >= lim.driver_limit_c {
                self.cooling = true;
                self.warned_still_heating = false;
                return Some(ThermalEvent::Cooling { coil_c, driver_c });
            }
            return None;
        }
        if coil_c <= lim.coil_resume_c && driver_c <= lim.driver_resume_c {
            self.cooling = false;
            return Some(ThermalEvent::Restored { coil_c, driver_c });
        }
        if !self.warned_still_heating
            && (coil_c >= lim.coil_limit_c + STILL_HEATING_MARGIN_C
                || driver_c >= lim.driver_limit_c + STILL_HEATING_MARGIN_C)
        {
            self.warned_still_heating = true;
            return Some(ThermalEvent::StillHeating { coil_c, driver_c });
        }
        None
    }

    /// Advance the ramp by `dt` seconds.
    pub fn step(&mut self, lim: &ThermalLimit, dt: f64) {
        let target = if self.cooling && lim.enabled {
            0.0
        } else {
            1.0
        };
        if lim.ramp_s <= 0.0 {
            self.level = target;
            return;
        }
        let rate = dt.max(0.0) / lim.ramp_s;
        self.level = if target < self.level {
            (self.level - rate).max(target)
        } else {
            (self.level + rate).min(target)
        };
    }

    /// The spring-torque cap (Nm) in force: `inf` when there is none.
    pub fn cap(&self, lim: &ThermalLimit) -> f64 {
        if !lim.enabled || self.level >= 1.0 {
            return f64::INFINITY;
        }
        let full = FULL_CAP_NM.max(lim.cool_cap_nm);
        lim.cool_cap_nm + (full - lim.cool_cap_nm) * self.level
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(g: &mut ThermalGuard, lim: &ThermalLimit, seconds: f64) {
        let dt = 1.0 / 240.0;
        for _ in 0..(seconds / dt).round() as usize {
            g.step(lim, dt);
        }
    }

    #[test]
    fn cool_until_resume_then_restore() {
        let lim = ThermalLimit::default();
        let mut g = ThermalGuard::default();
        assert_eq!(g.observe(&lim, 60.0, 50.0), None);
        run(&mut g, &lim, 1.0);
        assert_eq!(g.cap(&lim), f64::INFINITY);

        // Coil reaches its limit: the cap ramps down to the cool cap.
        assert_eq!(
            g.observe(&lim, 85.0, 50.0),
            Some(ThermalEvent::Cooling {
                coil_c: 85.0,
                driver_c: 50.0
            })
        );
        run(&mut g, &lim, 1.0);
        let half = g.cap(&lim);
        assert!(half > lim.cool_cap_nm && half < FULL_CAP_NM, "{half}");
        run(&mut g, &lim, 1.5);
        assert_eq!(g.cap(&lim), lim.cool_cap_nm);

        // Dropping below the limit isn't enough: it waits for the resume
        // temperature.
        assert_eq!(g.observe(&lim, 80.0, 50.0), None);
        run(&mut g, &lim, 5.0);
        assert_eq!(g.cap(&lim), lim.cool_cap_nm);
        assert!(g.cooling());

        assert_eq!(
            g.observe(&lim, 75.0, 50.0),
            Some(ThermalEvent::Restored {
                coil_c: 75.0,
                driver_c: 50.0
            })
        );
        run(&mut g, &lim, 1.0);
        assert!(g.cap(&lim) > lim.cool_cap_nm && g.cap(&lim).is_finite());
        run(&mut g, &lim, 1.5);
        assert_eq!(g.cap(&lim), f64::INFINITY);
    }

    #[test]
    fn the_driver_temperature_counts_too() {
        let lim = ThermalLimit::default();
        let mut g = ThermalGuard::default();
        assert!(matches!(
            g.observe(&lim, 40.0, 105.0),
            Some(ThermalEvent::Cooling { .. })
        ));
        // A cool coil with a hot driver keeps cooling.
        assert_eq!(g.observe(&lim, 40.0, 100.0), None);
        assert!(g.cooling());
        assert!(matches!(
            g.observe(&lim, 40.0, 95.0),
            Some(ThermalEvent::Restored { .. })
        ));
    }

    #[test]
    fn warns_once_if_it_keeps_heating() {
        let lim = ThermalLimit::default();
        let mut g = ThermalGuard::default();
        g.observe(&lim, 85.0, 50.0);
        assert_eq!(g.observe(&lim, 90.0, 50.0), None);
        assert!(matches!(
            g.observe(&lim, 95.0, 50.0),
            Some(ThermalEvent::StillHeating { .. })
        ));
        assert_eq!(g.observe(&lim, 97.0, 50.0), None);
    }

    #[test]
    fn off_never_caps() {
        let lim = ThermalLimit::parse(&["off"]).unwrap();
        let mut g = ThermalGuard::default();
        assert_eq!(g.observe(&lim, 99.0, 119.0), None);
        run(&mut g, &lim, 5.0);
        assert_eq!(g.cap(&lim), f64::INFINITY);
    }

    #[test]
    fn a_restore_mid_ramp_turns_back_from_where_it_is() {
        let lim = ThermalLimit::default();
        let mut g = ThermalGuard::default();
        g.observe(&lim, 85.0, 50.0);
        run(&mut g, &lim, 0.5);
        let low = g.cap(&lim);
        g.observe(&lim, 70.0, 50.0);
        g.step(&lim, 1.0 / 240.0);
        assert!(g.cap(&lim) > low);
    }

    #[test]
    fn parse_is_strict() {
        let lim = ThermalLimit::parse(&["85", "75", "105", "95", "1.5", "2"]).unwrap();
        assert_eq!(lim, ThermalLimit::default());
        assert!(ThermalLimit::parse(&["85", "75", "105", "95", "1.5"]).is_none());
        assert!(ThermalLimit::parse(&["75", "85", "105", "95", "1.5", "2"]).is_none());
        assert!(ThermalLimit::parse(&["85", "75", "105", "95", "0", "2"]).is_none());
        assert!(ThermalLimit::parse(&["85", "75", "105", "95", "nan", "2"]).is_none());
        assert!(ThermalLimit::parse(&["on"]).is_none());
    }

    #[test]
    fn garbage_readings_change_nothing() {
        let lim = ThermalLimit::default();
        let mut g = ThermalGuard::default();
        assert_eq!(g.observe(&lim, f64::NAN, 50.0), None);
        assert!(!g.cooling());
    }
}
