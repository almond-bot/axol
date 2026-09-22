"""Jelly's battery: state of charge from the pack voltage.

Jelly runs from two LiTime 24 V 50 Ah LiFePO4 packs in parallel — an 8S
LiFePO4 bank, 25.6 V nominal, 100 Ah (~2.56 kWh). Nothing on the robot talks
to the packs' Bluetooth BMS, so the charge is estimated from the one number
the host can read: the 24 V rail as measured by the jelly_legs lift board
(its ``GET_POWER`` telemetry, see :func:`almond_axol.robot.lift.decode_power`).

LiFePO4 holds an almost flat voltage above ~25 % charge (the whole span
from 25 % to full is ~0.5 V across the pack), so the estimate is only as good
as the reading:

- **Rest, not load.** Current sag and charger voltage both move the rail
  far more than a whole band of charge. The curve is the *resting*
  (open-circuit) one; samples taken while the lift or the wheels draw
  current are flagged ``under_load`` and never displace a resting estimate
  (see :class:`BatteryEstimator`).
- **Charging reads high.** A connected charger holds the rail near its
  29.2 V absorption voltage, well above any resting voltage; readings above
  :data:`CHARGING_VOLTS` report ``charging`` and clamp to 100 %.
- **ADC accuracy.** The board divides VM 100k/6.8k into a 12-bit ADC
  referenced to its own 3.3 V rail, so a few hundred millivolts of absolute
  error are possible; treat the percentage as a band, not a gauge.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

# Resting pack voltage -> state of charge for LiTime's 24 V LiFePO4 packs,
# from LiTime's own resting-voltage chart (0 %: 20-24 V, 25 %: 26.0-26.3 V,
# 50 %: 26.3-26.4 V, 75 %: 26.6-26.66 V, 100 %: >= 26.66 V): each band's
# midpoint, 0 % at the top of the empty band, 100 % at the full threshold.
# Linear between points. Checked on Jelly against the packs' BMS: 25.69 V
# at rest reads 20 % here, the LiTime app said 18 % (a generic 3.2 V/cell =
# 20 % curve read 26 %).
LIFEPO4_8S_CURVE: tuple[tuple[float, float], ...] = (
    (24.00, 0.0),
    (26.15, 25.0),
    (26.35, 50.0),
    (26.63, 75.0),
    (26.66, 100.0),
)

# Two 50 Ah packs in parallel.
CAPACITY_AH = 100.0

# 3.45 V/cell: above anything a resting LiFePO4 pack settles to, so the rail
# is being held up by a charger.
CHARGING_VOLTS = 27.6

# Below this the lift board reports no motor supply at all (its own
# DRIVER_VM_READY_VOLTS): the board is on USB power and the pack is
# disconnected or switched off, not empty.
PACK_ABSENT_VOLTS = 6.0

# Smoothing weight per resting sample. At the 1 Hz poll this is a ~5 s time
# constant: enough to hide ADC noise, short enough that a real change shows.
_SMOOTHING = 0.2


def battery_percent(
    volts: float, curve: tuple[tuple[float, float], ...] = LIFEPO4_8S_CURVE
) -> float:
    """State of charge (0-100) for a resting pack voltage, clamped to the curve."""
    if not math.isfinite(volts):
        raise ValueError("battery voltage must be a finite number")
    xs = [v for v, _ in curve]
    if volts <= xs[0]:
        return curve[0][1]
    if volts >= xs[-1]:
        return curve[-1][1]
    i = bisect.bisect_right(xs, volts)
    (v0, p0), (v1, p1) = curve[i - 1], curve[i]
    return p0 + (p1 - p0) * (volts - v0) / (v1 - v0)


@dataclass(frozen=True)
class BatteryStatus:
    """One battery estimate."""

    voltage: float  # pack volts (smoothed while resting)
    percent: float  # 0-100 state of charge from the resting curve
    charging: bool  # rail above any resting voltage: a charger is connected
    # The estimate comes from a sample taken while the lift or wheels drew
    # current (no resting sample yet), so it reads low.
    under_load: bool

    @property
    def remaining_ah(self) -> float:
        """Charge left in amp-hours, from :attr:`percent` and the pack capacity."""
        return CAPACITY_AH * self.percent / 100.0


def estimate_battery(volts: float, *, under_load: bool = False) -> BatteryStatus | None:
    """A single-sample estimate, or ``None`` when no pack is connected."""
    if not math.isfinite(volts) or volts < PACK_ABSENT_VOLTS:
        return None
    return BatteryStatus(
        voltage=volts,
        percent=battery_percent(volts),
        charging=volts >= CHARGING_VOLTS,
        under_load=under_load,
    )


class BatteryEstimator:
    """Smooths successive pack-voltage samples into a steady estimate.

    Resting samples are averaged (exponentially). A sample taken under load
    only counts while there is no resting estimate to keep; once the load
    ends the next resting sample replaces it outright. A pack that goes
    absent (VM gone) clears the estimate.
    """

    def __init__(self) -> None:
        self._volts: float | None = None
        self._under_load = False

    @property
    def status(self) -> BatteryStatus | None:
        """The current estimate, or ``None`` before any sample / with no pack."""
        if self._volts is None:
            return None
        return estimate_battery(self._volts, under_load=self._under_load)

    def update(self, volts: float, *, under_load: bool = False) -> BatteryStatus | None:
        """Fold one sample in and return the resulting estimate."""
        if not math.isfinite(volts) or volts < PACK_ABSENT_VOLTS:
            self.reset()
            return None
        if under_load:
            if self._volts is None or self._under_load:
                self._volts = volts
                self._under_load = True
        elif self._volts is None or self._under_load:
            self._volts = volts
            self._under_load = False
        elif volts >= CHARGING_VOLTS or self._volts >= CHARGING_VOLTS:
            # A charger connecting or leaving is a step, not noise.
            self._volts = volts
        else:
            self._volts += _SMOOTHING * (volts - self._volts)
        return self.status

    def reset(self) -> None:
        self._volts = None
        self._under_load = False
