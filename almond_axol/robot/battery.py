"""Jelly's battery: state of charge from the pack voltage.

Jelly runs from two LiTime 24 V 50 Ah LiFePO4 packs in parallel — an 8S
LiFePO4 bank, 25.6 V nominal, 100 Ah (~2.56 kWh). Nothing on the robot talks
to the packs' Bluetooth BMS, so the charge is estimated from the one number
the host can read: the 24 V rail as measured by the jelly_legs lift board
(its ``GET_POWER`` telemetry, see :func:`almond_axol.robot.lift.decode_power`).

LiFePO4 holds an almost flat voltage through the middle of its charge (from
40 % to 60 % the pack moves 0.16 V), so the estimate is only as good as the
reading:

- **Rest, not load.** Current sag and charger voltage both move the rail
  far more than a whole band of charge. The curve is the published
  *resting* (open-circuit) one; samples taken while the lift or the wheels
  draw current are flagged ``under_load`` and never displace a resting
  estimate (see :class:`BatteryEstimator`). Jelly is never fully at rest
  (the Jetson and holding arms always draw a little), so on the flat middle
  of the curve the estimate can sit well below the packs' own BMS.
- **Charging reads high.** A connected charger lifts the rail above any
  resting voltage, and the charger's 29.2 V absorption voltage is only
  reached at the very end. Measured on Jelly with this board: 27.5 V
  charging a nearly full pack, 27.0 V for the same pack full and unplugged.
  Readings at or above :data:`CHARGING_VOLTS` report ``charging`` (the
  percentage then reads near full and means nothing); the estimator keeps
  it until the rail drops under :data:`CHARGING_CLEAR_VOLTS` so it does not
  flicker. Early in a charge, a low pack can sit under the threshold and
  read as a resting (too high) percentage: voltage alone cannot tell that
  apart from a full pack at rest.
- **ADC accuracy.** The board divides VM 100k/6.8k into a 12-bit ADC
  referenced to its own 3.3 V rail, so a few hundred millivolts of absolute
  error are possible; treat the percentage as a band, not a gauge.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

# Resting cell voltage -> state of charge for LiFePO4, from EVE's published
# chart (https://www.evemall.eu/selection-guide/eve-lifepo4-state-charge-chart-discharge-curve-capacity-diagrams),
# the standard per-cell table. LiTime's own 24 V chart is coarser (four bands)
# and read a nearly full pack on Jelly as half empty. Linear between points.
_LIFEPO4_CELL_CURVE: tuple[tuple[float, float], ...] = (
    (2.50, 0.0),
    (3.00, 10.0),
    (3.20, 20.0),
    (3.22, 30.0),
    (3.25, 40.0),
    (3.26, 50.0),
    (3.27, 60.0),
    (3.30, 70.0),
    (3.32, 80.0),
    (3.35, 90.0),
    (3.40, 100.0),
)

# Jelly's packs are 8S: eight cells in series.
_CELLS = 8

LIFEPO4_8S_CURVE: tuple[tuple[float, float], ...] = tuple(
    (round(volts * _CELLS, 2), percent) for volts, percent in _LIFEPO4_CELL_CURVE
)

# Two 50 Ah packs in parallel.
CAPACITY_AH = 100.0

# Just above what the board reads for a full pack at rest (27.0 V: the ADC
# reads high and a just-charged pack settles slowly),
# well under a nearly full pack on the charger (27.5 V), both measured on
# Jelly. Kept low to catch as much of a charge as possible; the ADC noise is
# ~10 mV, well inside the margin.
CHARGING_VOLTS = 27.1

# Hysteresis: once charging, the rail must fall below this to clear it. Still
# above the 27.0 V a full resting pack reads.
CHARGING_CLEAR_VOLTS = 27.05

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
    ends the next resting sample replaces it outright. ``charging`` sets at
    :data:`CHARGING_VOLTS` and clears below :data:`CHARGING_CLEAR_VOLTS`. A
    pack that goes absent (VM gone) clears the estimate.
    """

    def __init__(self) -> None:
        self._volts: float | None = None
        self._under_load = False
        self._charging = False

    @property
    def status(self) -> BatteryStatus | None:
        """The current estimate, or ``None`` before any sample / with no pack."""
        if self._volts is None:
            return None
        return BatteryStatus(
            voltage=self._volts,
            percent=battery_percent(self._volts),
            charging=self._charging,
            under_load=self._under_load,
        )

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
        elif self._charging != self._charging_at(volts):
            # A charger connecting or leaving is a step, not noise.
            self._volts = volts
        else:
            self._volts += _SMOOTHING * (volts - self._volts)
        self._charging = self._charging_at(self._volts)
        return self.status

    def _charging_at(self, volts: float) -> bool:
        return volts >= (CHARGING_CLEAR_VOLTS if self._charging else CHARGING_VOLTS)

    def reset(self) -> None:
        self._volts = None
        self._under_load = False
        self._charging = False
