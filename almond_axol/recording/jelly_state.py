"""Jelly's lift and wheel positions as a dataset column.

``axol collect-data --record_jelly true`` adds one float32 feature,
``observation.jelly``, next to ``observation.state``:

========================  ==================================================
name                      value
========================  ==================================================
``jelly_lift.pos``        lift height as a fraction of its travel (0 = down)
``jelly_wheel_<n>.pos``   wheel ``n``'s measured angle in rad (ids 1-4, see
                          :mod:`almond_axol.robot.jelly`), continuous across
                          turns and zero where the session started
========================  ==================================================

It is a separate column rather than more entries in ``observation.state`` so
a policy trained on the arm state alone keeps the same input shape, and Run
policy (which has no Jelly readout) can still run it.
"""

from __future__ import annotations

import time
from typing import Any

JELLY_FEATURE = "observation.jelly"
LIFT_KEY = "jelly_lift.pos"
WHEEL_KEYS = tuple(f"jelly_wheel_{i}.pos" for i in range(1, 5))
JELLY_KEYS = (LIFT_KEY, *WHEEL_KEYS)


def jelly_features() -> dict[str, dict[str, Any]]:
    """The dataset feature the Jelly readout fills."""
    return {
        JELLY_FEATURE: {
            "dtype": "float32",
            "shape": (len(JELLY_KEYS),),
            "names": list(JELLY_KEYS),
        }
    }


def jelly_config_problem(jelly_cfg: Any) -> str | None:
    """Why a session with this (detected) Jelly config can't record Jelly.

    ``jelly_cfg`` is :func:`~almond_axol.robot.jelly.detect_jelly`'s result:
    ``None`` with no Jelly attached, else a config with the absent or
    switched-off parts cleared.
    """
    if jelly_cfg is None:
        return (
            "Record Jelly is on, but no Jelly is attached. Connect Jelly, or turn "
            "off Record Jelly."
        )
    missing = []
    if jelly_cfg.channel is None:
        missing.append("wheels")
    if not jelly_cfg.lift:
        missing.append("lift")
    if missing:
        parts = " and ".join(missing)
        return (
            f"Record Jelly is on, but Jelly's {parts} {'is' if missing == ['lift'] else 'are'} "
            f"off or not attached. Turn on Jelly {parts} and check the cable, or "
            "turn off Record Jelly."
        )
    return None


class JellyStateReader:
    """Reads :data:`JELLY_KEYS` from a running :class:`~almond_axol.robot.jelly.Jelly`.

    The lift reports a few times a second and is briefly unknown when a status
    frame is late, so each part holds its last reading. :meth:`wait_ready`
    makes sure there is a first reading before any row is recorded.
    """

    def __init__(self, jelly: Any) -> None:
        self._jelly = jelly
        self._lift: float | None = None
        self._wheels: tuple[float, ...] | None = None

    def _refresh(self) -> None:
        lift = self._jelly.lift_raised
        if lift is not None:
            self._lift = float(lift)
        wheels = self._jelly.wheel_positions
        if wheels is not None and len(wheels) == len(WHEEL_KEYS):
            self._wheels = tuple(float(v) for v in wheels)

    def missing(self) -> list[str]:
        """The parts with no reading yet (``"lift"``, ``"wheels"``)."""
        self._refresh()
        out = []
        if self._lift is None:
            out.append("lift")
        if self._wheels is None:
            out.append("wheels")
        return out

    def wait_ready(self, timeout_s: float = 3.0, poll_s: float = 0.05) -> None:
        """Block until both parts have reported, or raise ``RuntimeError``."""
        deadline = time.monotonic() + timeout_s
        while True:
            missing = self.missing()
            if not missing:
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(poll_s)
        problems = []
        if "lift" in missing:
            problems.append("the lift isn't reporting its height (run Home the lift)")
        if "wheels" in missing:
            problems.append(
                "the wheels aren't reporting their positions (update Axol, then "
                "check the wheel cable)"
            )
        raise RuntimeError(
            "Record Jelly is on, but "
            + " and ".join(problems)
            + ". Fix this and start again, or turn off Record Jelly."
        )

    def read(self) -> dict[str, float]:
        """The latest values by dataset name. Call :meth:`wait_ready` first."""
        self._refresh()
        if self._lift is None or self._wheels is None:
            raise RuntimeError("Jelly state read before its first reading")
        return {LIFT_KEY: self._lift, **dict(zip(WHEEL_KEYS, self._wheels))}
