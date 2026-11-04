"""Jelly's lift and wheel positions as a dataset column.

``axol collect-data --record_jelly true`` adds one float32 feature,
``observation.jelly``, next to ``observation.state``:

========================  ==================================================
name                      value
========================  ==================================================
``jelly_lift.pos``        lift height as a fraction of its travel (0 = down)
``jelly_wheel_<n>.pos``   wheel ``n``'s measured angle in rad (ids 1-4, see
                          :mod:`almond_axol.robot.jelly`), continuous across
                          turns and zero at the start of each episode
========================  ==================================================

It is a separate column rather than more entries in ``observation.state`` so
a policy trained on the arm state alone keeps the same input shape, and Run
policy (which has no Jelly readout) can still run it.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

JELLY_FEATURE = "observation.jelly"
LIFT_KEY = "jelly_lift.pos"
WHEEL_KEYS = tuple(f"jelly_wheel_{i}.pos" for i in range(1, 5))
JELLY_KEYS = (LIFT_KEY, *WHEEL_KEYS)

# Longest gap (s) a wheel sample is carried forward with its measured speed.
# Jelly's wheels report every 20 ms; a sample older than this means they
# stopped answering, and guessing further would invent motion.
MAX_EXTRAPOLATION_S = 0.1


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


def zero_episode_wheels(episode_buffer: Any) -> None:
    """Shift a buffered episode's wheel angles so its first row reads zero.

    ``episode_buffer`` is LeRobot's per-episode dict (feature key -> list of
    per-row arrays); rows are edited in place before save. The recorder opens
    an episode a moment after collect-data zeroes the wheels (it waits for
    the cameras), so a base that is already rolling would otherwise start the
    episode a little off zero. No-op for a session without the Jelly column.
    """
    if not isinstance(episode_buffer, dict):
        return
    rows = episode_buffer.get(JELLY_FEATURE)
    if not rows:
        return
    wheels = slice(1, len(JELLY_KEYS))
    origin = np.array(rows[0][wheels], dtype=np.float64)
    for row in rows:
        row[wheels] = (np.asarray(row[wheels], dtype=np.float64) - origin).astype(
            row.dtype
        )


class JellyStateReader:
    """Reads :data:`JELLY_KEYS` from a running :class:`~almond_axol.robot.jelly.Jelly`.

    The lift reports a few times a second and is briefly unknown when a status
    frame is late, so each part holds its last reading. :meth:`wait_ready`
    makes sure there is a first reading before any row is recorded.

    The wheels report at 50 Hz, which beats against a 30 fps recording: a
    frame would see one, two or no new samples. So :meth:`read` carries the
    latest sample forward to the read time with the measured wheel speeds
    (for at most :data:`MAX_EXTRAPOLATION_S`), and each row is the wheels'
    angle when the row was taken.

    The wheel angles are relative to the last :meth:`zero_wheels`. Call it
    before each episode's capture opens, so the values stay small however far
    the base was driven between takes; :func:`zero_episode_wheels` then makes
    the first recorded row exactly zero. The lift stays absolute: its height
    is a physical state of the scene.
    """

    def __init__(self, jelly: Any) -> None:
        self._jelly = jelly
        self._lift: float | None = None
        self._wheels: Any = None  # the latest WheelSample
        self._wheel_origin: tuple[float, ...] = (0.0,) * len(WHEEL_KEYS)

    def _refresh(self) -> None:
        lift = self._jelly.lift_raised
        if lift is not None:
            self._lift = float(lift)
        wheels = self._jelly.wheel_sample
        if wheels is not None:
            self._wheels = wheels

    def _wheel_angles(self, now: float) -> tuple[float, ...]:
        sample = self._wheels
        age = now - sample.at
        if sample.at <= 0 or not 0 <= age <= MAX_EXTRAPOLATION_S:
            age = 0.0
        return tuple(
            p + v * age
            for p, v in zip(sample.positions, sample.velocities, strict=True)
        )

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

    def zero_wheels(self, now: float | None = None) -> None:
        """Make the wheels' current angles read as zero from now on."""
        self._refresh()
        if self._wheels is None:
            raise RuntimeError("Jelly wheels zeroed before their first reading")
        self._wheel_origin = self._wheel_angles(
            time.monotonic() if now is None else now
        )

    def read(self, now: float | None = None) -> dict[str, float]:
        """The values at ``now`` (default: this moment, :func:`time.monotonic`)
        by dataset name. Call :meth:`wait_ready` first."""
        self._refresh()
        if self._lift is None or self._wheels is None:
            raise RuntimeError("Jelly state read before its first reading")
        angles = self._wheel_angles(time.monotonic() if now is None else now)
        wheels = (
            angle - origin
            for angle, origin in zip(angles, self._wheel_origin, strict=True)
        )
        return {LIFT_KEY: self._lift, **dict(zip(WHEEL_KEYS, wheels))}
