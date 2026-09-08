"""
base.calibrate

Measure the cart's per-wheel effective radii — and from them
``CartConfig.wheel_scale`` — with no tape measure, using the overhead ZED's
positional tracking as the ground truth.

Why. Four omni wheels never wear identically. A few percent of radius spread
is invisible from the wheels (each tracks its commanded speed perfectly) but
slides the base sideways by centimetres per metre even with the gyro heading
hold engaged, because the wheels' surface speeds are mutually inconsistent
and the free rollers absorb the difference (see the ``cart`` module
docstring). Encoder-only self-calibration — spinning a wheel free as an
odometer — was simulated and rejected: on an uneven floor the rolling
assumption it rests on is exactly what breaks, so it fits the floor rather
than the wheels. An external pose reference doesn't have that problem, and
the cart already carries one: the stereo ZED X on the overhead mount tracks
its own 6-DoF pose at 60 Hz with sub-millimetre stationary noise.

What it does. The cart performs a fixed set of strokes from wherever it is
standing — spin 90° each way, then drive forward, back, left and right by
``--distance`` — with the heading hold *off* so the wheels' raw kinematics
show. For each stroke it records the wheels' net rotation (motor odometry,
:meth:`Cart.read_wheels`) and the camera's net displacement and heading
change (ZED positional tracking). Inverting the x-drive kinematics, the
body's displacement is a fixed linear map of each wheel's surface travel
``R_i·φ_i``, so the strokes give a linear least-squares problem in the four
radii. Three nuisance quantities are solved alongside so the operator need
not measure anything:

- the camera's offset from the drive centre (it swings on a lever when the
  base turns, so the rotation strokes pin it down);
- the camera's mounting yaw relative to the body (a 1° error would otherwise
  masquerade as a 1.7% lateral slide — the very effect being measured; the
  forward/left stroke pair separates them, since a yaw offset mixes the two
  axes antisymmetrically while a radius error mixes them symmetrically);
- the wheels' rotation lever arm (wheelbase), unless ``--lever`` pins it.

Everything is relative to the camera's *own* heading at each stroke start, so
the mounting pitch (the overhead camera looks 60° down) doesn't matter; only
its roll is assumed small. The fit's per-stroke residuals are reported —
wheel slip during a stroke (an unloaded diagonal on a bad patch of floor)
shows up there, and the run should be repeated on flatter ground if they
exceed a few millimetres.

Output is the ``wheel_scale`` tuple to paste into ``--cart.wheel_scale``
(or the Advanced → Cart panel), the absolute radii as a sanity check, and
optionally (``--save``) the value written straight into the control panel's
saved settings. ``--out FILE`` dumps the raw strokes so the fit can be
recomputed offline with :func:`fit_calibration`.

Usage (on the ZED box; the wheels and the camera must both be free — stop
``axol serve``/teleop first):
    axol diag.base-calibrate
    axol diag.base-calibrate --distance 2 --repeat 2 --save
    uv run -m almond_axol.diagnostics.base.calibrate --serial 51617969 --out strokes.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...robot.cart import (
    _SESSION_PMAX,
    DEFAULT_CHANNEL,
    WHEELS,
    Cart,
    CartConfig,
    stroke_rows,
)

_logger = logging.getLogger(__name__)

# Rate at which strokes re-latch their command (the cart's own loop runs at
# CartConfig.frequency; commands older than command_timeout decay to a stop).
_COMMAND_HZ = 50.0
# Wheel speed below which the base counts as stopped, and how long the camera
# must see it motionless (< _SETTLE_MOVE_M) before a stroke's end is recorded.
_SETTLE_WHEEL_RAD_S = 0.05
_SETTLE_MOVE_M = 0.001
_SETTLE_TIMEOUT_S = 15.0
# A stroke must leave this much headroom to the ±PMAX position mapping, or a
# wheel could wrap mid-stroke and corrupt its odometry (~40 rad is 1.5 m).
_PMAX_HEADROOM_RAD = 80.0
# Fit quality above which the result is flagged as suspect.
_WARN_RESIDUAL_M = 0.004
_WARN_RESIDUAL_RAD = math.radians(0.3)
# Tracking-consistency limits (see consistency_report): spread of the
# camera-motion / wheel-turns ratio within one stroke direction, and how far
# off its axis a translation stroke may point while the heading barely moved.
_WARN_RATIO_SPREAD = 0.03
_WARN_OFF_AXIS_RAD = math.radians(5.0)

# --tracker choices → (sl.POSITIONAL_TRACKING_MODE name, enable_2d_ground_mode)
TRACKERS: dict[str, tuple[str, bool]] = {
    "gen3-2d": ("GEN_3", True),
    "gen3": ("GEN_3", False),
    "gen2": ("GEN_2", False),
    "gen1": ("GEN_1", False),
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Stroke:
    """One stroke's measurements, all relative to the cart's pose at its start.

    ``turns`` is each wheel's net rotation (rad, :data:`WHEELS` order, motor
    convention). ``dx_m``/``dy_m`` is the *camera's* horizontal displacement in
    the camera-start frame (x along the camera's heading, y to its left) and
    ``dtheta_rad`` the heading change (CCW positive), both from positional
    tracking. The body-frame displacement of the drive centre is what the fit
    recovers from these.
    """

    name: str
    turns: list[float]
    dx_m: float
    dy_m: float
    dtheta_rad: float
    duration_s: float = 0.0
    confidence: float = 0.0


@dataclass
class Calibration:
    """Result of :func:`fit_calibration`."""

    radii_m: list[float]
    wheel_scale: list[float]
    camera_offset_m: tuple[float, float]  # drive centre → camera, body frame
    camera_yaw_rad: float  # camera heading relative to body +x, CCW positive
    lever_m: float
    residuals: list[tuple[float, float, float]] = field(default_factory=list)
    rms_translation_m: float = 0.0
    rms_heading_rad: float = 0.0

    @property
    def radius_spread(self) -> float:
        r = np.asarray(self.radii_m)
        return float((r.max() - r.min()) / r.mean())

    @property
    def suspect(self) -> bool:
        return (
            self.rms_translation_m > _WARN_RESIDUAL_M
            or self.rms_heading_rad > _WARN_RESIDUAL_RAD
        )


def wheel_scale_arg(scale: list[float] | tuple[float, ...]) -> str:
    """Render a scale tuple the way ``--cart.wheel_scale`` wants it."""
    return "[" + ",".join(f"{s:.4f}" for s in scale) + "]"


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def arc_rows(
    turns: list[float], dtheta: float
) -> tuple[list[float], list[float], list[float]]:
    """:func:`stroke_rows` with the translation rows mapped from arc to chord.

    A stroke whose heading changes by ``dtheta`` while the body-frame velocity
    keeps its direction traces a circular arc; its chord (the measured
    start→end displacement, in the start frame) is the straight-line
    displacement rotated by ``dtheta/2`` and shortened by ``sinc(dtheta/2)``.
    """
    kx, ky, kw = stroke_rows(turns)
    h = 0.5 * dtheta
    sinc = math.sin(h) / h if abs(h) > 1e-9 else 1.0
    c, s = math.cos(h) * sinc, math.sin(h) * sinc
    kx_c = [c * x - s * y for x, y in zip(kx, ky)]
    ky_c = [s * x + c * y for x, y in zip(kx, ky)]
    return kx_c, ky_c, kw


def fit_calibration(strokes: list[Stroke], lever_m: float | None = None) -> Calibration:
    """Least-squares fit of the wheel radii and the camera's mount from strokes.

    Unknowns are the four effective radii ``R_i``, the camera's offset ``r``
    from the drive centre (body frame), its mounting yaw ``ψ``, and the wheels'
    rotation lever arm ``L`` unless ``lever_m`` gives it. Per stroke, with the
    kinematic rows ``(kx, ky, kw)`` of :func:`~almond_axol.robot.cart.stroke_rows`
    and the measured camera motion ``(d, Δθ)`` expressed in the body-start
    frame (``d_B = Rz(ψ)·d_C``):

        C(Δθ)·[kx; ky]·R + (Rz(Δθ) − I)·r  =  d_B   (drive centre + camera swing)
        kw·R                                =  Δθ·L

    ``C(Δθ) = Rz(Δθ/2)·sinc(Δθ/2)`` is the arc-to-chord map: with the heading
    hold off the base's heading drifts while it translates (that is the radius
    mismatch showing), and since body velocity and yaw rate stay proportional
    through the ramps the path is an exact circular arc, whose chord points
    along the *mean* heading. Without it a 1° drift over 1.5 m would leave a
    13 mm lateral residual and bias the radii.

    ``ψ`` enters through ``d_B``; it is linearized about the current estimate
    and the solve iterated, which converges in a couple of steps for any
    plausible mounting error. Raises ``ValueError`` if the strokes don't
    determine the unknowns (needs turning *and* translating in two directions).
    """
    if lever_m is not None and lever_m <= 0.0:
        raise ValueError("lever_m must be positive")
    n_wheels = len(WHEELS)
    fit_lever = lever_m is None
    n_unknowns = n_wheels + 3 + (1 if fit_lever else 0)
    if len(strokes) * 3 < n_unknowns:
        raise ValueError(
            f"{len(strokes)} strokes cannot determine {n_unknowns} unknowns — "
            "record a rotation plus forward and left strokes at least"
        )

    psi = 0.0
    solution: np.ndarray | None = None
    for _ in range(8):
        rows: list[list[float]] = []
        rhs: list[float] = []
        for s in strokes:
            kx, ky, kw = arc_rows(s.turns, s.dtheta_rad)
            c, sn = math.cos(psi), math.sin(psi)
            dx = c * s.dx_m - sn * s.dy_m
            dy = sn * s.dx_m + c * s.dy_m
            ct, st = math.cos(s.dtheta_rad), math.sin(s.dtheta_rad)
            # Unknown order: R (4), rx, ry, dpsi, [L]
            rows.append([*kx, ct - 1.0, -st, dy] + ([0.0] if fit_lever else []))
            rhs.append(dx)
            rows.append([*ky, st, ct - 1.0, -dx] + ([0.0] if fit_lever else []))
            rhs.append(dy)
            if fit_lever:
                rows.append([*kw, 0.0, 0.0, 0.0, -s.dtheta_rad])
                rhs.append(0.0)
            else:
                rows.append([*kw, 0.0, 0.0, 0.0])
                rhs.append(s.dtheta_rad * lever_m)
        a = np.array(rows)
        b = np.array(rhs)
        if np.linalg.matrix_rank(a) < n_unknowns:
            raise ValueError(
                "the strokes don't determine the wheel radii and camera mount — "
                "the set needs an in-place rotation plus translations in two "
                "directions (forward and left)"
            )
        solution, *_ = np.linalg.lstsq(a, b, rcond=None)
        dpsi = float(solution[n_wheels + 2])
        psi += dpsi
        if abs(dpsi) < 1e-9:
            break
    assert solution is not None

    radii = solution[:n_wheels]
    rx, ry = float(solution[n_wheels]), float(solution[n_wheels + 1])
    lever = float(solution[n_wheels + 3]) if fit_lever else float(lever_m)
    if not np.all(np.isfinite(solution)) or np.any(radii <= 0.0) or lever <= 0.0:
        raise ValueError(
            "the fit produced a non-positive radius or lever arm — check the "
            "wheel signs and that the camera tracked the whole run"
        )

    # Residuals of the converged system (dpsi ≈ 0 so the ψ column contributes
    # nothing): translation in metres, heading in radians.
    pred = a @ solution
    residuals: list[tuple[float, float, float]] = []
    for i in range(len(strokes)):
        ex = float(pred[3 * i] - b[3 * i])
        ey = float(pred[3 * i + 1] - b[3 * i + 1])
        et = float(pred[3 * i + 2] - b[3 * i + 2]) / lever
        residuals.append((ex, ey, et))
    rms_t = math.sqrt(
        sum(ex * ex + ey * ey for ex, ey, _ in residuals) / len(residuals)
    )
    rms_h = math.sqrt(sum(et * et for *_, et in residuals) / len(residuals))

    scales = radii.mean() / radii
    scales /= scales.mean()
    return Calibration(
        radii_m=[float(r) for r in radii],
        wheel_scale=[float(s) for s in scales],
        camera_offset_m=(rx, ry),
        camera_yaw_rad=psi,
        lever_m=lever,
        residuals=residuals,
        rms_translation_m=rms_t,
        rms_heading_rad=rms_h,
    )


# ---------------------------------------------------------------------------
# Camera-frame geometry
# ---------------------------------------------------------------------------


def camera_heading_axes(rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal unit vectors (forward, left) of the camera's heading.

    ``rotation`` is the camera's world←camera matrix in a Z-up, X-forward,
    Y-left frame. The heading is taken from the camera's *left* axis projected
    onto the ground plane: for a camera pitched steeply down that axis stays
    near-horizontal (only a small roll would tilt it), whereas the optical axis
    is mostly vertical. Forward is left rotated 90° clockwise.
    """
    left = np.array([rotation[0, 1], rotation[1, 1]], dtype=float)
    norm = float(np.linalg.norm(left))
    if norm < 0.2:
        raise ValueError("camera left axis is near-vertical — mount roll too large")
    left /= norm
    forward = np.array([left[1], -left[0]])
    return forward, left


def heading_change(r0: np.ndarray, r1: np.ndarray) -> float:
    """Yaw (about world Z, CCW positive) of the rotation taking ``r0`` to ``r1``."""
    rel = r1 @ r0.T
    return math.atan2(rel[1, 0], rel[0, 0])


# ---------------------------------------------------------------------------
# ZED positional tracking
# ---------------------------------------------------------------------------


@dataclass
class PoseSample:
    time: float  # monotonic
    xy: np.ndarray  # horizontal position, world frame (m)
    rotation: np.ndarray  # 3×3 world←camera
    ok: bool
    confidence: float


class ZedTracker:
    """Positional tracking on one stereo ZED, grabbed in a background thread.

    The camera is opened in a Z-up / X-forward / Y-left world frame with
    gravity as the origin, so the reported translation's x–y is the ground
    plane whatever the mount's pitch.

    ``tracker`` picks the SDK's tracking generation: ``gen3`` / ``gen3-2d``
    (visual-inertial SLAM, without or with the 2D ground constraint — which
    never initialized on the cart, whose base rocks; needs no depth) or
    ``gen2`` / ``gen1`` (the depth-based odometers, deprecated in SDK 5.x). ``depth_mode`` is any ``sl.DEPTH_MODE`` name; the
    neural modes trigger a one-off multi-minute model optimization on a
    Jetson. Stationary noise is tiny in every configuration (tens of microns
    on the ZED Box); what differs is how well *motion along the optical axis*
    is tracked, which only a driven comparison shows — see the consistency
    table in the report.
    """

    def __init__(
        self,
        serial: int,
        tracker: str = "gen3",
        depth_mode: str = "NONE",
        resolution: str = "SVGA",
        fps: int = 60,
    ) -> None:
        if tracker not in TRACKERS:
            raise ValueError(f"tracker must be one of {sorted(TRACKERS)}")
        self.serial = serial
        self.tracker = tracker
        self.depth_mode = depth_mode
        self.resolution = resolution
        self.fps = fps
        self._cam: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: PoseSample | None = None
        self._error: BaseException | None = None
        self.frames = 0

    def open(self) -> None:
        import pyzed.sl as sl

        cam = sl.Camera()
        init = sl.InitParameters()
        init.set_from_serial_number(self.serial)
        init.camera_resolution = getattr(sl.RESOLUTION, self.resolution)
        init.camera_fps = self.fps
        init.depth_mode = getattr(sl.DEPTH_MODE, self.depth_mode)
        init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
        init.coordinate_units = sl.UNIT.METER
        err = cam.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise ConnectionError(f"failed to open ZED {self.serial}: {err}")
        generation, ground_2d = TRACKERS[self.tracker]
        track = sl.PositionalTrackingParameters()
        track.enable_area_memory = False  # no relocalization jumps mid-stroke
        track.enable_imu_fusion = True
        track.set_gravity_as_origin = True
        track.enable_pose_smoothing = False
        track.mode = getattr(sl.POSITIONAL_TRACKING_MODE, generation)
        track.enable_2d_ground_mode = ground_2d
        err = cam.enable_positional_tracking(track)
        if err != sl.ERROR_CODE.SUCCESS:
            cam.close()
            raise ConnectionError(f"ZED {self.serial}: positional tracking: {err}")
        self._cam = cam
        self._thread = threading.Thread(
            target=self._run, name="zed-tracker", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        import pyzed.sl as sl

        pose = sl.Pose()
        runtime = sl.RuntimeParameters()
        try:
            while not self._stop.is_set():
                if self._cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    continue
                state = self._cam.get_position(pose, sl.REFERENCE_FRAME.WORLD)
                translation = np.asarray(pose.get_translation().get(), dtype=float)
                rotation = np.array(pose.get_rotation_matrix().r, dtype=float)
                sample = PoseSample(
                    time=time.monotonic(),
                    xy=translation[:2].copy(),
                    rotation=rotation.reshape(3, 3),
                    ok=state == sl.POSITIONAL_TRACKING_STATE.OK,
                    confidence=float(pose.pose_confidence),
                )
                with self._lock:
                    self._latest = sample
                    self.frames += 1
        except BaseException as exc:  # noqa: BLE001 - surfaced to the main loop
            self._error = exc

    def latest(self) -> PoseSample:
        """The most recent pose; raises if tracking has failed or gone quiet."""
        if self._error is not None:
            raise ConnectionError(f"ZED tracking thread died: {self._error!r}")
        with self._lock:
            sample = self._latest
        if sample is None:
            raise ConnectionError("no pose from the ZED yet")
        if time.monotonic() - sample.time > 1.0:
            raise ConnectionError("ZED tracking stalled (no pose for 1 s)")
        return sample

    async def wait_ready(self, timeout: float = 10.0) -> None:
        t0 = time.monotonic()
        while True:
            try:
                sample = self.latest()
                if sample.ok:
                    return
            except ConnectionError:
                if self._error is not None:
                    raise
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("ZED positional tracking did not report OK")
            await asyncio.sleep(0.1)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._cam is not None:
            try:
                self._cam.disable_positional_tracking()
            finally:
                self._cam.close()
            self._cam = None


def pick_serial(requested: int | None) -> int:
    """The ZED to track with: ``requested``, else the configured overhead
    camera if it is stereo, else the only stereo camera connected."""
    from ...zed import list_zed_devices

    try:
        devices = list_zed_devices()
    except ImportError:
        raise SystemExit(
            "pyzed is not installed — run this on the ZED box (axol zed.install)"
        ) from None
    stereo = {d["serial"] for d in devices if d["kind"] == "stereo"}
    if requested is not None:
        if requested not in stereo:
            raise SystemExit(
                f"ZED {requested} is not a connected stereo camera "
                f"(found: {sorted(d['serial'] for d in devices) or 'none'}); "
                "positional tracking needs a stereo ZED X."
            )
        return requested
    try:
        from ...serve.settings import SettingsStore

        cameras = SettingsStore().snapshot().get("cameras") or {}
        overhead = int(str((cameras.get("serials") or {}).get("overhead") or 0))
    except Exception:  # noqa: BLE001 - settings are optional here
        overhead = 0
    if overhead in stereo:
        return overhead
    if len(stereo) == 1:
        return next(iter(stereo))
    raise SystemExit(
        "could not pick a camera: pass --serial (connected stereo ZEDs: "
        f"{sorted(stereo) or 'none'})"
    )


# ---------------------------------------------------------------------------
# Driving the strokes
# ---------------------------------------------------------------------------


class StrokeError(RuntimeError):
    """A stroke could not be completed (tracking lost, timeout)."""


@dataclass
class StrokePlan:
    name: str
    command: tuple[float, float, float]
    target: float  # metres of camera travel, or radians of heading change
    rotation: bool


def make_plan(
    distance_m: float, angle_rad: float, speed: float, turn_speed: float, repeat: int
) -> list[StrokePlan]:
    one = [
        StrokePlan("rotate ccw", (0.0, 0.0, turn_speed), angle_rad, True),
        StrokePlan("rotate cw", (0.0, 0.0, -turn_speed), angle_rad, True),
        StrokePlan("forward", (speed, 0.0, 0.0), distance_m, False),
        StrokePlan("back", (-speed, 0.0, 0.0), distance_m, False),
        StrokePlan("left", (0.0, speed, 0.0), distance_m, False),
        StrokePlan("right", (0.0, -speed, 0.0), distance_m, False),
    ]
    return one * repeat


class Odometer:
    """Accumulates the camera's motion since a stroke began.

    Heading is integrated from successive samples rather than taken as the
    start→end angle, so a spin past 180° (a fast rotation stroke overshooting
    its ramp-down) can't wrap. ``progress`` is what the stroke loop compares
    with its target: heading for spins, straight-line distance otherwise.
    """

    def __init__(self, start: PoseSample, rotation: bool) -> None:
        self.start = start
        self.rotation = rotation
        self.theta = 0.0
        self._last_rotation = start.rotation
        self.latest = start

    def update(self, sample: PoseSample) -> None:
        self.theta += heading_change(self._last_rotation, sample.rotation)
        self._last_rotation = sample.rotation
        self.latest = sample

    @property
    def progress(self) -> float:
        if self.rotation:
            return abs(self.theta)
        return float(np.linalg.norm(self.latest.xy - self.start.xy))


def ramp_stop_time(cmd_norm: float, decel: float, jerk: float) -> float:
    """Seconds the cart's ramp needs to bring a command of ``cmd_norm`` to zero.

    The plain trapezoid takes ``cmd/decel``; with a jerk limit the rate has to
    build up and bleed off again, which adds up to ``decel/jerk`` (the exact
    S-curve is shorter when it never reaches ``decel``; the bound is fine for
    the stop-early estimate).
    """
    t = cmd_norm / decel
    if jerk > 0.0:
        t += decel / jerk
    return t


async def _settle(
    cart: Cart, tracker: ZedTracker, hold_s: float, odometer: Odometer | None = None
) -> PoseSample:
    """Command a stop and wait until the wheels and the camera are still.

    Keeps re-latching the zero command (so the cart's watchdog never has to)
    until the ramp has run out, every wheel reads slower than
    ``_SETTLE_WHEEL_RAD_S``, and the camera has moved less than
    ``_SETTLE_MOVE_M`` over the last ``hold_s`` seconds. Feeds every pose to
    ``odometer`` so the ramp-down's motion is counted. Returns the final pose.
    """
    # Slower than the drive loop: the zero command only has to beat the
    # cart's command_timeout, and the wheel poll is four feedback requests.
    interval = 0.05
    t0 = time.monotonic()
    anchor: PoseSample | None = None
    while True:
        cart.set_command(0.0, 0.0, 0.0)
        sample = tracker.latest()
        if odometer is not None:
            odometer.update(sample)
        ramp_done = all(abs(c) < 1e-3 for c in cart.body_cmd)
        wheels_still = False
        if ramp_done:
            _, velocities = await cart.read_wheels()
            wheels_still = all(abs(v) < _SETTLE_WHEEL_RAD_S for v in velocities)
        if not (ramp_done and wheels_still):
            anchor = None
        elif anchor is None:
            anchor = sample
        elif float(np.linalg.norm(sample.xy - anchor.xy)) > _SETTLE_MOVE_M:
            anchor = sample
        elif sample.time - anchor.time >= hold_s:
            return sample
        if time.monotonic() - t0 > _SETTLE_TIMEOUT_S:
            raise StrokeError("the cart did not come to rest after the stroke")
        await asyncio.sleep(interval)


async def run_stroke(
    cart: Cart,
    tracker: ZedTracker,
    plan: StrokePlan,
    *,
    timeout_s: float,
    pause_s: float,
) -> Stroke:
    """Drive one stroke to its target and return its measurements."""
    start_pose = await _settle(cart, tracker, pause_s)
    if not start_pose.ok:
        raise StrokeError("positional tracking is not OK at the stroke start")
    turns0, _ = await cart.read_wheels()
    if any(abs(p) + _PMAX_HEADROOM_RAD > _SESSION_PMAX for p in turns0):
        raise StrokeError(
            "a wheel's position is too close to the ±PMAX mapping limit for a "
            "stroke — power-cycle the base to reset wheel positions"
        )

    odometer = Odometer(start_pose, plan.rotation)
    interval = 1.0 / _COMMAND_HZ
    decel, jerk = cart.config.decel, cart.config.jerk
    t0 = time.monotonic()
    rate = 0.0  # smoothed progress rate (m/s or rad/s)
    last_progress, last_time = 0.0, t0
    while True:
        cart.set_command(*plan.command)
        sample = tracker.latest()
        if not sample.ok:
            cart.set_command(0.0, 0.0, 0.0)
            raise StrokeError(f"positional tracking lost during '{plan.name}'")
        odometer.update(sample)
        progress = odometer.progress
        now = time.monotonic()
        if now - last_time >= 0.1:
            rate += 0.5 * ((progress - last_progress) / (now - last_time) - rate)
            last_progress, last_time = progress, now
        # Stop early by the distance the ramp-down will still cover, so the
        # stroke lands near its target instead of overshooting by the whole
        # decel ramp (which at a brisk speed can be most of a metre).
        cmd_norm = math.sqrt(sum(c * c for c in cart.body_cmd))
        coast = 0.5 * rate * ramp_stop_time(cmd_norm, decel, jerk)
        if progress + coast >= plan.target:
            break
        if now - t0 > timeout_s:
            cart.set_command(0.0, 0.0, 0.0)
            raise StrokeError(
                f"'{plan.name}' did not reach its target in {timeout_s:.0f} s "
                f"(got {progress:.3f} of {plan.target:.3f})"
            )
        await asyncio.sleep(interval)
    duration = time.monotonic() - t0

    end_pose = await _settle(cart, tracker, pause_s, odometer)
    turns1, _ = await cart.read_wheels()

    forward, left = camera_heading_axes(start_pose.rotation)
    delta = end_pose.xy - start_pose.xy
    return Stroke(
        name=plan.name,
        turns=[float(b - a) for a, b in zip(turns0, turns1)],
        dx_m=float(delta @ forward),
        dy_m=float(delta @ left),
        dtheta_rad=odometer.theta,
        duration_s=duration,
        confidence=min(start_pose.confidence, end_pose.confidence),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _wheel_labels() -> list[str]:
    return ["".join(part[0].upper() for part in w.name.split("_")) for w in WHEELS]


def consistency_report(strokes: list[Stroke]) -> tuple[list[str], bool]:
    """Sanity-check the camera against the wheels, direction by direction.

    Within one stroke direction the wheels always turn in the same pattern,
    so the ratio of camera-measured motion to wheel rotation must come out
    the same every time (heading per radian for spins, metres per radian for
    translations) — whatever the radii are. A spread beyond a few percent, or
    a translation stroke whose measured displacement points well off its axis
    while the heading barely changed, means the tracker lost the plot in that
    direction (a down-looking camera sees little parallax for motion along its
    optical axis), or a wheel slipped. Either way the fit is not to be trusted.
    Returns the table lines and whether anything tripped.
    """
    groups: dict[str, list[Stroke]] = {}
    for s in strokes:
        key = (
            "spin"
            if s.name.startswith("rotate")
            else "forward/back"
            if s.name in ("forward", "back")
            else "left/right"
        )
        groups.setdefault(key, []).append(s)
    lines = ["  tracking consistency (camera motion per wheel radian, by direction):"]
    bad = False
    for key, group in groups.items():
        ratios = []
        off_axis = 0.0
        for s in group:
            turns = float(np.mean(np.abs(s.turns)))
            if turns <= 0.0:
                continue
            if key == "spin":
                ratios.append(abs(s.dtheta_rad) / turns)
            else:
                ratios.append(math.hypot(s.dx_m, s.dy_m) / turns)
                axis = math.atan2(s.dy_m, s.dx_m)
                if key == "left/right":
                    axis -= math.copysign(math.pi / 2, axis)
                axis = (axis + math.pi / 2) % math.pi - math.pi / 2
                if abs(s.dtheta_rad) < math.radians(2.0):
                    off_axis = max(off_axis, abs(axis))
        if not ratios:
            continue
        spread = (max(ratios) - min(ratios)) / (sum(ratios) / len(ratios))
        flag = ""
        if spread > _WARN_RATIO_SPREAD or off_axis > _WARN_OFF_AXIS_RAD:
            bad = True
            flag = "  <-- inconsistent"
        unit = "rad/rad" if key == "spin" else "m/rad"
        axis_txt = (
            "" if key == "spin" else f", worst off-axis {math.degrees(off_axis):.1f}°"
        )
        lines.append(
            f"    {key:<13} {min(ratios):.4f}–{max(ratios):.4f} {unit}  "
            f"spread {spread * 100:.1f}%{axis_txt}{flag}"
        )
    if bad:
        lines.append(
            "    The camera's motion doesn't agree with the wheels in the flagged "
            "direction(s): the tracker is unreliable there (try another --tracker /"
            " --depth-mode, more texture in view) or a wheel slipped. Radii fitted "
            "from these strokes are not meaningful."
        )
    return lines, bad


def format_report(strokes: list[Stroke], cal: Calibration, serial: int) -> str:
    labels = _wheel_labels()
    lines = [f"Wheel calibration — {len(strokes)} strokes, ZED {serial}", ""]
    lines.append(
        "  stroke        wheels (rad)                        camera dx/dy (m)   dθ (°)   resid (mm, mm, °)"
    )
    for s, (ex, ey, et) in zip(strokes, cal.residuals):
        turns = " ".join(f"{t:+7.2f}" for t in s.turns)
        lines.append(
            f"  {s.name:<12} {turns}   {s.dx_m:+.3f} {s.dy_m:+.3f}   "
            f"{math.degrees(s.dtheta_rad):+7.2f}   "
            f"{ex * 1e3:+5.1f} {ey * 1e3:+5.1f} {math.degrees(et):+.2f}"
        )
    lines.append("")
    consistency, inconsistent = consistency_report(strokes)
    lines += consistency
    lines.append("")
    radii = "  ".join(f"{lab} {r * 1e3:.2f}" for lab, r in zip(labels, cal.radii_m))
    scales = "  ".join(f"{lab} {s:.4f}" for lab, s in zip(labels, cal.wheel_scale))
    rx, ry = cal.camera_offset_m
    lines += [
        f"  effective radii (mm)  {radii}   spread {cal.radius_spread * 100:.2f}%",
        f"  wheel_scale           {scales}",
        f"  camera mount          {rx:+.3f} m fwd, {ry:+.3f} m left of the drive "
        f"centre, yaw {math.degrees(cal.camera_yaw_rad):+.2f}°",
        f"  rotation lever (a+b)/√2  {cal.lever_m:.3f} m",
        f"  fit residual          {cal.rms_translation_m * 1e3:.1f} mm rms, "
        f"{math.degrees(cal.rms_heading_rad):.2f}° rms",
    ]
    if cal.suspect or inconsistent:
        lines += [
            "",
            "  WARNING: do not apply this result — "
            + (
                "the camera and wheels disagree (see the consistency table)."
                if inconsistent
                else "residuals are large: the wheels slipped or the camera lost "
                "track during a stroke. Repeat on flatter floor."
            ),
        ]
    lines += [
        "",
        "Apply with:",
        f"  --cart.wheel_scale {wheel_scale_arg(cal.wheel_scale)}",
        "  (control panel: Advanced → Cart → wheel_scale; or rerun with --save)",
    ]
    return "\n".join(lines)


def save_wheel_scale(scale: list[float]) -> Path:
    """Persist the scale into the control panel's saved settings."""
    from ...serve.settings import SETTINGS_PATH, SettingsStore

    SettingsStore().update(advanced={"cart.wheel_scale": wheel_scale_arg(scale)})
    return SETTINGS_PATH


def write_strokes(
    path: Path,
    strokes: list[Stroke],
    serial: int,
    tracking: dict[str, Any] | None = None,
) -> None:
    payload = {
        "zed_serial": serial,
        "tracking": tracking or {},
        "strokes": [asdict(s) for s in strokes],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_strokes(path: Path) -> list[Stroke]:
    raw = json.loads(path.read_text())
    return [Stroke(**s) for s in raw["strokes"]]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _run(args: argparse.Namespace) -> int:
    serial = pick_serial(args.serial)
    plan = make_plan(
        args.distance,
        math.radians(args.angle),
        args.speed,
        args.turn_speed,
        args.repeat,
    )
    print(
        f"Clearance needed: {args.distance + 0.5:.1f} m ahead, behind, left and "
        f"right of the cart, and room to spin {args.angle:.0f}° each way.\n"
        f"Sequence ({len(plan)} strokes): "
        + ", ".join(p.name for p in plan[:6])
        + (f" × {args.repeat}" if args.repeat > 1 else "")
    )
    if not args.yes:
        input("Press Enter to start (Ctrl-C to abort) ")

    tracking = {
        "tracker": args.tracker,
        "depth_mode": args.depth_mode,
        "resolution": args.resolution,
    }
    tracker = ZedTracker(serial, **tracking)
    print(f"Opening ZED {serial} for positional tracking…", flush=True)
    tracker.open()
    strokes: list[Stroke] = []
    cart: Cart | None = None
    try:
        await tracker.wait_ready()
        # Heading hold off and no IMU: the fit wants the wheels' raw
        # kinematics, and the camera supplies the heading. Parking stays on so
        # the base holds still between strokes.
        cart = Cart(
            CartConfig(
                channel=args.channel,
                lift=False,
                imu=False,
                yaw_hold_gain=0.0,
                axis_snap_deg=0.0,
            )
        )
        await cart.enable()
        print("Cart enabled; tracking OK. Driving…", flush=True)
        for i, p in enumerate(plan, 1):
            stroke = await run_stroke(
                cart, tracker, p, timeout_s=args.stroke_timeout, pause_s=args.pause
            )
            strokes.append(stroke)
            print(
                f"  [{i}/{len(plan)}] {p.name:<11} camera {stroke.dx_m:+.3f} "
                f"{stroke.dy_m:+.3f} m  {math.degrees(stroke.dtheta_rad):+6.1f}°  "
                f"wheels " + " ".join(f"{t:+.1f}" for t in stroke.turns),
                flush=True,
            )
    except (StrokeError, ConnectionError, TimeoutError) as exc:
        print(f"\nAborted: {exc}")
        if strokes and args.out:
            write_strokes(Path(args.out), strokes, serial, tracking)
            print(f"Partial strokes written to {args.out}")
        return 1
    finally:
        if cart is not None:
            cart.set_command(0.0, 0.0, 0.0)
            await cart.disable()
        tracker.close()

    if args.out:
        write_strokes(Path(args.out), strokes, serial, tracking)
        print(f"Strokes written to {args.out}")
    print()
    return _report_and_save(strokes, serial, args)


def _report_and_save(
    strokes: list[Stroke], serial: int, args: argparse.Namespace
) -> int:
    cal = fit_calibration(strokes, args.lever)
    print(format_report(strokes, cal, serial))
    if not args.save:
        return 0
    _, inconsistent = consistency_report(strokes)
    if cal.suspect or inconsistent:
        if not sys.stdin.isatty():
            print("Not saving a suspect fit; rerun to retry.")
            return 1
        if input("The fit is suspect. Save anyway? [y/N] ").strip().lower() != "y":
            return 1
    path = save_wheel_scale(cal.wheel_scale)
    print(f"Saved cart.wheel_scale to {path}")
    return 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--channel",
        default=DEFAULT_CHANNEL,
        help=f"SocketCAN interface for the base (default: {DEFAULT_CHANNEL})",
    )
    parser.add_argument(
        "--serial",
        type=int,
        default=None,
        help="Serial of the stereo ZED to track with (default: the configured "
        "overhead camera, or the only stereo ZED connected)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=1.5,
        help="Length of each translation stroke in metres (default: 1.5)",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=90.0,
        help="Size of each rotation stroke in degrees (default: 90)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.25,
        help="Normalized translation command, 0–1 of CartConfig.max_speed "
        "(default: 0.25)",
    )
    parser.add_argument(
        "--turn-speed",
        type=float,
        default=0.2,
        help="Normalized rotation command for the spin strokes (default: 0.2)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the six-stroke sequence this many times (default: 1)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Seconds the base must sit still before a stroke's end is recorded "
        "(default: 1.0)",
    )
    parser.add_argument(
        "--stroke-timeout",
        type=float,
        default=40.0,
        help="Abort if a stroke hasn't reached its target in this many seconds "
        "(default: 40)",
    )
    parser.add_argument(
        "--lever",
        type=float,
        default=None,
        help="Wheel rotation lever arm (a+b)/√2 in metres, if known; otherwise "
        "it is fitted from the rotation strokes",
    )
    parser.add_argument(
        "--tracker",
        choices=sorted(TRACKERS),
        default="gen3",
        help="ZED positional-tracking generation: gen3 / gen3-2d (visual-inertial "
        "SLAM, no depth needed; 2d adds a ground-plane constraint that fails to "
        "initialize if the base rocks), gen2 / gen1 (depth-based odometry, both "
        "deprecated in SDK 5.x). Compare with the report's consistency table "
        "(default: gen3)",
    )
    parser.add_argument(
        "--depth-mode",
        default="NONE",
        help="ZED depth mode (sl.DEPTH_MODE name). gen3 needs none; gen1/gen2 want "
        "PERFORMANCE or a NEURAL mode (first use of a neural mode optimizes its "
        "model for minutes) (default: NONE)",
    )
    parser.add_argument(
        "--resolution",
        default="SVGA",
        help="ZED capture resolution (sl.RESOLUTION name; SVGA, HD1080, HD1200) "
        "(default: SVGA)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write the raw strokes as JSON here (refit later with --refit)",
    )
    parser.add_argument(
        "--refit",
        default=None,
        metavar="FILE",
        help="Skip driving: fit a strokes JSON written earlier with --out",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write the result into the control panel's saved settings "
        "(cart.wheel_scale under Advanced)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Don't ask before driving or saving a suspect fit (implied when "
        "there is no terminal, e.g. from the dashboard)",
    )


def run_cli(args: argparse.Namespace) -> None:
    if args.distance <= 0 or args.angle <= 0 or args.repeat < 1:
        raise SystemExit("--distance and --angle must be positive, --repeat >= 1")
    if not (0.0 < args.speed <= 1.0 and 0.0 < args.turn_speed <= 1.0):
        raise SystemExit("--speed and --turn-speed must be in (0, 1]")
    if not sys.stdin.isatty():
        args.yes = True

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    if args.refit:
        strokes = read_strokes(Path(args.refit))
        serial = json.loads(Path(args.refit).read_text()).get("zed_serial", 0)
        code = _report_and_save(strokes, serial, args)
        if code:
            raise SystemExit(code)
        return

    try:
        code = asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nInterrupted — cart disabled.")
        code = 130
    if code:
        raise SystemExit(code)


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register ``diag.base-calibrate`` for dashboard schema introspection."""
    parser = subparsers.add_parser(
        "diag.base-calibrate",
        help="Calibrate the cart's per-wheel radii with ZED positional tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    _add_arguments(parser)
    parser.set_defaults(func=run_cli)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="axol diag.base-calibrate",
        description="Calibrate the cart's per-wheel radii with ZED positional tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    _add_arguments(parser)
    run_cli(parser.parse_args(argv))


if __name__ == "__main__":
    main()
