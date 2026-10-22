"""Reference motions: joint trajectories replayed by ``axol tune.motion``.

A reference motion is a uniform-rate, both-arm joint trajectory stored as a
small ``.npz`` in :data:`MOTIONS_DIR` (``~/.almond/motions``, on the robot),
so the same motion can be replayed on that robot again and again and the
tracking metrics compared 1:1. Record one, or generate one (``axol
motion.chirp``, ``scripts/creep_motions.py``).

A few motions are built in (:data:`BUILTIN_MOTIONS`): generated in code, the
same on every robot, and always available by name without a file —
``slow_osc`` (the slow-motion acceptance test), ``fast_swing`` (large fast
swings, to check a change adds no buzz), each with a ``_left`` mirror, and
``hold``. A file of the same name in :data:`MOTIONS_DIR` takes precedence.

Motions are *built* from recorded sessions (``axol motion.build PREFIX
--name N`` postprocesses the flight-recorder capture), from either source:

* **Teleop** (``axol teleop --teleop.record PREFIX``): the final guarded
  command stream (``_cmd`` stage, ``out`` field), clipped to the engaged
  span.
* **Gravity comp** (``axol gravity-comp --record PREFIX``): the measured
  arm-joint positions of the hand-guided session (``_gc`` stage, ``qm``
  field); there is no engage state, so the still lead-in/lead-out is
  trimmed instead.

Either way the stream is resampled onto a uniform grid, zero-phase low-pass
smoothed (removing hand tremor and network jitter — the *intent* is what we
want to replay), and finally projected waypoint-by-waypoint through the same
collision-aware solver the teleop return-to-rest uses, so the stored motion
is joint-limit- and self-collision-safe by construction.

File format (``<name>.npz``):

* ``q``:    ``(N, 14) float32`` joint-frame targets — 7 left + 7 right arm
            joints in ``ARM_JOINTS`` order (no grippers).
* ``rate``: scalar float — playback rate in Hz (uniform).
* ``meta``: 0-d unicode array holding a JSON object (provenance: source
            recording, build parameters, build date, notes).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MOTIONS_DIR = Path.home() / ".almond" / "motions"

#: The start pose the generated motions (chirps, creeps) hold every other
#: joint at, in degrees, ARM_JOINTS order, 7 left then 7 right: shoulder_1
#: and the elbow raised and wrist_3 turned, the arms mirrored. It is the
#: first sample of the jelly robot's old ``slow_osc`` recording.
START_POSE_DEG = (
    *(-12.709, 0.0, 0.0, 25.943, 0.0, 0.0, -13.113),  # left
    *(12.709, 0.0, 0.0, -25.943, 0.0, 0.0, 13.113),  # right
)


def start_pose() -> np.ndarray:
    """:data:`START_POSE_DEG` in radians, shape ``(14,)``."""
    return np.radians(np.asarray(START_POSE_DEG, dtype=float))


# Width of a motion row: 7 left + 7 right arm joints (ARM_JOINTS order).
MOTION_WIDTH = 14


@dataclass
class ReferenceMotion:
    """One reference motion (see module docstring for the format)."""

    name: str
    rate: float
    q: np.ndarray  # (N, 14) float32, joint frame
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.q) / self.rate

    def times(self) -> np.ndarray:
        return np.arange(len(self.q)) / self.rate

    def peak_velocity(self) -> np.ndarray:
        """Per-joint peak |velocity| (rad/s), shape ``(14,)``."""
        if len(self.q) < 2:
            return np.zeros(MOTION_WIDTH)
        return np.max(np.abs(np.diff(self.q, axis=0)) * self.rate, axis=0)


def save_motion(motion: ReferenceMotion, path: Path | None = None) -> Path:
    """Write a motion to ``path`` (default: :data:`MOTIONS_DIR`)."""
    if path is None:
        path = MOTIONS_DIR / f"{motion.name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        q=np.asarray(motion.q, dtype=np.float32),
        rate=float(motion.rate),
        meta=json.dumps(motion.meta),
    )
    return path


def load_motion(name_or_path: str) -> ReferenceMotion:
    """Load a motion by path, by name in :data:`MOTIONS_DIR`, or a built-in."""
    path = Path(name_or_path)
    if not path.is_file():
        path = MOTIONS_DIR / f"{name_or_path}.npz"
    if not path.is_file() and name_or_path in BUILTIN_MOTIONS:
        return BUILTIN_MOTIONS[name_or_path]()
    if not path.is_file():
        known = ", ".join(m.name for m in list_motions()) or f"(none in {MOTIONS_DIR})"
        raise FileNotFoundError(
            f"No reference motion {name_or_path!r}. Known motions: {known}"
        )
    with np.load(path) as data:
        meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
        return ReferenceMotion(
            name=path.stem,
            rate=float(data["rate"]),
            q=np.asarray(data["q"], dtype=np.float32),
            meta=meta,
        )


def list_motions() -> list[ReferenceMotion]:
    """Every motion in :data:`MOTIONS_DIR` and the built-ins, alphabetically.

    ``q`` is loaded too (files are small); use this for listings and pickers.
    """
    files = sorted(MOTIONS_DIR.glob("*.npz")) if MOTIONS_DIR.is_dir() else []
    motions = {p.stem: load_motion(str(p)) for p in files}
    for name, make in BUILTIN_MOTIONS.items():
        motions.setdefault(name, make())
    return [motions[n] for n in sorted(motions)]


# --------------------------------------------------------------------- #
# Built-in motions: generated, identical on every robot                  #
# --------------------------------------------------------------------- #

_BUILTIN_RATE = 240.0
#: Peak speed of the largest joint move between two ``fast_swing`` poses.
_FAST_PEAK_DPS = 140.0
#: Right-arm poses ``fast_swing`` moves through (deg, ARM_JOINTS order),
#: taken from poses the old teleop recordings (``wirst_swing``,
#: ``shoulder_1_no_load``) passed through: the arm raised out to the side,
#: folded across the front, the forearm rolled both ways, and shoulder_1
#: swung far back. The inward folds stop at shoulder_3 +40° (the recordings
#: reached +76°, which brings the forearm and wrist ~25 mm closer to the base
#: than at rest in the collision model); the whole motion stays outside the
#: collision solver's activation shell, as ``tests/test_builtin_motions.py``
#: checks.
_FAST_SWING_POSES_DEG = (
    (-20.0, 78.0, -21.0, -15.0, -5.0, 6.0, 13.0),
    (-35.0, 77.0, -49.0, -72.0, -5.0, 6.0, 13.0),
    (-5.0, 22.0, -7.0, -117.0, 0.0, 7.0, 13.0),
    (-25.0, 8.0, 40.0, -108.0, 0.0, 26.0, 13.0),
    (-15.0, 29.0, 7.0, -132.0, 0.0, 16.0, 13.0),
    (-14.0, 69.0, -88.0, -55.0, 10.0, 15.0, 13.0),
    (11.0, 54.0, -101.0, -39.0, 17.0, 7.0, 13.0),
    (3.5, 14.0, -15.0, -110.0, 3.5, 22.0, 12.0),
    (-17.0, 2.0, 40.0, -95.0, 12.0, 16.0, 12.0),
    (0.0, 31.0, -13.0, -65.0, 4.0, -1.0, 10.0),
    (-28.0, 76.0, -35.0, -11.0, -7.0, 2.0, 13.0),
    (-61.0, 84.0, -46.0, -14.0, -14.0, 1.0, 13.0),
    (-93.0, 87.0, -63.0, -31.0, -31.0, 4.0, 13.0),
    (-80.0, 81.0, 9.0, -11.0, 10.0, 4.0, 13.0),
    (-63.0, 70.0, 2.0, -5.0, -5.0, 4.0, 13.0),
)


def _min_jerk(a: np.ndarray, b: np.ndarray, seconds: float) -> np.ndarray:
    """Rows from ``a`` to ``b`` (excluding ``b``), at rest at both ends."""
    s = np.arange(max(1, round(seconds * _BUILTIN_RATE))) / (seconds * _BUILTIN_RATE)
    s = 10 * s**3 - 15 * s**4 + 6 * s**5
    return a + (b - a) * s[:, None]


def _still(pose: np.ndarray, seconds: float) -> np.ndarray:
    return np.tile(pose, (round(seconds * _BUILTIN_RATE), 1))


def _right_arm(rows: np.ndarray) -> np.ndarray:
    """Right-arm rows (N, 7) with the left arm still at the start pose."""
    q = np.tile(start_pose(), (len(rows), 1))
    q[:, 7:] = rows
    return q


def _mirror(motion: ReferenceMotion) -> ReferenceMotion:
    """The motion onto the other arm: the arms swapped, every joint negated."""
    q = -np.concatenate([motion.q[:, 7:], motion.q[:, :7]], axis=1)
    meta = dict(
        motion.meta, source=motion.meta["source"] + ", mirrored to the left arm"
    )
    return ReferenceMotion(
        f"{motion.name}_left", motion.rate, q.astype(np.float32), meta
    )


def _builtin(name: str, q: np.ndarray, source: str) -> ReferenceMotion:
    meta = {"source": f"built-in: {source}", "builtin": True}
    return ReferenceMotion(name, _BUILTIN_RATE, q.astype(np.float32), meta)


def slow_osc_motion() -> ReferenceMotion:
    """``slow_osc``: the right arm's slow-motion acceptance test.

    Modelled on the jelly robot's recorded slow teleop sweep: from the start
    pose the arm bends up in front over 7 s (elbow ~-112°, wrist_3 rolled),
    then sweeps shoulder_1 slowly back and forth between +2° and -38° — two
    14 s periods, shoulder_1 at most ~9°/s — with the elbow opening 1.2° per
    degree as the arm swings out (~11°/s) and shoulder_3 / wrist_1 / wrist_2
    following a little: the slow motion an operator feels the 1-3 Hz sway
    in. Then it returns to the start pose over 7 s.
    """
    period, amp, centre = 14.0, 20.0, -18.0
    t = np.arange(round(2 * period * _BUILTIN_RATE)) / _BUILTIN_RATE
    s1 = centre + amp * np.cos(2 * np.pi * t / period)
    right = np.stack(
        [
            s1,
            np.full_like(s1, 20.0),
            -13.0 - 0.25 * s1,
            -110.0 - 1.2 * s1,
            -12.5 - 0.27 * s1,
            1.0 - 0.2 * s1,
            np.full_like(s1, 31.0),
        ],
        axis=1,
    )
    sweep = np.radians(right)
    rest = start_pose()[7:]
    rows = np.concatenate(
        [
            _still(rest, 1.0),
            _min_jerk(rest, sweep[0], 7.0),
            sweep,
            _min_jerk(sweep[0], rest, 7.0),
            _still(rest, 1.0),
        ]
    )
    return _builtin(
        "slow_osc",
        _right_arm(rows),
        "right shoulder_1 swept slowly between +2 and -38 deg (14 s period) "
        "with the elbow coupled, the arm bent up in front",
    )


def fast_swing_motion() -> ReferenceMotion:
    """``fast_swing``: large fast right-arm swings, for checking that a change
    adds no buzz on fast motion.

    Minimum-jerk moves through :data:`_FAST_SWING_POSES_DEG`, each timed so
    its largest joint move peaks at :data:`_FAST_PEAK_DPS`, stopping at every
    pose, from and back to the start pose.
    """
    rest = start_pose()[7:]
    poses = [rest, *np.radians(_FAST_SWING_POSES_DEG), rest]
    rows = [_still(rest, 1.0)]
    for a, b in zip(poses, poses[1:]):
        span = math.degrees(float(np.max(np.abs(b - a))))
        rows.append(_min_jerk(a, b, max(0.6, 1.875 * span / _FAST_PEAK_DPS)))
    rows.append(_still(rest, 1.0))
    return _builtin(
        "fast_swing",
        _right_arm(np.concatenate(rows)),
        f"right arm swung through {len(_FAST_SWING_POSES_DEG)} poses, "
        f"peaks near {_FAST_PEAK_DPS:.0f} deg/s",
    )


def hold_motion() -> ReferenceMotion:
    """``hold``: both arms still at the start pose for 40 s — the noise
    floor under the running controller (parked buzz, limit cycles)."""
    return _builtin("hold", _still(start_pose(), 40.0), "40 s hold at the start pose")


#: Built-in motions by name: generated on demand, no file needed.
BUILTIN_MOTIONS = {
    "slow_osc": slow_osc_motion,
    "slow_osc_left": lambda: _mirror(slow_osc_motion()),
    "fast_swing": fast_swing_motion,
    "fast_swing_left": lambda: _mirror(fast_swing_motion()),
    "hold": hold_motion,
}


# --------------------------------------------------------------------- #
# Build pipeline: flight-recorder capture -> reference motion            #
# --------------------------------------------------------------------- #


def _engaged_window(ik: dict[str, np.ndarray]) -> tuple[float, float] | None:
    """Time span of the longest fully-engaged stretch in the ik capture.

    Same convention as ``axol diag.teleop-jitter``: a tick counts as engaged
    when either arm's engage flag is set.
    """
    engaged = ik["engaged"].max(axis=1) > 0.5
    if not engaged.any():
        return None
    edges = np.diff(engaged.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if engaged[0]:
        starts.insert(0, 0)
    if engaged[-1]:
        ends.append(len(engaged))
    runs = sorted(zip(starts, ends), key=lambda se: se[1] - se[0])
    s, e = runs[-1]
    return float(ik["t"][s]), float(ik["t"][e - 1])


def _trim_still_ends(
    t: np.ndarray, q: np.ndarray, vel_threshold: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """Drop the still lead-in/lead-out (|v| below ``vel_threshold`` rad/s)."""
    if len(t) < 3:
        return t, q
    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    speed = np.nanmax(np.abs(np.diff(q, axis=0)) / dt[:, None], axis=1)
    moving = np.where(speed > vel_threshold)[0]
    if len(moving) == 0:
        return t, q
    s, e = int(moving[0]), int(moving[-1]) + 2
    return t[s:e], q[s:e]


def _zero_phase_lowpass(x: np.ndarray, rate: float, cutoff_hz: float) -> np.ndarray:
    """Forward-backward one-pole low-pass per column: zero phase, -40 dB/dec.

    Two passes of a one-pole filter (forward then reversed) cancel the phase
    lag exactly and square the magnitude response, so the stored motion is
    smoothed without being time-shifted relative to the operator's intent.
    """
    alpha = 1.0 / (1.0 + rate / (2.0 * math.pi * cutoff_hz))

    def one_pole(y: np.ndarray) -> np.ndarray:
        out = np.empty_like(y)
        acc = y[0].copy()
        for i in range(len(y)):
            acc = acc + alpha * (y[i] - acc)
            out[i] = acc
        return out

    return one_pole(one_pole(x)[::-1])[::-1]


def build_motion(
    prefix: str,
    name: str,
    *,
    rate: float = 100.0,
    smooth_cutoff_hz: float = 6.0,
    time_scale: float = 1.0,
    collision_project: bool = True,
    notes: str = "",
) -> tuple[ReferenceMotion, dict[str, np.ndarray]]:
    """Postprocess a flight-recorder capture into a reference motion.

    Args:
        prefix:            The recording prefix. A teleop capture
                           (``--teleop.record``) reads ``<prefix>_cmd.npz``
                           (the guarded command stream) and
                           ``<prefix>_ik.npz`` (optional, for the
                           engaged-span clip); a gravity-comp capture
                           (``--record``) reads ``<prefix>_gc.npz`` (the
                           hand-guided measured joints). ``_cmd`` wins when
                           both exist.
        name:              Motion name (file stem in the motions directory).
        rate:              Uniform playback rate (Hz).
        smooth_cutoff_hz:  Zero-phase low-pass cutoff. ~6 Hz keeps deliberate
                           motion and drops tremor/jitter.
        time_scale:        Stretch factor for playback time (2.0 = half
                           speed). Applied before the velocity report.
        collision_project: Project every waypoint through the collision-aware
                           solver (requires the kinematics stack; slow on
                           first call due to JIT).
        notes:             Free-form provenance stored in the metadata.

    Returns:
        ``(motion, raw)`` — the built motion, plus the clipped raw command
        stream it was built from (``{"t": (N,), "q": (N, 14)}``, rebased to
        the motion's timeline including ``time_scale``) so the caller can
        chart the before/after of the smoothing + projection passes.
    """
    cmd_path = Path(f"{prefix}_cmd.npz")
    gc_path = Path(f"{prefix}_gc.npz")
    if cmd_path.is_file():
        source_kind = "teleop"
        cmd = dict(np.load(cmd_path))
        t = np.asarray(cmd["t"], dtype=float)
        q = np.asarray(cmd["out"], dtype=float)  # final guarded command, (N, 14)
    elif gc_path.is_file():
        source_kind = "gravity-comp"
        gc = dict(np.load(gc_path))
        t = np.asarray(gc["t"], dtype=float)
        q = np.asarray(gc["qm"], dtype=float)  # hand-guided measured, (N, 14)
        print("  gravity-comp capture (hand-guided measured joints)")
    else:
        raise FileNotFoundError(
            f"neither {cmd_path} nor {gc_path} found — record a session with "
            "`axol teleop --teleop.record <prefix>` or "
            "`axol gravity-comp --record <prefix>` first."
        )
    if q.shape[1] != MOTION_WIDTH:
        raise ValueError(f"expected {MOTION_WIDTH}-wide command rows, got {q.shape}")
    if np.isnan(q).any():
        raise ValueError(
            "capture has gaps (NaN rows) — an arm was disabled or its "
            "telemetry never came up during the recording. Motions carry "
            "both arms; record with both connected."
        )

    ik_path = Path(f"{prefix}_ik.npz")
    span = None
    if ik_path.is_file():
        span = _engaged_window(dict(np.load(ik_path)))
    if span is not None:
        m = (t >= span[0]) & (t <= span[1])
        if m.sum() >= 3:
            t, q = t[m], q[m]
        print(f"  engaged span: {span[1] - span[0]:.1f} s")
    elif source_kind == "teleop":
        print("  no engaged span found — trimming still lead-in/out instead")
    t, q = _trim_still_ends(t, q)
    if len(t) < 3:
        raise ValueError("capture has no motion after clipping")

    # Uniform resample (per joint) at the target rate, with optional
    # time stretching.
    duration = (t[-1] - t[0]) * time_scale
    n = max(int(round(duration * rate)) + 1, 2)
    grid = np.linspace(t[0], t[-1], n)
    q_u = np.stack([np.interp(grid, t, q[:, i]) for i in range(q.shape[1])], axis=1)

    q_s = _zero_phase_lowpass(q_u, rate, smooth_cutoff_hz)

    if collision_project:
        print(f"  projecting {len(q_s)} waypoints through the collision solver ...")
        q_s = _project_waypoints(q_s, rate)

    # Hard-clamp to the joint limits motion_control enforces at replay. The
    # collision projection's limit term is a soft cost traded against staying
    # on the recorded waypoint (and the smoothing after it can re-cross the
    # boundary), and gravity-comp captures record *measured* positions, which
    # read slightly past a calibrated end stop when the joint rests against
    # it (zeroing tolerance + flex). Left unclamped, motion_control silently
    # clips those samples at replay and every tuning run scores the clip as
    # phantom tracking error on that joint.
    lo, hi = _motion_limits()
    past = np.count_nonzero((q_s < lo) | (q_s > hi))
    if past:
        worst = float(np.max(np.maximum(lo - q_s, q_s - hi)))
        q_s = np.clip(q_s, lo, hi)
        print(
            f"  clamped {past} past-limit samples to the joint limits "
            f"(worst excursion {math.degrees(worst):.2f}°)"
        )

    motion = ReferenceMotion(
        name=name,
        rate=rate,
        q=q_s.astype(np.float32),
        meta={
            "source": str(prefix),
            "source_kind": source_kind,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "smooth_cutoff_hz": smooth_cutoff_hz,
            "time_scale": time_scale,
            "collision_projected": collision_project,
            "engaged_span_s": round(duration, 2),
            "notes": notes,
        },
    )
    peak = motion.peak_velocity()
    print(
        f"  built {motion.name}: {len(motion.q)} waypoints, "
        f"{motion.duration:.1f} s at {rate:.0f} Hz, "
        f"peak joint velocity {peak.max():.2f} rad/s"
    )
    if peak.max() > 3.0:
        print(
            "  ! peak velocity is high — consider --time-scale to slow the "
            "playback down"
        )
    raw = {"t": (t - t[0]) * time_scale, "q": q.astype(np.float32)}
    return motion, raw


def _motion_limits() -> tuple[np.ndarray, np.ndarray]:
    """Per-column joint limits of a motion row, shape ``(14,)`` each.

    The same :func:`~almond_axol.robot.axol.arm_limits` bounds that
    ``motion_control`` clips commands to at replay — left 7 then right 7 in
    ``ARM_JOINTS`` order. Imported lazily: the robot stack is not needed to
    load or replay an already-built motion.
    """
    from ..constants import ARM_JOINTS
    from ..robot.axol import arm_limits

    lo = np.empty(MOTION_WIDTH)
    hi = np.empty(MOTION_WIDTH)
    for offset, is_left in ((0, True), (7, False)):
        for i, j in enumerate(ARM_JOINTS):
            lo[offset + i], hi[offset + i] = arm_limits(j, is_left)
    return lo, hi


def _project_waypoints(q: np.ndarray, rate: float) -> np.ndarray:
    """Project each waypoint onto the joint-limit/self-collision manifold.

    One :func:`~almond_axol.teleop.trajectory.solve_path_step` per waypoint,
    *seeded at that waypoint*: when the limit/collision costs are inactive
    there the solve is a no-op and the trajectory passes through untouched;
    where they activate, the waypoint slides off the obstacle. Seeding from
    the previous *projected* waypoint (the return-to-rest planner's pattern)
    is wrong here — the solver's cost-tolerance termination ignores the tiny
    per-waypoint deltas of an already-smooth motion, sticks for a stretch,
    then snaps, injecting velocity spikes into a motion meant to be a clean
    reference. A final zero-phase smoothing pass irons out any curvature the
    projection itself introduced. Imports the kinematics stack lazily (jax
    JIT on first call).
    """
    import jax.numpy as jnp

    from ..kinematics.config import KinematicsConfig
    from ..kinematics.model import collision_cost_params
    from ..kinematics.solver import KinematicsSolver
    from ..teleop.trajectory import solve_path_step

    # Projection is an explicit request (--project), so it always uses the
    # collision model regardless of the teleop default.
    solver = KinematicsSolver(KinematicsConfig(self_collision=True))
    assert solver.robot_coll is not None
    starts, widths = collision_cost_params(solver.robot, solver.robot_coll, 0.025)
    starts_jax, widths_jax = jnp.asarray(starts), jnp.asarray(widths)

    # Motion rows are left7 + right7 (ARM_JOINTS order); marshal through the
    # solver's full-N vector in case it carries extra joints.
    out = np.empty_like(q)
    q_full = np.zeros(solver.num_joints, dtype=np.float32)
    for i in range(len(q)):
        q_full[solver.left_indices] = q[i, :7]
        q_full[solver.right_indices] = q[i, 7:]
        q_pyroki = jnp.asarray(solver.to_pyroki_order(q_full))
        result = solve_path_step(
            solver.robot,
            solver.robot_coll,
            q_pyroki,
            q_pyroki,
            50.0,  # rest_weight — pull toward the recorded waypoint
            100.0,  # limit_weight
            starts_jax,
            widths_jax,
            100.0,  # collision_weight
            10,
        )
        projected = solver.from_pyroki_order(np.asarray(result, dtype=np.float32))
        out[i, :7] = projected[solver.left_indices]
        out[i, 7:] = projected[solver.right_indices]
    moved = float(np.max(np.abs(out - q)))
    print(f"  projection moved waypoints by at most {math.degrees(moved):.2f}°")
    if moved > 1e-4:
        # Smooth the projection's own kinks; a light pass barely re-enters
        # the collision margin (the cost activates well before contact).
        out = _zero_phase_lowpass(out, rate, 6.0)
    return out
