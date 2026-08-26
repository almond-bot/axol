"""Reference motions: committed joint trajectories, identical on every robot.

A reference motion is a uniform-rate, both-arm joint trajectory stored as a
small ``.npz`` in :data:`MOTIONS_DIR` (inside the package, committed to git),
so the exact same motion can be replayed on any robot, today or years from
now, and the tracking metrics compared 1:1.

Motions are *built* from recorded teleop sessions (``axol teleop
--teleop.record PREFIX`` writes the flight-recorder capture;
``axol motion.build PREFIX --name N`` postprocesses it): the final guarded
command stream (``_cmd`` stage, ``out`` field) is clipped to the engaged
span, resampled onto a uniform grid, zero-phase low-pass smoothed (removing
hand tremor and network jitter — the *intent* is what we want to replay),
and finally projected waypoint-by-waypoint through the same collision-aware
solver the teleop return-to-rest uses, so the stored motion is joint-limit-
and self-collision-safe by construction.

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

MOTIONS_DIR = Path(__file__).parent / "motions"

# Width of a motion row: 7 left + 7 right arm joints (ARM_JOINTS order).
MOTION_WIDTH = 14


@dataclass
class ReferenceMotion:
    """One committed reference motion (see module docstring for the format)."""

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
    """Write a motion to ``path`` (default: the committed motions directory)."""
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
    """Load a motion by committed name (``list_motions()``) or filesystem path."""
    path = Path(name_or_path)
    if not path.is_file():
        path = MOTIONS_DIR / f"{name_or_path}.npz"
    if not path.is_file():
        known = ", ".join(m.name for m in list_motions()) or "(none committed)"
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
    """All committed reference motions, alphabetically, without their arrays.

    ``q`` is loaded too (files are small); use this for listings and pickers.
    """
    if not MOTIONS_DIR.is_dir():
        return []
    return [load_motion(str(p)) for p in sorted(MOTIONS_DIR.glob("*.npz"))]


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
        prefix:            The ``--teleop.record`` prefix; reads
                           ``<prefix>_cmd.npz`` (required, the guarded command
                           stream) and ``<prefix>_ik.npz`` (optional, for the
                           engaged-span clip).
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
    if not cmd_path.is_file():
        raise FileNotFoundError(
            f"{cmd_path} not found — record a session with "
            "`axol teleop --teleop.record <prefix>` first."
        )
    cmd = dict(np.load(cmd_path))
    t = np.asarray(cmd["t"], dtype=float)
    q = np.asarray(cmd["out"], dtype=float)  # final guarded command, (N, 14)
    if q.shape[1] != MOTION_WIDTH:
        raise ValueError(f"expected {MOTION_WIDTH}-wide command rows, got {q.shape}")

    ik_path = Path(f"{prefix}_ik.npz")
    span = None
    if ik_path.is_file():
        span = _engaged_window(dict(np.load(ik_path)))
    if span is not None:
        m = (t >= span[0]) & (t <= span[1])
        if m.sum() >= 3:
            t, q = t[m], q[m]
        print(f"  engaged span: {span[1] - span[0]:.1f} s")
    else:
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

    motion = ReferenceMotion(
        name=name,
        rate=rate,
        q=q_s.astype(np.float32),
        meta={
            "source": str(prefix),
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

    from ..kinematics.model import collision_cost_params
    from ..kinematics.solver import KinematicsSolver
    from ..teleop.trajectory import solve_path_step

    solver = KinematicsSolver()
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
