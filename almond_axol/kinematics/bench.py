"""Offline IK backend benchmark for the teleop bake-off.

Replays scripted (or captured) ``VRFrame`` streams through the full
:class:`almond_axol.teleop.worker.IKWorker` step path — filters, engage snap,
target conditioning, and the selected IK backend — and scores each backend on
the failure modes reported from real teleop sessions:

- ``figure8``       nominal pick-place-speed motion (baseline tracking/smoothness)
- ``fast-jitter``   fast multi-frequency hand motion + tracking noise (jitter)
- ``singularity``   sweep through full arm extension and back (lockups)
- ``bounds``        targets far past reach and wrists rolled past limits (bounds)
- ``shelf``         high wrist + raised elbow, the elbow-up shelf/box pose (swivel)
- ``torso-graze``   hands sweeping close across the torso (collision freezes)

Metrics per backend per scenario: EE tracking RMSE / max error, joint jerk
RMS, >3 Hz spectral power fraction of the joint velocities, freeze events
(unchanged q while the target keeps moving), joint-limit and collision-margin
violations, achieved elbow rise, and solve-time p50/p99.

Usage::

    python -m almond_axol.kinematics.bench                       # all backends
    python -m almond_axol.kinematics.bench --backends pink-qp
    python -m almond_axol.kinematics.bench --scenarios shelf --json out.json
    python -m almond_axol.kinematics.bench --replay session.jsonl

Synthetic scenarios are defined as world-frame EE/elbow displacement
trajectories relative to the rest pose and converted *exactly* into VR
controller streams by inverting the worker's relative-target mapping, so what
the bench commands is what a headset would have commanded.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..teleop.config import VRTeleopConfig
from ..teleop.worker import IKWorker
from ..vr.models import VRFrame, VRPose, VRPosition, VRQuaternion
from .base import CANONICAL_JOINT_NAMES, FKFrames, frame_body_names
from .config import KinematicsConfig
from .mujoco_model import arm_torso_geom_pairs, canonical_qpos_indices, load_mj_model

_logger = logging.getLogger(__name__)

# Base VR controller positions at engage (headset frame: X=Down, Y=Left,
# Z=Forward). Arbitrary — the worker's mapping is relative to the engage snap.
_L_CTRL0 = np.array([0.0, 0.2, 0.3])
_R_CTRL0 = np.array([0.0, -0.2, 0.3])
# Elbows hang below and behind the wrists at engage.
_L_ELBOW0 = _L_CTRL0 + np.array([0.25, 0.0, -0.2])
_R_ELBOW0 = _R_CTRL0 + np.array([0.25, 0.0, -0.2])

# Axis-permutation matrices of the worker's coordinate plumbing, used to
# invert it exactly:
# - ``_D`` maps VR axes (X=Down, Y=Left, Z=Forward) to FLU: the worker's
#   ``_vr_to_flu_np`` rotation conversion is ``A_flu = D @ m @ D.T``.
# - ``_C`` relates the FLU-converted controller rotation ``A`` (relative to
#   its engage snap) to the world-frame rotation delta applied to the EE
#   target: ``R_delta = C @ A @ C`` (C is symmetric and involutive).
# A commanded world rotation delta therefore requires the VR quaternion of
# ``m = D.T @ (C @ R_delta @ C) @ D``.
_C = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
_D = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

_FREEZE_EPS_RAD = 1e-6
_FREEZE_TARGET_DRIFT_M = 0.005
_HF_CUTOFF_HZ = 3.0


@dataclass
class ArmCmd:
    """World-frame command for one arm at time ``t``, relative to rest FK.

    Attributes:
        dpos:   ``(3,)`` EE displacement (m) from the rest EE position.
        drot:   Optional ``(3,3)`` EE rotation delta (about the rest EE
                orientation); identity when ``None``.
        delbow: ``(3,)`` elbow displacement (m) from the rest elbow position.
    """

    dpos: np.ndarray
    drot: np.ndarray | None = None
    delbow: np.ndarray = field(default_factory=lambda: np.zeros(3))


ScenarioFn = Callable[[float], tuple[ArmCmd, ArmCmd]]
"""t (seconds since engage) -> (left command, right command)."""


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _mirror(dpos: np.ndarray) -> np.ndarray:
    """Mirror a left-arm displacement for the right arm (lateral axis flips)."""
    return np.array([-dpos[0], dpos[1], dpos[2]])


def _scn_figure8(t: float) -> tuple[ArmCmd, ArmCmd]:
    """Nominal pick-place: 0.4 Hz figure-eight, 20 cm wide, elbows following.

    Centred 10 cm *above* rest — the rest pose is close to full downward
    extension, so motion below it is unreachable by construction.
    """
    w = 2 * math.pi * 0.4
    d = np.array(
        [
            0.10 * math.sin(w * t),
            0.08 * math.sin(2 * w * t),
            0.10 - 0.10 * math.cos(w * t),
        ]
    )
    left = ArmCmd(dpos=d, delbow=0.5 * d)
    return left, ArmCmd(dpos=_mirror(d), delbow=0.5 * _mirror(d))


def _tracking_noise(t: float, phase: float) -> np.ndarray:
    """Deterministic ~1.5 mm pseudo-noise (same stream for every backend)."""
    return 0.0015 * np.array(
        [
            math.sin(2 * math.pi * 8.7 * t + phase),
            math.sin(2 * math.pi * 11.3 * t + 2.1 + phase),
            math.sin(2 * math.pi * 9.9 * t + 4.2 + phase),
        ]
    )


def _scn_fast_jitter(t: float) -> tuple[ArmCmd, ArmCmd]:
    """Fast multi-frequency hand motion plus millimetre tracking noise."""
    d = np.array(
        [
            0.12 * math.sin(2 * math.pi * 1.1 * t)
            + 0.05 * math.sin(2 * math.pi * 2.3 * t + 1.0),
            0.10 * math.sin(2 * math.pi * 0.9 * t + 0.5)
            + 0.04 * math.sin(2 * math.pi * 2.7 * t),
            0.10 * math.sin(2 * math.pi * 1.4 * t + 2.0)
            + 0.04 * math.sin(2 * math.pi * 2.1 * t + 0.7),
        ]
    )
    left = ArmCmd(dpos=d + _tracking_noise(t, 0.0), delbow=0.5 * d)
    d_r = _mirror(d) + _tracking_noise(t, 1.0)
    return left, ArmCmd(dpos=d_r, delbow=0.5 * _mirror(d))


def _scn_singularity(t: float) -> tuple[ArmCmd, ArmCmd]:
    """Sweep to full extension (straight-arm singularity) and back, twice.

    The commanded EE runs 45 cm outward-down-forward from rest — past full
    extension — holds, and returns. Solvers without damping lock up or spin
    the wrist at the straight-arm boundary.
    """
    s = 0.45 * 0.5 * (1 - math.cos(2 * math.pi * t / 5.0))  # 0->0.45->0 per 5 s
    d = s * np.array([0.4, 0.5, -0.75])
    left = ArmCmd(dpos=d, delbow=0.6 * d)
    return left, ArmCmd(dpos=_mirror(d), delbow=0.6 * _mirror(d))


def _scn_bounds(t: float) -> tuple[ArmCmd, ArmCmd]:
    """Push far past reach while rolling the wrists past their joint limits."""
    s = 0.5 * (1 - math.cos(2 * math.pi * t / 6.0))
    d = s * np.array([0.2, 0.7, 0.4])  # ~0.85 m past rest at peak
    roll = s * 2.4  # past the ±1.57 wrist limits
    drot = Rotation.from_rotvec([0.0, 0.0, roll]).as_matrix()
    left = ArmCmd(dpos=d, drot=drot, delbow=0.5 * d)
    right = ArmCmd(
        dpos=_mirror(d),
        drot=Rotation.from_rotvec([0.0, 0.0, -roll]).as_matrix(),
        delbow=0.5 * _mirror(d),
    )
    return left, right


def _scn_shelf(t: float) -> tuple[ArmCmd, ArmCmd]:
    """Shelf/box pose: wrists high in front, elbows raised to shoulder level.

    Ramps up over 2 s, holds with a small placing motion. The elbow command
    rises ~45 cm — an elbow-up swivel the raw-position elbow tracking could
    never reach. Scores the achieved elbow rise.
    """
    ramp = min(t / 2.0, 1.0)
    ramp = ramp * ramp * (3 - 2 * ramp)  # smoothstep
    place = 0.03 * math.sin(2 * math.pi * 0.5 * max(t - 2.0, 0.0))
    d = ramp * np.array([0.05, 0.30, 0.55]) + np.array([0.0, place, 0.0])
    delbow = ramp * np.array([0.10, 0.15, 0.45])
    left = ArmCmd(dpos=d, delbow=delbow)
    return left, ArmCmd(dpos=_mirror(d), delbow=_mirror(delbow))


def _scn_torso_graze(t: float) -> tuple[ArmCmd, ArmCmd]:
    """Sweep the hands inward toward the torso, grazing the collision margin."""
    s = math.sin(2 * math.pi * t / 4.0)
    # Inward (toward the torso mid-plane, stopping short of the far arm's
    # half) and slightly forward/up.
    d = np.array([-0.18 * max(s, 0.0), 0.10, 0.15 * max(s, 0.0)])
    left = ArmCmd(dpos=d, delbow=0.5 * d)
    return left, ArmCmd(dpos=_mirror(d), delbow=0.5 * _mirror(d))


SCENARIOS: dict[str, tuple[ScenarioFn, float]] = {
    "figure8": (_scn_figure8, 8.0),
    "fast-jitter": (_scn_fast_jitter, 8.0),
    "singularity": (_scn_singularity, 10.0),
    "bounds": (_scn_bounds, 6.0),
    "shelf": (_scn_shelf, 8.0),
    "torso-graze": (_scn_torso_graze, 8.0),
}


# ---------------------------------------------------------------------------
# VRFrame synthesis (exact inverse of the worker's relative-target mapping)
# ---------------------------------------------------------------------------


def _vr_pose(pos: np.ndarray, quat_xyzw: np.ndarray) -> VRPose:
    return VRPose(
        position=VRPosition(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
        quaternion=VRQuaternion(
            x=float(quat_xyzw[0]),
            y=float(quat_xyzw[1]),
            z=float(quat_xyzw[2]),
            w=float(quat_xyzw[3]),
        ),
    )


class FrameSynth:
    """Convert world-frame arm commands into the VR stream a headset would send.

    The worker maps controller deltas through the engage-snapshot FK frame
    (see ``_relative_target_np``); this class inverts that mapping so a
    commanded world displacement/rotation lands exactly on the intended EE
    target. Rest FK orientation columns come from the backend under test at
    the engage pose, which the worker also uses — so the inversion is exact.
    """

    def __init__(self, rest_frames: FKFrames, position_multiplier: float) -> None:
        self._rot_fk = {
            "left": np.asarray(rest_frames.left_ee[1], dtype=np.float64),
            "right": np.asarray(rest_frames.right_ee[1], dtype=np.float64),
        }
        self._mult = position_multiplier

    def _ctrl_delta(self, side: str, dpos_world: np.ndarray) -> np.ndarray:
        """VR-frame controller displacement producing ``dpos_world`` at the EE."""
        R = self._rot_fk[side]
        w = np.asarray(dpos_world, dtype=np.float64) / self._mult
        # Worker: new_t = fk0*d[2] - fk1*d[1] + fk2*d[0] with d the FLU
        # controller delta (orthonormal columns => project), and the FLU
        # delta of a VR delta (dx, dy, dz) is (dz, dy, -dx).
        d_flu = np.array([w @ R[:, 2], -(w @ R[:, 1]), w @ R[:, 0]])
        return np.array([-d_flu[2], d_flu[1], d_flu[0]])

    def _ctrl_rot(self, side: str, drot_world: np.ndarray | None) -> np.ndarray:
        """Controller quaternion (xyzw) producing ``drot_world`` about the snap."""
        if drot_world is None:
            return np.array([0.0, 0.0, 0.0, 1.0])
        A = _C @ np.asarray(drot_world, dtype=np.float64) @ _C
        m = _D.T @ A @ _D
        return Rotation.from_matrix(m).as_quat()

    @staticmethod
    def _elbow_delta(dpos_world: np.ndarray) -> np.ndarray:
        """VR-frame elbow displacement for a world displacement (FLU inverse)."""
        f = np.asarray(dpos_world, dtype=np.float64)
        return np.array([-f[2], f[1], f[0]])

    def frame(self, left: ArmCmd, right: ArmCmd, lock: bool = True) -> VRFrame:
        lq = self._ctrl_rot("left", left.drot)
        rq = self._ctrl_rot("right", right.drot)
        return VRFrame(
            l_ee=_vr_pose(_L_CTRL0 + self._ctrl_delta("left", left.dpos), lq),
            r_ee=_vr_pose(_R_CTRL0 + self._ctrl_delta("right", right.dpos), rq),
            l_elbow=VRPosition(
                x=float(_L_ELBOW0[0] + self._elbow_delta(left.delbow)[0]),
                y=float(_L_ELBOW0[1] + self._elbow_delta(left.delbow)[1]),
                z=float(_L_ELBOW0[2] + self._elbow_delta(left.delbow)[2]),
            ),
            r_elbow=VRPosition(
                x=float(_R_ELBOW0[0] + self._elbow_delta(right.delbow)[0]),
                y=float(_R_ELBOW0[1] + self._elbow_delta(right.delbow)[1]),
                z=float(_R_ELBOW0[2] + self._elbow_delta(right.delbow)[2]),
            ),
            l_lock=lock,
            r_lock=lock,
        )


# ---------------------------------------------------------------------------
# Reference model for metrics (backend-independent)
# ---------------------------------------------------------------------------


class MetricModel:
    """Backend-independent FK + limit/collision measurement on MuJoCo."""

    def __init__(self) -> None:
        model = load_mj_model(with_meshes=True)
        self._model = model
        self._data = mujoco.MjData(model)
        self._qidx = canonical_qpos_indices(model)
        names = frame_body_names()
        self._bids = {
            k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, v)
            for k, v in names.items()
        }
        jnt_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in CANONICAL_JOINT_NAMES
        ]
        self.lower = model.jnt_range[jnt_ids, 0].copy()
        self.upper = model.jnt_range[jnt_ids, 1].copy()
        # Measurement pair set: every arm<->torso pair that is separable at
        # home (buffer 0) — a superset of any backend's constraint pairs.
        self._pairs = [
            (p[0][0], p[1][0])
            for p in arm_torso_geom_pairs(model, margin=0.0, threshold=0.0)
        ]

    def _set(self, q: np.ndarray) -> None:
        self._data.qpos[:] = 0.0
        self._data.qpos[self._qidx] = np.asarray(q, dtype=np.float64)
        mujoco.mj_kinematics(self._model, self._data)

    def ee_and_elbow(
        self, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(left EE pos, right EE pos, left elbow pos, right elbow pos)."""
        self._set(q)
        return (
            self._data.xpos[self._bids["left_ee"]].copy(),
            self._data.xpos[self._bids["right_ee"]].copy(),
            self._data.xpos[self._bids["left_elbow"]].copy(),
            self._data.xpos[self._bids["right_elbow"]].copy(),
        )

    def min_clearance(self, q: np.ndarray) -> float:
        """Minimum arm<->torso separation (m); negative when penetrating."""
        self._set(q)
        fromto = np.empty(6)
        best = np.inf
        for g, t in self._pairs:
            d = mujoco.mj_geomDistance(self._model, self._data, g, t, 0.2, fromto)
            best = min(best, d)
        return float(best)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Metrics for one backend on one scenario."""

    backend: str
    scenario: str
    rmse_mm: float
    max_err_mm: float
    jerk_rms: float
    hf_frac: float
    freezes: int
    limit_viol_ticks: int
    min_clearance_mm: float
    penetration_ticks: int
    elbow_rise_cm: float
    solve_ms_p50: float
    solve_ms_p99: float

    def row(self) -> list[str]:
        return [
            self.backend,
            f"{self.rmse_mm:.1f}",
            f"{self.max_err_mm:.0f}",
            f"{self.jerk_rms:.0f}",
            f"{self.hf_frac * 100:.1f}%",
            str(self.freezes),
            str(self.limit_viol_ticks),
            f"{self.min_clearance_mm:.0f}",
            str(self.penetration_ticks),
            f"{self.elbow_rise_cm:.1f}",
            f"{self.solve_ms_p50:.2f}",
            f"{self.solve_ms_p99:.2f}",
        ]


_HEADER = [
    "backend",
    "rmse_mm",
    "max_mm",
    "jerk",
    "hf>3Hz",
    "freezes",
    "lim_viol",
    "clear_mm",
    "penetr",
    "elbow_cm",
    "p50_ms",
    "p99_ms",
]


def _make_worker(backend: str, frequency: float) -> tuple[IKWorker, VRTeleopConfig]:
    cfg = VRTeleopConfig(frequency=frequency)
    kin = KinematicsConfig(backend=backend)
    return IKWorker(cfg, kin), cfg


def _metrics_from_run(
    backend: str,
    scenario: str,
    dt: float,
    qs: np.ndarray,
    targets_l: np.ndarray,
    targets_r: np.ndarray,
    solve_ms: np.ndarray,
    metric_model: MetricModel,
    rest_elbow_z: float,
) -> RunResult:
    """Score one recorded run (T ticks of q, targets, and solve times)."""
    T = qs.shape[0]
    ee_l = np.zeros((T, 3))
    ee_r = np.zeros((T, 3))
    elb_l = np.zeros((T, 3))
    clearance = np.full(T, np.inf)
    for i in range(T):
        pl, pr, pel, _ = metric_model.ee_and_elbow(qs[i])
        ee_l[i], ee_r[i], elb_l[i] = pl, pr, pel
        if i % 2 == 0:
            clearance[i] = metric_model.min_clearance(qs[i])

    err = np.concatenate(
        [
            np.linalg.norm(ee_l - targets_l, axis=1),
            np.linalg.norm(ee_r - targets_r, axis=1),
        ]
    )
    rmse_mm = float(np.sqrt(np.mean(err**2)) * 1e3)
    max_err_mm = float(np.max(err) * 1e3)

    dq = np.diff(qs, axis=0) / dt
    jerk = np.diff(qs, n=3, axis=0) / dt**3
    jerk_rms = float(np.sqrt(np.mean(jerk**2)))

    # Fraction of joint-velocity spectral power above the cutoff.
    spec = np.abs(np.fft.rfft(dq, axis=0)) ** 2
    freqs = np.fft.rfftfreq(dq.shape[0], d=dt)
    total = float(np.sum(spec))
    hf_frac = float(np.sum(spec[freqs > _HF_CUTOFF_HZ])) / total if total > 0 else 0.0

    # Freeze events: runs of bit-identical q during which the commanded target
    # kept moving by more than the threshold.
    freezes = 0
    run_start: int | None = None
    for i in range(1, T):
        if np.max(np.abs(qs[i] - qs[i - 1])) < _FREEZE_EPS_RAD:
            if run_start is None:
                run_start = i - 1
        elif run_start is not None:
            drift = max(
                float(np.linalg.norm(targets_l[i - 1] - targets_l[run_start])),
                float(np.linalg.norm(targets_r[i - 1] - targets_r[run_start])),
            )
            if drift > _FREEZE_TARGET_DRIFT_M:
                freezes += 1
            run_start = None
    if run_start is not None:
        drift = max(
            float(np.linalg.norm(targets_l[-1] - targets_l[run_start])),
            float(np.linalg.norm(targets_r[-1] - targets_r[run_start])),
        )
        if drift > _FREEZE_TARGET_DRIFT_M:
            freezes += 1

    limit_viol = int(
        np.sum(
            np.any(
                (qs < metric_model.lower - 1e-4) | (qs > metric_model.upper + 1e-4),
                axis=1,
            )
        )
    )

    measured = clearance[np.isfinite(clearance)]
    min_clearance_mm = float(np.min(measured) * 1e3) if measured.size else float("nan")
    penetration_ticks = int(np.sum(measured < 0.0))

    elbow_rise_cm = float((np.max(elb_l[:, 2]) - rest_elbow_z) * 1e2)

    return RunResult(
        backend=backend,
        scenario=scenario,
        rmse_mm=rmse_mm,
        max_err_mm=max_err_mm,
        jerk_rms=jerk_rms,
        hf_frac=hf_frac,
        freezes=freezes,
        limit_viol_ticks=limit_viol,
        min_clearance_mm=min_clearance_mm,
        penetration_ticks=penetration_ticks,
        elbow_rise_cm=elbow_rise_cm,
        solve_ms_p50=float(np.percentile(solve_ms, 50)),
        solve_ms_p99=float(np.percentile(solve_ms, 99)),
    )


def run_scenario(
    backend: str,
    scenario: str,
    metric_model: MetricModel,
    frequency: float = 120.0,
    worker: IKWorker | None = None,
) -> RunResult:
    """Run one synthetic scenario through the full worker path for one backend."""
    fn, duration = SCENARIOS[scenario]
    dt = 1.0 / frequency
    if worker is None:
        worker, _ = _make_worker(backend, frequency)

    q = worker.get_rest_q()
    rest_frames = worker._backend.fk_frames(q)
    synth = FrameSynth(rest_frames, worker._config.position_multiplier)
    rest_l = np.asarray(rest_frames.left_ee[0], dtype=np.float64)
    rest_r = np.asarray(rest_frames.right_ee[0], dtype=np.float64)
    rest_elbow_z = float(rest_frames.left_elbow[2])

    # Engage at the scenario's t=0 command so there is no initial jump.
    l0, r0 = fn(0.0)
    q = worker.step(synth.frame(l0, r0), q)

    n_ticks = int(duration * frequency)
    qs = np.zeros((n_ticks, worker._backend.num_joints), dtype=np.float32)
    targets_l = np.zeros((n_ticks, 3))
    targets_r = np.zeros((n_ticks, 3))
    solve_ms = np.zeros(n_ticks)
    for i in range(n_ticks):
        t = (i + 1) * dt
        left, right = fn(t)
        frame = synth.frame(left, right)
        t0 = time.perf_counter()
        q = worker.step(frame, q)
        solve_ms[i] = (time.perf_counter() - t0) * 1e3
        qs[i] = q
        targets_l[i] = rest_l + left.dpos
        targets_r[i] = rest_r + right.dpos

    worker.reset()
    return _metrics_from_run(
        backend,
        scenario,
        dt,
        qs,
        targets_l,
        targets_r,
        solve_ms,
        metric_model,
        rest_elbow_z,
    )


def _latch_engage(frames: list[VRFrame]) -> list[VRFrame]:
    """Rewrite a raw capture's lock flags with the live engage-toggle state.

    A capture stores the *raw* grip-button levels, which are only held for the
    ~0.2 s of the engage press. Live, ``VRTeleopCore.update_engage`` latches
    those presses into a toggle (both grips together → enable, either alone →
    disable) and rewrites the frame locks before the worker sees them; without
    mirroring that here the worker ignores everything after the press and a
    replay of a real session is a no-op.
    """
    out: list[VRFrame] = []
    enabled = prev_both = prev_either = False
    for frame in frames:
        both = frame.l_lock and frame.r_lock
        either = frame.l_lock or frame.r_lock
        if not enabled:
            if both and not prev_both:
                enabled = True
        elif either and not prev_either:
            enabled = False
        prev_both = both
        prev_either = either
        out.append(frame.model_copy(update={"l_lock": enabled, "r_lock": enabled}))
    return out


def run_replay(
    backend: str,
    frames: list[VRFrame],
    metric_model: MetricModel,
    frequency: float = 120.0,
) -> RunResult:
    """Replay a captured VRFrame session through one backend.

    A capture has no ground-truth EE targets, so the tracking-error columns
    are scored against the achieved EE positions (i.e. reported as ~0);
    jerk, freeze, limit, collision, and solve-time metrics are fully valid.
    """
    worker, _ = _make_worker(backend, frequency)
    q = worker.get_rest_q()
    qs_list: list[np.ndarray] = []
    solve_list: list[float] = []
    for frame in _latch_engage(frames):
        t0 = time.perf_counter()
        q = worker.step(frame, q)
        solve_list.append((time.perf_counter() - t0) * 1e3)
        qs_list.append(q.copy())
    qs = np.asarray(qs_list, dtype=np.float32)
    T = qs.shape[0]
    # No ground-truth targets in a capture: reuse achieved EE so rmse == 0.
    ee_l = np.zeros((T, 3))
    ee_r = np.zeros((T, 3))
    for i in range(T):
        pl, pr, _, _ = metric_model.ee_and_elbow(qs[i])
        ee_l[i], ee_r[i] = pl, pr
    rest_frames = worker._backend.fk_frames(worker.get_rest_q())
    return _metrics_from_run(
        backend,
        "replay",
        1.0 / frequency,
        qs,
        ee_l,
        ee_r,
        np.asarray(solve_list),
        metric_model,
        float(rest_frames.left_elbow[2]),
    )


def load_capture(path: str) -> list[VRFrame]:
    """Load a JSONL VRFrame capture (see ``VRServerConfig.capture_path``)."""
    frames: list[VRFrame] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj.pop("recv_t", None)
            frames.append(VRFrame.model_validate(obj))
    return frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_table(scenario: str, results: Iterable[RunResult]) -> None:
    rows = [_HEADER] + [r.row() for r in results]
    widths = [max(len(row[c]) for row in rows) for c in range(len(_HEADER))]
    print(f"\n=== {scenario} ===")
    for i, row in enumerate(rows):
        print("  ".join(cell.ljust(widths[c]) for c, cell in enumerate(row)))
        if i == 0:
            print("  ".join("-" * widths[c] for c in range(len(_HEADER))))


def main(argv: list[str] | None = None) -> None:
    from .backends import BACKEND_NAMES

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--backends",
        default=",".join(BACKEND_NAMES),
        help="Comma-separated backend names (default: all).",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help="Comma-separated scenario names (default: all).",
    )
    parser.add_argument("--frequency", type=float, default=120.0)
    parser.add_argument("--json", default=None, help="Write results JSON here.")
    parser.add_argument(
        "--replay", default=None, help="Replay a captured VRFrame JSONL session."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, force=True)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for s in scenarios:
        if s not in SCENARIOS:
            parser.error(f"unknown scenario {s!r}; choose from {list(SCENARIOS)}")

    metric_model = MetricModel()
    all_results: list[RunResult] = []

    if args.replay is not None:
        frames = load_capture(args.replay)
        print(f"Replaying {len(frames)} captured frames.")
        results = []
        for backend in backends:
            try:
                results.append(
                    run_replay(backend, frames, metric_model, args.frequency)
                )
            except Exception as exc:  # noqa: BLE001 - missing optional dep
                print(f"[skip] {backend}: {exc}")
        _print_table("replay", results)
        all_results.extend(results)
    else:
        # One worker per backend, reused across scenarios (worker.reset()
        # between runs) — backend construction (JIT compile / model load) is
        # by far the most expensive part.
        by_scenario: dict[str, list[RunResult]] = {s: [] for s in scenarios}
        for backend in backends:
            try:
                worker, _ = _make_worker(backend, args.frequency)
            except Exception as exc:  # noqa: BLE001 - missing optional dep
                print(f"[skip] {backend}: {exc}")
                continue
            for scenario in scenarios:
                r = run_scenario(
                    backend,
                    scenario,
                    metric_model,
                    frequency=args.frequency,
                    worker=worker,
                )
                by_scenario[scenario].append(r)
        for scenario in scenarios:
            _print_table(scenario, by_scenario[scenario])
            all_results.extend(by_scenario[scenario])

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([vars(r) for r in all_results], f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
