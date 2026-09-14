"""
Headless benchmark for the Mantis absolute-mode IK tracking pipeline.

Drives the real ``IKWorker`` (the exact code path ``teleop --mantis`` and
``collect-data --mantis`` run) with synthetic-but-realistic hand trajectories and
measures what the dataset would contain: the residual between the solved
joints' FK and where the physical hand-held gripper actually was.

The primary metric is the JAW-TIP position error (position error plus the
orientation error leveraged through the 14.5 cm tip protrusion), because the
fingertips are what interact with the scene — a dataset is replayable on the
robot only if FK(recorded joints) puts the tips where the camera saw them.

This is a tracking/convergence gate. It preserves the production collision
objective in its reference solve, but does not measure physical clearance or
replace on-robot collision validation.

Run:
    uv run python scripts/mantis_ik_bench.py --configs oracle_mantis mantis
    uv run python scripts/mantis_ik_bench.py --configs default balanced  # diagnostics
    uv run python scripts/mantis_ik_bench.py --list
"""

from __future__ import annotations

# ruff: noqa: E402 — the XLA/threading env vars must be set before the first
# JAX import (mirrors run_ik_worker), which forces imports below that block.
# Mirror run_ik_worker's environment so solve timings match production.
import os

_xla = os.environ.get("XLA_FLAGS", "")
if "xla_cpu_multi_thread_eigen" not in _xla:
    os.environ["XLA_FLAGS"] = f"{_xla} --xla_cpu_multi_thread_eigen=false".strip()
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from almond_axol.constants import GRIPPER_TIP_OFFSET, URDF_PATH
from almond_axol.kinematics.config import (
    KinematicsConfig,
    apply_mantis_kinematics_profile,
)
from almond_axol.teleop.config import VRTeleopConfig, apply_mantis_teleop_profile
from almond_axol.teleop.worker import IKWorker
from almond_axol.vr.models import VRFrame, VRPose, VRPosition, VRQuaternion

logging.basicConfig(level=logging.WARNING)

_TIP = np.asarray(GRIPPER_TIP_OFFSET, dtype=np.float64)
_BENCHMARK_SCHEMA = 1


def _model_sha256() -> str:
    """Hash the URDF and meshes because both affect FK/collision results."""
    digest = hashlib.sha256()
    model_root = URDF_PATH.parent
    for path in sorted(
        candidate for candidate in model_root.rglob("*") if candidate.is_file()
    ):
        digest.update(path.relative_to(model_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


_MODEL_SHA256 = _model_sha256()

# Engage pose: operator standing at the origin of a WebXR local-floor space
# (+x right, +y up, -z forward), hands at a comfortable start pose.
_ENGAGE_L = np.array([-0.17, 1.00, -0.35])
_ENGAGE_R = np.array([+0.17, 1.00, -0.35])


# ---------------------------------------------------------------------------
# Solver config presets
# ---------------------------------------------------------------------------


def _cfg(**kw: float) -> KinematicsConfig:
    return dataclasses.replace(KinematicsConfig(), **kw)


def _mantis_cfg() -> KinematicsConfig:
    cfg = KinematicsConfig()
    apply_mantis_kinematics_profile(cfg)
    return cfg


def _mantis_oracle_cfg() -> KinematicsConfig:
    """Convergence floor with the exact shipping Mantis objective intact."""
    return dataclasses.replace(
        _mantis_cfg(),
        max_joint_delta=10.0,
        max_iterations=64,
    )


PRESETS: dict[str, KinematicsConfig] = {
    # Shipping defaults (tuned for the physical arm).
    "default": _cfg(),
    # The over-aggressive attempt that destabilised the null space.
    "aggressive_fail": _cfg(
        pos_weight=300.0,
        ori_weight=100.0,
        rest_weight=1.0,
        posture_weight=0.5,
        max_joint_delta=0.1,
        max_iterations=24,
    ),
    # Raised tracking weights, regularizers left at defaults.
    "balanced": _cfg(
        pos_weight=200.0,
        ori_weight=60.0,
        max_joint_delta=0.05,
        max_iterations=16,
    ),
    # The shipped Mantis profile (kinematics/config.py) — what --mantis runs.
    "mantis": _mantis_cfg(),
    # Convergence counterpart with the exact shipping objective. Only the
    # per-tick step bound and iteration count are relaxed. Raising the oracle's
    # pose weights without raising its collision weight let it push much farther
    # through the torso capsules and made a less conservative solution look
    # feasible.
    "oracle_mantis": _mantis_oracle_cfg(),
    # Aggressive task-priority diagnostic. Unlike oracle_mantis this does not
    # preserve the shipping pose/collision weight ratios, so it is neither a
    # safety-matched floor nor production-scored. rest_weight=1 keeps the
    # normal equations damped (near-zero regularization destabilises the solve).
    "oracle": _cfg(
        pos_weight=1000.0,
        ori_weight=300.0,
        rest_weight=1.0,
        posture_weight=0.0,
        manipulability_weight=0.0,
        max_joint_delta=10.0,
        max_iterations=64,
        cost_tolerance=1e-4,
    ),
}


def _schedule_configs(
    todo: dict[str, KinematicsConfig],
) -> dict[str, KinematicsConfig]:
    """Put a fresh reference first whenever production is benchmarked."""
    scheduled = dict(todo)
    reference = scheduled.pop("oracle_mantis", None)
    if "mantis" in scheduled:
        return {
            "oracle_mantis": reference or PRESETS["oracle_mantis"],
            **scheduled,
        }
    if reference is not None:
        return {"oracle_mantis": reference, **scheduled}
    return scheduled


# ---------------------------------------------------------------------------
# Trajectory generation — displacement + rotation streams applied to both
# hands from the engage pose. All are C1-smooth (human hands don't teleport).
# ---------------------------------------------------------------------------


def _min_jerk(tau: np.ndarray) -> np.ndarray:
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _segments_to_stream(
    fps: float, segs: list[tuple[float, "np.ndarray | None", "tuple | None"]]
) -> tuple[np.ndarray, np.ndarray]:
    """Build (T,3) displacement + (T,3,3) rotation streams from move segments.

    Each segment is ``(duration_s, target_disp_3 | None, (axis_3, angle) | None)``;
    ``None`` holds the current value. Motion within a segment is min-jerk.
    """
    d_list: list[np.ndarray] = []
    r_list: list[np.ndarray] = []
    cur_d = np.zeros(3)
    cur_ang = 0.0
    cur_axis = np.array([0.0, 0.0, 1.0])
    for dur, tgt_d, tgt_r in segs:
        n = max(1, int(round(dur * fps)))
        tau = _min_jerk(np.arange(1, n + 1) / n)
        nxt_d = cur_d if tgt_d is None else np.asarray(tgt_d, dtype=np.float64)
        for i in range(n):
            d_list.append(cur_d + (nxt_d - cur_d) * tau[i])
        if tgt_r is None:
            r_list.extend([_rot_axis(cur_axis, cur_ang)] * n)
        else:
            axis, ang = np.asarray(tgt_r[0], dtype=np.float64), float(tgt_r[1])
            for i in range(n):
                # Blend the angle; axis switches take effect through the blend.
                a = cur_ang + (ang - cur_ang) * tau[i]
                r_list.append(_rot_axis(axis, a))
            cur_axis, cur_ang = axis, ang
        cur_d = nxt_d
    return np.asarray(d_list), np.asarray(r_list)


def make_trajectories(fps: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Named (T,3) displacement / (T,3,3) rotation streams (VR world frame).

    Amplitudes exercise both the comfortable workspace and collision/reach
    boundaries around it. The objective-matched reference separates error
    imposed by the shipping objective from lag caused by the production
    step/iteration budget. Both hands move together (constant separation),
    like carrying a rigidly-held object.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # 1. Hold still — engage convergence + null-space drift.
    out["hold"] = _segments_to_stream(fps, [(3.0, None, None)])

    # 2. Slow lateral/vertical wave (tray-carry sweep), peak ~0.2 m/s.
    t = np.arange(int(8.0 * fps)) / fps
    ramp = _min_jerk(t / 1.0)  # soft start
    d = np.stack(
        [
            0.06 * np.sin(2 * math.pi * 0.4 * t) * ramp,
            0.04 * np.sin(2 * math.pi * 0.8 * t) * ramp,
            np.zeros_like(t),
        ],
        axis=1,
    )
    r = np.broadcast_to(np.eye(3), (len(t), 3, 3)).copy()
    out["slow_wave"] = (d, r)

    # 3. Forward reach and return, normal speed (peak ~0.25 m/s).
    out["reach"] = _segments_to_stream(
        fps,
        [
            (1.0, np.array([0.0, 0.0, -0.10]), None),
            (0.5, None, None),
            (1.0, np.zeros(3), None),
            (0.5, None, None),
        ]
        * 2,
    )

    # 4. Fast transport (peak ~0.55 m/s) — where lag shows up.
    out["fast_reach"] = _segments_to_stream(
        fps,
        [
            (0.45, np.array([0.0, -0.03, -0.11]), None),
            (0.4, None, None),
            (0.45, np.zeros(3), None),
            (0.4, None, None),
        ]
        * 2,
    )

    # 5. Wrist rotations at fixed position (roll then pitch, ~60°/s peak).
    t = np.arange(int(6.0 * fps)) / fps
    ramp = _min_jerk(t / 1.0)
    d = np.zeros((len(t), 3))
    r = np.empty((len(t), 3, 3))
    for i, ti in enumerate(t):
        roll = math.radians(25.0) * math.sin(2 * math.pi * 0.5 * ti) * ramp[i]
        pitch = math.radians(18.0) * math.sin(2 * math.pi * 0.3 * ti) * ramp[i]
        r[i] = _rot_axis(np.array([0.0, 0.0, 1.0]), roll) @ _rot_axis(
            np.array([1.0, 0.0, 0.0]), pitch
        )
    out["wrist_twist"] = (d, r)

    # 6. Pick-and-place: reach down + yaw, hold (grasp), lift + carry, place.
    out["pick_place"] = _segments_to_stream(
        fps,
        [
            (1.4, np.array([0.0, -0.07, -0.06]), (np.array([0.0, 1.0, 0.0]), 0.5)),
            (0.8, None, None),
            (1.2, np.array([0.07, 0.01, -0.04]), (np.array([0.0, 1.0, 0.0]), -0.25)),
            (0.8, None, None),
            (1.4, np.zeros(3), (np.array([0.0, 1.0, 0.0]), 0.0)),
            (0.4, None, None),
        ],
    )

    # 7. Speed ramp: same 6 cm sweep at climbing frequency → err vs speed.
    t = np.arange(int(8.0 * fps)) / fps
    freq = 0.2 + (1.3 * t / t[-1])  # 0.2 → 1.5 Hz
    phase = 2 * math.pi * np.cumsum(freq) / fps
    ramp = _min_jerk(t / 1.0) * _min_jerk((t[-1] - t) / 1.0)
    d = np.stack(
        [0.06 * np.sin(phase) * ramp, np.zeros_like(t), np.zeros_like(t)], axis=1
    )
    r = np.broadcast_to(np.eye(3), (len(t), 3, 3)).copy()
    out["speed_ramp"] = (d, r)

    return out


def _trajectory_signatures(
    trajs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, str]:
    """Content hashes that identify the generated paths in saved results."""
    signatures: dict[str, str] = {}
    for name, arrays in trajs.items():
        digest = hashlib.sha256()
        for array in arrays:
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.shape).encode())
            digest.update(contiguous.dtype.str.encode())
            digest.update(contiguous.tobytes())
        signatures[name] = digest.hexdigest()
    return signatures


def _tremor(n: int, fps: float, rms_m: float, seed: int) -> np.ndarray:
    """(n,3) physiological-tremor-like positional noise (8–11 Hz sinusoids)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    out = np.zeros((n, 3))
    for ax in range(3):
        for f in (8.3, 9.7, 10.9):
            out[:, ax] += rng.normal(0, 1) * np.sin(
                2 * math.pi * f * t + rng.uniform(0, 2 * math.pi)
            )
    out *= rms_m / max(1e-9, np.sqrt(np.mean(out**2)))
    return out


def _trajectory_seed(name: str) -> int:
    """Stable tremor seed; Python's randomized ``hash()`` is not reproducible."""
    digest = hashlib.sha256(name.encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Bench core
# ---------------------------------------------------------------------------


def _quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        return (
            float(R[2, 1] - R[1, 2]) / (4 * w),
            float(R[0, 2] - R[2, 0]) / (4 * w),
            float(R[1, 0] - R[0, 1]) / (4 * w),
            w,
        )
    # Fall back for 180° rotations.
    x = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    return (x, y, z, 0.0)


def _frame(l_pos, l_rot, r_pos, r_rot) -> VRFrame:
    def pose(p, R):
        q = _quat_xyzw(R)
        return VRPose(
            position=VRPosition(x=float(p[0]), y=float(p[1]), z=float(p[2])),
            quaternion=VRQuaternion(x=q[0], y=q[1], z=q[2], w=q[3]),
        )

    zero = VRPosition(x=0.0, y=0.0, z=0.0)
    return VRFrame(
        l_ee=pose(l_pos, l_rot),
        r_ee=pose(r_pos, r_rot),
        l_elbow=zero,
        r_elbow=zero,
        l_lock=True,
        r_lock=True,
    )


def _ori_err_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) * 0.5
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _summ(a: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def _benchmark_teleop_config(fps: float) -> VRTeleopConfig:
    """Build a host-independent Mantis config for synthetic tracker poses."""
    identity_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    config = VRTeleopConfig(
        tcp_transform_left=identity_transform.copy(),
        tcp_transform_right=identity_transform.copy(),
    )
    apply_mantis_teleop_profile(config)
    config.frequency = fps
    # Use engage-relative controller motion, independent of any machine-local
    # or factory tracker-to-gripper transform.
    config.tcp_transform_left = None
    config.tcp_transform_right = None
    return config


def run_config(
    name: str,
    kcfg: KinematicsConfig,
    trajs: dict[str, tuple[np.ndarray, np.ndarray]],
    fps: float,
    tremor_rms: float,
) -> dict:
    tcfg = _benchmark_teleop_config(fps)

    t0 = time.perf_counter()
    worker = IKWorker(tcfg, kcfg)
    init_s = time.perf_counter() - t0
    solver = worker._solver

    results: dict = {
        "benchmark_schema": _BENCHMARK_SCHEMA,
        "config": dataclasses.asdict(kcfg),
        "fps": fps,
        "tremor_rms": tremor_rms,
        "model_sha256": _MODEL_SHA256,
        "trajectory_sha256": _trajectory_signatures(trajs),
        "init_s": init_s,
        "trajs": {},
    }

    for traj_name, (disp, rots) in trajs.items():
        n = len(disp)
        tremor_seed = _trajectory_seed(traj_name)
        noise = (
            _tremor(n, fps, tremor_rms, seed=tremor_seed)
            if tremor_rms > 0
            else np.zeros((n, 3))
        )

        # Fresh engage per trajectory: worker.reset() then a rising edge at
        # the engage pose (mirrors an operator squeezing both grips).
        worker.reset()
        q = worker.get_rest_q()
        q = worker.step(_frame(_ENGAGE_L, np.eye(3), _ENGAGE_R, np.eye(3)), q)

        m = {
            k: np.zeros(n)
            for k in (
                "tip_l",
                "tip_r",
                "pos_l",
                "pos_r",
                "ori_l",
                "ori_r",
                "hand_speed",
                "joint_speed",
                "elbow_speed",
                "solve_ms",
            )
        }
        prev_q = q.copy()
        prev_elbows: np.ndarray | None = None
        q_hist = np.zeros((n, len(q)))

        for i in range(n):
            l_pos = _ENGAGE_L + disp[i] + noise[i]
            r_pos = _ENGAGE_R + disp[i] + noise[i]
            R = rots[i]

            t0 = time.perf_counter()
            q = worker.step(_frame(l_pos, R, r_pos, R), q)
            m["solve_ms"][i] = (time.perf_counter() - t0) * 1e3

            # Ground truth: the physical hand pose (streamed, unfiltered)
            # mapped through the engage-rigid transform — exactly where the
            # gripper (and its camera) actually is.
            tgt_l = worker._absolute_target("left", l_pos, R)
            tgt_r = worker._absolute_target("right", r_pos, R)
            import jaxlie  # local import keeps module import light

            fk = solver.robot.forward_kinematics(np.asarray(solver.to_pyroki_order(q)))
            poses = {}
            for side, idx in (("l", solver.l_ee_idx), ("r", solver.r_ee_idx)):
                T = jaxlie.SE3(fk[idx])
                poses[side] = (
                    np.asarray(T.translation(), dtype=np.float64),
                    np.asarray(T.rotation().as_matrix(), dtype=np.float64),
                )
            for side, tgt in (("l", tgt_l), ("r", tgt_r)):
                fp, fR = poses[side]
                tp = np.asarray(tgt[0], dtype=np.float64)
                tR = np.asarray(tgt[1], dtype=np.float64)
                m[f"pos_{side}"][i] = np.linalg.norm(fp - tp) * 1e3
                m[f"ori_{side}"][i] = _ori_err_deg(fR, tR)
                m[f"tip_{side}"][i] = (
                    np.linalg.norm((fp + fR @ _TIP) - (tp + tR @ _TIP)) * 1e3
                )

            if i > 0:
                m["hand_speed"][i] = np.linalg.norm(disp[i] - disp[i - 1]) * fps
            m["joint_speed"][i] = float(np.max(np.abs(q - prev_q))) * fps
            prev_q = q.copy()
            q_hist[i] = q
            elbows = np.concatenate(
                [
                    np.asarray(jaxlie.SE3(fk[solver.l_elbow_idx]).translation()),
                    np.asarray(jaxlie.SE3(fk[solver.r_elbow_idx]).translation()),
                ]
            )
            if prev_elbows is not None:
                m["elbow_speed"][i] = float(np.max(np.abs(elbows - prev_elbows))) * fps
            prev_elbows = elbows

        tip = np.concatenate([m["tip_l"], m["tip_r"]])
        ori = np.concatenate([m["ori_l"], m["ori_r"]])
        # Skip the first 0.5 s (engage catch-up transient) for the summary.
        s = int(0.5 * fps)
        tip_ss = np.concatenate([m["tip_l"][s:], m["tip_r"][s:]])
        ori_ss = np.concatenate([m["ori_l"][s:], m["ori_r"][s:]])

        r: dict = {
            "ticks": n,
            "tremor_seed": tremor_seed,
            "tip_mm": _summ(tip_ss),
            "pos_mm": _summ(np.concatenate([m["pos_l"][s:], m["pos_r"][s:]])),
            "ori_deg": _summ(ori_ss),
            "solve_ms": _summ(m["solve_ms"][1:]),
            "tip_mm_incl_engage": _summ(tip),
            "ori_deg_incl_engage": _summ(ori),
        }
        # Null-space stability: joint / elbow motion while the hand is slow.
        slow = m["hand_speed"] < 0.05
        slow[: int(1.5 * fps)] = False  # ignore engage catch-up
        if np.any(slow):
            r["drift_joint_rad_s"] = float(np.percentile(m["joint_speed"][slow], 99))
            r["drift_elbow_m_s"] = float(np.percentile(m["elbow_speed"][slow], 99))
        r["elbow_speed_p99_m_s"] = float(np.percentile(m["elbow_speed"][1:], 99))
        if traj_name == "hold":
            # Net null-space drift over the final second (rad/s). Uses the
            # displacement between two window-averaged configurations so
            # zero-mean tremor response doesn't read as drift.
            w = int(0.25 * fps)
            a = q_hist[-int(fps) - w : -int(fps)].mean(axis=0)
            b = q_hist[-w:].mean(axis=0)
            r["hold_drift_rad_s"] = float(np.max(np.abs(b - a)))
        results["trajs"][traj_name] = r

    return results


# ---------------------------------------------------------------------------
# Spec + reporting
# ---------------------------------------------------------------------------

SPEC = {
    # Solver excess over the objective-matched convergence floor: the part of
    # the tip error caused by the shipping iteration and per-tick step bounds.
    # The reference solve keeps the production pose/collision/limit weight
    # ratios, so lower tracking error cannot be bought by weakening that
    # objective. This benchmark does not assert physical collision clearance.
    "tip_excess_mm_p95": 6.0,
    "tip_excess_mm_max": 15.0,
    # Absolute bounds, enforced where the floor itself is small (feasible).
    "tip_mm_p95": 10.0,
    "tip_mm_max": 20.0,
    "feasible_floor_mm": 5.0,
    "ori_deg_p95": 3.0,
    "hold_drift_rad_s": 0.02,  # net null-space drift at rest, rad/s
    "solve_ms_p95": 33.0,  # one 30 Hz collection tick (CPU-JAX Jetson budget)
}

# Trajectories the spec is scored on.
SPEC_TRAJS = ("hold", "slow_wave", "reach", "wrist_twist", "pick_place")


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def check_spec(res: dict, oracle: dict | None) -> tuple[bool, list[str]]:
    fails: list[str] = []
    if oracle is None:
        return False, ["objective-matched reference floor is missing"]

    trajectories = res.get("trajs")
    floor_trajectories = oracle.get("trajs")
    if not isinstance(trajectories, dict):
        return False, ["production trajectory results are missing"]
    if not isinstance(floor_trajectories, dict):
        return False, ["reference trajectory results are missing"]

    def metric(trajectory: object, group: str, name: str, label: str) -> float | None:
        if not isinstance(trajectory, dict):
            fails.append(f"{label}: result is missing")
            return None
        summary = trajectory.get(group)
        value = summary.get(name) if isinstance(summary, dict) else None
        if not _is_finite_number(value):
            fails.append(f"{label}: {group}.{name} is missing or non-finite")
            return None
        return float(value)

    scored = 0
    for tname in SPEC_TRAJS:
        if tname not in trajectories:
            fails.append(f"{tname}: production result is missing")
            continue
        scored += 1
        trajectory = trajectories[tname]
        floor_trajectory = floor_trajectories.get(tname)
        tip_p95 = metric(trajectory, "tip_mm", "p95", tname)
        tip_max = metric(trajectory, "tip_mm", "max", tname)
        floor_p95 = metric(floor_trajectory, "tip_mm", "p95", f"{tname} floor")
        floor_max = metric(floor_trajectory, "tip_mm", "max", f"{tname} floor")
        ori_p95 = metric(trajectory, "ori_deg", "p95", tname)
        solve_p95 = metric(trajectory, "solve_ms", "p95", tname)

        if None not in (tip_p95, tip_max, floor_p95, floor_max):
            assert tip_p95 is not None
            assert tip_max is not None
            assert floor_p95 is not None
            assert floor_max is not None
            excess_p95 = tip_p95 - floor_p95
            excess_max = tip_max - floor_max
            if excess_p95 > SPEC["tip_excess_mm_p95"]:
                fails.append(f"{tname}: tip excess p95 {excess_p95:.1f}mm over floor")
            if excess_max > SPEC["tip_excess_mm_max"]:
                fails.append(f"{tname}: tip excess max {excess_max:.1f}mm over floor")
            if floor_p95 < SPEC["feasible_floor_mm"]:
                if tip_p95 > SPEC["tip_mm_p95"]:
                    fails.append(f"{tname}: tip p95 {tip_p95:.1f}mm")
                if tip_max > SPEC["tip_mm_max"]:
                    fails.append(f"{tname}: tip max {tip_max:.1f}mm")
        if ori_p95 is not None and ori_p95 > SPEC["ori_deg_p95"]:
            fails.append(f"{tname}: ori p95 {ori_p95:.2f}°")
        if solve_p95 is not None and solve_p95 > SPEC["solve_ms_p95"]:
            fails.append(f"{tname}: solve p95 {solve_p95:.1f}ms")

        if tname == "hold":
            hold_drift = (
                trajectory.get("hold_drift_rad_s")
                if isinstance(trajectory, dict)
                else None
            )
            if not _is_finite_number(hold_drift):
                fails.append("hold: hold_drift_rad_s is missing or non-finite")
            elif float(hold_drift) > SPEC["hold_drift_rad_s"]:
                fails.append(f"hold: null-space drift {float(hold_drift):.3f}rad/s")

    if scored == 0:
        fails.append("no scored production trajectories were run")
    return (not fails), fails


def print_report(
    name: str,
    res: dict,
    oracle: dict | None,
    *,
    score: bool = False,
    unscored_label: str = "DIAGNOSTIC CONFIG",
) -> bool | None:
    print(f"\n=== {name} (init {res['init_s']:.1f}s) ===")
    hdr = (
        f"{'trajectory':<12} {'tip mm (mean/p95/max)':>24} {'floor p95':>10}"
        f" {'ori° p95':>9} {'solve ms p95':>13}"
    )
    print(hdr)
    for tname, t in res["trajs"].items():
        tip = t["tip_mm"]
        fl = (
            f"{oracle['trajs'][tname]['tip_mm']['p95']:>10.1f}"
            if oracle and tname in oracle.get("trajs", {})
            else f"{'—':>10}"
        )
        print(
            f"{tname:<12} {tip['mean']:>7.1f} /{tip['p95']:>6.1f} /{tip['max']:>7.1f}"
            f" {fl} {t['ori_deg']['p95']:>9.2f} {t['solve_ms']['p95']:>13.1f}"
        )
    if score:
        ok, fails = check_spec(res, oracle)
        print("SPEC:", "PASS" if ok else "FAIL — " + "; ".join(fails))
        return ok
    print(f"SPEC: {unscored_label} (not production-scored)")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="*", default=["mantis"], help="Preset names")
    ap.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="K=V",
        help="Ad-hoc config: overrides on KinematicsConfig, named adhoc",
    )
    ap.add_argument("--fps", type=float, default=120.0)
    ap.add_argument("--tremor-mm", type=float, default=1.5)
    ap.add_argument(
        "--trajs",
        nargs="*",
        default=None,
        help="Subset of trajectories to run (default: all)",
    )
    ap.add_argument("--out", type=Path, default=Path("/tmp/mantis_ik_bench"))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in PRESETS.items():
            print(k, dataclasses.asdict(v))
        return

    todo: dict[str, KinematicsConfig] = {}
    for c in args.configs:
        todo[c] = PRESETS[c]
    if args.set:
        kw: dict[str, float] = {}
        for kv in args.set:
            k, v = kv.split("=")
            kw[k] = int(v) if k == "max_iterations" else float(v)
        todo["adhoc"] = _cfg(**kw)
    if not todo:
        ap.error("select at least one --configs preset or provide --set overrides")

    trajs = make_trajectories(args.fps)
    if args.trajs:
        trajs = {k: trajs[k] for k in args.trajs}
    args.out.mkdir(parents=True, exist_ok=True)

    # Always compute a fresh objective-matched convergence floor in the same
    # process as the production run. Reusing a saved floor could compare
    # results produced by different solver, transform, dependency, or runtime
    # implementations even when the serialized config and model are unchanged.
    oracle: dict | None = None
    todo = _schedule_configs(todo)
    production_failed = False
    for name, kcfg in todo.items():
        res = run_config(name, kcfg, trajs, args.fps, args.tremor_mm * 1e-3)
        if name == "oracle_mantis":
            oracle = res
        outcome = print_report(
            name,
            res,
            oracle if name != "oracle_mantis" else None,
            score=name == "mantis",
            unscored_label=(
                "REFERENCE FLOOR" if name == "oracle_mantis" else "DIAGNOSTIC CONFIG"
            ),
        )
        production_failed |= outcome is False
        (args.out / f"{name}.json").write_text(json.dumps(res, indent=1))
        print(f"(saved {args.out / (name + '.json')})")

    if production_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
