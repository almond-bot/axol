"""Velocity-QP tracking backend using Mink and DAQP.

Per-tick differential IK uses hard joint and velocity constraints, global
Tikhonov damping for singularity robustness, a rest-pose posture attractor
and optional torso-to-arm collision constraints. Forward kinematics and
reset planning use MuJoCo and separate Mink state.

The interface maps joint vectors to MuJoCo qpos by name. Non-arm degrees
of freedom are frozen by equality constraints and copied from the input.
Per-iteration velocity limits bound the call's total joint displacement,
leaving the caller's componentwise clamp as a backstop.

Each iteration first solves with collision constraints, retries without
collision constraints if necessary, then returns the seed unchanged if
both attempts fail. The failure counter makes these holds observable.
"""

from __future__ import annotations

import logging

import mink
import mujoco
import numpy as np

from almond_axol.constants import Joint, urdf_body_name

from .ik_config import PINNED_URDF
from .ik_config import MinkIKConfig as KinematicsConfig
from .ik_mujoco_model import body_geom_ids, load_mj_model, qpos_indices

_logger = logging.getLogger(__name__)

# Arm bodies guarded against the torso, excluding fingers and TCP frames.
# The s2 shoulder yoke rotates inside the torso mount with 0-10 mm clearance;
# joint limits constrain this designed overlap rather than collision IK.
_ARM_BODY_SUFFIXES = ("s3", "e1", "e2", "w0", "w1", "w2", "gripper")
_TORSO_BODIES = ("base", "s1")

# Fraction of mink_max_joint_delta the QP may use per call; the remaining
# margin guarantees the caller's componentwise clamp stays a no-op backstop.
_DELTA_BUDGET = 0.98


class MinkTracker:
    """Velocity-QP tracker over a caller-specified joint ordering.

    Args:
        config: Solver weights and limits (``mink_*`` fields).
        joint_names: Actuated joint names in the caller's vector order.
        arm_joint_names: The 14 arm joints that the QP may move; other degrees
            of freedom remain frozen.
        ee_bodies: Left and right end-effector body names.
        elbow_bodies: Left and right elbow body names for optional hint tasks.
    """

    def __init__(
        self,
        config: KinematicsConfig,
        joint_names: list[str],
        arm_joint_names: list[str],
        ee_bodies: tuple[str, str],
        elbow_bodies: tuple[str, str],
    ) -> None:
        self._config = config
        self._model = load_mj_model(PINNED_URDF)
        self._configuration = mink.Configuration(self._model)

        # Name-mapped views between the caller's q order and MuJoCo qpos.
        self._q_names = list(joint_names)
        self._qadr = qpos_indices(self._model, self._q_names)
        # Per-mapped-joint ranges: the seed must be clamped into them before
        # the QP sees it. An out-of-range seed makes ConfigurationLimit's
        # retreat row (0.95×violation per iteration) contradict the velocity
        # rows for violations over ~1°, turning every solve infeasible — a
        # permanent hold. Clamping recovers at the outer clamp's walk-back
        # rate while retaining hard joint constraints.
        jids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self._q_names
        ]
        limited = self._model.jnt_limited[jids].astype(bool)
        self._q_lo = np.where(limited, self._model.jnt_range[jids, 0], -np.inf)
        self._q_hi = np.where(limited, self._model.jnt_range[jids, 1], np.inf)
        arm_set = set(arm_joint_names)
        self._frozen_dofs = [
            int(self._model.jnt_dofadr[j])
            for j in range(self._model.njnt)
            if mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, j)
            not in arm_set
        ]

        lm = config.mink_lm_damping
        self._ee_tasks = {
            side: mink.FrameTask(
                frame_name=name,
                frame_type="body",
                position_cost=config.mink_position_cost,
                orientation_cost=config.mink_orientation_cost,
                lm_damping=lm,
            )
            for side, name in zip(("left", "right"), ee_bodies, strict=True)
        }
        self._elbow_tasks = {
            side: mink.FrameTask(
                frame_name=name,
                frame_type="body",
                position_cost=config.mink_elbow_cost,
                orientation_cost=0.0,
                lm_damping=lm,
            )
            for side, name in zip(("left", "right"), elbow_bodies, strict=True)
        }
        self._posture_task = mink.PostureTask(
            self._model, cost=config.mink_posture_cost
        )
        self._posture_task.set_target(np.zeros(self._model.nq))
        # Speed-gated posture: remember the last EE targets to estimate the
        # demanded speed; the attractor's effective cost rolls off
        # quadratically above mink_posture_speed_gate (see config). The
        # applied scale is additionally rate-limited through a one-pole
        # low-pass (mink_posture_gate_tau) so a speed-gate crossing shifts
        # the QP optimum over ~tau instead of snapping it in one solve.
        self._gate_scale: float | None = None
        self._last_ee_target: dict[str, np.ndarray | None] = {
            "left": None,
            "right": None,
        }

        # Elbow-hint projection: with position_multiplier > 1 the scaled
        # human elbow hint orbits well outside the robot elbow's reachable
        # sphere. An unreachable point target tilts the whole arm instead
        # of shaping the elbow direction.
        # Project every hint onto the sphere about the (fixed) shoulder
        # center so the task carries direction only.
        data = mujoco.MjData(self._model)
        data.qpos[:] = 0.0
        mujoco.mj_forward(self._model, data)
        self._shoulder_pos = {}
        self._elbow_bid = {}
        self._elbow_radius = {}  # qpos=0 fallback; solve() uses per-pose
        for side, is_left, elbow_body in zip(
            ("left", "right"), (True, False), elbow_bodies, strict=True
        ):
            sid = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_BODY,
                urdf_body_name(Joint.SHOULDER_1, is_left=is_left),
            )
            eid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, elbow_body)
            self._shoulder_pos[side] = data.xpos[sid].copy()
            self._elbow_bid[side] = eid
            self._elbow_radius[side] = float(
                np.linalg.norm(data.xpos[eid] - data.xpos[sid])
            )

        self._constraints = (
            [mink.DofFreezingTask(self._model, dof_indices=self._frozen_dofs)]
            if self._frozen_dofs
            else []
        )

        # Per-iteration velocity cap sized so the whole call's displacement
        # stays inside mink_max_joint_delta (the outer clamp never binds).
        per_iter_vmax = (config.mink_max_joint_delta * _DELTA_BUDGET) / (
            config.mink_iterations * config.mink_dt
        )
        base_limits: list[object] = [
            mink.ConfigurationLimit(self._model),
            mink.VelocityLimit(
                self._model, dict.fromkeys(arm_joint_names, per_iter_vmax)
            ),
        ]
        self._limits = list(base_limits)
        self._limits_no_collision = list(base_limits)
        # Kept torso<->arm geom-id pairs (introspection/diagnostics).
        self.collision_pairs: list[tuple[int, int]] = []
        if config.mink_collision:
            pairs = self._home_feasible_collision_pairs()
            self.collision_pairs = [(a[0], b[0]) for a, b in pairs]
            if pairs:
                self._limits.append(
                    mink.CollisionAvoidanceLimit(
                        self._model,
                        geom_pairs=pairs,
                        minimum_distance_from_collisions=(
                            config.mink_collision_min_distance
                        ),
                        collision_detection_distance=(
                            config.mink_collision_detection_distance
                        ),
                    )
                )
        self._fail_count = 0

    def _home_feasible_collision_pairs(
        self,
    ) -> list[tuple[list[int], list[int]]]:
        """Select torso-to-arm geometry pairs that are feasible at home.

        Pairs already inside the clearance margin at home are excluded: their
        conservative geometry would make the home pose infeasible for hard
        collision constraints.
        """
        torso = body_geom_ids(self._model, list(_TORSO_BODIES))
        arms = body_geom_ids(
            self._model,
            [
                f"{side}_{sfx}"
                for side in ("left", "right")
                for sfx in _ARM_BODY_SUFFIXES
            ],
        )
        data = mujoco.MjData(self._model)
        data.qpos[:] = 0.0
        mujoco.mj_forward(self._model, data)
        fromto = np.empty(6)
        pairs: list[tuple[list[int], list[int]]] = []
        dropped = 0
        for tg in torso:
            for ag in arms:
                d = mujoco.mj_geomDistance(self._model, data, tg, ag, 1.0, fromto)
                if d <= self._config.mink_collision_min_distance:
                    dropped += 1
                    continue
                pairs.append(([tg], [ag]))
        if dropped:
            _logger.info(
                "mink collision rows: %d torso<->arm geom pairs (%d dropped "
                "as infeasible at home)",
                len(pairs),
                dropped,
            )
        return pairs

    def set_posture(self, q: np.ndarray) -> None:
        """Engage-edge posture update; a no-op in ``"rest"`` target mode."""
        if self._config.mink_posture_target == "engage":
            self._posture_task.set_target(self._to_qpos(q))

    def reset_tracking_state(self) -> None:
        """Clear cross-call posture-gate memory at a target discontinuity.

        Previous targets estimate demand speed. A handover, episode boundary or
        seed reset can change the target abruptly; retained history would cause
        a false speed spike that decays over ``mink_posture_gate_tau``. Warmup
        also clears this history before control begins.
        """
        self._last_ee_target = {"left": None, "right": None}
        self._gate_scale = None

    @property
    def fail_count(self) -> int:
        """Number of solves that exhausted both attempts and held the seed.

        A held command alone does not distinguish solver failure from a static
        target; callers can expose this counter in session diagnostics.
        """
        return self._fail_count

    def set_rest_posture(self, q: np.ndarray) -> None:
        """Pin the posture attractor to the (settled) rest configuration."""
        self._posture_task.set_target(self._to_qpos(q))

    def solve(
        self,
        q_current: np.ndarray,
        left_pose: tuple[np.ndarray, np.ndarray] | None,
        right_pose: tuple[np.ndarray, np.ndarray] | None,
        left_elbow_pos: np.ndarray | None,
        right_elbow_pos: np.ndarray | None,
    ) -> np.ndarray:
        """Return one tracking step in the caller's joint ordering.

        Frozen non-arm entries pass through from ``q_current`` unchanged.
        """
        cfg = self._config
        self._configuration.update(self._to_qpos(q_current))

        # Posture speed gate: estimate the fastest EE-target speed this call
        # (per-call target delta x the configured solve cadence,
        # cfg.mink_solve_hz) and roll the attractor off.
        if cfg.mink_posture_speed_gate > 0.0:
            speed = 0.0
            for side, pose in (("left", left_pose), ("right", right_pose)):
                if pose is None:
                    continue
                pos = np.asarray(pose[0], dtype=np.float64)
                last = self._last_ee_target[side]
                if last is not None:
                    speed = max(
                        speed, float(np.linalg.norm(pos - last)) * cfg.mink_solve_hz
                    )
                self._last_ee_target[side] = pos
            scale = 1.0 / (1.0 + (speed / cfg.mink_posture_speed_gate) ** 2)
            if cfg.mink_posture_gate_tau > 0.0:
                if self._gate_scale is None:
                    self._gate_scale = scale
                else:
                    alpha = min(
                        (1.0 / cfg.mink_solve_hz) / cfg.mink_posture_gate_tau, 1.0
                    )
                    self._gate_scale += alpha * (scale - self._gate_scale)
                scale = self._gate_scale
            self._posture_task.set_cost(cfg.mink_posture_cost * scale)

        tasks: list[object] = [self._posture_task]
        for side, pose in (("left", left_pose), ("right", right_pose)):
            if pose is None:
                continue
            pos, rot = pose
            task = self._ee_tasks[side]
            task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3.from_matrix(np.asarray(rot, dtype=np.float64)),
                    np.asarray(pos, dtype=np.float64),
                )
            )
            tasks.append(task)
        if cfg.mink_elbow_cost > 0.0:
            for side, hint in (
                ("left", left_elbow_pos),
                ("right", right_elbow_pos),
            ):
                if hint is None:
                    continue
                center = self._shoulder_pos[side]
                d = np.asarray(hint, dtype=np.float64) - center
                norm = float(np.linalg.norm(d))
                if norm < 0.05:
                    # A hint within 5 cm of the shoulder center carries no
                    # usable direction — a few mm of tracking noise would
                    # swing the projected target across the whole sphere.
                    continue
                # The shoulder joint cluster is not concentric (s2/s3
                # anchors sit ~7 cm off the s1 axis): the true elbow radius
                # varies ±5 cm with pose, so measure it from the CURRENT
                # configuration rather than the qpos=0 constant.
                elbow_now = self._configuration.data.xpos[self._elbow_bid[side]]
                radius = float(np.linalg.norm(elbow_now - center))
                if radius < 0.05:
                    radius = self._elbow_radius[side]
                projected = center + d * (radius / norm)
                task = self._elbow_tasks[side]
                task.set_target(
                    mink.SE3.from_rotation_and_translation(
                        mink.SO3.identity(), projected
                    )
                )
                tasks.append(task)

        for _ in range(cfg.mink_iterations):
            vel = self._solve_step(tasks)
            if vel is None:
                return np.asarray(q_current, dtype=np.float32).copy()
            self._configuration.integrate_inplace(vel, cfg.mink_dt)

        out = np.asarray(q_current, dtype=np.float64).copy()
        out[: len(self._qadr)] = self._configuration.q[self._qadr]
        return out.astype(np.float32)

    def _solve_step(self, tasks: list[object]) -> np.ndarray | None:
        """Failure ladder: full -> no collision rows -> hold (None)."""
        cfg = self._config
        last_exc: Exception | None = None
        for limits in (self._limits, self._limits_no_collision):
            try:
                return mink.solve_ik(
                    self._configuration,
                    tasks,
                    cfg.mink_dt,
                    "daqp",
                    damping=cfg.mink_damping,
                    safety_break=False,
                    limits=limits,
                    constraints=self._constraints,
                )
            except Exception as exc:  # noqa: BLE001 - drop-and-warn semantics
                last_exc = exc
                continue
        self._fail_count += 1
        if self._fail_count == 1 or self._fail_count % 100 == 0:
            _logger.warning(
                "mink solve failed (%d so far), holding last q: %s",
                self._fail_count,
                last_exc,
            )
        return None

    def _to_qpos(self, q: np.ndarray) -> np.ndarray:
        """Caller-order joint vector -> full MuJoCo qpos (unmapped dofs 0).

        Values are clamped into the joint ranges — see the seed-clamp note
        in ``__init__``.
        """
        qpos = np.zeros(self._model.nq)
        qpos[self._qadr] = np.clip(
            np.asarray(q, dtype=np.float64)[: len(self._qadr)],
            self._q_lo,
            self._q_hi,
        )
        return qpos
