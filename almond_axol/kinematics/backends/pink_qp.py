"""``pink-qp`` backend: Pinocchio QP differential IK (bake-off winner).

Velocity-level IK solved as one small QP per tick on the Pinocchio rigid-body
library: end-effector / elbow / posture tasks in the objective, joint position
and velocity limits as hard QP inequality constraints, plus an arm<->torso
self-collision barrier (control-barrier-function inequality on convex pair
distances; arm links versus base/torso only, never arm<->arm — see
``KinematicsConfig.diff_collision_margin``).
"""

from __future__ import annotations

import logging

import numpy as np
import pinocchio as pin
import pink
from pink.barriers import SelfCollisionBarrier
from pink.limits import ConfigurationLimit, VelocityLimit
from pink.tasks import FrameTask, PostureTask

from ...constants import URDF_PATH, Joint, urdf_joint_name
from ..base import CANONICAL_JOINT_NAMES, FKFrames, IKBackend, frame_body_names
from ..config import KinematicsConfig

_logger = logging.getLogger(__name__)

_QP_SOLVER = "daqp"

_TORSO_LINKS: tuple[str, ...] = ("base", "s1")

_SHOULDER_SUFFIXES: tuple[str, ...] = ("_s2", "_s3")
"""Arm links excluded from the barrier: the shoulder clusters orbit the base
column at ~18-22 mm across the whole joint range (inside any useful margin),
so a minimum-distance constraint on them wedges permanently and shoves the
arm's null space; joint limits, not the barrier, keep them safe. Measured:
with them included the barrier rode ``s3<->base`` at exactly the margin and
displaced the solution up to 115 deg from the unconstrained one."""

_BARRIER_GAIN = 10.0
"""CBF gain: max approach speed toward the margin is ``gain * (d - d_min)``.

The gain sets where braking starts, so it trades penetration against how
much of the workspace the constraint slows down: low gains brake from tens
of centimetres away and drag on ordinary motion, very high gains brake so
late that the discrete 120 Hz step overshoots the margin. 10 is the value
that ran the most real-hardware hours without either a reported contact or
a complaint attributable to the barrier. Verify the bench ``torso-graze``
scenario when changing this.
"""

_PRUNE_THRESHOLD = 0.025
"""Home-pose clearance (m) a pair needs to stay in the barrier.

Fixed — deliberately *not* derived from the runtime margin. The pair set
must not change with the margin: deriving the threshold as ``margin +
buffer`` meant a small margin re-admitted the shoulder-adjacent pairs that
sit 18-22 mm from the column across their whole joint range (they wedge the
QP — see ``_SHOULDER_SUFFIXES``), while a large margin silently pruned the
forearm<->base pairs that provide the actual protection. 0.025 sits between
the shoulder-cluster clearance (~22 mm) and the forearm rest clearance
(~27 mm), giving the same pair set at any margin.
"""


def _build_collision_model(
    model: pin.Model, margin: float, threshold: float | None = None
) -> pin.GeometryModel | None:
    """Convex-hull collision model restricted to feasible arm<->torso pairs.

    Mirrors :func:`almond_axol.kinematics.mujoco_model.arm_torso_geom_pairs`:
    only arm<->torso pairs matter, and pairs already within
    ``_PRUNE_THRESHOLD`` at the home pose are dropped (close by construction;
    a hard minimum-distance constraint on them would permanently fight the
    task). ``margin`` is only logged — the pair set is margin-independent.
    """
    urdf = URDF_PATH.read_text().replace(
        "package://assembly/meshes/", str(URDF_PATH.parent / "meshes") + "/"
    )
    geom_model = pin.buildGeomFromUrdfString(model, urdf, pin.GeometryType.COLLISION)
    for gobj in geom_model.geometryObjects:
        gobj.geometry.buildConvexRepresentation(False)
        gobj.geometry = gobj.geometry.convex

    torso_gids: list[int] = []
    arm_gids: list[int] = []
    for gid, gobj in enumerate(geom_model.geometryObjects):
        link = model.frames[gobj.parentFrame].name
        if link in _TORSO_LINKS:
            torso_gids.append(gid)
        elif link.startswith(("left_", "right_")) and not link.endswith(
            _SHOULDER_SUFFIXES
        ):
            arm_gids.append(gid)
    for a in arm_gids:
        for t in torso_gids:
            geom_model.addCollisionPair(pin.CollisionPair(a, t))

    data = model.createData()
    geom_data = pin.GeometryData(geom_model)
    pin.computeDistances(model, data, geom_model, geom_data, np.zeros(model.nq))
    threshold = _PRUNE_THRESHOLD if threshold is None else threshold
    keep = [
        k
        for k in range(len(geom_model.collisionPairs))
        if geom_data.distanceResults[k].min_distance > threshold
    ]
    pruned = len(geom_model.collisionPairs) - len(keep)
    for k in sorted(
        set(range(len(geom_model.collisionPairs))) - set(keep), reverse=True
    ):
        geom_model.removeCollisionPair(geom_model.collisionPairs[k])
    _logger.info(
        "pink collision: %d active arm<->torso pairs (%d pruned at home, "
        "margin %.0f mm).",
        len(geom_model.collisionPairs),
        pruned,
        margin * 1e3,
    )
    if not geom_model.collisionPairs:
        return None
    return geom_model


class _ArmTorsoBarrier(SelfCollisionBarrier):
    """``SelfCollisionBarrier`` with a fast Jacobian for arm<->torso pairs.

    The stock implementation costs ~2.5 ms per call here: a Python loop with
    ~20 small-array numpy ops per pair (each a few microseconds on this ARM
    CPU), an ``np.allclose`` per pair, closest-pair re-sorting, and a full
    two-body Jacobian even though our torso geoms sit on fixed links whose
    Jacobian is identically zero. This override produces bit-identical rows
    vectorised across pairs — one wrench-matrix product per moving joint —
    and skips fixed links.
    """

    def compute_barrier(self, configuration: pink.Configuration) -> np.ndarray:
        results = configuration.collision_data.distanceResults
        return np.array([results[k].min_distance - self.d_min for k in range(self.dim)])

    def compute_jacobian(self, configuration: pink.Configuration) -> np.ndarray:
        model = configuration.model
        data = configuration.data
        cm = configuration.collision_model
        cd = configuration.collision_data

        # Static structure (the collision model never changes after
        # construction): the moving joint of each pair's arm-side geometry,
        # remapped to a compact list of unique joints. The torso-side
        # geometries all sit on fixed links (parentJoint 0, zero Jacobian),
        # so only the arm-side term of the distance gradient survives.
        maps = getattr(self, "_pair_maps", None)
        if maps is None:
            pair_jids = []
            for cp in cm.collisionPairs:
                jids = [
                    cm.geometryObjects[gid].parentJoint for gid in (cp.first, cp.second)
                ]
                moving = [j for j in jids if j != 0]
                if len(moving) != 1 or cm.geometryObjects[cp.second].parentJoint != 0:
                    raise RuntimeError(
                        "_ArmTorsoBarrier expects pairs of (moving arm geom, "
                        "fixed torso geom)."
                    )
                pair_jids.append(moving[0])
            unique = sorted(set(pair_jids))
            maps = (unique, np.array([unique.index(j) for j in pair_jids]))
            self._pair_maps = maps
        unique_jids, pair_to_unique = maps

        K = self.dim
        w1 = np.empty((K, 3))
        w2 = np.empty((K, 3))
        for k in range(K):
            dr = cd.distanceResults[k]
            w1[k] = dr.getNearestPoint1()
            w2[k] = dr.getNearestPoint2()
        sep = w1 - w2
        dist = np.linalg.norm(sep, axis=1)
        # Touching pairs (distance ~0) have an undefined gradient; their
        # normal is zeroed so the row stays zero (matches the stock code).
        n = np.where((dist > 1e-9)[:, None], sep / np.maximum(dist, 1e-9)[:, None], 0.0)

        jacs = np.stack(
            [
                pin.getJointJacobian(
                    model, data, jid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                for jid in unique_jids
            ]
        )
        origins = np.stack([data.oMi[jid].translation for jid in unique_jids])
        r = w1 - origins[pair_to_unique]
        wrench = np.hstack([n, np.cross(r, n)])
        return np.einsum("kf,kfv->kv", wrench, jacs[pair_to_unique])


class PinkBackend(IKBackend):
    """Differential IK on Pinocchio via pink, both arms in a single QP."""

    name = "pink-qp"

    def __init__(self, config: KinematicsConfig, dt: float) -> None:
        self._config = config
        self._dt = dt

        model = pin.buildModelFromUrdf(str(URDF_PATH))
        # The URDF's 30 rad/s placeholder velocity limits are far beyond the
        # hardware; enforce the configured teleop limit instead.
        model.velocityLimit[:] = config.diff_max_joint_vel
        # Shoulder yaw/roll get half the global limit. They are the arm's
        # internal-reconfiguration pair: near the singular shoulder
        # (|shoulder_2| ~ 90 deg, crossed by every reach above shoulder
        # height) the swivel self-motion collapses onto an s1/s3
        # counter-rotation whose gain diverges, and the solver answers a
        # slowly moving hand with a violent whole-arm unwind at whatever
        # speed it is allowed — measured 300+ deg/s on captured sessions
        # while the hand crawled at 0.15 m/s. Ordinary teleop keeps these
        # joints under 135 deg/s (p99 across captured sessions), so pi rad/s
        # slows only the pathological reconfigurations, spreading them over
        # time instead of snapping.
        for joint in (Joint.SHOULDER_1, Joint.SHOULDER_3):
            for is_left in (True, False):
                jid = model.getJointId(urdf_joint_name(joint, is_left=is_left))
                model.velocityLimit[model.joints[jid].idx_v] = min(
                    config.diff_max_joint_vel, np.pi
                )
        self._model = model
        self._qidx = np.array(
            [model.joints[model.getJointId(n)].idx_q for n in CANONICAL_JOINT_NAMES],
            dtype=np.int64,
        )
        self._fk_data = model.createData()

        names = frame_body_names()
        self._body_names = names
        self._frame_ids = {key: model.getFrameId(frame) for key, frame in names.items()}

        self._barriers: list[SelfCollisionBarrier] = []
        collision_model = (
            _build_collision_model(model, config.diff_collision_margin)
            if config.diff_collision_margin > 0.0
            else None
        )
        if collision_model is not None:
            self._configuration = pink.Configuration(
                model,
                model.createData(),
                np.zeros(model.nq),
                collision_model=collision_model,
                collision_data=pin.GeometryData(collision_model),
            )
            self._barriers.append(
                _ArmTorsoBarrier(
                    n_collision_pairs=len(collision_model.collisionPairs),
                    gain=_BARRIER_GAIN,
                    safe_displacement_gain=0.0,
                    d_min=config.diff_collision_margin,
                )
            )
        else:
            self._configuration = pink.Configuration(
                model, model.createData(), np.zeros(model.nq)
            )
        self._l_ee_task = FrameTask(
            names["left_ee"],
            position_cost=config.diff_position_cost,
            orientation_cost=config.diff_orientation_cost,
            lm_damping=config.diff_lm_damping,
        )
        self._r_ee_task = FrameTask(
            names["right_ee"],
            position_cost=config.diff_position_cost,
            orientation_cost=config.diff_orientation_cost,
            lm_damping=config.diff_lm_damping,
        )
        self._l_elbow_task = FrameTask(
            names["left_elbow"],
            position_cost=config.diff_elbow_cost,
            orientation_cost=0.0,
            lm_damping=config.diff_lm_damping,
        )
        self._r_elbow_task = FrameTask(
            names["right_elbow"],
            position_cost=config.diff_elbow_cost,
            orientation_cost=0.0,
            lm_damping=config.diff_lm_damping,
        )
        # Canonical indices of the two shoulder_2 joints, whose angle is the
        # conditioning of the swivel null space (see _gate_elbow_tasks).
        self._l_s2_idx = CANONICAL_JOINT_NAMES.index(
            urdf_joint_name(Joint.SHOULDER_2, is_left=True)
        )
        self._r_s2_idx = CANONICAL_JOINT_NAMES.index(
            urdf_joint_name(Joint.SHOULDER_2, is_left=False)
        )
        self._posture_task = PostureTask(cost=config.diff_posture_cost)
        self._tasks = [
            self._l_ee_task,
            self._r_ee_task,
            self._posture_task,
        ]
        # With the elbow weight at zero the tasks would only add rows of
        # zeros to the QP; keep them out entirely.
        if config.diff_elbow_cost > 0.0:
            self._tasks[2:2] = [self._l_elbow_task, self._r_elbow_task]
        self._limits = [ConfigurationLimit(model), VelocityLimit(model)]

        frames = self.fk_frames(np.zeros(self.num_joints, dtype=np.float32))
        pin.framesForwardKinematics(model, self._fk_data, np.zeros(model.nq))
        self.left_shoulder_pos = np.asarray(
            self._fk_data.oMf[self._frame_ids["left_shoulder"]].translation,
            dtype=np.float32,
        )
        self.right_shoulder_pos = np.asarray(
            self._fk_data.oMf[self._frame_ids["right_shoulder"]].translation,
            dtype=np.float32,
        )

        self.set_posture_pose(np.zeros(self.num_joints, dtype=np.float32))
        # Warm-up solve (builds the QP once).
        self.ik(
            np.zeros(self.num_joints, dtype=np.float32),
            left_pose=frames.left_ee,
            right_pose=frames.right_ee,
        )
        _logger.info("pink-qp backend ready.")

    # -- Internal helpers -----------------------------------------------------

    def _to_pin(self, q_canonical: np.ndarray) -> np.ndarray:
        q = np.zeros(self._model.nq, dtype=np.float64)
        q[self._qidx] = np.asarray(q_canonical, dtype=np.float64)
        return q

    @staticmethod
    def _se3(pos: np.ndarray, rot: np.ndarray | None) -> pin.SE3:
        R = np.eye(3) if rot is None else np.asarray(rot, dtype=np.float64)
        return pin.SE3(R, np.asarray(pos, dtype=np.float64))

    def _gate_elbow_tasks(self, q_current: np.ndarray) -> None:
        """Fade each elbow task with its shoulder's conditioning.

        The elbow reference only ever asks for a swivel — the arm's single
        self-motion — and near ``shoulder_2 = +/-90 deg`` that self-motion is a
        shoulder_1/shoulder_3 counter-rotation whose joint gain diverges (the
        shoulder's contribution to end-effector pose is exactly rank-deficient
        there, measured ``sigma_min = 0``). A full-strength elbow task at that
        point answers a centimetre of swivel error with hundreds of deg/s of
        shoulder rotation — it saturates the velocity limit, and the operator
        sees the shoulder snap round while the hand barely moves. Every reach
        above shoulder height crosses this configuration, so it cannot be
        avoided, only handled.

        Scaling the task's weight by the conditioning makes it *singularity
        consistent*: full authority where the swivel is cheap to move (so the
        posture actually tracks), fading to none where it is not (so the arm
        coasts through holding whatever swivel it arrived with, which is also
        the swivel it leaves with — no branch flip on the way back down).
        ``|cos(shoulder_2)|`` is the conditioning: measured against the true
        singular value it tracks it within 7% over the whole range.
        """
        q = np.asarray(q_current, dtype=np.float64)
        for task, idx, cost in (
            (self._l_elbow_task, self._l_s2_idx, self._config.diff_elbow_cost),
            (self._r_elbow_task, self._r_s2_idx, self._config.diff_elbow_cost),
        ):
            task.set_position_cost(cost * abs(float(np.cos(q[idx]))))

    # -- IKBackend interface ----------------------------------------------------

    def ik(
        self,
        q_current: np.ndarray,
        left_pose: tuple[np.ndarray, np.ndarray] | None = None,
        right_pose: tuple[np.ndarray, np.ndarray] | None = None,
        left_elbow_pos: np.ndarray | None = None,
        right_elbow_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        if left_pose is None and right_pose is None:
            return np.asarray(q_current, dtype=np.float32).copy()

        cfg = self._config
        self._configuration.update(self._to_pin(q_current))

        if left_pose is not None:
            self._l_ee_task.set_target(self._se3(left_pose[0], left_pose[1]))
        else:
            self._l_ee_task.set_target_from_configuration(self._configuration)
        if right_pose is not None:
            self._r_ee_task.set_target(self._se3(right_pose[0], right_pose[1]))
        else:
            self._r_ee_task.set_target_from_configuration(self._configuration)
        if cfg.diff_elbow_cost > 0.0:
            if left_elbow_pos is not None:
                self._l_elbow_task.set_target(self._se3(left_elbow_pos, None))
            else:
                self._l_elbow_task.set_target_from_configuration(self._configuration)
            if right_elbow_pos is not None:
                self._r_elbow_task.set_target(self._se3(right_elbow_pos, None))
            else:
                self._r_elbow_task.set_target_from_configuration(self._configuration)
            self._gate_elbow_tasks(q_current)

        n_iters = max(1, cfg.diff_iters)
        sub_dt = self._dt / n_iters
        for _ in range(n_iters):
            try:
                vel = pink.solve_ik(
                    self._configuration,
                    self._tasks,
                    sub_dt,
                    solver=_QP_SOLVER,
                    damping=cfg.diff_damping,
                    limits=self._limits,
                    barriers=self._barriers,
                    safety_break=False,
                )
            except pink.exceptions.NoSolutionFound:
                # Infeasible QP (possible when the collision barrier is
                # enabled and wedged against the task): hold the current
                # configuration for this tick instead of crashing teleop.
                _logger.warning("pink QP infeasible; holding configuration.")
                return np.asarray(q_current, dtype=np.float32).copy()
            self._configuration.integrate_inplace(vel, sub_dt)

        return np.asarray(self._configuration.q[self._qidx], dtype=np.float32)

    def fk_frames(self, q: np.ndarray) -> FKFrames:
        pin.framesForwardKinematics(self._model, self._fk_data, self._to_pin(q))

        def _pose(key: str) -> tuple[np.ndarray, np.ndarray]:
            M = self._fk_data.oMf[self._frame_ids[key]]
            return (
                np.asarray(M.translation, dtype=np.float32),
                np.asarray(M.rotation, dtype=np.float32),
            )

        l_ee = _pose("left_ee")
        r_ee = _pose("right_ee")
        return FKFrames(
            left_ee=l_ee,
            right_ee=r_ee,
            left_elbow=_pose("left_elbow")[0],
            right_elbow=_pose("right_elbow")[0],
        )

    def set_posture_pose(self, q: np.ndarray) -> None:
        self._posture_task.set_target(self._to_pin(q))
