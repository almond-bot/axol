"""Process-wide shared robot model for the kinematics entry points.

The bundled URDF never changes at runtime, but ``pyroki.Robot.from_urdf``
mints a fresh ``JointVar`` class on every call — and that class is a
*static* field of the ``Robot`` pytree. Two ``Robot`` instances built from
the same URDF therefore never compare equal for JIT-cache purposes: every
jitted function that takes a ``Robot`` (the IK solve, forward kinematics)
re-traces and re-runs jaxls problem analysis per instance, costing seconds
of pure Python work each time even though the compiled XLA executable is
identical (and served from the persistent cache, see :mod:`.jax_cache`).

Building the robot once per process and handing the same instance to every
:class:`~almond_axol.kinematics.solver.KinematicsSolver` /
:class:`~almond_axol.kinematics.fk.AxolForwardKinematics` makes each
instance after the first hit JAX's in-memory jit cache, so constructing
additional objects is close to free.

The collision model is built lazily and separately: forward-kinematics-only
users (observation recording) never pay for capsule fitting.
"""

from __future__ import annotations

import logging
import threading

import jax.numpy as jnp
import numpy as np
import pyroki as pk
import yourdfpy

from ..constants import URDF_PATH

_logger = logging.getLogger(__name__)

_TORSO_LINKS: tuple[str, ...] = ("base", "s1")
"""Static body links that the arms must not collide into.

Self-collision on Axol is restricted to ``arm <-> torso`` pairs only.
"""

# The two shoulder links mount directly onto the torso. Their conservative
# capsules overlap the base by construction and cannot be used as collision
# constraints. Distal links must remain protected even when their fitted
# capsule overlaps at the straight-down home pose: their clearance relative
# to that known-safe pose still detects an arm folding into the base.
_SHOULDER_MOUNT_SUFFIXES = ("_s2", "_s3")

# Per-pair collision-cost activation distances are derived from each pair's
# clearance at the home pose: ``min(home_clearance - _MARGIN_REST_BUFFER,
# default_margin)``. Pairs that *live* near their activation shell (the
# wrists and grippers pass within 10-17 mm of the base during ordinary
# close-over-table work, and the elbow capsules graze at rest) would
# otherwise keep the cost hinge active across the entire work envelope —
# replaying recorded deburring sessions showed base<->wrist pairs inside a
# uniform 25 mm activation for up to 59% of all frames, and every crossing of
# the shell pulses the collision gradient into the arm, which reads as
# path-specific jitter. Deriving the margin from the home clearance keeps
# such pairs silent in their normal envelope while pairs with generous
# clearance keep the full early-warning distance.
#
# The activation may be *negative*: ``base <-> e2`` (the elbow) reads
# -1.4 mm at home because pyroki's conservative capsule fits already
# interpenetrate there while the physical parts hang in free air. pyroki's
# stock cost cannot activate below zero (its smoothing ramp spans [0,
# margin]), so this pair used to be clamped to a +8 mm activation — leaving
# its gradient permanently active at rest and across the whole
# gripper-in-front-of-torso envelope, where it measurably tripled the
# solver's per-tick output acceleration and kicked the elbow at every shell
# crossing. The custom residual in .solver decouples the activation start
# from the ramp width, so such pairs activate only when the pose actually
# gets closer than home, while keeping the full linear pushback deeper in.
_RAMP_WIDTH_MIN = 0.008
_MARGIN_REST_BUFFER = 0.002

_lock = threading.RLock()
_urdf: yourdfpy.URDF | None = None
_robot: pk.Robot | None = None
_robot_coll: pk.collision.RobotCollision | None = None


def _load_urdf() -> yourdfpy.URDF:
    global _urdf
    if _urdf is None:
        _logger.info("Loading Axol URDF...")
        _urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(URDF_PATH.parent))
    return _urdf


def shared_robot() -> pk.Robot:
    """The pyroki robot for the bundled Axol URDF, built once per process."""
    global _robot
    with _lock:
        if _robot is None:
            _robot = pk.Robot.from_urdf(_load_urdf())
        return _robot


def shared_robot_collision() -> pk.collision.RobotCollision:
    """The torso<->arm collision model, built once per process."""
    global _robot_coll
    with _lock:
        if _robot_coll is None:
            _robot_coll = _build_robot_collision(_load_urdf())
        return _robot_coll


def _build_robot_collision(urdf: yourdfpy.URDF) -> pk.collision.RobotCollision:
    """Build ``RobotCollision`` with self-collision restricted to torso<->arm pairs.

    Each Axol arm is a serial chain attached to a static torso (``base`` +
    ``s1``). pyroki's PCA capsule fit produces conservative single-capsule-
    per-link shapes that always overlap at adjacent-link joint interfaces,
    so blanket self-collision causes persistent jitter the IK cannot
    resolve. We restrict the active pair set to the only collisions that
    actually matter: any link pair where exactly one side is the torso
    and the other is an arm link. Within-arm, cross-arm, and torso<->torso
    pairs are filtered out (cross-arm contacts are unreachable, within-arm
    is constrained by joint limits, and torso<->torso is rigidly fixed).

    Only the shoulder-mount links (``s2``/``s3``) are excluded. The old model
    also discarded every distal pair whose conservative capsule happened to
    overlap at the safe straight-down home pose. That removed ``base/s1 <->
    e1`` — the upper-arm pair that physically contacted the base during a
    cross-body reach — as well as ``base <-> w2``. The custom residual uses a
    per-pair activation relative to home clearance, so those negative home
    distances are usable: the cost is off at home and activates only when the
    arm moves closer to the base than that known-safe reference.
    """
    link_names = [link.name for link in urdf.robot.links]

    def is_arm(n: str) -> bool:
        return n.startswith("left_") or n.startswith("right_")

    def is_torso(n: str) -> bool:
        return n in _TORSO_LINKS

    ignore: set[tuple[str, str]] = set()
    for i, a in enumerate(link_names):
        for b in link_names[i + 1 :]:
            keep = (is_torso(a) and is_arm(b)) or (is_torso(b) and is_arm(a))
            arm_link = b if is_torso(a) else a if is_torso(b) else ""
            if arm_link.endswith(_SHOULDER_MOUNT_SUFFIXES):
                keep = False
            if not keep:
                ignore.add((a, b))

    rc = pk.collision.RobotCollision.from_urdf(urdf, user_ignore_pairs=tuple(ignore))
    _logger.info(
        "RobotCollision: restricted to %d torso<->arm pairs.",
        len(rc.active_idx_i),
    )
    return rc


def collision_cost_params(
    robot: pk.Robot, rc: pk.collision.RobotCollision, default_margin: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair ``(activation_start, ramp_width)`` for the smoothed collision cost.

    ``activation_start = min(home_clearance - _MARGIN_REST_BUFFER,
    default_margin)`` for each active pair — possibly negative for pairs
    whose conservative capsules already interpenetrate at home (see the
    constants above). ``ramp_width`` is the distance over which the residual
    ramps smoothly from zero to its full linear slope: equal to the activation
    start where that start is generous, floored at ``_RAMP_WIDTH_MIN`` so
    near-shell pairs keep a protective gradient onset.

    Cheap (one collision-distance evaluation), so callers compute it per
    solver instance rather than caching it here: the result depends on the
    configured ``default_margin``.
    """
    q0 = jnp.zeros(robot.joints.num_actuated_joints)
    d = np.asarray(rc.compute_self_collision_distance(robot, q0))
    starts = np.minimum(d - _MARGIN_REST_BUFFER, default_margin).astype(np.float32)
    widths = np.maximum(starts, _RAMP_WIDTH_MIN).astype(np.float32)
    _logger.info(
        "Collision activation: %d of %d pairs below the default %.0f mm "
        "(%d activate only inside their home-pose clearance).",
        int((starts < default_margin).sum()),
        len(starts),
        1e3 * default_margin,
        int((starts <= 0.0).sum()),
    )
    return starts, widths
