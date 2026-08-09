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
            _robot_coll = _build_robot_collision(_load_urdf(), shared_robot())
        return _robot_coll


def _build_robot_collision(
    urdf: yourdfpy.URDF, robot: pk.Robot
) -> pk.collision.RobotCollision:
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

    A second pass excludes any remaining pair that is already penetrating
    at the home pose — those are over-conservative capsule fits the IK
    can never separate (e.g. ``base <-> shoulder`` capsules that overlap
    by construction because the arms mount onto the torso).
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
            if not keep:
                ignore.add((a, b))

    rc = pk.collision.RobotCollision.from_urdf(urdf, user_ignore_pairs=tuple(ignore))
    q0 = jnp.zeros(robot.joints.num_actuated_joints)
    d = np.asarray(rc.compute_self_collision_distance(robot, q0))
    ai = np.asarray(rc.active_idx_i)
    aj = np.asarray(rc.active_idx_j)
    for k in np.where(d < 0.0)[0]:
        ignore.add((rc.link_names[ai[k]], rc.link_names[aj[k]]))

    rc = pk.collision.RobotCollision.from_urdf(urdf, user_ignore_pairs=tuple(ignore))
    _logger.info(
        "RobotCollision: restricted to %d torso<->arm pairs.",
        len(rc.active_idx_i),
    )
    return rc
