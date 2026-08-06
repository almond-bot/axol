"""Shared, cached pyroki robot + collision model for the Axol URDF.

One pyroki :class:`~pyroki.Robot` / :class:`~pyroki.collision.RobotCollision`
pair per process, shared by the pyroki IK backends and the collision-aware
reset-trajectory planner so the (expensive) URDF parse and capsule fit happen
once regardless of which IK backend is selected.
"""

from __future__ import annotations

import functools
import logging

import jax.numpy as jnp
import numpy as np
import pyroki as pk
import yourdfpy

from ..constants import URDF_PATH
from .base import CANONICAL_JOINT_NAMES

_logger = logging.getLogger(__name__)


_TORSO_LINKS: tuple[str, ...] = ("base", "s1")
"""Static body links that the arms must not collide into.

Self-collision on Axol is restricted to ``arm <-> torso`` pairs only.
"""


def build_robot_collision(
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


@functools.lru_cache(maxsize=1)
def load_pyroki_model() -> tuple[yourdfpy.URDF, pk.Robot, pk.collision.RobotCollision]:
    """Load the bundled Axol URDF into pyroki once per process (cached)."""
    _logger.info("Loading Axol URDF into pyroki...")
    urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(URDF_PATH.parent))
    robot = pk.Robot.from_urdf(urdf)
    robot_coll = build_robot_collision(urdf, robot)
    return urdf, robot, robot_coll


def canonical_to_pyroki(robot: pk.Robot) -> np.ndarray:
    """Index array mapping canonical joint order into pyroki's actuated order.

    ``q_pyroki[perm] = q_canonical`` and ``q_canonical = q_pyroki[perm]``
    do NOT both hold; use the returned ``perm`` as
    ``q_pyroki = q_canonical[inverse]`` via the helpers below instead.

    Returns ``perm`` such that ``q_canonical[i] == q_pyroki[perm[i]]``.
    """
    actuated = list(robot.joints.actuated_names)
    return np.array([actuated.index(n) for n in CANONICAL_JOINT_NAMES], dtype=np.int64)


def to_pyroki_order(q_canonical: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Reorder a canonical ``(14,)`` joint vector into pyroki's actuated order."""
    q_pyroki = np.empty_like(q_canonical)
    q_pyroki[perm] = q_canonical
    return q_pyroki


def to_canonical_order(q_pyroki: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Reorder a pyroki-ordered ``(14,)`` joint vector into canonical order."""
    return np.asarray(q_pyroki)[perm]
