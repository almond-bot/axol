"""Shared MuJoCo model of the Axol URDF for the mink / dls IK backends.

Loads the bundled URDF into an :class:`mujoco.MjModel`, rewriting the
``package://`` mesh URIs so MuJoCo can resolve the STL collision meshes
(needed by mink's collision-avoidance limit and the bench's clearance
metrics). The gravity compensator keeps its own stripped-mesh loader; this
one is cached separately because the IK backends need the collision geoms.
"""

from __future__ import annotations

import functools
import logging
import re

import mujoco
import numpy as np

from ..constants import URDF_PATH
from .base import CANONICAL_JOINT_NAMES

_logger = logging.getLogger(__name__)

_TORSO_BODIES: tuple[str, ...] = ("base", "s1")
"""Static torso bodies; arm<->torso is the only self-collision that matters
(see :mod:`almond_axol.kinematics.pyroki_model`)."""


@functools.lru_cache(maxsize=2)
def load_mj_model(with_meshes: bool = True) -> mujoco.MjModel:
    """Load the Axol URDF into MuJoCo (cached per process).

    Args:
        with_meshes: Keep the STL collision meshes (required for collision
            avoidance). ``False`` strips all geoms for a lighter kinematics-only
            model.

    Returns:
        The compiled model. Joint (qpos/dof) order matches the canonical
        left-then-right ``ARM_JOINTS`` order, but callers should still index
        via :func:`canonical_qpos_indices` rather than assume it.
    """
    text = URDF_PATH.read_text()
    if with_meshes:
        mesh_dir = str((URDF_PATH.parent / "meshes").resolve())
        text = text.replace("package://assembly/meshes/", "")
        compiler = (
            f'<mujoco><compiler meshdir="{mesh_dir}" balanceinertia="true" '
            'discardvisual="true" fusestatic="false"/></mujoco>'
        )
    else:
        text = re.sub(r"<visual>.*?</visual>", "", text, flags=re.DOTALL)
        text = re.sub(r"<collision>.*?</collision>", "", text, flags=re.DOTALL)
        compiler = (
            '<mujoco><compiler balanceinertia="true" fusestatic="false"/></mujoco>'
        )
    text = re.sub(r"(<robot[^>]*>)", r"\1" + compiler, text, count=1)
    model = mujoco.MjModel.from_xml_string(text)
    _logger.info(
        "MuJoCo model loaded: %d joints, %d geoms (meshes=%s).",
        model.njnt,
        model.ngeom,
        with_meshes,
    )
    return model


def canonical_qpos_indices(model: mujoco.MjModel) -> np.ndarray:
    """qpos (== dof, all hinges) index for each canonical joint name."""
    idx = []
    for name in CANONICAL_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint {name!r} not found in the MuJoCo model")
        idx.append(int(model.jnt_qposadr[jid]))
    return np.array(idx, dtype=np.int64)


def body_id(model: mujoco.MjModel, name: str) -> int:
    """Body id by name, raising if absent."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"Body {name!r} not found in the MuJoCo model")
    return bid


def arm_torso_geom_pairs(
    model: mujoco.MjModel,
    margin: float,
    # Fixed home-pose clearance a pair needs to stay active — deliberately
    # NOT derived from the margin, so the pair set is margin-independent
    # (mirrors _PRUNE_THRESHOLD in the pink backend: 0.025 sits between the
    # shoulder-cluster clearance ~22 mm and the forearm rest clearance
    # ~27 mm).
    threshold: float = 0.025,
) -> list[tuple[list[int], list[int]]]:
    """Geom pairs for arm<->torso collision avoidance, pruned for feasibility.

    Mirrors the pyroki collision model's restriction: only arm<->torso pairs
    are considered (cross-arm contacts are unreachable, within-arm is
    constrained by joint limits). The shoulder clusters (``*_s2``/``*_s3``)
    are excluded outright — they orbit the base column inside any useful
    margin across the whole joint range, so a hard minimum-distance
    constraint on them wedges permanently (see ``_SHOULDER_SUFFIXES`` in the
    pink backend). Any remaining pair whose separation at the home pose is
    already below ``threshold`` is also dropped as close by construction.

    Args:
        model: MuJoCo model with collision geoms.
        margin: The minimum-distance value the caller will enforce (m);
            logged only — it does not affect the pair set.
        threshold: Home-pose clearance a pair must have to stay active (m).

    Returns:
        A list of single-pair ``([arm_geom], [torso_geom])`` tuples.
    """
    torso_ids = {body_id(model, n) for n in _TORSO_BODIES}
    torso_geoms: list[int] = []
    arm_geoms: list[int] = []
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if b in torso_ids:
            torso_geoms.append(g)
        elif bname.startswith(("left_", "right_")) and not bname.endswith(
            ("_s2", "_s3")
        ):
            arm_geoms.append(g)

    pairs: list[tuple[list[int], list[int]]] = []
    pruned = 0
    fromto = np.empty(6)
    for g in arm_geoms:
        for t in torso_geoms:
            d = mujoco.mj_geomDistance(model, data, g, t, threshold + 0.1, fromto)
            if d > threshold:
                pairs.append(([g], [t]))
            else:
                pruned += 1
    _logger.info(
        "Collision pairs: %d active arm<->torso pairs (%d pruned at home, "
        "margin %.0f mm).",
        len(pairs),
        pruned,
        margin * 1e3,
    )
    return pairs
