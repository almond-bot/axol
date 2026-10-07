"""Load the bundled Axol Mink model with collision geometry and fixed frames.

MuJoCo's URDF importer needs compiler overrides for this model:

- ``strippath`` and ``meshdir`` resolve ``package://`` mesh basenames in
  the bundled ``meshes`` directory.
- ``fusestatic="false"`` retains fixed-joint gripper and TCP bodies that
  end-effector tasks need to address by name.
- ``discardvisual="true"`` removes duplicate visual geometry while
  retaining collision geometry for Mink collision constraints.

The bundled model uses a separate root-frame convention from Axol world
poses. Conversion belongs at the interface boundary; all solver geometry
and forward kinematics use this one model.
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

from .ik_config import PINNED_URDF

# Injected as the first child of <robot>: MuJoCo reads an embedded <mujoco>
# extension element from URDF files for compiler settings.
_COMPILER_TMPL = (
    '<mujoco><compiler strippath="true" meshdir="{meshdir}" '
    'fusestatic="false" discardvisual="true"/></mujoco>'
)


def load_mj_model(urdf_path: Path = PINNED_URDF) -> mujoco.MjModel:
    """Load an Axol URDF as a :class:`mujoco.MjModel` with fixed frames kept.

    Args:
        urdf_path: URDF file to load. Meshes resolve from the sibling
            ``meshes`` directory; the default is the bundled Mink model.

    Returns:
        A compiled model retaining named EE/TCP bodies and collision geometry.
        Visual-only duplicates are discarded.
    """
    text = urdf_path.read_text(encoding="utf-8")
    inject = _COMPILER_TMPL.format(meshdir=str(urdf_path.parent / "meshes"))
    patched, n = re.subn(r"(<robot\b[^>]*>)", r"\1" + inject, text, count=1)
    if n != 1:
        raise ValueError(f"{urdf_path} has no <robot> element to patch")
    return mujoco.MjModel.from_xml_string(patched)


def qpos_indices(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    """qpos addresses of the named (1-DOF) joints, in the given order."""
    return np.array(
        [model.jnt_qposadr[_joint_id(model, n)] for n in joint_names], dtype=int
    )


def dof_indices(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    """Velocity-space (dof) addresses of the named joints, in the given order."""
    return np.array(
        [model.jnt_dofadr[_joint_id(model, n)] for n in joint_names], dtype=int
    )


def body_geom_ids(model: mujoco.MjModel, body_names: list[str]) -> list[int]:
    """All geom ids attached to the named bodies (skips geom-less bodies)."""
    ids: list[int] = []
    for name in body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"body {name!r} not in model")
        start = model.body_geomadr[bid]
        ids.extend(range(start, start + model.body_geomnum[bid]))
    return ids


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"joint {name!r} not in model")
    return jid
