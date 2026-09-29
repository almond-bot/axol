"""
Build the bundled Axol-on-Jelly URDFs from their Onshape export.

The Onshape export (``almond_urdf/urdf/almond_urdf.urdf`` + ``meshes/``) is a
faithful CAD tree — ~140 links including every screw, a two-stage prismatic
lift, prismatic gripper fingers, four wheel joints, and link frames wherever
Onshape put them. None of that matches what the control stack expects, so
this script distils it into three URDFs that share one set of meshes
(``almond_axol/kinematics/urdf/meshes/jelly/``):

- ``axol_jelly.urdf`` — **arm IK** (the default): only the 14 arm joints move;
  base, lift, fingers and wheels are frozen (lift fully raised), rooted at
  the classic torso-relative world frame.
- ``axol_jelly_whole_body.urdf`` — **whole-body IK**: the same torso and arms
  on the Jelly's planar base (``base_x`` / ``base_y`` / ``base_yaw`` about the
  Jelly centre) and its lift (``lift``, the second stage mimicking it; 0 =
  fully raised, positive lowers each stage). Its world frame is the arm
  model's with every body joint at zero, so the two agree at startup.
- ``axol_jelly_sim.urdf`` — **simulation** (e.g. Isaac Sim): a physical
  robot rooted at the Jelly on the floor, with the lift, coupled gripper
  fingers (``left_finger_1`` + mimic) and spinning wheels as joints, masses
  and CoMs from :class:`~almond_axol.robot.config.AxolConfig`, contact
  geometry per part, and relative mesh paths. See ``SIM_NOTE``.

All three:

- **Arms are the classic arms.** Every arm link, joint name, joint frame, axis
  and inertial is copied verbatim from ``axol.urdf``; only the joint *limits*
  come from the export. The script first proves the two arm chains are the
  same mechanism (it fits the rigid transform between the exports' joint
  axes, checks every arm link moves identically under random joint angles,
  and reports the per-joint sign flips between the two naming schemes), so
  FK, IK targets, recorded Cartesian frames and gravity compensation are
  unchanged between the versions.
- **Meshes are the export's**, re-expressed in the bundled link frames: each
  rigid group of export links (a link plus its fastened children) is merged
  per colour into one decimated visual STL, and convex hulls for collision.
  Fasteners and connector bits under ``--min-part-mm`` are dropped.
- **Camera optical frames** for both wrist cameras and both head-camera
  lenses (``*_optical``: +z along the optical axis, +x image right, +y image
  down), from the CAD lens geometry — see :func:`camera_frames`.

The IK models split the body for capsule collision (pyroki fits one capsule
per link): ``s1`` (shoulder bar), ``lift_plate`` (the bracket it bolts onto),
``head`` (head-camera mount) and the lift column — one link (``base``) in the
arm model, where the Jelly deck is visual only (``ARM_NOTE``), and one per
stage (``base``, ``lift_stage``, ``jelly``) plus deck strips in the
whole-body model. Their fingers collide as one cylinder spanning the full
stroke.

Run (``fast-simplification`` is only needed here, not at runtime)::

    uv run --with fast-simplification python scripts/build_jelly_urdfs.py \\
        path/to/almond_urdf
"""

from __future__ import annotations

import argparse
import itertools
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import trimesh
import yourdfpy

REPO = Path(__file__).resolve().parents[1]
URDF_DIR = REPO / "almond_axol" / "kinematics" / "urdf"
CLASSIC_URDF = URDF_DIR / "axol.urdf"
OUT_ARM = URDF_DIR / "axol_jelly.urdf"
OUT_WHOLE_BODY = URDF_DIR / "axol_jelly_whole_body.urdf"
OUT_SIM = URDF_DIR / "axol_jelly_sim.urdf"
OUT_SIM_DRIVES = URDF_DIR / "axol_jelly_sim_drives.json"
MESH_SUBDIR = "meshes/jelly"
PACKAGE = "package://assembly/"
# Travel of each planar base joint in the whole-body model (m): far beyond
# any single session, so the joint limits never shape the solve.
BASE_TRAVEL = 10.0

# Classic arm joints (ARM_JOINTS order) and their export counterparts.
CLASSIC_JOINTS = ("s1_0", "s2_0", "s3_0", "e1_0", "e2_0", "w1_0", "w2_0")
EXPORT_JOINTS = ("s1", "s2", "s3", "e1", "w1", "w2", "w3")

# Export link -> bundled link. Every other export link joins the group of its
# nearest listed ancestor.
ANCHORS: dict[str, str] = {
    "jelly_base": "jelly",
    "jelly_stage_2_1": "lift_stage",
    "jelly_stage_3": "base",
    "s1": "s1",
    "part_1": "head",
    "s2_left": "left_s2",
    "s3_left": "left_s3",
    "e1_left": "left_e1",
    "e2_left": "left_e2",
    "w0_left": "left_w0",
    "w1_left": "left_w1",
    "w2_1": "left_w2",
    "gripper_base": "left_gripper",
    "gripper_tip_1": "left_finger_1",
    "gripper_tip": "left_finger_2",
    "wrist_mount": "left_wrist_camera",
    "s2_right": "right_s2",
    "s3_right": "right_s3",
    "e1_right": "right_e1",
    "e2_right": "right_e2",
    "w0_right": "right_w0",
    "w1_right": "right_w1",
    "w2": "right_w2",
    "gripper_base_1": "right_gripper",
    "gripper_tip_3": "right_finger_1",
    "gripper_tip_2": "right_finger_2",
    "wrist_mount_1": "right_wrist_camera",
    "wheel": "wheel_1_link",
    "wheel_2": "wheel_2_link",
    "wheel_3": "wheel_3_link",
    "wheel_1": "wheel_4_link",
}

# Bundled links that exist only in the Jelly models, and the classic link each
# is rigidly attached to with an identity origin (their meshes are placed in
# the parent's frame).
EXTRA_LINKS: dict[str, str] = {
    "head": "s1",
    "lift_plate": "base",
    # Split off the gripper so each part gets its own (much tighter) capsule.
    "left_wrist_camera": "left_gripper",
    "right_wrist_camera": "right_gripper",
    # IK models only: collision-only link holding the finger cylinder.
    "left_fingers": "left_gripper",
    "right_fingers": "right_gripper",
}

# The Jelly body links: frames world-aligned at the ground under the Jelly
# centre (the export root), in every model.
BODY_FRAME_LINKS = ("jelly", "lift_stage")


@dataclass(frozen=True)
class Moving:
    """A bundled link that moves in the export: its frame and joint come from it."""

    export_link: str
    export_joint: str
    parent: str
    mimic: str | None = None


# Fingers and wheels keep the export's link frames (their joint frames), so
# the sim model can move them along the export's own axes.
MOVING: dict[str, Moving] = {
    "left_finger_1": Moving("gripper_tip_1", "gripper_tip_1", "left_gripper"),
    "left_finger_2": Moving(
        "gripper_tip", "gripper_tip_2", "left_gripper", "left_finger_1"
    ),
    "right_finger_1": Moving("gripper_tip_3", "gripper_tip_1_1", "right_gripper"),
    "right_finger_2": Moving(
        "gripper_tip_2", "gripper_tip_2_1", "right_gripper", "right_finger_1"
    ),
    "wheel_1_link": Moving("wheel", "wheel_1", "jelly"),
    "wheel_2_link": Moving("wheel_2", "wheel_2", "jelly"),
    "wheel_3_link": Moving("wheel_3", "wheel_3", "jelly"),
    "wheel_4_link": Moving("wheel_1", "wheel_4", "jelly"),
}

# Collision pieces: each export mesh feeds the piece named after its group,
# unless split here by a mesh-local box ``(lo, hi)`` into the piece inside
# and the piece outside it. Each model builds its collision bodies from
# pieces (:func:`collision_plan`).
# - The top of the last lift stage is the bracket the shoulder bar bolts onto;
#   folded into the column's hull it would inflate the column capsule from
#   ~85 mm to ~120 mm, so it is its own piece (and link).
# - The Jelly base mesh includes the fixed bottom section of the lift column,
#   which is within reach at gripper height, separate from the deck.
_INF = np.inf
_ALL = ((-_INF,) * 3, (_INF,) * 3)
PIECE_SPLITS: dict[str, tuple[tuple[float, ...], tuple[float, ...], str, str]] = {
    "Jelly_Stage_3.stl": (
        (-_INF, -_INF, 1.22),
        (_INF,) * 3,
        "lift_plate",
        "column_top",
    ),
    "Jelly_Base.stl": (
        (-0.041, -0.073, 0.2),
        (0.041, 0.073, _INF),
        "column_stub",
        "deck",
    ),
    "Jelly_Stage_2.stl": (*_ALL, "column_mid", ""),
}

# Collision body specs: a hull over pieces, a cylinder along a link axis over
# pieces (``stroke``: include the fingers fully open), or deck strips.
_ARM_PARTS = ("s2", "s3", "e1", "e2", "w0", "w1", "w2", "gripper", "wrist_camera")
COMMON_HULLS = (
    *(f"{side}_{part}" for side in ("left", "right") for part in _ARM_PARTS),
    "s1",
    "head",
    "lift_plate",
)
# The deck (0.6 x 0.6 m, ~0.24 m tall) as strips across the robot for capsule
# collision: one capsule over the whole deck would swallow the arms; three
# along y stay within a few cm of it.
DECK_STRIPS = 3

# Visual meshes the wrist cameras see up close (5-10 cm): simplified to the
# tight tolerance so renders from the wrist match real images.
CAMERA_VIEWED = ("_gripper", "_finger_1", "_finger_2")

_CYLINDER_RPY = {0: "0 1.5707963267948966 0", 1: "-1.5707963267948966 0 0", 2: "0 0 0"}

# Frame at the ground under the robot (the export's root), in the IK models.
FLOOR_LINK = "floor"

ARM_NOTE = (
    "Arm IK model: the base, lift, fingers and wheels are frozen, the lift "
    "fully raised (its CAD pose). At that height the Jelly deck sits below "
    "anything the arms can reach, so it is visual only; the arms collide with "
    "the shoulder bar (s1), the lift column (base), its top plate (lift_plate) "
    "and the head-camera mount (head), exactly as the classic arms collide "
    "with the classic base and s1."
)
WHOLE_BODY_NOTE = (
    "Whole-body IK model: the Jelly base moves in the plane (base_x, base_y, "
    "base_yaw about the Jelly centre) and the lift lowers the torso (lift, "
    "0 = fully raised, per stage; the second stage mimics it). The world "
    "frame is the arm model's with every body joint at zero. The arms "
    "collide with the shoulder bar, top plate, head camera, each lift-column "
    "section and the deck (deck_0..2)."
)
SIM_NOTE = (
    "Simulation model: rooted at the Jelly on the floor (link jelly, at the "
    "Jelly centre). Moving: the 14 arm joints, the lift (lift, lift_2 "
    "mimics), each gripper's fingers (left_finger_1 / right_finger_1, the "
    "second finger mimics; 0 = closed, 0.0588 m = open) and the four wheels "
    "(continuous). Arm joints carry AxolConfig's measured friction as "
    "<dynamics friction> (Coulomb, N·m) and damping (viscous, N·m·s/rad), "
    "per arm; the motors' impedance gains (kp, kd), which URDF cannot hold, "
    "are in axol_jelly_sim_drives.json next to this file. The x-drive's "
    "omni rollers are not modelled: plain wheels "
    "cannot strafe, so drive the base kinematically or with a holonomic "
    "controller. Arm masses and CoMs are AxolConfig's tuned gravity-comp "
    "values (the wrist-3 body is split so the gripper, fingers and wrist "
    "camera carry their share while the assembly's total mass and CoM stay "
    "as tuned); every other mass is an ESTIMATE (SIM_MASSES in the "
    "generator) - replace with measured values. Inertia tensors assume "
    "uniform density over each part's convex hull. Effort/velocity: arm "
    "joints keep axol.urdf's values; the rest are estimates. Camera frames "
    "(*_optical) use the ROS optical convention (+z forward, +x right, +y "
    "down); an Isaac Sim camera attached with ROS camera axes needs no extra "
    "rotation. Mesh paths are relative to this file."
)

WRIST_CAMERA_NOTE = (
    "Wrist camera frames (*_wrist_camera_optical, ZED X One S fisheye): the "
    "origin is Stereolabs' fisheye-lens reference frame from the camera's "
    "CAD, 18.9 mm ahead of the body's seal ring on the lens axis; fitting "
    "real wrist images through each camera's factory calibration put the "
    "projection centre within 0.6 mm of it on both arms. Roll puts the "
    "gripper at the bottom of the image, as in real frames."
)

# Masses (kg) of the non-arm links in the sim model: ESTIMATES, not measured.
# The wrist-3 entries are carved out of AxolConfig's wrist_3 mass (which
# lumps the whole gripper assembly), so they redistribute rather than add.
SIM_MASSES: dict[str, float] = {
    "jelly": 40.0,  # drive base + two 24 V 50 Ah LiFePO4 packs
    "lift_stage": 3.0,
    "base": 4.0,  # top lift stage
    "lift_plate": 1.0,
    "s1": 3.0,
    "head": 0.3,
    "wheel_1_link": 1.0,
    "wheel_2_link": 1.0,
    "wheel_3_link": 1.0,
    "wheel_4_link": 1.0,
}
SIM_WRIST_PARTS: dict[str, float] = {
    "gripper": 0.25,
    "finger_1": 0.04,
    "finger_2": 0.04,
    "wrist_camera": 0.08,
}
# Joint effort (N·m or N) / velocity (rad/s or m/s) in the sim model for the
# joints axol.urdf does not define: ESTIMATES.
SIM_LIMITS: dict[str, tuple[float, float]] = {
    "lift": (1000.0, 0.03),
    "finger": (50.0, 0.1),
    "wheel": (20.0, 30.0),
}


def _load(path: Path) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        str(path), load_meshes=False, build_collision_scene_graph=False
    )


def _zero(urdf: yourdfpy.URDF, **overrides: float) -> None:
    cfg = {n: 0.0 for n in urdf.actuated_joint_names}
    cfg.update(overrides)
    urdf.update_cfg(cfg)


def _names(side: str) -> tuple[list[str], list[str]]:
    long = "left" if side == "l" else "right"
    return [f"{long}_{j}" for j in CLASSIC_JOINTS], [
        f"{j}_{side}" for j in EXPORT_JOINTS
    ]


def _axis_line(urdf: yourdfpy.URDF, joint: str) -> tuple[np.ndarray, np.ndarray]:
    j = urdf.joint_map[joint]
    tf = urdf.get_transform(j.parent, urdf.base_link) @ j.origin
    return tf[:3, 3], tf[:3, :3] @ np.asarray(j.axis, dtype=float)


def align(
    classic: yourdfpy.URDF, export: yourdfpy.URDF
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid transform export-root -> classic-root, and per-joint sign flips.

    Fits the 14 arm joint axes as lines: rotation by Kabsch over the (sign
    resolved) axis directions, translation by least squares on the
    perpendicular offsets between corresponding lines.
    """
    _zero(classic)
    _zero(export)
    lines = []
    for side in "lr":
        c_names, e_names = _names(side)
        for c, e in zip(c_names, e_names):
            lines.append((*_axis_line(classic, c), *_axis_line(export, e)))
    d_c = np.array([ln[1] for ln in lines])
    d_e = np.array([ln[3] for ln in lines])
    best = None
    for signs in itertools.product((1.0, -1.0), repeat=len(CLASSIC_JOINTS)):
        sg = np.array(signs * 2)
        u, _, vt = np.linalg.svd((d_e * sg[:, None]).T @ d_c)
        rot = (u @ vt).T
        if np.linalg.det(rot) < 0:
            continue
        err = np.linalg.norm((rot @ (d_e * sg[:, None]).T).T - d_c)
        if best is None or err < best[0]:
            best = (err, sg, rot)
    assert best is not None
    err, sg, rot = best
    if err > 1e-3:
        raise SystemExit(
            f"arm joint axes do not match the classic arms (err {err:.2e})"
        )
    rows, rhs = [], []
    for p_c, d, p_e, _ in lines:
        m = np.eye(3) - np.outer(d, d)
        rows.append(m)
        rhs.append(m @ (p_c - rot @ p_e))
    t = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
    tf = np.eye(4)
    tf[:3, :3] = rot
    tf[:3, 3] = t
    worst = max(
        np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (p_c - (rot @ p_e + t)))
        for p_c, d, p_e, _ in lines
    )
    print(
        f"export -> classic root: t={np.round(t, 5)}, worst axis offset {worst * 1e3:.2f} mm"
    )
    if worst > 3e-3:
        raise SystemExit("arm joint axes are more than 3 mm from the classic arms")
    if not np.allclose(rot, np.eye(3), atol=1e-3):
        raise SystemExit("export root is not level with the classic world frame")
    # The export's root is level with the classic frame: keep only the offset
    # (the fit's residual rotation is CAD noise).
    tf[:3, :3] = np.eye(3)
    return tf, sg


def check_equivalent(
    classic: yourdfpy.URDF, export: yourdfpy.URDF, tf: np.ndarray, sg: np.ndarray
) -> None:
    """Every arm link must move identically in both models (within CAD slop)."""
    rng = np.random.default_rng(0)
    for side, off in (("l", 0), ("r", 7)):
        c_names, e_names = _names(side)
        prefix = "left_" if side == "l" else "right_"
        pairs = [
            (bundled, exported)
            for exported, bundled in ANCHORS.items()
            if bundled.startswith(prefix)
            and bundled not in EXTRA_LINKS
            and bundled not in MOVING
            and not bundled.endswith("_gripper")
        ]
        ref = None
        worst = 0.0
        for k in range(25):
            q = np.zeros(7) if k == 0 else rng.uniform(-1.5, 1.5, 7)
            _zero(classic, **dict(zip(c_names, q)))
            _zero(export, **dict(zip(e_names, sg[off : off + 7] * q)))
            xs = [
                np.linalg.inv(classic.get_transform(c, classic.base_link))
                @ tf
                @ export.get_transform(e, export.base_link)
                for c, e in pairs
            ]
            if ref is None:
                ref = xs
            worst = max(worst, max(float(np.abs(x - r).max()) for x, r in zip(xs, ref)))
        print(
            f"{side} arm: worst link deviation over random poses {worst * 1e3:.2f} mm"
        )
        if worst > 5e-3:
            raise SystemExit(f"{side} arm kinematics differ from the classic arm")
    _zero(classic)
    _zero(export)


def check_lift(export: yourdfpy.URDF) -> float:
    """The export's two lift stages: same travel, both lowering along -z."""
    stages = [export.joint_map[n] for n in ("stage_1", "stage_2")]
    travel = {round(float(j.limit.upper), 6) for j in stages}
    if len(travel) != 1 or any(abs(float(j.limit.lower)) > 1e-6 for j in stages):
        raise SystemExit(f"lift stages differ: {travel}")
    _zero(export)
    for j in stages:
        axis = (
            export.get_transform(j.parent, export.base_link)[:3, :3]
            @ j.origin[:3, :3]
            @ j.axis
        )
        if not np.allclose(axis, [0, 0, -1], atol=1e-6):
            raise SystemExit(f"{j.name} does not lower along -z: {axis}")
    print(f"lift: two stages of {travel.pop():.3f} m each, 0 = fully raised")
    return float(stages[0].limit.upper)


def export_limits(
    export: yourdfpy.URDF, sg: np.ndarray
) -> dict[str, tuple[float, float]]:
    out = {}
    for side, off in (("l", 0), ("r", 7)):
        c_names, e_names = _names(side)
        for k, (c, e) in enumerate(zip(c_names, e_names)):
            lim = export.joint_map[e].limit
            lo, hi = float(lim.lower), float(lim.upper)
            if sg[off + k] < 0:
                lo, hi = -hi, -lo
            out[c] = (round(lo, 4), round(hi, 4))
    return out


def groups(export: yourdfpy.URDF) -> dict[str, str]:
    """Export link -> bundled link (nearest anchored ancestor)."""
    parent = {j.child: j.parent for j in export.robot.joints}
    out = {}
    for link in export.link_map:
        cur = link
        while cur not in ANCHORS and cur in parent:
            cur = parent[cur]
        if cur in ANCHORS:
            out[link] = ANCHORS[cur]
    return out


def _mesh(export_dir: Path, filename: str) -> trimesh.Trimesh:
    rel = re.sub(r"^package://[^/]+/", "", filename)
    return trimesh.load(export_dir / rel, force="mesh")


def _cluster(mesh: trimesh.Trimesh, cell: float) -> trimesh.Trimesh:
    """Vertex clustering on a ``cell``-sized grid (drops collapsed faces)."""
    keys = np.round(mesh.vertices / cell).astype(np.int64)
    _, index, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    faces = inverse.reshape(-1)[mesh.faces]
    ok = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    out = trimesh.Trimesh(mesh.vertices[index], faces[ok], process=False)
    out.update_faces(out.unique_faces())
    return out


def _simplify(mesh: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    """Quadric decimation, falling back to grid clustering where it stalls."""
    import fast_simplification

    if len(mesh.faces) <= target:
        return mesh
    v, f = fast_simplification.simplify(
        mesh.vertices, mesh.faces, 1.0 - target / len(mesh.faces)
    )
    out = trimesh.Trimesh(v, f)
    cell = 5e-4
    while len(out.faces) > target * 1.2:
        out = _cluster(out, cell)
        cell *= 1.5
    return out


def _deviation(a: trimesh.Trimesh, b: trimesh.Trimesh, n: int = 4000) -> float:
    """Symmetric surface deviation (m, 99th percentile of sampled distances)."""
    pa = trimesh.sample.sample_surface(a, n, seed=0)[0]
    pb = trimesh.sample.sample_surface(b, n, seed=1)[0]
    d = np.r_[
        trimesh.proximity.closest_point(b, pa)[1],
        trimesh.proximity.closest_point(a, pb)[1],
    ]
    return float(np.percentile(d, 99))


def _decimate(
    mesh: trimesh.Trimesh, target: int, tolerance: float
) -> tuple[trimesh.Trimesh, float]:
    """The smallest simplification within ``tolerance`` of the CAD surface.

    Simplifying CAD parts is lossy in ways that matter at camera range (the
    wrist camera sees the gripper from 5-10 cm): an unchecked pass once left
    the gripper body 7 mm short and up to 27 mm off the CAD surface. The face
    budget doubles until the result is within ``tolerance``; parts that never
    get there keep their full CAD mesh.
    """
    budget = target
    while budget < len(mesh.faces):
        candidate = _simplify(mesh, budget)
        error = _deviation(mesh, candidate)
        if error <= tolerance:
            return candidate, error
        budget *= 2
    return mesh, 0.0


class Frames:
    """Bundled link frames in the classic root frame, every joint at zero."""

    def __init__(
        self, classic: yourdfpy.URDF, export: yourdfpy.URDF, tf: np.ndarray
    ) -> None:
        _zero(classic)
        _zero(export)
        self.classic, self.export, self.tf = classic, export, tf
        self.jelly_center = tf[:3, 3].copy()

    def __call__(self, link: str) -> np.ndarray:
        if link in BODY_FRAME_LINKS or link.startswith("deck_"):
            frame = np.eye(4)
            frame[:3, 3] = self.jelly_center
            return frame
        if link in MOVING:
            return self.tf @ self.export.get_transform(
                MOVING[link].export_link, self.export.base_link
            )
        return self.classic.get_transform(
            EXTRA_LINKS.get(link, link), self.classic.base_link
        )

    def relative(self, parent: str, child: str) -> np.ndarray:
        return np.linalg.inv(self(parent)) @ self(child)


def build_meshes(
    export: yourdfpy.URDF,
    export_dir: Path,
    frames: Frames,
    min_part: float,
    visual_faces: int,
    tolerance: float,
    loose_tolerance: float,
) -> tuple[
    dict[str, list[tuple[str, tuple[float, ...]]]],
    dict[str, tuple[str, list[np.ndarray]]],
    dict[str, np.ndarray],
]:
    """Write the shared visual meshes.

    Returns them, the collision pieces (``{name: (link, [points in that
    link's frame])}``), and every export link's vertices in the classic root
    frame (for the camera frames).
    """
    owner = groups(export)
    by_color: dict[tuple[str, tuple[float, ...]], list[trimesh.Trimesh]] = defaultdict(
        list
    )
    pieces: dict[str, tuple[str, list[np.ndarray]]] = {}
    in_root: dict[str, list[np.ndarray]] = defaultdict(list)

    def add(piece: str, link: str, pts: np.ndarray) -> None:
        if piece:
            pieces.setdefault(piece, (link, []))[1].append(pts)

    for link_name, link in export.link_map.items():
        target = owner.get(link_name)
        if target is None:
            continue
        in_target = np.linalg.inv(frames(target)) @ frames.tf
        for vis in link.visuals:
            mesh = _mesh(export_dir, vis.geometry.mesh.filename)
            origin = vis.origin if vis.origin is not None else np.eye(4)
            pose = export.get_transform(link_name, export.base_link)
            placed = mesh.copy().apply_transform(in_target @ pose @ origin)
            in_root[link_name].append(
                trimesh.transform_points(mesh.vertices, frames.tf @ pose @ origin)
            )
            split = PIECE_SPLITS.get(Path(vis.geometry.mesh.filename).name)
            if split is None:
                add(target, target, placed.vertices)
            else:
                lo, hi, inside_piece, outside_piece = split
                inside = np.all((mesh.vertices >= lo) & (mesh.vertices <= hi), axis=1)
                add(inside_piece, target, placed.vertices[inside])
                add(outside_piece, target, placed.vertices[~inside])
            if np.linalg.norm(mesh.extents) * 1e3 < min_part:
                continue
            rgba = (0.6, 0.6, 0.6, 1.0)
            if vis.material is not None and vis.material.color is not None:
                rgba = tuple(round(float(c), 4) for c in vis.material.color.rgba)
            by_color[(target, rgba)].append(placed)

    (URDF_DIR / MESH_SUBDIR).mkdir(parents=True, exist_ok=True)
    for old in (URDF_DIR / MESH_SUBDIR).glob("*.stl"):
        old.unlink()
    visuals: dict[str, list[tuple[str, tuple[float, ...]]]] = defaultdict(list)
    counters: dict[str, int] = defaultdict(int)
    worst = tight = (0.0, "")
    for (target, rgba), meshes in sorted(by_color.items()):
        merged = trimesh.util.concatenate(meshes)
        merged.merge_vertices()
        tol = tolerance if target.endswith(CAMERA_VIEWED) else loose_tolerance
        merged, error = _decimate(merged, visual_faces, tol)
        worst = max(worst, (error, target))
        if target.endswith(CAMERA_VIEWED):
            tight = max(tight, (error, target))
        name = f"{target}_{counters[target]}.stl"
        counters[target] += 1
        merged.export(URDF_DIR / MESH_SUBDIR / name)
        visuals[target].append((f"{MESH_SUBDIR}/{name}", rgba))
    print(
        f"visual meshes: worst deviation {worst[0] * 1e3:.2f} mm ({worst[1]}); "
        f"camera-viewed {tight[0] * 1e3:.2f} mm ({tight[1]})"
    )
    return visuals, pieces, {k: np.vstack(v) for k, v in in_root.items()}


# -- camera frames -----------------------------------------------------------


def _frame(
    origin: np.ndarray,
    z: np.ndarray,
    x_hint: np.ndarray | None = None,
    y_hint: np.ndarray | None = None,
) -> np.ndarray:
    """Right-handed frame at ``origin`` with +z along ``z`` and +x ~ ``x_hint``
    (or +y ~ ``y_hint``)."""
    z = z / np.linalg.norm(z)
    frame = np.eye(4)
    if y_hint is not None:
        y = y_hint - (y_hint @ z) * z
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
    else:
        assert x_hint is not None
        x = x_hint - (x_hint @ z) * z
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
    frame[:3, 0], frame[:3, 1], frame[:3, 2], frame[:3, 3] = x, y, z, origin
    return frame


def _thin_normal(pts: np.ndarray) -> np.ndarray:
    centred = pts - pts.mean(axis=0)
    return np.linalg.eigh(centred.T @ centred)[1][:, 0]


def camera_frames(
    in_root: dict[str, np.ndarray], frames: Frames
) -> dict[str, tuple[str, np.ndarray]]:
    """Optical frames (ROS convention: +z out of the lens, +x image right,
    +y image down), as ``{link: (parent, pose in parent)}``.

    - Wrist ZED X One S (mono fisheye): at Stereolabs' fisheye-lens
      reference frame from the camera's CAD, which sits on the lens axis
      18.9 mm ahead of the body's seal ring; +z along the ring's normal away
      from the camera body. Fitting real wrist images (both arms, each
      camera's factory calibration) put the projection centre within
      0.6 mm of it. The sensor's roll
      cannot be read from the CAD (the body is square); it is set from real
      wrist images, which show the gripper at the bottom of the frame with
      the fingers pointing up into the scene: +y (image down) points from the
      lens toward that gripper's fingers.
    - Head ZED X Mini (stereo): one frame per lens at the centre of its front
      cover, +z out of the camera, +x from the left lens to the right one
      (the ZED convention: the right camera sits at +x of the left). The two
      covers are checked to be the 50 mm baseline apart.
    """
    out: dict[str, tuple[str, np.ndarray]] = {}
    for seal, lens, rear, fingers, lens_frame in (
        (
            "camera_component___seal___zed_x_one",
            "fisheye_lens",
            "camera_component___body_zed_x_one___rear",
            ("gripper_tip", "gripper_tip_1"),
            "fisheye_lens__1_",
        ),
        (
            "camera_component___seal___zed_x_one_1",
            "fisheye_lens_1",
            "camera_component___body_zed_x_one___rear_1",
            ("gripper_tip_2", "gripper_tip_3"),
            "fisheye_lens__1__1",
        ),
    ):
        ring = in_root[seal]
        centre = (ring.min(axis=0) + ring.max(axis=0)) / 2
        z = _thin_normal(ring)
        outward = in_root[lens].mean(axis=0) - in_root[rear].mean(axis=0)
        if z @ outward < 0:
            z = -z
        side = "left" if centre[1] > 0 else "right"
        parent = f"{side}_wrist_camera"
        # The optical centre: Stereolabs' own fisheye-lens reference frame in
        # the camera's CAD, on the ring's axis ahead of it. Fitting real wrist
        # images through each camera's factory calibration put the
        # projection centre +19.0-19.5 mm ahead of the ring on both arms;
        # this frame is +18.9 mm.
        lens_origin = (
            frames.tf @ frames.export.get_transform(lens_frame, frames.export.base_link)
        )[:3, 3]
        ahead = float((lens_origin - centre) @ z)
        off_axis = float(np.linalg.norm((lens_origin - centre) - ahead * z))
        if not (0.010 < ahead < 0.030) or off_axis > 5e-4:
            raise SystemExit(
                f"{lens_frame} is not on the lens axis ahead of the seal ring "
                f"({ahead * 1e3:.1f} mm ahead, {off_axis * 1e3:.2f} mm off axis)"
            )
        tips = np.vstack([in_root[f] for f in fingers]).mean(axis=0)
        world = _frame(lens_origin, z, y_hint=tips - centre)
        out[f"{side}_wrist_camera_optical"] = (
            parent,
            np.linalg.inv(frames(parent)) @ world,
        )
    covers = [
        in_root["_zedx_mini_rev2__plastron_v2"],
        in_root["_zedx_mini_rev2__plastron_v2_2"],
    ]
    centres = [(c.min(axis=0) + c.max(axis=0)) / 2 for c in covers]
    baseline = float(np.linalg.norm(centres[0] - centres[1]))
    if abs(baseline - 0.050) > 1e-3:
        raise SystemExit(f"head lens covers are {baseline * 1e3:.1f} mm apart, not 50")
    left, right = sorted(centres, key=lambda c: -c[1])  # robot left = +y
    case = in_root["_zedx_mini_rev2__upper_case_v8"].mean(axis=0)
    z = _thin_normal(covers[0])
    if z @ (left - case) < 0:
        z = -z
    for name, centre in (("left", left), ("right", right)):
        world = _frame(centre, z, right - left)
        if world[:3, 1] @ np.array([0.0, 0.0, -1.0]) <= 0:
            raise SystemExit("head camera image-down does not point down")
        out[f"head_camera_{name}_optical"] = (
            "head",
            np.linalg.inv(frames("head")) @ world,
        )
    return out


# -- collision -----------------------------------------------------------------


Spec = tuple  # ("mesh", path) | ("cylinder", radius, length, xyz, rpy)


def _fit_cylinder(pts: np.ndarray, axis: int) -> tuple[float, float, np.ndarray]:
    """Smallest capsule along link axis ``axis`` enclosing ``pts``.

    Returned as the URDF cylinder pyroki turns into that capsule: its radius,
    its length (the capsule's straight section) and its centre.
    """
    others = [i for i in range(3) if i != axis]
    centre = (pts.min(axis=0) + pts.max(axis=0)) / 2
    rho = np.linalg.norm(pts[:, others] - centre[others], axis=1)
    radius = float(rho.max())
    # A point beyond the straight section is covered by the hemispherical
    # cap as long as its axial overhang is within sqrt(r^2 - rho^2).
    overhang = np.abs(pts[:, axis] - centre[axis]) - np.sqrt(
        np.maximum(radius**2 - rho**2, 0.0)
    )
    half = max(float(overhang.max()), 0.0)
    return round(radius, 5), round(2 * half, 5), centre


class Collision:
    """Builds collision bodies from pieces, sharing hull files across models."""

    def __init__(
        self,
        frames: Frames,
        pieces: dict[str, tuple[str, list[np.ndarray]]],
        export: yourdfpy.URDF,
    ) -> None:
        self.frames, self.pieces, self.export = frames, pieces, export
        self.written: dict[tuple[str, ...], str] = {}

    def points(self, link: str, names: tuple[str, ...]) -> np.ndarray:
        return np.vstack(
            [
                trimesh.transform_points(
                    np.vstack(self.pieces[n][1]),
                    self.frames.relative(link, self.pieces[n][0]),
                )
                for n in names
            ]
        )

    def hull(self, link: str, names: tuple[str, ...]) -> Spec:
        # Snapping to a 2 mm grid keeps the hull to a few hundred faces
        # (<= 1.7 mm of shape error, far inside the capsule fit's slack).
        key = (link, *names)
        if key not in self.written:
            snapped = np.unique(
                np.round(self.points(link, names) / 2e-3) * 2e-3, axis=0
            )
            hull = trimesh.PointCloud(snapped).convex_hull
            stem = link if names == (link,) else "_".join(names)
            if stem != link:
                stem = f"{link}__{stem}"
            path = f"{MESH_SUBDIR}/{stem}_collision.stl"
            hull.export(URDF_DIR / path)
            self.written[key] = path
        return ("mesh", self.written[key])

    def finger_cylinder(self, link: str, fingers: tuple[str, ...]) -> Spec:
        """One cylinder along the gripper's jaw axis over both fingers, closed
        and fully open (fingers move along their joint axis, in their frame)."""
        pts = []
        for finger in fingers:
            closed = np.vstack(self.pieces[finger][1])
            joint = self.export.joint_map[MOVING[finger].export_joint]
            opened = closed + np.asarray(joint.axis) * float(joint.limit.upper)
            rel = self.frames.relative(link, finger)
            pts += [trimesh.transform_points(p, rel) for p in (closed, opened)]
        radius, length, centre = _fit_cylinder(np.vstack(pts), 0)
        return ("cylinder", radius, length, centre, _CYLINDER_RPY[0])

    def deck_strips(self) -> dict[str, list[Spec]]:
        # The deck mesh is a box with vertices only at its edges, so strip
        # its bounding box rather than binning vertices.
        pts = self.points("jelly", ("deck",))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        edges = np.linspace(lo[0], hi[0], DECK_STRIPS + 1)
        out = {}
        for i in range(DECK_STRIPS):
            corners = np.array(
                list(
                    itertools.product(
                        (edges[i], edges[i + 1]), (lo[1], hi[1]), (lo[2], hi[2])
                    )
                )
            )
            radius, length, centre = _fit_cylinder(corners, 1)
            out[f"deck_{i}"] = [("cylinder", radius, length, centre, _CYLINDER_RPY[1])]
        return out


def collision_plan(model: str, col: Collision) -> dict[str, list[Spec]]:
    """Each model's collision bodies (link -> specs)."""
    plan: dict[str, list[Spec]] = {
        link: [col.hull(link, (link,))] for link in COMMON_HULLS
    }
    fingers = {
        side: (f"{side}_finger_1", f"{side}_finger_2") for side in ("left", "right")
    }
    if model == "arm":
        plan["base"] = [col.hull("base", ("column_top", "column_mid", "column_stub"))]
    else:
        plan["base"] = [col.hull("base", ("column_top",))]
        plan["lift_stage"] = [col.hull("lift_stage", ("column_mid",))]
        plan["jelly"] = [col.hull("jelly", ("column_stub",))]
    if model in ("arm", "whole_body"):
        for side, pair in fingers.items():
            plan[f"{side}_fingers"] = [col.finger_cylinder(f"{side}_gripper", pair)]
    if model == "whole_body":
        plan.update(col.deck_strips())
    if model == "sim":
        plan["jelly"].append(col.hull("jelly", ("deck",)))
        for pair in fingers.values():
            for finger in pair:
                plan[finger] = [col.hull(finger, (finger,))]
        for wheel in (n for n in MOVING if n.startswith("wheel_")):
            plan[wheel] = [col.hull(wheel, (wheel,))]
    return plan


def report(model: str, plan: dict[str, list[Spec]]) -> None:
    print(f"{model} collision:")
    for link, specs in sorted(plan.items()):
        for spec in specs:
            if spec[0] == "cylinder":
                print(
                    f"  {link:22s} cylinder r={spec[1] * 1e3:5.1f} mm h={spec[2] * 1e3:5.1f} mm"
                )
            else:
                hull = trimesh.load(URDF_DIR / spec[1])
                cyl = trimesh.bounds.minimum_cylinder(hull)
                print(
                    f"  {link:22s} hull {len(hull.faces):4d} faces, capsule "
                    f"r={cyl['radius'] * 1e3:5.1f} mm h={cyl['height'] * 1e3:5.1f} mm"
                )


# -- sim inertials ---------------------------------------------------------------


def sim_inertials(
    col: Collision, plan: dict[str, list[Spec]]
) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    """``{link: (mass, com, inertia 3x3 about the com)}`` for the sim model."""
    from almond_axol.constants import ARM_JOINTS, urdf_body_name
    from almond_axol.robot.config import AxolConfig

    def hull_of(link: str) -> trimesh.Trimesh:
        meshes = [trimesh.load(URDF_DIR / s[1]) for s in plan[link] if s[0] == "mesh"]
        return trimesh.util.concatenate(meshes).convex_hull

    def inertia(link: str, mass: float, com: np.ndarray) -> np.ndarray:
        hull = hull_of(link)
        hull.density = mass / max(hull.volume, 1e-9)
        # About the hull centroid, then moved to the given CoM.
        tensor = hull.moment_inertia
        d = com - hull.center_mass
        return tensor + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    out: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    cfg = AxolConfig()
    for side, arm in (("left", cfg.left), ("right", cfg.right)):
        is_left = side == "left"
        for joint in ARM_JOINTS:
            body = urdf_body_name(joint, is_left=is_left)
            jc = getattr(arm, joint.value)
            out[body] = (float(jc.mass), np.asarray(jc.com, dtype=float), None)
        # wrist_3's configured mass lumps the whole gripper assembly: carve the
        # gripper, fingers and camera out of it, keeping the total mass and
        # CoM (what gravity comp was tuned against) unchanged.
        w2 = f"{side}_w2"
        total, com_total, _ = out[w2]
        carved_mass, carved_moment = 0.0, np.zeros(3)
        for part, mass in SIM_WRIST_PARTS.items():
            link = f"{side}_{part}"
            com_link = hull_of(link).center_mass
            com_w2 = trimesh.transform_points(
                com_link[None], col.frames.relative(w2, link)
            )[0]
            out[link] = (mass, com_link, None)
            carved_mass += mass
            carved_moment += mass * com_w2
        rest = total - carved_mass
        if rest <= 0:
            raise SystemExit(f"{w2}: wrist parts outweigh the tuned wrist_3 mass")
        out[w2] = (rest, (total * com_total - carved_moment) / rest, None)
    for link, mass in SIM_MASSES.items():
        out[link] = (mass, hull_of(link).center_mass, None)
    return {
        link: (mass, com, inertia(link, mass, com))
        for link, (mass, com, _) in out.items()
    }


def sim_dynamics() -> dict[str, dict[str, float]]:
    """Per arm joint, from AxolConfig (the calibrated defaults, stiffness
    blend applied): measured friction and the motors' impedance gains.

    Only what a simulator can use as physics or as a joint drive goes in.
    The friction model's constant offset ``fo``, the acceleration
    feedforward ``j_eff`` (a controller term — as inertia it would count the
    links twice) and the host-side damping ``kd_host`` (a pose-scheduled,
    band-passed controller term) are left out.
    """
    from almond_axol.constants import ARM_JOINTS, urdf_joint_name
    from almond_axol.robot.config import AxolConfig

    cfg = AxolConfig().resolved()
    out: dict[str, dict[str, float]] = {}
    for arm, is_left in ((cfg.left, True), (cfg.right, False)):
        for joint in ARM_JOINTS:
            jc = getattr(arm, joint.value)
            out[urdf_joint_name(joint, is_left=is_left)] = {
                "friction": round(float(jc.friction.fc), 6),
                "damping": round(float(jc.friction.fv), 6),
                "stiffness": float(jc.kp),
                "drive_damping": float(jc.kd),
            }
    return out


def write_sim_drives(dynamics: dict[str, dict[str, float]]) -> None:
    import json

    doc = {
        "note": (
            "Joint drive gains for axol_jelly_sim.urdf, from AxolConfig "
            "(generated by scripts/build_jelly_urdfs.py). stiffness (Nm/rad) "
            "and damping (Nm·s/rad) are the motors' impedance-control kp / "
            "kd: set them as each joint's position-drive gains after import. "
            "Friction lives in the URDF (<dynamics>)."
        ),
        "joints": {
            name: {"stiffness": d["stiffness"], "damping": d["drive_damping"]}
            for name, d in dynamics.items()
        },
    }
    OUT_SIM_DRIVES.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_SIM_DRIVES.relative_to(REPO)}")


# -- writer ----------------------------------------------------------------------


def _fmt(v: np.ndarray) -> str:
    return " ".join(f"{x:.6g}" if abs(x) > 1e-12 else "0" for x in v)


def _rpy(rot: np.ndarray) -> np.ndarray:
    """URDF rpy (fixed-axis x-y-z) of a rotation matrix."""
    pitch = np.arcsin(-np.clip(rot[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) < 1e-9:
        return np.array([np.arctan2(-rot[1, 2], rot[1, 1]), pitch, 0.0])
    return np.array(
        [np.arctan2(rot[2, 1], rot[2, 2]), pitch, np.arctan2(rot[1, 0], rot[0, 0])]
    )


def _joint(
    robot: ET.Element,
    name: str,
    kind: str,
    parent: str,
    child: str,
    pose: np.ndarray | None = None,
    axis: np.ndarray | None = None,
    limits: tuple[float, float] | None = None,
    mimic: str | None = None,
    effort_velocity: tuple[float, float] = (1.0, 1.0),
) -> None:
    joint = ET.SubElement(robot, "joint", name=name, type=kind)
    pose = np.eye(4) if pose is None else pose
    ET.SubElement(joint, "origin", xyz=_fmt(pose[:3, 3]), rpy=_fmt(_rpy(pose[:3, :3])))
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    if axis is not None:
        ET.SubElement(joint, "axis", xyz=_fmt(np.asarray(axis, dtype=float)))
    if limits is not None or kind == "continuous":
        attrs = dict(
            effort=f"{effort_velocity[0]:g}", velocity=f"{effort_velocity[1]:g}"
        )
        if limits is not None:
            attrs.update(lower=f"{limits[0]:.6g}", upper=f"{limits[1]:.6g}")
        ET.SubElement(joint, "limit", **attrs)
    if mimic is not None:
        ET.SubElement(joint, "mimic", joint=mimic, multiplier="1", offset="0")


def _translation(xyz: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = xyz
    return pose


def write_urdf(
    model: str,
    out_path: Path,
    visuals: dict[str, list[tuple[str, tuple[float, ...]]]],
    plan: dict[str, list[Spec]],
    limits: dict[str, tuple[float, float]],
    frames: Frames,
    cameras: dict[str, tuple[str, np.ndarray]],
    export: yourdfpy.URDF,
    lift_travel: float,
    inertials: dict[str, tuple[float, np.ndarray, np.ndarray]] | None = None,
    dynamics: dict[str, dict[str, float]] | None = None,
) -> None:
    tree = ET.parse(CLASSIC_URDF)
    robot = tree.getroot()
    robot.set("name", "assembly")
    links = {ln.get("name"): ln for ln in robot.findall("link")}
    centre = frames.jelly_center
    # Mesh paths: package:// for the IK models (the control panel and headset
    # map "assembly" onto this directory); relative for the sim model, whose
    # importers resolve files next to the URDF.
    prefix = "" if model == "sim" else PACKAGE

    def link(name: str) -> None:
        links[name] = ET.SubElement(robot, "link", name=name)

    for name, parent in EXTRA_LINKS.items():
        if model == "sim" and name.endswith("_fingers"):
            continue
        link(name)
        _joint(robot, f"{name}_0", "fixed", parent, name)
    for name in (*BODY_FRAME_LINKS, *MOVING, *cameras):
        link(name)
    for name, (parent, pose) in cameras.items():
        _joint(robot, f"{name}_0", "fixed", parent, name, pose)

    lift = ("prismatic", np.array([0, 0, -1.0]), (0.0, lift_travel))
    if model == "arm":
        link(FLOOR_LINK)
        for name in (FLOOR_LINK, *BODY_FRAME_LINKS):
            _joint(robot, f"{name}_0", "fixed", "root", name, _translation(centre))
    else:
        if model == "whole_body":
            # world -> planar base -> jelly -> lift -> torso root (the classic
            # tree). At every body joint's zero, "root" is the world origin.
            for name in ("world", "base_x_link", "base_y_link", FLOOR_LINK):
                link(name)
            _joint(
                robot,
                f"{FLOOR_LINK}_0",
                "fixed",
                "world",
                FLOOR_LINK,
                _translation(centre),
            )
            travel = (-BASE_TRAVEL, BASE_TRAVEL)
            _joint(
                robot,
                "base_x",
                "prismatic",
                "world",
                "base_x_link",
                _translation(centre),
                np.array([1.0, 0, 0]),
                travel,
            )
            _joint(
                robot,
                "base_y",
                "prismatic",
                "base_x_link",
                "base_y_link",
                None,
                np.array([0, 1.0, 0]),
                travel,
            )
            _joint(
                robot,
                "base_yaw",
                "revolute",
                "base_y_link",
                "jelly",
                None,
                np.array([0, 0, 1.0]),
                (-2 * np.pi, 2 * np.pi),
            )
            for i in range(DECK_STRIPS):
                link(f"deck_{i}")
                _joint(robot, f"deck_{i}_0", "fixed", "jelly", f"deck_{i}")
        lift_limits = SIM_LIMITS["lift"] if model == "sim" else (1.0, 1.0)
        _joint(
            robot,
            "lift",
            *lift[:1],
            "jelly",
            "lift_stage",
            None,
            *lift[1:],
            effort_velocity=lift_limits,
        )
        _joint(
            robot,
            "lift_2",
            *lift[:1],
            "lift_stage",
            "root",
            _translation(-centre),
            *lift[1:],
            mimic="lift",
            effort_velocity=lift_limits,
        )

    for name, moving in MOVING.items():
        joint = export.joint_map[moving.export_joint]
        pose = frames.relative(moving.parent, name)
        if model != "sim":
            _joint(robot, f"{name}_0", "fixed", moving.parent, name, pose)
            continue
        if name.startswith("wheel_"):
            _joint(
                robot,
                moving.export_joint,
                "continuous",
                moving.parent,
                name,
                pose,
                joint.axis,
                effort_velocity=SIM_LIMITS["wheel"],
            )
        else:
            _joint(
                robot,
                f"{name}_joint",
                "prismatic",
                moving.parent,
                name,
                pose,
                joint.axis,
                (0.0, float(joint.limit.upper)),
                mimic=f"{moving.mimic}_joint" if moving.mimic else None,
                effort_velocity=SIM_LIMITS["finger"],
            )

    for name, element in links.items():
        for tag in ("visual", "collision"):
            for el in element.findall(tag):
                element.remove(el)
        if inertials is not None:
            for el in element.findall("inertial"):
                element.remove(el)
            if name in inertials:
                mass, com, tensor = inertials[name]
                inertial = ET.Element("inertial")
                ET.SubElement(inertial, "origin", xyz=_fmt(com), rpy="0 0 0")
                ET.SubElement(inertial, "mass", value=f"{mass:.6g}")
                ET.SubElement(
                    inertial,
                    "inertia",
                    ixx=f"{tensor[0, 0]:.6g}",
                    ixy=f"{tensor[0, 1]:.6g}",
                    ixz=f"{tensor[0, 2]:.6g}",
                    iyy=f"{tensor[1, 1]:.6g}",
                    iyz=f"{tensor[1, 2]:.6g}",
                    izz=f"{tensor[2, 2]:.6g}",
                )
                element.insert(0, inertial)
        for path, rgba in visuals.get(name, []):
            vis = ET.SubElement(element, "visual")
            ET.SubElement(vis, "origin", xyz="0 0 0", rpy="0 0 0")
            geom = ET.SubElement(vis, "geometry")
            ET.SubElement(geom, "mesh", filename=prefix + path, scale="1 1 1")
            mat = ET.SubElement(
                vis, "material", name="c_" + "_".join(f"{c:g}" for c in rgba)
            )
            ET.SubElement(mat, "color", rgba=" ".join(f"{c:g}" for c in rgba))
        for spec in plan.get(name, []):
            col = ET.SubElement(element, "collision")
            if spec[0] == "mesh":
                ET.SubElement(col, "origin", xyz="0 0 0", rpy="0 0 0")
                geom = ET.SubElement(col, "geometry")
                ET.SubElement(geom, "mesh", filename=prefix + spec[1], scale="1 1 1")
            else:
                _, radius, length, xyz, rpy = spec
                ET.SubElement(col, "origin", xyz=_fmt(xyz), rpy=rpy)
                geom = ET.SubElement(col, "geometry")
                ET.SubElement(
                    geom, "cylinder", radius=f"{radius:g}", length=f"{length:g}"
                )
    for joint in robot.findall("joint"):
        lim = limits.get(joint.get("name", ""))
        if lim is not None:
            joint.find("limit").set("lower", f"{lim[0]:.4f}")
            joint.find("limit").set("upper", f"{lim[1]:.4f}")
        dyn = (dynamics or {}).get(joint.get("name", ""))
        if dyn is not None:
            ET.SubElement(
                joint,
                "dynamics",
                damping=f"{dyn['damping']:g}",
                friction=f"{dyn['friction']:g}",
            )
    ET.indent(tree, space="    ")
    body = ET.tostring(robot, encoding="unicode")
    note = (
        "Axol on the Jelly mobile base. GENERATED by scripts/build_jelly_urdfs.py "
        "from the Onshape export; do not edit by hand. Arm links, joints and "
        "joint frames are the classic axol.urdf's, so FK and gravity "
        "compensation match it exactly; joint limits and meshes are the "
        "export's. "
        + {"arm": ARM_NOTE, "whole_body": WHOLE_BODY_NOTE, "sim": SIM_NOTE}[model]
        + " "
        + WRIST_CAMERA_NOTE
    )
    header = "<!-- " + textwrap.fill(note, 76, subsequent_indent="     ") + "\n-->\n"
    out_path.write_text(header + body + "\n")
    print(f"wrote {out_path.relative_to(REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "export_dir", type=Path, help="Onshape export (contains urdf/ and meshes/)"
    )
    ap.add_argument("--min-part-mm", type=float, default=15.0)
    ap.add_argument("--visual-faces", type=int, default=2000)
    ap.add_argument(
        "--visual-tolerance-mm",
        type=float,
        default=0.3,
        help="largest deviation from the CAD of what the wrist cameras see "
        "(grippers and fingers)",
    )
    ap.add_argument(
        "--loose-tolerance-mm",
        type=float,
        default=1.0,
        help="largest deviation from the CAD of every other visual mesh",
    )
    args = ap.parse_args()
    urdfs = sorted((args.export_dir / "urdf").glob("*.urdf"))
    if len(urdfs) != 1:
        raise SystemExit(f"expected one URDF under {args.export_dir / 'urdf'}")
    classic = _load(CLASSIC_URDF)
    export = _load(urdfs[0])
    tf, sg = align(classic, export)
    check_equivalent(classic, export, tf, sg)
    lift_travel = check_lift(export)
    limits = export_limits(export, sg)
    for name, (lo, hi) in limits.items():
        old = classic.joint_map[name].limit
        mark = (
            ""
            if (round(old.lower, 4), round(old.upper, 4)) == (lo, hi)
            else "  <- changed"
        )
        print(f"  {name:12s} [{lo:+.4f}, {hi:+.4f}]{mark}")
    frames = Frames(classic, export, tf)
    visuals, pieces, in_root = build_meshes(
        export,
        args.export_dir,
        frames,
        args.min_part_mm,
        args.visual_faces,
        args.visual_tolerance_mm * 1e-3,
        args.loose_tolerance_mm * 1e-3,
    )
    cameras = camera_frames(in_root, frames)
    col = Collision(frames, pieces, export)
    for model, out_path in (
        ("arm", OUT_ARM),
        ("whole_body", OUT_WHOLE_BODY),
        ("sim", OUT_SIM),
    ):
        plan = collision_plan(model, col)
        report(model, plan)
        write_urdf(
            model,
            out_path,
            visuals,
            plan,
            limits,
            frames,
            cameras,
            export,
            lift_travel,
            sim_inertials(col, plan) if model == "sim" else None,
            sim_dynamics() if model == "sim" else None,
        )
    write_sim_drives(sim_dynamics())


if __name__ == "__main__":
    main()
