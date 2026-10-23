"""
Build the bundled Axol URDFs from their Onshape exports: Axol on Jelly (with
the parallel or the parcel gripper), and Axol standalone, on the Ox cart and
on the Owl mount — one file per version.

An Onshape export (``almond_urdf/urdf/almond_urdf.urdf`` + ``meshes/``) is a
faithful CAD tree — ~140 links including every screw, prismatic lift stages
and fingers, wheel joints, and link frames wherever Onshape put them. None of
that matches what the control stack expects, so this script distils each
into one full model (``axol_jelly.urdf``, ``axol_jelly_parcel.urdf``,
``axol_standalone.urdf``, ``axol_ox_cart.urdf``, ``axol_owl.urdf``) that IK,
the viewers and physics simulators (Isaac Sim, MuJoCo) all load:

- **Arms are the classic arms.** Every arm link, joint name, joint frame and
  axis is copied verbatim from ``axol.urdf``; only the joint *limits* come
  from the export. The script first proves the two arm chains are the same
  mechanism (it fits the rigid transform between the exports' joint axes,
  checks every arm link moves identically under random joint angles, and
  reports the per-joint sign flips between the two naming schemes), so FK,
  IK targets and recorded Cartesian frames are unchanged between versions.
- **Every moving part is a joint:** the Jelly base in the plane (``base_x``
  / ``base_y`` / ``base_yaw``) and its two-stage lift (``lift``, ``lift_2``
  mimicking it), the Ox cart's lift, the fingers or parcel plate, and the
  wheels. With every body joint at zero the torso root is the world
  origin, so the world frame is the classic one.
- **Two collision sets** (see ``MODEL_NOTE``): one convex hull per part in
  ``<collision>``, for physics, and capsule-friendly pieces in the custom
  ``<ik_collision>`` (pyroki fits one capsule per link): the shoulder bar,
  the head-camera mount, the lift column's top plate and each section, the
  Jelly deck and the Owl foot plate as strips, and each gripper's jaws as
  one body over their whole motion. IK (``almond_axol.kinematics.model``)
  freezes every joint but the arms' and collides the second set.
- **Masses and drives** from :class:`~almond_axol.robot.config.AxolConfig`
  for the arms (wrist_3's lump split across the gripper parts), estimates
  elsewhere (``Body.masses``); friction and impedance gains on the arm
  joints.
- **Meshes are the export's**, re-expressed in the bundled link frames: each
  rigid group of export links (a link plus its fastened children) is merged
  per colour into one decimated visual STL. Fasteners and connector bits
  under ``--min-part-mm`` are dropped. The body and arms are the same in
  every export, so later exports reuse the first one's meshes
  (``meshes/jelly/``) wherever the source parts match and keep only their
  own under ``meshes/<version>/``. Mesh paths are relative.
- **Camera optical frames** for both wrist cameras and both head-camera
  lenses (``*_optical``: +z along the optical axis, +x image right, +y image
  down), from the CAD lens geometry — see :func:`camera_frames`.

Each export is a body (a :class:`Body`: Jelly, standalone, Ox cart, Owl) and
a gripper (a :class:`Variant`); the script tells them apart by their links.

Run (``fast-simplification`` is only needed here, not at runtime) with
every export at once, the Jelly parallel gripper's first, so the others
reuse its meshes::

    uv run --with fast-simplification python scripts/build_urdfs.py \\
        jelly/almond_urdf parcel/almond_urdf axol/almond_urdf \\
        owl/almond_urdf ox_cart/almond_urdf
"""

from __future__ import annotations

import argparse
import itertools
import re
import textwrap
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import trimesh
import yourdfpy

REPO = Path(__file__).resolve().parents[1]
URDF_DIR = REPO / "almond_axol" / "kinematics" / "urdf"
CLASSIC_URDF = URDF_DIR / "axol.urdf"
MESH_SUBDIR = "meshes/jelly"
# Travel of each planar base joint (m): far beyond any single session.
BASE_TRAVEL = 10.0

# Classic arm joints (ARM_JOINTS order) and their export counterparts.
CLASSIC_JOINTS = ("s1_0", "s2_0", "s3_0", "e1_0", "e2_0", "w1_0", "w2_0")
EXPORT_JOINTS = ("s1", "s2", "s3", "e1", "w1", "w2", "w3")

# Export link -> bundled link. Every other export link joins the group of its
# nearest listed ancestor. The shoulder bar, head and arms, common to every
# export; each body's and each gripper's own links are in their Body and
# Variant.
ARM_ANCHORS: dict[str, str] = {
    "s1": "s1",
    "part_1": "head",
    "s2_left": "left_s2",
    "s3_left": "left_s3",
    "e1_left": "left_e1",
    "e2_left": "left_e2",
    "w0_left": "left_w0",
    "w1_left": "left_w1",
    "w2_1": "left_w2",
    "s2_right": "right_s2",
    "s3_right": "right_s3",
    "e1_right": "right_e1",
    "e2_right": "right_e2",
    "w0_right": "right_w0",
    "w1_right": "right_w1",
    "w2": "right_w2",
}

# Bundled links that are not in axol.urdf, and the classic link each is
# rigidly attached to with an identity origin (their meshes are placed in the
# parent's frame). Each Body and Variant adds its own.
HEAD_LINKS: dict[str, str] = {"head": "s1"}
# IK models only: collision-only link per gripper holding its jaws' swept
# volume (see Collision.finger_cylinder / plate_sweep).
FINGER_LINKS: dict[str, str] = {
    "left_fingers": "left_gripper",
    "right_fingers": "right_gripper",
}


@dataclass(frozen=True)
class Moving:
    """A bundled link that moves in the export: its frame and joint come from it."""

    export_link: str
    export_joint: str
    parent: str
    mimic: str | None = None


# Fingers and wheels keep the export's link frames (their joint frames), so
# the sim model can move them along the export's own axes. Each Body and
# Variant lists its moving parts.

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
Split = tuple[tuple[float, ...], tuple[float, ...], str, str]

# Collision body specs: a hull over pieces, a cylinder along a link axis over
# pieces (``stroke``: include the fingers fully open), or deck strips.
_ARM_PARTS = ("s2", "s3", "e1", "e2", "w0", "w1", "w2", "gripper")
# A flat part (the Jelly deck, 0.6 x 0.6 m, ~0.24 m tall; the Owl mount's
# foot) as strips across the robot for capsule collision: one capsule over it
# would swallow the arms; three along y stay within a few cm of it.
DECK_STRIPS = 3


@dataclass(frozen=True)
class Body:
    """What an export mounts the shoulder bar on, and how it collides.

    Collision pieces: each export mesh feeds the piece named after its group,
    unless ``splits`` splits it by a mesh-local box ``(lo, hi)`` into the
    piece inside and the piece outside it. Each link gets two collision
    sets: ``sim_hulls`` (one convex hull per part, for physics) and
    ``ik_hulls`` + ``strips`` (capsule-friendly, for IK; see IK_NOTE).
    """

    key: str  # AxolModel value
    marker: str | None  # an export link only this body has (None: no body)
    anchors: dict[str, str]  # its export links -> bundled links
    extra_links: dict[str, str]  # its fixed extra links (see HEAD_LINKS)
    moving: dict[str, Moving]  # its moving parts besides the lift (wheels)
    # Links world-aligned at the export root (the ground under the robot
    # when it stands on it).
    frame_links: tuple[str, ...]
    splits: dict[str, Split]
    # IK body hulls besides s1 and head: link -> pieces (one hull).
    ik_hulls: dict[str, tuple[str, ...]]
    # IK: flat pieces as strips: piece -> (parent link, strip prefix, count).
    strips: dict[str, tuple[str, str, int]]
    # Physics: link -> its parts' hulls (each a tuple of pieces).
    sim_hulls: dict[str, tuple[tuple[str, ...], ...]]
    # The lift: its export joints (equal stages, 0 = fully raised, lowering
    # along -z) and the bundled links it runs through, base to torso
    # ("root"); one stage per pair of neighbours, the later ones mimicking
    # the first. Empty: no lift.
    lift_joints: tuple[str, ...]
    lift_links: tuple[str, ...]
    # Masses (kg) of its links and s1 / head: ESTIMATES, not measured.
    masses: dict[str, float]
    # Whether the body stands on the ground: a floor link goes under its
    # lowest point (not the export root, which is not always there — the Ox
    # cart's is its frame, 112 mm above the casters' bottoms). The Owl mount
    # and standalone Axol stand on whatever they are bolted to, above where
    # the arms hang.
    floor: bool
    title: str  # what the URDF's header calls the robot
    note: str  # what IK keeps the arms clear of, for the header
    # The Jelly base drives in the plane: base_x / base_y / base_yaw joints
    # a simulator can drive.
    planar: bool = False


_TORSO_MASSES = {"s1": 3.0, "head": 0.3}

JELLY = Body(
    key="mobile",
    marker="jelly_base",
    anchors={
        "jelly_base": "jelly",
        "jelly_stage_2_1": "lift_stage",
        "jelly_stage_3": "base",
        "wheel": "wheel_1_link",
        "wheel_2": "wheel_2_link",
        "wheel_3": "wheel_3_link",
        "wheel_1": "wheel_4_link",
    },
    extra_links={"lift_plate": "base"},
    moving={
        "wheel_1_link": Moving("wheel", "wheel_1", "jelly"),
        "wheel_2_link": Moving("wheel_2", "wheel_2", "jelly"),
        "wheel_3_link": Moving("wheel_3", "wheel_3", "jelly"),
        "wheel_4_link": Moving("wheel_1", "wheel_4", "jelly"),
    },
    frame_links=("jelly", "lift_stage"),
    # - The top of the last lift stage is the bracket the shoulder bar bolts
    #   onto; folded into the column's hull it would inflate the column
    #   capsule from ~85 mm to ~120 mm, so it is its own piece (and link).
    # - The Jelly base mesh includes the fixed bottom section of the lift
    #   column, which is within reach at gripper height, separate from the
    #   deck.
    splits={
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
    },
    # Each lift-column section rides its own stage.
    ik_hulls={
        "lift_plate": ("lift_plate",),
        "base": ("column_top",),
        "lift_stage": ("column_mid",),
        "jelly": ("column_stub",),
    },
    # The deck stays on the ground while the lift carries the arms down, so
    # it is in reach with the lift lowered (IK places it from the live lift
    # height, see almond_axol.kinematics.model.lift_link_shift).
    strips={"deck": ("jelly", "deck", DECK_STRIPS)},
    sim_hulls={
        "lift_plate": (("lift_plate",),),
        "base": (("column_top",),),
        "lift_stage": (("column_mid",),),
        "jelly": (("column_stub",), ("deck",)),
    },
    lift_joints=("stage_1", "stage_2"),
    lift_links=("jelly", "lift_stage", "root"),
    masses={
        **_TORSO_MASSES,
        "jelly": 40.0,  # drive base + two 24 V 50 Ah LiFePO4 packs
        "lift_stage": 3.0,
        "base": 4.0,  # top lift stage
        "lift_plate": 1.0,
        "wheel_1_link": 1.0,
        "wheel_2_link": 1.0,
        "wheel_3_link": 1.0,
        "wheel_4_link": 1.0,
    },
    floor=True,
    title="Axol on the Jelly mobile base",
    note=(
        "IK keeps the arms clear of the shoulder bar (s1), the head-camera "
        "mount (head), the lift column's top plate (lift_plate) and its three "
        "sections (base, lift_stage, jelly), and the deck as strips (deck_*). "
        "IK freezes the base and lift at zero, the lift fully raised, and "
        "moves the parts the lift leaves behind (the deck and the lower "
        "column) up to the arms by the live lift height."
    ),
    planar=True,
)

# Axol on its own: the shoulder bar and head, nothing below them.
STANDALONE = Body(
    key="standalone",
    marker=None,
    anchors={},
    extra_links={},
    moving={},
    frame_links=(),
    splits={},
    ik_hulls={},
    strips={},
    sim_hulls={},
    lift_joints=(),
    lift_links=(),
    masses=dict(_TORSO_MASSES),
    floor=False,
    title="Axol standalone",
    note=(
        "Nothing below the shoulder bar: IK keeps the arms clear of it (s1) "
        "and the head-camera mount (head)."
    ),
)

# The Owl mount: a column under the shoulder bar on a foot plate.
OWL = Body(
    key="owl",
    marker="owl_base",
    anchors={"owl_base": "base"},
    extra_links={},
    moving={},
    frame_links=(),
    # The foot plate (255 x 275 mm, 22 mm thick) at the bottom of the column.
    splits={"Owl_Base.stl": ((-_INF, -_INF, -0.37), (_INF,) * 3, "base", "foot")},
    ik_hulls={"base": ("base",)},
    # Six strips: the capsule round each is ~24 mm for the 22 mm plate (three
    # would be 44 mm, and swallow the upper arms at the rest pose).
    strips={"foot": ("base", "foot", 6)},
    sim_hulls={"base": (("base",), ("foot",))},
    lift_joints=(),
    lift_links=(),
    masses={**_TORSO_MASSES, "base": 3.0},
    floor=False,
    title="Axol on the Owl mount",
    note=(
        "IK keeps the arms clear of the shoulder bar (s1), the head-camera "
        "mount (head), the mount's column (base) and its foot plate "
        "(foot_0..5)."
    ),
)

# The Ox cart: a lift column on a caster frame.
OX_CART = Body(
    key="ox_cart",
    marker="ox_cart_base",
    anchors={"ox_cart_lift": "base", "ox_cart_base": "cart"},
    extra_links={"lift_plate": "base"},
    moving={},
    frame_links=("cart",),
    # The cart's fixed column section, which the lift column rides on, apart
    # from the caster frame below it.
    # - The lift's top 38 mm is the plate the shoulder bar bolts onto,
    #   reaching 130 mm behind the 148 x 75 mm column: in the column's hull
    #   it tilts the fitted capsule off the vertical and widens it to 127 mm,
    #   so it is its own piece (and link), as on Jelly.
    # - The cart's fixed column section, which the lift column rides on,
    #   apart from the caster frame below it.
    splits={
        "Ox_Cart_Lift.stl": (
            (-_INF, -_INF, 1.075),
            (_INF,) * 3,
            "lift_plate",
            "column_top",
        ),
        "Ox_Cart_Base.stl": (
            (-0.09, -0.08, 0.23),
            (0.09, 0.11, _INF),
            "column_stub",
            "cart_frame",
        ),
    },
    # IK freezes the lift fully raised: one hull over the column and the
    # stub it rides on (in the torso's frame, valid at that height).
    ik_hulls={
        "lift_plate": ("lift_plate",),
        "base": ("column_top", "column_stub"),
    },
    strips={},
    sim_hulls={
        "lift_plate": (("lift_plate",),),
        "base": (("column_top",),),
        "cart": (("column_stub",), ("cart_frame",)),
    },
    lift_joints=("ox_lift",),
    lift_links=("cart", "root"),
    masses={**_TORSO_MASSES, "base": 5.0, "lift_plate": 1.0, "cart": 30.0},
    floor=True,
    title="Axol on the Ox cart",
    note=(
        "IK freezes the lift fully raised (its CAD pose), where the cart's "
        "caster frame sits below anything the arms can reach, and keeps the "
        "arms clear of the shoulder bar (s1), the head-camera mount (head), "
        "the lift column (base) and its top plate (lift_plate). Lowering the lift brings the frame up "
        "toward the hands, which IK does not see."
    ),
)
BODIES = (JELLY, OX_CART, OWL, STANDALONE)  # matched in order; standalone last


# ZED X One S fisheye: where the camera's rays converge, along its optical
# axis from the camera's CAD origin (m). From Stereolabs' model
# (ZED_X_ONE_S_Fisheye.step): the planes of its "Simple FOV" solid meet on the
# axis at z = 10.696 mm (2.2 mm behind the front glass). The export carries
# that CAD origin as the zed_x_one_gs_compact frame, 12.6 mm ahead of the
# body's seal ring.
PROJECTION_CENTRE_Z = 0.010696
# ZED X Mini Wide (head, stereo): the same, behind each lens's front cover
# (m). From Stereolabs' model (ZED_X_Mini_Wide.step): the planes of each
# lens's "FOV" solid meet on its axis 7.2 mm behind the mid-plane of its
# "Plastron v2" cover (the 22 x 22 x 1 mm part the export carries too).
HEAD_PROJECTION_BEHIND_COVER = 0.007205

_CYLINDER_RPY = {0: "0 1.5707963267948966 0", 1: "-1.5707963267948966 0 0", 2: "0 0 0"}

# Frame at the ground under the robot (the export's root).
FLOOR_LINK = "floor"

# One model serves IK, the viewers and physics simulators.
MODEL_NOTE = (
    "One model for IK, the viewers and physics simulators. Moving: the 14 "
    "arm joints, {body_joints}{gripper_joints}. "
    "With every body joint at zero the torso root (link root) is the "
    "world origin, so poses match the classic axol.urdf. IK "
    "(almond_axol.kinematics.model) freezes every joint but the arms' and "
    "collides capsule-friendly pieces, the custom ik_collision elements, "
    "instead of each part's collision hull; importers ignore them. {ik} "
    "Physics: arm joints carry AxolConfig's measured friction as <dynamics "
    "friction> (Coulomb, N·m) and damping (viscous, N·m·s/rad), per arm, and "
    "the motors' impedance gains as <drive stiffness> (kp, N·m/rad) and "
    "damping (kd, N·m·s/rad): a custom element no importer reads, so set "
    "them as each joint's position-drive gains after import. Arm masses and "
    "CoMs are AxolConfig's tuned gravity-comp values (the wrist-3 body is "
    "split so {wrist_parts} carry their share while the assembly's total "
    "mass and CoM stay as tuned); every other mass is an ESTIMATE (Body."
    "masses in the generator) - replace with measured values. Inertia "
    "tensors assume uniform density over each part's convex hull. "
    "Effort/velocity: arm joints keep axol.urdf's values; the rest are "
    "estimates. {sim}Camera frames (*_optical) use the ROS optical "
    "convention (+z forward, +x right, +y down); an Isaac Sim camera "
    "attached with ROS camera axes needs no extra rotation. Mesh paths are "
    "relative to this file."
)
# Per body: its joints, and what a simulator should know about them.
BODY_JOINTS_NOTE = {
    "mobile": (
        "the Jelly base in the plane (base_x, base_y, base_yaw about the "
        "Jelly centre, in the world frame), the lift (lift, 0 = fully raised, "
        "lowering each of the two stages; lift_2 mimics it), the four wheels "
        "(continuous), "
    ),
    "ox_cart": "the cart's lift (lift, 0 = fully raised, positive lowers), ",
}
SIM_BODY_NOTE = {
    "mobile": (
        "The x-drive's omni rollers are not modelled: plain wheels cannot "
        "strafe, so drive the base through base_x / base_y / base_yaw (or "
        "kinematically). "
    ),
}

HEAD_CAMERA_NOTE = (
    "Head camera frames (head_camera_*_optical, ZED X Mini Wide): each lens's "
    "projection centre from Stereolabs' model, 7.2 mm behind its front cover."
)
WRIST_CAMERA_NOTE = (
    "Wrist camera frames (*_wrist_camera_optical, ZED X One S fisheye): the "
    "origin is the projection centre from Stereolabs' model of the camera "
    "(its FOV solid's rays meet on the lens axis 2.2 mm behind the front "
    "glass); fitting real wrist images through each camera's factory "
    "calibration agrees within the fit's ~3 mm resolution. Roll puts the "
    "gripper at the bottom of the image, as in real frames. " + HEAD_CAMERA_NOTE
)

# Joint effort (N·m or N) / velocity (rad/s or m/s) in the sim model for the
# joints axol.urdf does not define: ESTIMATES.
SIM_LIMITS: dict[str, tuple[float, float]] = {
    "lift": (1000.0, 0.03),
    "finger": (50.0, 0.1),
    "plate": (5.0, 3.0),
    "wheel": (20.0, 30.0),
    "base": (1000.0, 1.0),
    "base_yaw": (1000.0, 1.0),
}


@dataclass(frozen=True)
class Variant:
    """One export — a body and a gripper: the links the gripper adds to the
    body and arms, and the URDFs and meshes made from it."""

    body: Body
    gripper: str  # AxolConfig.gripper
    marker: str  # an export link only this gripper's export has
    anchors: dict[str, str]  # its export links -> bundled links
    extra_links: dict[str, str]  # its fixed extra links (see BODY_EXTRA_LINKS)
    moving: dict[str, Moving]  # its moving parts (sim joints)
    hulls: tuple[str, ...]  # its per-side collision hulls besides _ARM_PARTS
    # Visual meshes simplified to the tight tolerance (link-name suffixes).
    tight: tuple[str, ...]
    # Masses (kg, ESTIMATES) carved out of AxolConfig's wrist_3 mass, which
    # lumps the whole gripper assembly, so they redistribute rather than add.
    wrist_parts: dict[str, float]
    output: Path
    mesh_subdir: str
    # Set to reference the meshes earlier exports in the run wrote from the
    # same parts instead of writing copies (the body and arms are the same in
    # every export): the first export's subdirectory, which they all share.
    share_subdir: str | None
    wrist_cameras: bool
    gripper_joints: str  # MODEL_NOTE's description of the gripper joints
    wrist_parts_note: str  # ... and of wrist_parts

    @property
    def all_anchors(self) -> dict[str, str]:
        return {**ARM_ANCHORS, **self.body.anchors, **self.anchors}

    @property
    def all_extra_links(self) -> dict[str, str]:
        return {
            **HEAD_LINKS,
            **self.body.extra_links,
            **self.extra_links,
            **FINGER_LINKS,
        }

    @property
    def all_moving(self) -> dict[str, Moving]:
        return {**self.moving, **self.body.moving}


PARALLEL = Variant(
    body=JELLY,
    gripper="parallel",
    marker="gripper_base",
    anchors={
        "gripper_base": "left_gripper",
        "gripper_tip_1": "left_finger_1",
        "gripper_tip": "left_finger_2",
        "wrist_mount": "left_wrist_camera",
        "gripper_base_1": "right_gripper",
        "gripper_tip_3": "right_finger_1",
        "gripper_tip_2": "right_finger_2",
        "wrist_mount_1": "right_wrist_camera",
    },
    # Split off the gripper so each part gets its own (much tighter) capsule.
    extra_links={
        "left_wrist_camera": "left_gripper",
        "right_wrist_camera": "right_gripper",
    },
    moving={
        "left_finger_1": Moving("gripper_tip_1", "gripper_tip_1", "left_gripper"),
        "left_finger_2": Moving(
            "gripper_tip", "gripper_tip_2", "left_gripper", "left_finger_1"
        ),
        "right_finger_1": Moving("gripper_tip_3", "gripper_tip_1_1", "right_gripper"),
        "right_finger_2": Moving(
            "gripper_tip_2", "gripper_tip_2_1", "right_gripper", "right_finger_1"
        ),
    },
    hulls=("wrist_camera",),
    # What the wrist cameras see up close (5-10 cm): renders from the wrist
    # should match real images.
    tight=("_gripper", "_finger_1", "_finger_2"),
    wrist_parts={
        "gripper": 0.25,
        "finger_1": 0.04,
        "finger_2": 0.04,
        "wrist_camera": 0.08,
    },
    output=URDF_DIR / "axol_jelly.urdf",
    mesh_subdir=MESH_SUBDIR,
    share_subdir=None,
    wrist_cameras=True,
    gripper_joints=(
        "each gripper's fingers (left_finger_1 / right_finger_1, the second "
        "finger mimics; 0 = closed, 0.0588 m = open)"
    ),
    wrist_parts_note="the gripper, fingers and wrist camera",
)

# The parcel gripper (AxolConfig.gripper = "parcel"): a fixed jaw and a plate
# hinged on it, no wrist camera. Its export's body and arms are the parallel
# export's, so its URDFs reuse those meshes.
PARCEL = Variant(
    body=JELLY,
    gripper="parcel",
    marker="thumb_l",
    anchors={
        "finger_l": "left_gripper",
        "thumb_l": "left_plate",
        "finger_r": "right_gripper",
        "thumb_r": "right_plate",
    },
    extra_links={},
    moving={
        "left_plate": Moving("thumb_l", "g_l", "left_gripper"),
        "right_plate": Moving("thumb_r", "g_r", "right_gripper"),
    },
    hulls=(),
    tight=("_gripper", "_plate"),
    wrist_parts={"gripper": 0.30, "plate": 0.08},
    output=URDF_DIR / "axol_jelly_parcel.urdf",
    mesh_subdir="meshes/jelly_parcel",
    share_subdir=MESH_SUBDIR,
    wrist_cameras=False,
    gripper_joints=(
        "each gripper's hinged plate (left_plate_joint / right_plate_joint, "
        "revolute over the export's half turn: left [-pi, 0], right [0, pi], "
        "0 = folded flat along the jaw)"
    ),
    wrist_parts_note="the jaw and plate",
)


def _mount(body: Body) -> Variant:
    """The parallel gripper on a non-Jelly body: its own body meshes, and
    everything else shared with Jelly's."""
    return replace(
        PARALLEL,
        body=body,
        output=URDF_DIR / f"axol_{body.key}.urdf",
        mesh_subdir=f"meshes/{body.key}",
        share_subdir=MESH_SUBDIR,
    )


VARIANTS = (PARALLEL, PARCEL, *(_mount(b) for b in (STANDALONE, OWL, OX_CART)))


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
    fits = []
    for signs in itertools.product((1.0, -1.0), repeat=len(CLASSIC_JOINTS)):
        sg = np.array(signs * 2)
        u, _, vt = np.linalg.svd((d_e * sg[:, None]).T @ d_c)
        rot = (u @ vt).T
        if np.linalg.det(rot) < 0:
            continue
        err = np.linalg.norm((rot @ (d_e * sg[:, None]).T).T - d_c)
        fits.append((err, sg, rot))
    err = min(f[0] for f in fits)
    if err > 1e-3:
        raise SystemExit(
            f"arm joint axes do not match the classic arms (err {err:.2e})"
        )
    # Axis directions carry no sign, and the two arms mirror each other, so
    # the robot turned half a turn fits the axes as well; which of those
    # tied fits scores lowest is float noise. The export is level and faces
    # forward (checked below), so take the fit closest to no rotation.
    _, sg, rot = max(
        (f for f in fits if f[0] < err + 1e-6), key=lambda f: np.trace(f[2])
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
    classic: yourdfpy.URDF,
    export: yourdfpy.URDF,
    tf: np.ndarray,
    sg: np.ndarray,
    variant: Variant,
) -> None:
    """Every arm link must move identically in both models (within CAD slop)."""
    rng = np.random.default_rng(0)
    for side, off in (("l", 0), ("r", 7)):
        c_names, e_names = _names(side)
        prefix = "left_" if side == "l" else "right_"
        pairs = [
            (bundled, exported)
            for exported, bundled in variant.all_anchors.items()
            if bundled.startswith(prefix)
            and bundled not in variant.all_extra_links
            and bundled not in variant.all_moving
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


def check_lift(export: yourdfpy.URDF, names: tuple[str, ...]) -> float:
    """The export's lift stages: same travel, each lowering along -z from 0
    (fully raised). Returns the travel per stage (0 without a lift)."""
    if not names:
        return 0.0
    stages = [export.joint_map[n] for n in names]
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
    print(f"lift: {len(stages)} stage(s) of {travel.pop():.3f} m, 0 = fully raised")
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


def groups(export: yourdfpy.URDF, anchors: dict[str, str]) -> dict[str, str]:
    """Export link -> bundled link (nearest anchored ancestor)."""
    parent = {j.child: j.parent for j in export.robot.joints}
    out = {}
    for link in export.link_map:
        cur = link
        while cur not in anchors and cur in parent:
            cur = parent[cur]
        if cur in anchors:
            out[link] = anchors[cur]
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
        self,
        classic: yourdfpy.URDF,
        export: yourdfpy.URDF,
        tf: np.ndarray,
        variant: Variant,
    ) -> None:
        _zero(classic)
        _zero(export)
        self.classic, self.export, self.tf = classic, export, tf
        self.variant = variant
        # The export root in the classic frame: the ground under the robot
        # when its body stands on it.
        self.jelly_center = tf[:3, 3].copy()
        # The floor link's pose, if the body stands on the floor; set by build().
        self.floor: np.ndarray | None = None
        # Strip links (see Body.strips) -> the link they are fixed to.
        self.strip_parent = {
            f"{prefix}_{i}": parent
            for parent, prefix, count in variant.body.strips.values()
            for i in range(count)
        }

    def __call__(self, link: str) -> np.ndarray:
        if link in self.variant.body.frame_links or link.startswith("deck_"):
            frame = np.eye(4)
            frame[:3, 3] = self.jelly_center
            return frame
        if link in self.strip_parent:
            return self(self.strip_parent[link])
        moving = self.variant.all_moving
        if link in moving:
            return self.tf @ self.export.get_transform(
                moving[link].export_link, self.export.base_link
            )
        return self.classic.get_transform(
            self.variant.all_extra_links.get(link, link), self.classic.base_link
        )

    def relative(self, parent: str, child: str) -> np.ndarray:
        return np.linalg.inv(self(parent)) @ self(child)


# Every mesh written this run: file name -> [(source geometry, path)], so an
# export whose part matches one already written references it rather than
# writing a copy.
_WRITTEN: dict[str, list[tuple[tuple[np.ndarray, ...], str]]] = defaultdict(list)
# Sources closer than this are the same part: two exports of one assembly
# place it a few µm apart (their CAD poses are rounded differently).
SAME_SOURCE = 1e-4


def _offset(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...]) -> np.ndarray | None:
    """The translation taking source ``b`` onto ``a`` (zero for the same
    source), or None when they are not the same part. The first array is
    the vertices; the rest must match as they are."""
    if a[0].shape != b[0].shape or not all(
        x.shape == y.shape and np.allclose(x, y, rtol=0.0, atol=SAME_SOURCE)
        for x, y in zip(a[1:], b[1:], strict=True)
    ):
        return None
    shift = (a[0] - b[0]).mean(axis=0)
    if not np.allclose(a[0], b[0] + shift, rtol=0.0, atol=SAME_SOURCE):
        return None
    return np.where(np.abs(shift) < SAME_SOURCE, 0.0, shift)


def _write_mesh(
    name: str,
    source: tuple[np.ndarray, ...],
    variant: Variant,
    make: Callable[[], trimesh.Trimesh],
    shift: bool = False,
) -> tuple[str, np.ndarray]:
    """Write ``make()`` as ``variant``'s ``name``; returns its URDF-relative
    path and the offset to place it at (zero unless ``shift``).

    ``source`` is the geometry it is made from. When an earlier export in
    this run wrote the same file from the same source (and the variant
    shares, ``variant.share_subdir``) — the body and arms are the same in every export — that file is
    referenced instead (simplification and hulls are not bit-reproducible,
    so the outputs themselves cannot be compared). With ``shift``, a source
    that is the same part moved (the fingers: some exports put their link
    frames elsewhere on the finger) is referenced too, at the offset.
    """
    for other, path in _WRITTEN[name]:
        if variant.share_subdir is None or path.startswith(f"{variant.mesh_subdir}/"):
            continue
        offset = _offset(source, other)
        if offset is not None and (shift or not offset.any()):
            return path, offset
    path = f"{variant.mesh_subdir}/{name}"
    make().export(URDF_DIR / path)
    _WRITTEN[name].append((source, path))
    return path, np.zeros(3)


def build_meshes(
    export: yourdfpy.URDF,
    export_dir: Path,
    frames: Frames,
    min_part: float,
    visual_faces: int,
    tolerance: float,
    loose_tolerance: float,
) -> tuple[
    dict[str, list[tuple[str, tuple[float, ...], np.ndarray]]],
    dict[str, tuple[str, list[np.ndarray]]],
    dict[str, np.ndarray],
]:
    """Write the shared visual meshes.

    Returns them, the collision pieces (``{name: (link, [points in that
    link's frame])}``), and every export link's vertices in the classic root
    frame (for the camera frames).
    """
    variant = frames.variant
    owner = groups(export, variant.all_anchors)
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
            split = variant.body.splits.get(Path(vis.geometry.mesh.filename).name)
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

    (URDF_DIR / variant.mesh_subdir).mkdir(parents=True, exist_ok=True)
    for old in (URDF_DIR / variant.mesh_subdir).glob("*.stl"):
        old.unlink()
    visuals: dict[str, list[tuple[str, tuple[float, ...], np.ndarray]]] = defaultdict(
        list
    )
    counters: dict[str, int] = defaultdict(int)
    worst = tight = (0.0, "")
    for (target, rgba), meshes in sorted(by_color.items()):
        merged = trimesh.util.concatenate(meshes)
        merged.merge_vertices()
        is_tight = target.endswith(variant.tight)
        tol = tolerance if is_tight else loose_tolerance
        name = f"{target}_{counters[target]}.stl"
        counters[target] += 1

        def make(merged=merged, target=target, is_tight=is_tight, tol=tol):
            nonlocal worst, tight
            out, error = _decimate(merged, visual_faces, tol)
            worst = max(worst, (error, target))
            if is_tight:
                tight = max(tight, (error, target))
            return out

        source = (merged.vertices, merged.faces, np.asarray(rgba))
        path, offset = _write_mesh(
            name, source, variant, make, shift=target in variant.moving
        )
        visuals[target].append((path, rgba, offset))
    print(
        f"visual meshes: worst deviation {worst[0] * 1e3:.2f} mm ({worst[1]}); "
        f"gripper {tight[0] * 1e3:.2f} mm ({tight[1]})"
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

    - Wrist ZED X One S (mono fisheye): at the projection centre from
      Stereolabs' model of the camera (``PROJECTION_CENTRE_Z``), on the lens
      axis 23.3 mm ahead of the body's seal ring and 2.2 mm behind the front
      glass; +z along the ring's normal away from the camera body. Fitting
      real wrist images through each camera's factory calibration agrees to
      within the fit's ~3 mm resolution. The sensor's roll
      cannot be read from the CAD (the body is square); it is set from real
      wrist images, which show the gripper at the bottom of the frame with
      the fingers pointing up into the scene: +y (image down) points from the
      lens toward that gripper's fingers.
    - Head ZED X Mini Wide (stereo): one frame per lens at its projection
      centre from Stereolabs' model (``HEAD_PROJECTION_BEHIND_COVER`` behind
      the centre of the lens's front cover, on its axis), +z out of the
      camera, +x from the left lens to the right one (the ZED convention:
      the right camera sits at +x of the left). The two covers are checked
      to be the 50 mm baseline apart.

    The wrist frames only for a gripper with wrist cameras.
    """
    out: dict[str, tuple[str, np.ndarray]] = {}
    wrists = (
        (
            "camera_component___seal___zed_x_one",
            "fisheye_lens",
            "camera_component___body_zed_x_one___rear",
            ("gripper_tip", "gripper_tip_1"),
            "zed_x_one_gs_compact__1_",
        ),
        (
            "camera_component___seal___zed_x_one_1",
            "fisheye_lens_1",
            "camera_component___body_zed_x_one___rear_1",
            ("gripper_tip_2", "gripper_tip_3"),
            "zed_x_one_gs_compact__1__1",
        ),
    )
    for seal, lens, rear, fingers, camera_frame in (
        wrists if frames.variant.wrist_cameras else ()
    ):
        ring = in_root[seal]
        centre = (ring.min(axis=0) + ring.max(axis=0)) / 2
        z = _thin_normal(ring)
        outward = in_root[lens].mean(axis=0) - in_root[rear].mean(axis=0)
        if z @ outward < 0:
            z = -z
        side = "left" if centre[1] > 0 else "right"
        parent = f"{side}_wrist_camera"
        # The optical centre, from Stereolabs' own model of the camera
        # (ZED_X_ONE_S_Fisheye.step): its "Simple FOV" solid's rays converge
        # on the lens axis PROJECTION_CENTRE_Z ahead of the camera's CAD
        # origin, which the export carries as the camera sub-assembly frame.
        cam = frames.tf @ frames.export.get_transform(
            camera_frame, frames.export.base_link
        )
        if abs(cam[:3, 2] @ z) < 1 - 1e-6:
            raise SystemExit(f"{camera_frame}'s z is not the lens axis")
        lens_origin = cam[:3, 3] + cam[:3, 2] * PROJECTION_CENTRE_Z
        ahead = float((lens_origin - centre) @ z)
        off_axis = float(np.linalg.norm((lens_origin - centre) - ahead * z))
        if not (0.015 < ahead < 0.030) or off_axis > 5e-4:
            raise SystemExit(
                f"projection centre is not on the lens axis ahead of the seal "
                f"ring ({ahead * 1e3:.1f} mm ahead, {off_axis * 1e3:.2f} mm off axis)"
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
    for name, cover in (("left", left), ("right", right)):
        centre = cover - z * HEAD_PROJECTION_BEHIND_COVER
        depth = (in_root["_zedx_mini_rev2__upper_case_v8"] - centre) @ z
        if not depth.min() < 0 < depth.max():
            raise SystemExit("head projection centre is not inside the camera")
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
            stem = link if names == (link,) else "_".join(names)
            if stem != link:
                stem = f"{link}__{stem}"
            self.written[key], _ = _write_mesh(
                f"{stem}_collision.stl",
                (self.points(link, names),),
                self.frames.variant,
                lambda: trimesh.PointCloud(snapped).convex_hull,
            )
        return ("mesh", self.written[key])

    def finger_cylinder(self, link: str, fingers: tuple[str, ...]) -> Spec:
        """One cylinder along the gripper's jaw axis over both fingers, closed
        and fully open (fingers move along their joint axis, in their frame)."""
        pts = []
        for finger in fingers:
            closed = np.vstack(self.pieces[finger][1])
            joint = self.export.joint_map[
                self.frames.variant.moving[finger].export_joint
            ]
            # The export's pose is anywhere in the stroke: take both ends.
            ends = [
                closed + np.asarray(joint.axis) * float(q)
                for q in (joint.limit.lower, joint.limit.upper, 0.0)
            ]
            rel = self.frames.relative(link, finger)
            pts += [trimesh.transform_points(p, rel) for p in ends]
        radius, length, centre = _fit_cylinder(np.vstack(pts), 0)
        return ("cylinder", radius, length, centre, _CYLINDER_RPY[0])

    def plate_sweep(self, link: str, plate: str, steps: int = 37) -> Spec:
        """One cylinder over a hinged plate's whole sweep (it turns about its
        joint axis through its frame's origin), along the link axis that
        gives the smallest capsule."""
        rest = np.vstack(self.pieces[plate][1])
        joint = self.export.joint_map[self.frames.variant.moving[plate].export_joint]
        rel = self.frames.relative(link, plate)
        pts = []
        for angle in np.linspace(joint.limit.lower, joint.limit.upper, steps):
            turn = trimesh.transformations.rotation_matrix(angle, joint.axis)
            pts.append(trimesh.transform_points(rest, rel @ turn))
        pts = np.vstack(pts)
        fits = [(_fit_cylinder(pts, axis), axis) for axis in range(3)]
        (radius, length, centre), axis = min(
            fits, key=lambda f: f[0][0] ** 2 * (f[0][1] + 4 / 3 * f[0][0])
        )
        return ("cylinder", radius, length, centre, _CYLINDER_RPY[axis])

    def deck_strips(
        self,
        parent: str = "jelly",
        piece: str = "deck",
        prefix: str = "deck",
        count: int = DECK_STRIPS,
    ) -> dict[str, list[Spec]]:
        """``piece`` as ``count`` strips, cut front to back (along world x)
        and each a cylinder across the robot (along world y), in ``parent``'s
        frame — so the strips mirror left to right like the arms."""
        # The deck mesh is a box with vertices only at its edges, so strip
        # its bounding box rather than binning vertices.
        pts = self.points(parent, (piece,))
        rot = self.frames(parent)[:3, :3]
        cut, along = (int(np.argmax(np.abs(rot[k]))) for k in (0, 1))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        edges = np.linspace(lo[cut], hi[cut], count + 1)
        out = {}
        for i in range(count):
            ranges = [(lo[k], hi[k]) for k in range(3)]
            ranges[cut] = (edges[i], edges[i + 1])
            corners = np.array(list(itertools.product(*ranges)))
            radius, length, centre = _fit_cylinder(corners, along)
            out[f"{prefix}_{i}"] = [
                ("cylinder", radius, length, centre, _CYLINDER_RPY[along])
            ]
        return out


def collision_plan(kind: str, col: Collision) -> dict[str, list[Spec]]:
    """The model's collision bodies (link -> specs): ``"ik"`` the
    capsule-friendly set IK collides (<ik_collision>), ``"sim"`` one hull per
    part for physics (<collision>)."""
    variant = col.frames.variant
    body = variant.body
    hulls = (
        *(
            f"{side}_{part}"
            for side in ("left", "right")
            for part in (*_ARM_PARTS, *variant.hulls)
        ),
        *HEAD_LINKS,
        "s1",
    )
    plan: dict[str, list[Spec]] = {link: [col.hull(link, (link,))] for link in hulls}
    # Each gripper's moving parts (fingers or plate).
    jaws = {
        side: tuple(n for n in variant.moving if n.startswith(f"{side}_"))
        for side in ("left", "right")
    }
    if kind == "ik":
        for link, pieces in body.ik_hulls.items():
            plan[link] = [col.hull(link, pieces)]
        for piece, (parent, prefix, count) in body.strips.items():
            plan.update(col.deck_strips(parent, piece, prefix, count))
        # IK freezes the jaws, so each gripper's collide as one body over
        # their whole motion.
        for side, parts in jaws.items():
            plan[f"{side}_fingers"] = [
                col.plate_sweep(f"{side}_gripper", *parts)
                if variant.gripper == "parcel"
                else col.finger_cylinder(f"{side}_gripper", parts)
            ]
    else:
        for link, parts in body.sim_hulls.items():
            plan[link] = [col.hull(link, pieces) for pieces in parts]
        for parts in jaws.values():
            for part in parts:
                plan[part] = [col.hull(part, (part,))]
        for wheel in body.moving:
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
        for part, mass in col.frames.variant.wrist_parts.items():
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
    for link, mass in col.frames.variant.body.masses.items():
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


def _collision(element: ET.Element, tag: str, spec: Spec) -> None:
    col = ET.SubElement(element, tag)
    if spec[0] == "mesh":
        ET.SubElement(col, "origin", xyz="0 0 0", rpy="0 0 0")
        geom = ET.SubElement(col, "geometry")
        ET.SubElement(geom, "mesh", filename=spec[1], scale="1 1 1")
    else:
        _, radius, length, xyz, rpy = spec
        ET.SubElement(col, "origin", xyz=_fmt(xyz), rpy=rpy)
        geom = ET.SubElement(col, "geometry")
        ET.SubElement(geom, "cylinder", radius=f"{radius:g}", length=f"{length:g}")


def _inertial(element: ET.Element, mass: float, com: np.ndarray, tensor) -> None:
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


# The planar base's two intermediate links carry no part; physics engines
# want some mass on every moving link.
_VIRTUAL_MASS = 0.01


def write_urdf(
    out_path: Path,
    visuals: dict[str, list[tuple[str, tuple[float, ...], np.ndarray]]],
    ik_plan: dict[str, list[Spec]],
    sim_plan: dict[str, list[Spec]],
    limits: dict[str, tuple[float, float]],
    frames: Frames,
    cameras: dict[str, tuple[str, np.ndarray]],
    export: yourdfpy.URDF,
    lift_travel: float,
    inertials: dict[str, tuple[float, np.ndarray, np.ndarray]],
    dynamics: dict[str, dict[str, float]],
) -> None:
    """The variant's one model: every moving part a joint, both collision
    sets, masses and drives (see MODEL_NOTE)."""
    tree = ET.parse(CLASSIC_URDF)
    robot = tree.getroot()
    robot.set("name", "assembly")
    links = {ln.get("name"): ln for ln in robot.findall("link")}
    centre = frames.jelly_center
    variant = frames.variant
    body = variant.body

    def link(name: str) -> None:
        links[name] = ET.SubElement(robot, "link", name=name)

    for name, parent in variant.all_extra_links.items():
        link(name)
        _joint(robot, f"{name}_0", "fixed", parent, name)
    for name in (*body.frame_links, *variant.all_moving, *cameras):
        link(name)
    for name, (parent, pose) in cameras.items():
        _joint(robot, f"{name}_0", "fixed", parent, name, pose)
    for strip, parent in frames.strip_parent.items():
        link(strip)
        _joint(robot, f"{strip}_0", "fixed", parent, strip)

    # A body the torso rides on: world -> (planar base ->) the body's base
    # link -> lift stages -> the torso root (the classic tree). At every
    # body joint's zero, "root" is the world origin.
    if body.lift_links:
        base_link = body.lift_links[0]
        link("world")
        if frames.floor is not None:
            link(FLOOR_LINK)
            _joint(robot, f"{FLOOR_LINK}_0", "fixed", "world", FLOOR_LINK, frames.floor)
        if body.planar:
            for name in ("base_x_link", "base_y_link"):
                link(name)
            travel = (-BASE_TRAVEL, BASE_TRAVEL)
            planar = SIM_LIMITS["base"]
            _joint(
                robot,
                "base_x",
                "prismatic",
                "world",
                "base_x_link",
                _translation(centre),
                np.array([1.0, 0, 0]),
                travel,
                effort_velocity=planar,
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
                effort_velocity=planar,
            )
            _joint(
                robot,
                "base_yaw",
                "revolute",
                "base_y_link",
                base_link,
                None,
                np.array([0, 0, 1.0]),
                (-2 * np.pi, 2 * np.pi),
                effort_velocity=SIM_LIMITS["base_yaw"],
            )
        else:
            _joint(
                robot,
                f"{base_link}_0",
                "fixed",
                "world",
                base_link,
                _translation(centre),
            )
        stages = list(zip(body.lift_links, body.lift_links[1:]))
        for k, (parent, child) in enumerate(stages):
            _joint(
                robot,
                "lift" if k == 0 else f"lift_{k + 1}",
                "prismatic",
                parent,
                child,
                _translation(-centre) if child == "root" else None,
                np.array([0, 0, -1.0]),
                (0.0, lift_travel),
                mimic="lift" if k else None,
                effort_velocity=SIM_LIMITS["lift"],
            )
    elif frames.floor is not None:
        link(FLOOR_LINK)
        _joint(robot, f"{FLOOR_LINK}_0", "fixed", "root", FLOOR_LINK, frames.floor)

    for name, moving in variant.all_moving.items():
        joint = export.joint_map[moving.export_joint]
        pose = frames.relative(moving.parent, name)
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
        elif joint.type == "revolute":
            # The parcel gripper's hinged plate: the export's range as is.
            _joint(
                robot,
                f"{name}_joint",
                "revolute",
                moving.parent,
                name,
                pose,
                joint.axis,
                (float(joint.limit.lower), float(joint.limit.upper)),
                effort_velocity=SIM_LIMITS["plate"],
            )
        else:
            # Fingers: 0 = the export's pose, along its axis over its range.
            _joint(
                robot,
                f"{name}_joint",
                "prismatic",
                moving.parent,
                name,
                pose,
                joint.axis,
                (float(joint.limit.lower), float(joint.limit.upper)),
                mimic=f"{moving.mimic}_joint" if moving.mimic else None,
                effort_velocity=SIM_LIMITS["finger"],
            )

    virtual = {"base_x_link", "base_y_link"} & set(links)
    for name, element in links.items():
        for tag in ("visual", "collision", "inertial"):
            for el in element.findall(tag):
                element.remove(el)
        if name in inertials:
            _inertial(element, *inertials[name])
        elif name in virtual:
            _inertial(element, _VIRTUAL_MASS, np.zeros(3), np.eye(3) * 1e-6)
        for path, rgba, offset in visuals.get(name, []):
            vis = ET.SubElement(element, "visual")
            ET.SubElement(vis, "origin", xyz=_fmt(offset), rpy="0 0 0")
            geom = ET.SubElement(vis, "geometry")
            ET.SubElement(geom, "mesh", filename=path, scale="1 1 1")
            mat = ET.SubElement(
                vis, "material", name="c_" + "_".join(f"{c:g}" for c in rgba)
            )
            ET.SubElement(mat, "color", rgba=" ".join(f"{c:g}" for c in rgba))
        for spec in sim_plan.get(name, []):
            _collision(element, "collision", spec)
        # Not URDF: importers ignore it; IK collides these instead.
        for spec in ik_plan.get(name, []):
            _collision(element, "ik_collision", spec)
    for joint in robot.findall("joint"):
        lim = limits.get(joint.get("name", ""))
        if lim is not None:
            joint.find("limit").set("lower", f"{lim[0]:.4f}")
            joint.find("limit").set("upper", f"{lim[1]:.4f}")
        dyn = dynamics.get(joint.get("name", ""))
        if dyn is not None:
            ET.SubElement(
                joint,
                "dynamics",
                damping=f"{dyn['damping']:g}",
                friction=f"{dyn['friction']:g}",
            )
            # Not URDF: importers ignore it, so the gains travel with the
            # model and are applied as position-drive gains after import.
            ET.SubElement(
                joint,
                "drive",
                stiffness=f"{dyn['stiffness']:g}",
                damping=f"{dyn['drive_damping']:g}",
            )
    ET.indent(tree, space="    ")
    xml = ET.tostring(robot, encoding="unicode")
    gripper = (
        "With the parcel gripper (AxolConfig.gripper = parcel): a fixed jaw "
        "and a plate hinged on it, no wrist cameras. "
        if variant.gripper == "parcel"
        else ""
    )
    model_note = MODEL_NOTE.format(
        body_joints=BODY_JOINTS_NOTE.get(body.key, ""),
        gripper_joints=variant.gripper_joints,
        ik=body.note,
        wrist_parts=variant.wrist_parts_note,
        sim=SIM_BODY_NOTE.get(body.key, ""),
    )
    note = (
        f"{body.title}. GENERATED by scripts/build_urdfs.py "
        "from the Onshape export; do not edit by hand. "
        + gripper
        + "Arm links, joints and "
        "joint frames are the classic axol.urdf's, so FK and gravity "
        "compensation match it exactly; joint limits and meshes are the "
        "export's. "
        + model_note
        + " "
        + (WRIST_CAMERA_NOTE if variant.wrist_cameras else HEAD_CAMERA_NOTE)
    )
    header = "<!-- " + textwrap.fill(note, 76, subsequent_indent="     ") + "\n-->\n"
    out_path.write_text(header + xml + "\n")
    print(f"wrote {out_path.relative_to(REPO)}")


def build(args: argparse.Namespace, export_dir: Path) -> None:
    """Every URDF of one export (its gripper's Variant)."""
    urdfs = sorted((export_dir / "urdf").glob("*.urdf"))
    if len(urdfs) != 1:
        raise SystemExit(f"expected one URDF under {export_dir / 'urdf'}")
    classic = _load(CLASSIC_URDF)
    export = _load(urdfs[0])
    body = next(b for b in BODIES if b.marker is None or b.marker in export.link_map)
    found = [v for v in VARIANTS if v.body is body and v.marker in export.link_map]
    if len(found) != 1:
        raise SystemExit(f"{export_dir}: not a known body and gripper's export")
    variant = found[0]
    print(f"== {export_dir} ({body.title}, {variant.gripper} gripper)")
    tf, sg = align(classic, export)
    check_equivalent(classic, export, tf, sg, variant)
    lift_travel = check_lift(export, body.lift_joints)
    limits = export_limits(export, sg)
    for name, (lo, hi) in limits.items():
        old = classic.joint_map[name].limit
        mark = (
            ""
            if (round(old.lower, 4), round(old.upper, 4)) == (lo, hi)
            else "  <- changed"
        )
        print(f"  {name:12s} [{lo:+.4f}, {hi:+.4f}]{mark}")
    frames = Frames(classic, export, tf, variant)
    visuals, pieces, in_root = build_meshes(
        export,
        export_dir,
        frames,
        args.min_part_mm,
        args.visual_faces,
        args.visual_tolerance_mm * 1e-3,
        args.loose_tolerance_mm * 1e-3,
    )
    if body.floor:
        # Under the export root, at the body's lowest point (the wheels' or
        # casters' bottoms).
        owner = groups(export, variant.all_anchors)
        mine = [
            pts for n, pts in in_root.items() if owner.get(n) in body.anchors.values()
        ]
        frames.floor = _translation(frames.jelly_center)
        frames.floor[2, 3] = float(np.vstack(mine)[:, 2].min())
    cameras = camera_frames(in_root, frames)
    col = Collision(frames, pieces, export)
    ik_plan = collision_plan("ik", col)
    sim_plan = collision_plan("sim", col)
    report("ik", ik_plan)
    report("sim", sim_plan)
    write_urdf(
        variant.output,
        visuals,
        ik_plan,
        sim_plan,
        limits,
        frames,
        cameras,
        export,
        lift_travel,
        sim_inertials(col, sim_plan),
        sim_dynamics(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "export_dirs",
        type=Path,
        nargs="+",
        help="Onshape exports (each contains urdf/ and meshes/), one per "
        "gripper; the parallel gripper's first, since the others reuse its "
        "body meshes",
    )
    ap.add_argument("--min-part-mm", type=float, default=15.0)
    ap.add_argument("--visual-faces", type=int, default=2000)
    ap.add_argument(
        "--visual-tolerance-mm",
        type=float,
        default=0.3,
        help="largest deviation from the CAD of the grippers' visual meshes "
        "(what the wrist cameras see)",
    )
    ap.add_argument(
        "--loose-tolerance-mm",
        type=float,
        default=1.0,
        help="largest deviation from the CAD of every other visual mesh",
    )
    args = ap.parse_args()
    for export_dir in args.export_dirs:
        build(args, export_dir)


if __name__ == "__main__":
    main()
