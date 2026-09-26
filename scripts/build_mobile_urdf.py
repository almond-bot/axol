"""
Build the bundled Axol Mobile URDFs from their Onshape export.

The Onshape export (``almond_urdf/urdf/almond_urdf.urdf`` + ``meshes/``) is a
faithful CAD tree — ~140 links including every screw, a two-stage prismatic
lift, prismatic gripper fingers, and link frames wherever Onshape put them.
None of that matches what the control stack expects, so this script distils it
into two URDFs that share one set of visual meshes:

- ``axol_mobile.urdf`` — **arm IK** (the default): only the 14 arm joints
  move; the base and lift are frozen (lift fully raised), rooted at the
  classic torso-relative world frame.
- ``axol_mobile_whole_body.urdf`` — **whole-body IK**: the same torso and
  arms on top of the Jelly's planar base (``base_x`` / ``base_y`` /
  ``base_yaw`` about the Jelly centre) and its lift (``lift``, with the
  second stage mimicking it; 0 = fully raised, positive lowers each stage).
  Its world frame is where the arm model's frame is with the body joints at
  zero, so the two agree at startup.

Both:

- **Arms are the classic arms.** Every arm link, joint name, joint frame, axis
  and inertial is copied verbatim from ``axol.urdf``; only the joint *limits*
  come from the export. The script first proves the two arm chains are the
  same mechanism (it fits the rigid transform between the exports' joint
  axes, checks every arm link moves identically under random joint angles,
  and reports the per-joint sign flips between the two naming schemes), so
  FK, IK targets, recorded Cartesian frames and gravity compensation are
  unchanged between the two versions.
- **Meshes are the export's**, re-expressed in the classic link frames: each
  rigid group of export links (a link plus its fastened children) is merged
  per colour into one decimated visual STL, and one convex hull for
  collision. Fasteners and connector bits under ``--min-part-mm`` are dropped.
- **The body is split for capsule collision.** pyroki fits one capsule per
  link, so the body is kept as capsule-friendly pieces: ``s1`` (the shoulder
  bar), ``lift_plate`` (the bracket it bolts onto), ``head`` (the head-camera
  mount) and the lift column. In the arm model the column is one link
  (``base``) and the Jelly deck is visual only (see ``ARM_NOTE``); in the
  whole-body model each column section rides its own stage (``base``,
  ``lift_stage``, ``jelly``) and the deck is covered by ``deck_0..2``.
- **The fingers and wheels are frozen** (fingers closed; wheel spin is not a
  pose — base motion is the ``base_*`` joints). The fingers' collision
  cylinder spans their full stroke.

Run (``fast-simplification`` is only needed here, not at runtime)::

    uv run --with fast-simplification python scripts/build_mobile_urdf.py \\
        path/to/almond_urdf
"""

from __future__ import annotations

import argparse
import itertools
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import trimesh
import yourdfpy

REPO = Path(__file__).resolve().parents[1]
URDF_DIR = REPO / "almond_axol" / "kinematics" / "urdf"
CLASSIC_URDF = URDF_DIR / "axol.urdf"
OUT_URDF = URDF_DIR / "axol_mobile.urdf"
OUT_WHOLE_BODY_URDF = URDF_DIR / "axol_mobile_whole_body.urdf"
# Travel of each planar base joint in the whole-body model (m): far beyond
# any single session, so the joint limits never shape the solve.
BASE_TRAVEL = 10.0
MESH_SUBDIR = "meshes/mobile"
PACKAGE = "package://assembly/"

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
    "gripper_tip": "left_fingers",
    "gripper_tip_1": "left_fingers",
    "wrist_mount": "left_wrist_camera",
    "s2_right": "right_s2",
    "s3_right": "right_s3",
    "e1_right": "right_e1",
    "e2_right": "right_e2",
    "w0_right": "right_w0",
    "w1_right": "right_w1",
    "w2": "right_w2",
    "gripper_base_1": "right_gripper",
    "gripper_tip_2": "right_fingers",
    "gripper_tip_3": "right_fingers",
    "wrist_mount_1": "right_wrist_camera",
}

# Bundled links that exist only in the mobile model, and the classic link each
# is rigidly attached to (identity origin: their meshes are placed in the
# parent's frame).
EXTRA_LINKS: dict[str, str] = {
    "head": "s1",
    "lift_plate": "base",
    # Split off the gripper so each gets its own (much tighter) capsule.
    "left_fingers": "left_gripper",
    "right_fingers": "right_gripper",
    "left_wrist_camera": "left_gripper",
    "right_wrist_camera": "right_gripper",
}

# The Jelly body links: frames world-aligned at the ground under the Jelly
# centre (the export root), in both models. In the arm model they are fixed
# to the torso-relative root; in the whole-body model ``jelly`` is the child
# of the planar base joints and ``lift_stage`` rides the first lift stage.
BODY_FRAME_LINKS = ("jelly", "lift_stage")

# Collision pieces: each export mesh feeds the piece named after its group,
# unless split here by a mesh-local box ``(lo, hi)`` into the piece inside
# and the piece outside it ("" = no collision). Each model then builds its
# collision links from pieces (ARM_COLLISION / WHOLE_BODY_COLLISION).
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
    "Wheel.stl": (*_ALL, "", ""),
}

# Collision link -> the pieces it covers. Links absent here (the arms, s1,
# head, lift_plate) collide as their own piece.
ARM_COLLISION: dict[str, tuple[str, ...]] = {
    "base": ("column_top", "column_mid", "column_stub"),
}
WHOLE_BODY_COLLISION: dict[str, tuple[str, ...]] = {
    "base": ("column_top",),
    "lift_stage": ("column_mid",),
    "jelly": ("column_stub",),
}
# The deck (0.6 x 0.6 m, ~0.24 m tall) as strips across the robot: one
# capsule over the whole deck would swallow the arms; three along y stay
# within a few cm of it.
DECK_STRIPS = 3

# Links whose collision body is an explicit cylinder along a link-frame axis
# (0 = x) rather than a hull. pyroki fits one minimum-volume cylinder per
# link, and for the near-cubic closed fingers that fit is ambiguous — mirror-
# image hulls landed on different tilted axes, so the two hands collided
# differently. A cylinder's fit is itself: deterministic and symmetric. It
# runs along the jaw-opening axis and is sized over the full finger stroke,
# so it covers the fingers open as well as closed.
COLLISION_CYLINDERS: dict[str, int] = {"left_fingers": 0, "right_fingers": 0}
_CYLINDER_RPY = {0: "0 1.5707963267948966 0", 1: "-1.5707963267948966 0 0", 2: "0 0 0"}

# Frame at the ground under the robot (the export's root).
FLOOR_LINK = "floor"

ARM_NOTE = (
    "Arm IK model: the base and lift are frozen, the lift fully raised (its "
    "CAD pose). At that height the Jelly deck sits below anything the arms "
    "can reach, so it is visual only; the arms collide with the shoulder bar "
    "(s1), the lift column (base), its top plate (lift_plate) and the "
    "head-camera mount (head), exactly as the classic arms collide with the "
    "classic base and s1."
)
WHOLE_BODY_NOTE = (
    "Whole-body IK model: the Jelly base moves in the plane (base_x, base_y, "
    "base_yaw about the Jelly centre) and the lift lowers the torso (lift, "
    "0 = fully raised, per stage; the second stage mimics it). The world "
    "frame is the arm model's with every body joint at zero. The arms "
    "collide with the shoulder bar, top plate, head camera, each lift-column "
    "section and the deck (deck_0..2)."
)


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


def _decimate(mesh: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    """Quadric decimation, falling back to grid clustering where it stalls.

    CAD fasteners and connectors (thread flanks, knurling) often defeat the
    quadric simplifier; clustering always converges and is plenty for a
    viewer mesh.
    """
    import fast_simplification

    if len(mesh.faces) <= target:
        return mesh
    v, f = fast_simplification.simplify(
        mesh.vertices, mesh.faces, 1.0 - target / len(mesh.faces)
    )
    mesh = trimesh.Trimesh(v, f)
    cell = 5e-4
    while len(mesh.faces) > target * 1.2:
        mesh = _cluster(mesh, cell)
        cell *= 1.5
    return mesh


def build_meshes(
    export: yourdfpy.URDF,
    export_dir: Path,
    classic: yourdfpy.URDF,
    tf: np.ndarray,
    min_part: float,
    visual_faces: int,
) -> tuple[
    dict[str, list[tuple[str, tuple[float, ...]]]],
    dict[str, tuple[str, list[np.ndarray]]],
]:
    """Write the shared visual meshes; return them and the collision pieces.

    Pieces are ``{name: (link, [points in that link's frame])}``; each
    model groups them into its own collision links (:func:`collisions`).
    """
    owner = groups(export)
    _zero(classic)

    by_color: dict[tuple[str, tuple[float, ...]], list[trimesh.Trimesh]] = defaultdict(
        list
    )
    pieces: dict[str, tuple[str, list[np.ndarray]]] = {}

    def add(piece: str, link: str, pts: np.ndarray) -> None:
        if piece:
            pieces.setdefault(piece, (link, []))[1].append(pts)

    stroke = {
        j.child: j
        for j in export.robot.joints
        if j.type == "prismatic" and "gripper" in j.name
    }
    for link_name, link in export.link_map.items():
        target = owner.get(link_name)
        if target is None:
            continue
        in_target = np.linalg.inv(link_frame(classic, tf, target)) @ tf
        for vis in link.visuals:
            mesh = _mesh(export_dir, vis.geometry.mesh.filename)
            origin = vis.origin if vis.origin is not None else np.eye(4)
            pose = export.get_transform(link_name, export.base_link)
            placed = mesh.copy().apply_transform(in_target @ pose @ origin)
            if target in COLLISION_CYLINDERS and link_name in stroke:
                joint = stroke[link_name]
                _zero(export, **{joint.name: float(joint.limit.upper)})
                opened = export.get_transform(link_name, export.base_link)
                _zero(export)
                add(
                    target,
                    target,
                    trimesh.transform_points(
                        mesh.vertices, in_target @ opened @ origin
                    ),
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
    for (target, rgba), meshes in sorted(by_color.items()):
        merged = trimesh.util.concatenate(meshes)
        merged.merge_vertices()
        merged = _decimate(merged, visual_faces)
        name = f"{target}_{counters[target]}.stl"
        counters[target] += 1
        merged.export(URDF_DIR / MESH_SUBDIR / name)
        visuals[target].append((f"{MESH_SUBDIR}/{name}", rgba))
    return visuals, pieces


def link_frame(classic: yourdfpy.URDF, tf: np.ndarray, link: str) -> np.ndarray:
    """A bundled link's frame in the classic root frame (every joint at zero)."""
    if link in BODY_FRAME_LINKS or link.startswith("deck_"):
        frame = np.eye(4)
        frame[:3, 3] = tf[:3, 3]
        return frame
    return classic.get_transform(EXTRA_LINKS.get(link, link), classic.base_link)


CollisionSpec = str | tuple[float, float, np.ndarray, str]


def collisions(
    classic: yourdfpy.URDF,
    tf: np.ndarray,
    pieces: dict[str, tuple[str, list[np.ndarray]]],
    grouping: dict[str, tuple[str, ...]],
    *,
    deck: bool,
    tag: str,
) -> dict[str, CollisionSpec]:
    """One model's collision bodies from the shared pieces.

    Hulls are written as ``<link>[_<tag>]_collision.stl``; the tag only
    appears where a link's hull differs from the arm model's.
    """
    grouped = {p for ps in grouping.values() for p in ps}
    plan: dict[str, tuple[str, ...]] = dict(grouping)
    for piece, (link, _) in pieces.items():
        if piece not in grouped and piece in (link, "lift_plate"):
            plan.setdefault(piece, (piece,))

    def points(link: str, names: tuple[str, ...]) -> np.ndarray:
        dst = np.linalg.inv(link_frame(classic, tf, link))
        return np.vstack(
            [
                trimesh.transform_points(
                    np.vstack(pieces[n][1]), dst @ link_frame(classic, tf, pieces[n][0])
                )
                for n in names
            ]
        )

    out: dict[str, CollisionSpec] = {}
    for link, names in sorted(plan.items()):
        pts = points(link, names)
        if link in COLLISION_CYLINDERS:
            out[link] = _fit_cylinder(pts, COLLISION_CYLINDERS[link])
            radius, length = out[link][:2]
            print(
                f"  {link:18s} cylinder r={radius * 1e3:5.1f} mm h={length * 1e3:5.1f} mm"
            )
            continue
        # Snapping to a 2 mm grid first keeps the hull to a few hundred faces
        # (<= 1.7 mm of shape error, far inside the capsule fit's slack).
        snapped = np.unique(np.round(pts / 2e-3) * 2e-3, axis=0)
        hull = trimesh.PointCloud(snapped).convex_hull
        suffix = (
            f"_{tag}" if tag and grouping.get(link) != ARM_COLLISION.get(link) else ""
        )
        name = f"{link}{suffix}_collision.stl"
        hull.export(URDF_DIR / MESH_SUBDIR / name)
        out[link] = f"{MESH_SUBDIR}/{name}"
        cyl = trimesh.bounds.minimum_cylinder(hull)
        print(
            f"  {link:18s} hull {len(hull.faces):4d} faces, capsule "
            f"r={cyl['radius'] * 1e3:5.1f} mm h={cyl['height'] * 1e3:5.1f} mm"
        )
    if deck:
        # The deck mesh is a box with vertices only at its edges, so strip
        # its bounding box rather than binning vertices.
        pts = points("jelly", ("deck",))
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        edges = np.linspace(lo[0], hi[0], DECK_STRIPS + 1)
        for i in range(DECK_STRIPS):
            corners = np.array(
                list(
                    itertools.product(
                        (edges[i], edges[i + 1]), (lo[1], hi[1]), (lo[2], hi[2])
                    )
                )
            )
            out[f"deck_{i}"] = _fit_cylinder(corners, 1)
            radius, length = out[f"deck_{i}"][:2]
            print(
                f"  deck_{i:<13d} cylinder r={radius * 1e3:5.1f} mm h={length * 1e3:5.1f} mm"
            )
    return out


def _fit_cylinder(pts: np.ndarray, axis: int) -> tuple[float, float, np.ndarray, str]:
    """Smallest capsule along link axis ``axis`` enclosing ``pts``.

    Returned as the URDF cylinder pyroki turns into that capsule: its radius,
    its length (the capsule's straight section), its centre, and the rpy that
    turns the cylinder's z onto ``axis``.
    """
    others = [i for i in range(3) if i != axis]
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2
    rho = np.linalg.norm(pts[:, others] - center[others], axis=1)
    radius = float(rho.max())
    # A point beyond the straight section is covered by the hemispherical
    # cap as long as its axial overhang is within sqrt(r^2 - rho^2).
    overhang = np.abs(pts[:, axis] - center[axis]) - np.sqrt(
        np.maximum(radius**2 - rho**2, 0.0)
    )
    half = max(float(overhang.max()), 0.0)
    return round(radius, 5), round(2 * half, 5), center, _CYLINDER_RPY[axis]


def _fmt(v: np.ndarray) -> str:
    return " ".join(f"{x:.6g}" if abs(x) > 1e-12 else "0" for x in v)


def _joint(
    robot: ET.Element,
    name: str,
    kind: str,
    parent: str,
    child: str,
    xyz: np.ndarray | None = None,
    axis: str | None = None,
    limits: tuple[float, float] | None = None,
    mimic: str | None = None,
) -> None:
    joint = ET.SubElement(robot, "joint", name=name, type=kind)
    ET.SubElement(
        joint, "origin", xyz="0 0 0" if xyz is None else _fmt(xyz), rpy="0 0 0"
    )
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    if axis is not None:
        ET.SubElement(joint, "axis", xyz=axis)
    if limits is not None:
        ET.SubElement(
            joint,
            "limit",
            lower=f"{limits[0]:.6g}",
            upper=f"{limits[1]:.6g}",
            effort="1",
            velocity="1",
        )
    if mimic is not None:
        ET.SubElement(joint, "mimic", joint=mimic, multiplier="1", offset="0")


def write_urdf(
    out_path: Path,
    classic_path: Path,
    visuals: dict[str, list[tuple[str, tuple[float, ...]]]],
    collision: dict[str, CollisionSpec],
    limits: dict[str, tuple[float, float]],
    tf: np.ndarray,
    *,
    whole_body: bool,
    lift_travel: float,
) -> None:
    tree = ET.parse(classic_path)
    robot = tree.getroot()
    robot.set("name", "assembly")
    links = {ln.get("name"): ln for ln in robot.findall("link")}
    jelly_center = tf[:3, 3]

    def link(name: str) -> ET.Element:
        links[name] = ET.SubElement(robot, "link", name=name)
        return links[name]

    # Where the wheels meet the ground (the export's root), world-fixed.
    # Viewers draw their ground plane here; the world frame itself stays the
    # classic one (0.86 m below the shoulders).
    link(FLOOR_LINK)
    for name, parent in EXTRA_LINKS.items():
        link(name)
        _joint(robot, f"{name}_0", "fixed", parent, name)
    link("jelly")
    link("lift_stage")
    if not whole_body:
        _joint(robot, f"{FLOOR_LINK}_0", "fixed", "root", FLOOR_LINK, jelly_center)
        _joint(robot, "jelly_0", "fixed", "root", "jelly", jelly_center)
        _joint(robot, "lift_stage_0", "fixed", "root", "lift_stage", jelly_center)
    else:
        # world -> planar base -> jelly -> lift -> torso root (the classic
        # tree). At every body joint's zero, "root" sits at the world origin.
        link("world")
        link("base_x_link")
        link("base_y_link")
        _joint(robot, f"{FLOOR_LINK}_0", "fixed", "world", FLOOR_LINK, jelly_center)
        _joint(
            robot,
            "base_x",
            "prismatic",
            "world",
            "base_x_link",
            jelly_center,
            "1 0 0",
            (-BASE_TRAVEL, BASE_TRAVEL),
        )
        _joint(
            robot,
            "base_y",
            "prismatic",
            "base_x_link",
            "base_y_link",
            None,
            "0 1 0",
            (-BASE_TRAVEL, BASE_TRAVEL),
        )
        _joint(
            robot,
            "base_yaw",
            "revolute",
            "base_y_link",
            "jelly",
            None,
            "0 0 1",
            (-2 * np.pi, 2 * np.pi),
        )
        _joint(
            robot,
            "lift",
            "prismatic",
            "jelly",
            "lift_stage",
            None,
            "0 0 -1",
            (0.0, lift_travel),
        )
        _joint(
            robot,
            "lift_2",
            "prismatic",
            "lift_stage",
            "root",
            -jelly_center,
            "0 0 -1",
            (0.0, lift_travel),
            mimic="lift",
        )
        for i in range(DECK_STRIPS):
            link(f"deck_{i}")
            _joint(robot, f"deck_{i}_0", "fixed", "jelly", f"deck_{i}")
    for name, element in links.items():
        for tag in ("visual", "collision"):
            for el in element.findall(tag):
                element.remove(el)
        for path, rgba in visuals.get(name, []):
            vis = ET.SubElement(element, "visual")
            ET.SubElement(vis, "origin", xyz="0 0 0", rpy="0 0 0")
            geom = ET.SubElement(vis, "geometry")
            ET.SubElement(geom, "mesh", filename=PACKAGE + path, scale="1 1 1")
            mat = ET.SubElement(
                vis, "material", name="c_" + "_".join(f"{c:g}" for c in rgba)
            )
            ET.SubElement(mat, "color", rgba=" ".join(f"{c:g}" for c in rgba))
        if name in collision:
            spec = collision[name]
            col = ET.SubElement(element, "collision")
            if isinstance(spec, str):
                ET.SubElement(col, "origin", xyz="0 0 0", rpy="0 0 0")
                geom = ET.SubElement(col, "geometry")
                ET.SubElement(geom, "mesh", filename=PACKAGE + spec, scale="1 1 1")
            else:
                radius, length, center, rpy = spec
                ET.SubElement(col, "origin", xyz=_fmt(center), rpy=rpy)
                geom = ET.SubElement(col, "geometry")
                ET.SubElement(
                    geom, "cylinder", radius=f"{radius:g}", length=f"{length:g}"
                )
    for joint in robot.findall("joint"):
        lim = limits.get(joint.get("name", ""))
        if lim is not None:
            joint.find("limit").set("lower", f"{lim[0]:.4f}")
            joint.find("limit").set("upper", f"{lim[1]:.4f}")
    ET.indent(tree, space="    ")
    body = ET.tostring(robot, encoding="unicode")
    note = (
        "Axol Mobile (Axol on the Jelly mobile base). GENERATED by "
        "scripts/build_mobile_urdf.py from the Onshape export; do not edit by "
        "hand. Arm links, joints and inertials are the classic axol.urdf's, so "
        "FK and gravity compensation match it exactly; joint limits and meshes "
        "are the export's. " + (WHOLE_BODY_NOTE if whole_body else ARM_NOTE)
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
    args = ap.parse_args()
    urdfs = sorted((args.export_dir / "urdf").glob("*.urdf"))
    if len(urdfs) != 1:
        raise SystemExit(f"expected one URDF under {args.export_dir / 'urdf'}")
    classic = _load(CLASSIC_URDF)
    export = _load(urdfs[0])
    tf, sg = align(classic, export)
    if not np.allclose(tf[:3, :3], np.eye(3), atol=1e-3):
        raise SystemExit("export root is not level with the classic world frame")
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
    visuals, pieces = build_meshes(
        export, args.export_dir, classic, tf, args.min_part_mm, args.visual_faces
    )
    print("arm IK collision:")
    arm = collisions(classic, tf, pieces, ARM_COLLISION, deck=False, tag="")
    write_urdf(
        OUT_URDF,
        CLASSIC_URDF,
        visuals,
        arm,
        limits,
        tf,
        whole_body=False,
        lift_travel=lift_travel,
    )
    print("whole-body IK collision:")
    body = collisions(
        classic, tf, pieces, WHOLE_BODY_COLLISION, deck=True, tag="whole_body"
    )
    write_urdf(
        OUT_WHOLE_BODY_URDF,
        CLASSIC_URDF,
        visuals,
        body,
        limits,
        tf,
        whole_body=True,
        lift_travel=lift_travel,
    )


def check_lift(export: yourdfpy.URDF) -> float:
    """The export's two lift stages: same travel, both lowering along -z."""
    stages = [export.joint_map[n] for n in ("stage_1", "stage_2")]
    travel = {round(float(j.limit.upper), 6) for j in stages}
    if len(travel) != 1 or any(abs(float(j.limit.lower)) > 1e-6 for j in stages):
        raise SystemExit(f"lift stages differ: {travel}")
    for j in stages:
        _zero(export)
        axis = export.get_transform(j.parent, export.base_link)[:3, :3] @ np.asarray(
            j.origin[:3, :3] @ j.axis
        )
        if not np.allclose(axis, [0, 0, -1], atol=1e-6):
            raise SystemExit(f"{j.name} does not lower along -z: {axis}")
    print(f"lift: two stages of {travel.pop():.3f} m each, 0 = fully raised")
    return float(stages[0].limit.upper)


if __name__ == "__main__":
    main()
