"""
Build the bundled Axol Mobile URDF from its Onshape export.

The Onshape export (``almond_urdf/urdf/almond_urdf.urdf`` + ``meshes/``) is a
faithful CAD tree — ~140 links including every screw, a two-stage prismatic
lift, prismatic gripper fingers, and link frames wherever Onshape put them.
None of that matches what the control stack expects, so this script distils it
into ``almond_axol/kinematics/urdf/axol_mobile.urdf``:

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
  link, so the static body is kept as capsule-friendly pieces: ``s1`` (the
  shoulder bar), ``base`` (the lift column), ``lift_plate`` (the bracket the
  shoulder bar bolts onto) and ``head`` (the head-camera mount).
  The Jelly deck (``jelly``) is visual only — see ``LIFT_NOTE``.
- **The lift and fingers are frozen** at their CAD pose (lift fully raised,
  fingers closed — the classic gripper model is closed too): the kinematic
  model has exactly the 14 arm joints.

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
MESH_SUBDIR = "meshes/mobile"
PACKAGE = "package://assembly/"

# Classic arm joints (ARM_JOINTS order) and their export counterparts.
CLASSIC_JOINTS = ("s1_0", "s2_0", "s3_0", "e1_0", "e2_0", "w1_0", "w2_0")
EXPORT_JOINTS = ("s1", "s2", "s3", "e1", "w1", "w2", "w3")

# Export link -> bundled link. Every other export link joins the group of its
# nearest listed ancestor.
ANCHORS: dict[str, str] = {
    "jelly_base": "jelly",
    "jelly_stage_2_1": "base",
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
    "jelly": "base",
    "lift_plate": "base",
    # Split off the gripper so each gets its own (much tighter) capsule.
    "left_fingers": "left_gripper",
    "right_fingers": "right_gripper",
    "left_wrist_camera": "left_gripper",
    "right_wrist_camera": "right_gripper",
}

# Collision-only splits of one export mesh: vertices inside the mesh-local box
# ``(lo, hi)`` go to another link's hull instead of their own.
# - The top of the last lift stage is the bracket the shoulder bar bolts onto;
#   folded into the column's hull it would inflate the column capsule from
#   ~85 mm to ~120 mm, so it gets its own link.
# - The Jelly base mesh includes the fixed bottom section of the lift column,
#   which is within reach at gripper height; it joins the column's hull while
#   the deck itself stays visual only.
_INF = np.inf
HULL_SPLITS: dict[str, tuple[tuple[float, ...], tuple[float, ...], str]] = {
    "Jelly_Stage_3.stl": ((-_INF, -_INF, 1.22), (_INF, _INF, _INF), "lift_plate"),
    "Jelly_Base.stl": ((-0.041, -0.073, 0.2), (0.041, 0.073, _INF), "base"),
}

# Links carrying a collision hull. The Jelly deck is deliberately absent.
LIFT_NOTE = (
    "The lift is frozen fully raised (its CAD pose). At that height the Jelly "
    "deck sits below anything the arms can reach, so it is visual only; the "
    "arms collide with the shoulder bar (s1), the lift column (base), its "
    "top plate (lift_plate) and the head-camera mount (head), exactly as the "
    "classic arms collide with the classic base and s1."
)
NO_COLLISION = {"jelly"}


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
) -> tuple[dict[str, list[tuple[str, tuple[float, ...]]]], dict[str, str]]:
    owner = groups(export)
    _zero(classic)

    def frame(link: str) -> np.ndarray:
        return classic.get_transform(EXTRA_LINKS.get(link, link), classic.base_link)

    by_color: dict[tuple[str, tuple[float, ...]], list[trimesh.Trimesh]] = defaultdict(
        list
    )
    hull_pts: dict[str, list[np.ndarray]] = defaultdict(list)
    for link_name, link in export.link_map.items():
        target = owner.get(link_name)
        if target is None:
            continue
        in_target = np.linalg.inv(frame(target)) @ tf
        for vis in link.visuals:
            mesh = _mesh(export_dir, vis.geometry.mesh.filename)
            origin = vis.origin if vis.origin is not None else np.eye(4)
            pose = export.get_transform(link_name, export.base_link)
            placed = mesh.copy().apply_transform(in_target @ pose @ origin)
            split = HULL_SPLITS.get(Path(vis.geometry.mesh.filename).name)
            if split is None:
                hull_pts[target].append(placed.vertices)
            else:
                lo, hi, other = split
                inside = np.all((mesh.vertices >= lo) & (mesh.vertices <= hi), axis=1)
                hull_pts[target].append(placed.vertices[~inside])
                to_other = np.linalg.inv(frame(other)) @ frame(target)
                hull_pts[other].append(
                    trimesh.transform_points(placed.vertices[inside], to_other)
                )
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
    collisions = {}
    for target, pts in sorted(hull_pts.items()):
        if target in NO_COLLISION:
            continue
        # Snapping to a 2 mm grid first keeps the hull to a few hundred faces
        # (<= 1.7 mm of shape error, far inside the capsule fit's slack).
        snapped = np.unique(np.round(np.vstack(pts) / 2e-3) * 2e-3, axis=0)
        hull = trimesh.PointCloud(snapped).convex_hull
        name = f"{target}_collision.stl"
        hull.export(URDF_DIR / MESH_SUBDIR / name)
        collisions[target] = f"{MESH_SUBDIR}/{name}"
        cyl = trimesh.bounds.minimum_cylinder(hull)
        print(
            f"  {target:15s} hull {len(hull.faces):4d} faces, capsule r={cyl['radius'] * 1e3:5.1f} mm "
            f"h={cyl['height'] * 1e3:5.1f} mm"
        )
    return visuals, collisions


def _fmt(v: np.ndarray) -> str:
    return " ".join(f"{x:.6g}" if abs(x) > 1e-12 else "0" for x in v)


def write_urdf(
    classic_path: Path,
    visuals: dict[str, list[tuple[str, tuple[float, ...]]]],
    collisions: dict[str, str],
    limits: dict[str, tuple[float, float]],
) -> None:
    tree = ET.parse(classic_path)
    robot = tree.getroot()
    robot.set("name", "assembly")
    links = {ln.get("name"): ln for ln in robot.findall("link")}
    for name, parent in EXTRA_LINKS.items():
        link = ET.SubElement(robot, "link", name=name)
        joint = ET.SubElement(robot, "joint", name=f"{name}_0", type="fixed")
        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=name)
        links[name] = link
    for name, link in links.items():
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)
        for path, rgba in visuals.get(name, []):
            vis = ET.SubElement(link, "visual")
            ET.SubElement(vis, "origin", xyz="0 0 0", rpy="0 0 0")
            geom = ET.SubElement(vis, "geometry")
            ET.SubElement(geom, "mesh", filename=PACKAGE + path, scale="1 1 1")
            mat = ET.SubElement(
                vis, "material", name="c_" + "_".join(f"{c:g}" for c in rgba)
            )
            ET.SubElement(mat, "color", rgba=" ".join(f"{c:g}" for c in rgba))
        if name in collisions:
            col = ET.SubElement(link, "collision")
            ET.SubElement(col, "origin", xyz="0 0 0", rpy="0 0 0")
            geom = ET.SubElement(col, "geometry")
            ET.SubElement(
                geom, "mesh", filename=PACKAGE + collisions[name], scale="1 1 1"
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
        "are the export's. " + LIFT_NOTE
    )
    header = "<!-- " + textwrap.fill(note, 76, subsequent_indent="     ") + "\n-->\n"
    OUT_URDF.write_text(header + body + "\n")
    print(f"wrote {OUT_URDF.relative_to(REPO)}")


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
    check_equivalent(classic, export, tf, sg)
    limits = export_limits(export, sg)
    for name, (lo, hi) in limits.items():
        old = classic.joint_map[name].limit
        mark = (
            ""
            if (round(old.lower, 4), round(old.upper, 4)) == (lo, hi)
            else "  <- changed"
        )
        print(f"  {name:12s} [{lo:+.4f}, {hi:+.4f}]{mark}")
    visuals, collisions = build_meshes(
        export, args.export_dir, classic, tf, args.min_part_mm, args.visual_faces
    )
    write_urdf(CLASSIC_URDF, visuals, collisions, limits)


if __name__ == "__main__":
    main()
