"""
Show the IK collision shapes over the robot model in a browser.

The Pyroki IK backend (the default) keeps each arm's shapes off the body's
(:func:`almond_axol.kinematics.model.shared_robot_collision`): on Axol on
Jelly, a capsule fitted to each arm part and finger against a box or capsule
fitted to each body part; on the other versions, pyroki's one capsule per
link. This viewer draws exactly that model on top of the full visual URDF,
so you can see where the boundaries sit relative to the real parts:

- body shapes (the links in :func:`almond_axol.constants.torso_links`) in
  blue, arm shapes in green, links IK never checks (the shoulder mounts) in
  grey;
- each shape recoloured by its worst pair: amber once that pair is inside
  its activation distance (the soft cost is pushing it away), red when it is
  also overlapping, magenta when an upper arm passes its hard-stop floor.
  Pairs that already overlap at home activate only closer than home, so they
  stay green/blue at home;
- a line between the closest points of every pair whose cost is active;
- a table of the closest pairs with their clearance and activation distance.

Sliders drive the 14 arm joints, the lift and the grippers, the way the
solver places them: IK freezes the lift raised and moves the parts it leaves
behind (the deck and the lower column) up toward the arms by the live lift
height, and freezes the fingers closed and slides them by the live opening
(:func:`almond_axol.kinematics.solver.shift_links`).

The Mink backend uses its own MuJoCo model (``policy/mink_ik/assets``) and is
not shown here.

Run:
    uv run python scripts/view_ik_collision.py                  # Axol on Jelly
    uv run python scripts/view_ik_collision.py --gripper parcel
    uv run python scripts/view_ik_collision.py --model owl --port 8004
Then open http://localhost:8003.
"""

from __future__ import annotations

import argparse
import logging
import time

import jax.numpy as jnp
import jaxlie
import numpy as np
import trimesh
import viser
import yourdfpy
from viser.extras import ViserUrdf

from almond_axol.constants import (
    AxolModel,
    torso_links,
    upper_arm_guard_links,
    urdf_arm_joint_names,
    urdf_path,
)
from almond_axol.kinematics.config import KinematicsConfig
from almond_axol.kinematics.model import (
    collision_cost_params,
    finger_slide,
    lift_link_shift,
    shared_robot,
    shared_robot_collision,
    upper_arm_guard_floor,
)
from almond_axol.kinematics.solver import body_collision_distance, shift_links
from almond_axol.teleop.config import VRTeleopConfig

_logger = logging.getLogger("view_ik_collision")

# RGB 0-255: viser casts colour property updates to uint8.
_BODY = (70, 130, 240)
_ARM = (60, 200, 110)
_UNCHECKED = (150, 150, 150)
_ACTIVE = (255, 175, 20)
_OVERLAP = (240, 50, 50)
_GUARD = (230, 50, 230)

_LIFT_JOINT = "lift"


def _segment_closest_points(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closest points between segments p1-q1 and p2-q2 (Ericson, RTCD 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    eps = 1e-12
    if a <= eps and e <= eps:
        return p1, p2
    if a <= eps:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = d1 @ r
        if e <= eps:
            s, t = float(np.clip(-c / a, 0.0, 1.0)), 0.0
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = float(np.clip((b * f - c * e) / den, 0.0, 1.0)) if den > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                s, t = float(np.clip(-c / a, 0.0, 1.0)), 0.0
            elif t > 1.0:
                s, t = float(np.clip((b - c) / a, 0.0, 1.0)), 1.0
    return p1 + s * d1, p2 + t * d2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument(
        "--model",
        default=AxolModel.MOBILE.value,
        choices=[m.value for m in AxolModel],
        help="Axol version (default: mobile, Axol on Jelly).",
    )
    parser.add_argument(
        "--gripper",
        default="parallel",
        choices=["parallel", "parcel"],
        help="Fitted gripper (mobile only).",
    )
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument(
        "--margin",
        type=float,
        default=KinematicsConfig().self_collision_margin,
        help="Default activation distance (m), as kinematics.self_collision_margin.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    model = AxolModel(args.model)
    gripper = args.gripper if model is AxolModel.MOBILE else "parallel"

    # --- IK collision model, exactly as the solver builds it -------------
    robot = shared_robot(model, gripper=gripper)
    rc = shared_robot_collision(model, gripper=gripper)
    lift_shift, travel = lift_link_shift(model, gripper=gripper)
    finger_axis, finger_side, finger_travel = finger_slide(model, gripper=gripper)

    def link_shift(lowered: float, grip: tuple[float, float]) -> jnp.ndarray:
        # As KinematicsSolver._make_link_shift.
        opening = np.where(
            finger_side >= 0, np.asarray(grip)[np.maximum(finger_side, 0)], 0.0
        )
        local = finger_axis * (opening * finger_travel)[:, None]
        return jnp.asarray(
            np.concatenate([lift_shift * lowered, local], axis=1), dtype=jnp.float32
        )

    # As KinematicsSolver.collision_params.
    starts, _widths = collision_cost_params(
        robot,
        rc,
        args.margin,
        home_shifts=(link_shift(0.0, (0.0, 0.0)), link_shift(0.0, (1.0, 1.0))),
    )

    names = list(rc.link_names)
    idx_i = np.asarray(rc.active_idx_i)
    idx_j = np.asarray(rc.active_idx_j)
    n_act = robot.joints.num_actuated_joints
    q_home = jnp.zeros(n_act)
    home_d = np.asarray(rc.compute_self_collision_distance(robot, q_home))

    torso = set(torso_links(model))
    guard_body = set(upper_arm_guard_links(model))
    guard_floor = np.full(len(idx_i), -np.inf)
    for k, (i, j) in enumerate(zip(idx_i, idx_j)):
        a, b = names[i], names[j]
        arm = b if a in guard_body else a if b in guard_body else ""
        if arm.endswith("_e1"):
            guard_floor[k] = upper_arm_guard_floor(home_d[k])

    # Shapes: ("c", k) capsule k, ("b", k) box k; each pair's two shapes.
    cap_size = np.asarray(rc.capsules.size)  # radius, segment length
    box_ext = 2 * np.asarray(rc.boxes.half_extent) if rc.boxes is not None else None
    shape_link = {("c", k): li for k, li in enumerate(rc.capsule_link)}
    shape_link |= {("b", k): li for k, li in enumerate(rc.box_link)}
    pair_shapes = [(("c", i), ("c", j)) for i, j in rc.capsule_pairs]
    pair_shapes += [(("c", i), ("b", j)) for i, j in rc.box_pairs]
    checked = {names[i] for i in idx_i} | {names[j] for j in idx_j}

    def base_color(name: str) -> tuple[int, int, int]:
        if name not in checked:
            return _UNCHECKED
        return _BODY if name in torso else _ARM

    # Map pyroki's actuated joint order to left-then-right arm order.
    act_names = list(robot.joints.actuated_names)
    arm_order = urdf_arm_joint_names(is_left=True) + urdf_arm_joint_names(is_left=False)
    to_pk = [act_names.index(n) for n in arm_order]
    lower = np.asarray(robot.joints.lower_limits)
    upper = np.asarray(robot.joints.upper_limits)

    # --- Full visual URDF -------------------------------------------------
    path = urdf_path(model, gripper=gripper)
    full = yourdfpy.URDF.load(str(path), mesh_dir=str(path.parent))
    torso_root = "root" if "root" in full.link_map else full.base_link
    has_lift = travel > 0.0 and _LIFT_JOINT in full.joint_map
    # The finger joints that drive the visual fingers (the rest mimic them).
    finger_joints = {
        side: name
        for side in ("left", "right")
        for name, joint in full.joint_map.items()
        if name.startswith(f"{side}_finger_")
        and joint.type == "prismatic"
        and joint.mimic is None
    }

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # Front three-quarter view of the arms and the column.
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = (2.1, -1.6, 0.5)
        client.camera.look_at = (0.0, 0.0, -0.3)

    viser_urdf = ViserUrdf(
        server, urdf_or_path=full, root_node_name="/robot", load_meshes=True
    )
    viser_order = viser_urdf.get_actuated_joint_names()
    floor = (
        tuple(float(v) for v in full.get_transform("floor")[:3, 3])
        if "floor" in full.link_map
        else (0.0, 0.0, 0.0)
    )
    server.scene.add_grid("/grid", width=2.0, height=2.0, position=floor)

    # Shapes live under /ik, posed at the torso root (IK solves in it).
    ik_frame = server.scene.add_frame("/ik", show_axes=False)
    shape_handles: dict[tuple[str, int], viser.MeshHandle] = {}
    shell_handles: dict[tuple[str, int], viser.MeshHandle] = {}

    # --- GUI ----------------------------------------------------------------
    rest = VRTeleopConfig()
    rest_q = np.concatenate([rest.rest_pose_left, rest.rest_pose_right])

    with server.gui.add_folder("Display"):
        show_robot = server.gui.add_checkbox("Robot meshes", True)
        show_caps = server.gui.add_checkbox("IK shapes", True)
        show_unchecked = server.gui.add_checkbox("Unchecked links", False)
        show_shell = server.gui.add_checkbox("Activation shell (body)", False)
        show_lines = server.gui.add_checkbox("Closest-point lines", True)
        opacity = server.gui.add_slider("Shape opacity", 0.05, 1.0, 0.05, 0.45)
    with server.gui.add_folder("Pose"):
        preset = server.gui.add_button_group("Preset", ["Home", "Rest"])
        lift = (
            server.gui.add_slider("Lift lowered (m)", 0.0, travel, 0.005, 0.0)
            if has_lift
            else None
        )
        grips = (
            [
                server.gui.add_slider(
                    f"{side.title()} gripper open", 0.0, 1.0, 0.01, 1.0
                )
                for side in ("left", "right")
            ]
            if finger_travel > 0.0
            else []
        )
    sliders: list[viser.GuiInputHandle[float]] = []
    for side, folder in (("left", "Left arm"), ("right", "Right arm")):
        with server.gui.add_folder(folder):
            for name in arm_order:
                if not name.startswith(side):
                    continue
                k = act_names.index(name)
                sliders.append(
                    server.gui.add_slider(
                        name.removeprefix(f"{side}_").removesuffix("_0"),
                        float(lower[k]),
                        float(upper[k]),
                        0.005,
                        0.0,
                    )
                )
    with server.gui.add_folder("Closest pairs"):
        table = server.gui.add_markdown("")

    def shape_mesh(shape: tuple[str, int], grow: float = 0.0) -> trimesh.Trimesh:
        kind, k = shape
        if kind == "c":
            r, h = float(cap_size[k, 0]), float(cap_size[k, 1])
            return trimesh.creation.capsule(
                height=max(h, 1e-6), radius=r + grow, count=[24, 24]
            )
        assert box_ext is not None
        return trimesh.creation.box(extents=box_ext[k] + 2 * grow)

    for shape, li in shape_link.items():
        name = names[li]
        kind, k = shape
        if kind == "c" and cap_size[k, 0] <= 0.0:
            continue  # a frame-only link's zero capsule (legacy model)
        mesh = shape_mesh(shape)
        shape_handles[shape] = server.scene.add_mesh_simple(
            f"/ik/shapes/{name}_{kind}{k}",
            mesh.vertices,
            mesh.faces,
            color=base_color(name),
            opacity=opacity.value,
            flat_shading=kind == "b",
        )
        if name in torso:
            # A box's shell is drawn as a grown box (the true one is rounded).
            shell = shape_mesh(shape, args.margin)
            shell_handles[shape] = server.scene.add_mesh_simple(
                f"/ik/shell/{name}_{kind}{k}",
                shell.vertices,
                shell.faces,
                color=_BODY,
                opacity=0.12,
                wireframe=True,
            )

    lines = server.scene.add_line_segments(
        "/ik/lines", np.zeros((0, 2, 3)), _OVERLAP, line_width=3.0
    )
    dirty = {"v": True}

    def mark(_=None) -> None:
        dirty["v"] = True

    for s in (*sliders, *grips):
        s.on_update(mark)
    for c in (show_robot, show_caps, show_unchecked, show_shell, show_lines):
        c.on_update(mark)
    if lift is not None:
        lift.on_update(mark)

    @opacity.on_update
    def _(_) -> None:
        for h in shape_handles.values():
            h.opacity = opacity.value

    @preset.on_click
    def _(_) -> None:
        q = rest_q if preset.value == "Rest" else np.zeros(len(sliders))
        with server.atomic():
            for s, v in zip(sliders, q):
                s.value = float(np.clip(v, s.min, s.max))
        mark()

    def update() -> None:
        q_arm = np.array([s.value for s in sliders])
        q_pk = np.zeros(n_act)
        q_pk[to_pk] = q_arm
        lowered = float(lift.value) if lift is not None else 0.0
        grip = (grips[0].value, grips[1].value) if grips else (0.0, 0.0)

        # Visual model: arms, the lift and the fingers.
        cfg = np.zeros(len(viser_order))
        for vi, name in enumerate(viser_order):
            if name in arm_order:
                cfg[vi] = q_arm[arm_order.index(name)]
            elif name == _LIFT_JOINT:
                cfg[vi] = lowered
            for g, side in zip(grip, ("left", "right")):
                if name == finger_joints.get(side):
                    cfg[vi] = g * finger_travel
        viser_urdf.update_cfg(cfg)
        viser_urdf.show_visual = show_robot.value
        full_cfg = {n: 0.0 for n in full.actuated_joint_names}
        for name, v in zip(arm_order, q_arm):
            full_cfg[name] = float(v)
        if has_lift:
            full_cfg[_LIFT_JOINT] = lowered
        full.update_cfg(full_cfg)
        world_T_torso = full.get_transform(torso_root, full.base_link)
        T = jaxlie.SE3.from_matrix(world_T_torso)
        ik_frame.wxyz = np.asarray(T.rotation().wxyz)
        ik_frame.position = np.asarray(T.translation())

        # IK model: FK in the torso frame, with the lift and fingers moved.
        shift = link_shift(lowered, grip)
        poses = shift_links(robot.forward_kinematics(jnp.asarray(q_pk)), shift)
        caps, boxes = rc.shapes_at(poses)
        placed = {
            ("c", k): (p, w)
            for k, (p, w) in enumerate(
                zip(
                    np.asarray(caps.pose.translation()),
                    np.asarray(caps.pose.rotation().wxyz),
                )
            )
        }
        cap_axis = np.asarray(caps.pose.rotation().as_matrix())[:, :, 2]
        if boxes is not None:
            box_rot = np.asarray(boxes.pose.rotation().as_matrix())
            placed |= {
                ("b", k): (p, w)
                for k, (p, w) in enumerate(
                    zip(
                        np.asarray(boxes.pose.translation()),
                        np.asarray(boxes.pose.rotation().wxyz),
                    )
                )
            }
        d = np.asarray(body_collision_distance(robot, rc, jnp.asarray(q_pk), shift))

        def closest(a: tuple[str, int], b: tuple[str, int]):
            """Closest points of a capsule's segment and the other shape's
            axis (capsule) or volume (box)."""
            pa = placed[a][0]
            ha = cap_size[a[1], 1] / 2 * cap_axis[a[1]]
            if b[0] == "c":
                pb = placed[b][0]
                hb = cap_size[b[1], 1] / 2 * cap_axis[b[1]]
                return _segment_closest_points(pa - ha, pa + ha, pb - hb, pb + hb)
            rot, half = box_rot[b[1]], box_ext[b[1]] / 2
            t = np.linspace(-1.0, 1.0, 201)[:, None]
            seg = (pa + t * ha - placed[b][0]) @ rot
            on_box = np.clip(seg, -half, half)
            i = int(np.argmin(np.linalg.norm(seg - on_box, axis=1)))
            return pa + t[i] * ha, placed[b][0] + rot @ on_box[i]

        # Worst status per shape: 0 clear, 1 cost active, 2 overlap, 3 guard.
        status = {shape: 0 for shape in shape_link}
        segs, rows = [], []
        for k, (a, b) in enumerate(pair_shapes):
            # Overlap only counts once the pair is inside its activation
            # distance: a pair that overlaps at home (see
            # collision_cost_params) is left alone there.
            s = 0
            if d[k] < starts[k]:
                s = 2 if d[k] < 0.0 else 1
            if d[k] < guard_floor[k]:
                s = 3
            status[a] = max(status[a], s)
            status[b] = max(status[b], s)
            if s >= 1:
                segs.append(closest(a, b))
            rows.append((d[k], starts[k], names[idx_i[k]], names[idx_j[k]], s))

        with server.atomic():
            for shape, h in shape_handles.items():
                name = names[shape_link[shape]]
                h.position, h.wxyz = placed[shape]
                h.visible = show_caps.value and (
                    name in checked or show_unchecked.value
                )
                h.color = (base_color(name), _ACTIVE, _OVERLAP, _GUARD)[status[shape]]
            for shape, h in shell_handles.items():
                h.position, h.wxyz = placed[shape]
                h.visible = show_shell.value
            lines.points = (
                np.array(segs, dtype=np.float32)
                if segs
                else np.zeros((0, 2, 3), dtype=np.float32)
            )
            lines.visible = show_lines.value
            table.content = _table(rows)

    _logger.info("Open http://localhost:%d", args.port)
    while True:
        if dirty["v"]:
            dirty["v"] = False
            update()
        time.sleep(1 / 30)


_STATUS = ("clear", "**cost active**", "**overlap**", "**hard stop**")


def _table(rows: list[tuple], limit: int = 10) -> str:
    rows = sorted(rows, key=lambda r: r[0] - r[1])[:limit]
    out = [
        "| Pair | Clear (mm) | Activates (mm) | |",
        "|---|---:|---:|---|",
    ]
    for d, start, a, b, s in rows:
        label = _STATUS[s]
        if s == 0 and start < 0.0:
            label = "clear (overlaps at home)"
        out.append(f"| {a} / {b} | {1e3 * d:.0f} | {1e3 * start:.0f} | {label} |")
    out.append("")
    out.append("Sorted by distance to the activation shell. Negative clearance")
    out.append("is overlap. A pair that overlaps at home activates only when")
    out.append("the arm moves closer than home.")
    return "\n".join(out)


if __name__ == "__main__":
    main()
