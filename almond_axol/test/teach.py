"""Interactive teach mode — pose the robot in the browser and record joint positions.

Run:
    python -m almond_axol.test.teach

Open http://localhost:8080, drag the sliders to pose each arm, then click
"Record waypoint" to print the joint arrays to the terminal.
"""

import threading

import numpy as np

from ..robot.axol import (
    ELBOW_LEFT_LIMITS,
    ELBOW_RIGHT_LIMITS,
    LIMITS,
    SHOULDER_2_LEFT_LIMITS,
    SHOULDER_2_RIGHT_LIMITS,
)
from ..shared import ARM_JOINTS, URDF_PATH, Joint

try:
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf
except ImportError as e:
    raise ImportError("viser is required. Install with: uv sync --extra sim") from e

_LEFT_LIMITS: list[tuple[float, float]] = [
    LIMITS[Joint.SHOULDER_1],
    SHOULDER_2_LEFT_LIMITS,
    LIMITS[Joint.SHOULDER_3],
    ELBOW_LEFT_LIMITS,
    LIMITS[Joint.WRIST_1],
    LIMITS[Joint.WRIST_2],
    LIMITS[Joint.WRIST_3],
    (0.0, 1.0),  # gripper normalized
]

_RIGHT_LIMITS: list[tuple[float, float]] = [
    LIMITS[Joint.SHOULDER_1],
    SHOULDER_2_RIGHT_LIMITS,
    LIMITS[Joint.SHOULDER_3],
    ELBOW_RIGHT_LIMITS,
    LIMITS[Joint.WRIST_1],
    LIMITS[Joint.WRIST_2],
    LIMITS[Joint.WRIST_3],
    (0.0, 1.0),  # gripper normalized
]

_LABELS = [j.value.replace("_", " ").title() for j in Joint]

_LEFT_URDF = [
    "left_s1_0",
    "left_s2_0",
    "left_s3_0",
    "left_e1_0",
    "left_e2_0",
    "left_w1_0",
    "left_w2_0",
]
_RIGHT_URDF = [
    "right_s1_0",
    "right_s2_0",
    "right_s3_0",
    "right_e1_0",
    "right_e2_0",
    "right_w1_0",
    "right_w2_0",
]


def main() -> None:
    server = viser.ViserServer(port=8080)

    urdf = yourdfpy.URDF.load(str(URDF_PATH), mesh_dir=str(URDF_PATH.parent))
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf,
        root_node_name="/robot",
        load_meshes=True,
        load_collision_meshes=False,
    )
    server.scene.add_grid("/grid", width=2.0, height=2.0)

    # Map viser joint order → robot_order index (mirrors Sim._build_q)
    robot_order = _LEFT_URDF + _RIGHT_URDF
    viser_order = viser_urdf.get_actuated_joint_names()
    viser_to_robot = []
    for name in viser_order:
        try:
            viser_to_robot.append(robot_order.index(name))
        except ValueError:
            viser_to_robot.append(-1)

    n_arm = len(ARM_JOINTS)  # 7, no gripper

    def _update_viz(ls: list, rs: list) -> None:
        q_robot = np.array(
            [s.value for s in ls[:n_arm]] + [s.value for s in rs[:n_arm]],
            dtype=float,
        )
        q_viser = np.zeros(len(viser_order))
        for vi, ri in enumerate(viser_to_robot):
            if ri >= 0:
                q_viser[vi] = q_robot[ri]
        viser_urdf.update_cfg(q_viser)

    # Build sliders
    left_sliders: list = []
    right_sliders: list = []

    with server.gui.add_folder("Left arm"):
        for label, (lo, hi) in zip(_LABELS, _LEFT_LIMITS):
            left_sliders.append(
                server.gui.add_slider(
                    label, min=lo, max=hi, step=0.001, initial_value=0.0
                )
            )

    with server.gui.add_folder("Right arm"):
        for label, (lo, hi) in zip(_LABELS, _RIGHT_LIMITS):
            right_sliders.append(
                server.gui.add_slider(
                    label, min=lo, max=hi, step=0.001, initial_value=0.0
                )
            )

    for s in left_sliders + right_sliders:

        @s.on_update
        def _(event, ls=left_sliders, rs=right_sliders):
            _update_viz(ls, rs)

    # Record button — prints arrays ready to paste into code
    count = [0]
    btn = server.gui.add_button("⬤  Record waypoint")

    @btn.on_click
    def _(event, ls=left_sliders, rs=right_sliders):
        count[0] += 1
        ql = np.round([s.value for s in ls], 4).tolist()
        qr = np.round([s.value for s in rs], 4).tolist()
        print(f"\n--- Waypoint {count[0]} ---")
        print(f"left  = np.array({ql}, dtype=np.float32)")
        print(f"right = np.array({qr}, dtype=np.float32)")

    print("\n=== TEACH MODE ===")
    print("Viser server running at  http://localhost:8080")
    print("Drag the sliders to pose the robot, then click 'Record waypoint'.")
    print("Press Ctrl+C to exit.\n")

    threading.Event().wait()


if __name__ == "__main__":
    main()
