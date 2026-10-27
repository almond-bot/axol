"""The simulated robot seen from its own cameras.

The sim has no cameras, so this renders them: the robot's URDF (with its
meshes) is loaded into MuJoCo with a floor and a table of blocks in front,
and a camera is placed on each camera mount the URDF defines (the head
camera's left lens and each wrist camera, ROS optical frames). Every frame
poses the model from the live sim state (arm joints, grippers, lift and
base) and renders the streams that are being watched.

Rendering runs in a spawned subprocess, off the control loop's GIL, with
MuJoCo's software (OSMesa) renderer unless ``MUJOCO_GL`` says otherwise, so
it works on a machine with no display.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import re
import threading
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)

# Stream -> (URDF optical frame link, vertical field of view in degrees).
_CAMERAS = {
    "head": ("head_camera_left_optical", 70.0),
    "left": ("left_wrist_camera_optical", 65.0),
    "right": ("right_wrist_camera_optical", 65.0),
}
# Table in front of the robot: centre x, half sizes and its top above the floor.
_TABLE_X, _TABLE_HALF, _TABLE_TOP = 0.6, (0.22, 0.45), 0.72
_BLOCKS = (
    (-0.05, 0.12, (0.85, 0.25, 0.2, 1.0)),
    (0.02, -0.1, (0.2, 0.45, 0.85, 1.0)),
    (-0.08, 0.0, (0.25, 0.7, 0.35, 1.0)),
)
_ARM_JOINTS = ("s1_0", "s2_0", "s3_0", "e1_0", "e2_0", "w1_0", "w2_0")
_FINGER_OPEN_M = 0.058804

SimState = dict[str, Any]


def build_scene(urdf: str) -> tuple[Any, Any, tuple[str, ...]]:
    """MuJoCo model and data for the robot plus props; and its streams."""
    import mujoco

    from ..constants import URDF_DIR

    compiler = (
        f'<mujoco><compiler meshdir="{URDF_DIR}" balanceinertia="true" '
        'discardvisual="false" fusestatic="false" strippath="false"/></mujoco>'
    )
    text = re.sub(r"(<robot[^>]*>)", r"\1" + compiler, urdf, count=1)
    spec = mujoco.MjSpec.from_string(text)
    names = {body.name for body in spec.bodies}
    streams = []
    for stream, (link, fovy) in _CAMERAS.items():
        if link not in names:
            continue
        camera = spec.body(link).add_camera()
        camera.name = stream
        # ROS optical (+z forward, +y down) to MuJoCo's (-z forward, +y up).
        camera.quat = [0.0, 1.0, 0.0, 0.0]
        camera.fovy = fovy
        streams.append(stream)

    probe = spec.compile()
    probe_data = mujoco.MjData(probe)
    mujoco.mj_kinematics(probe, probe_data)
    floor_z = float(probe_data.body("floor").xpos[2]) if "floor" in names else -0.75
    world = spec.worldbody
    texture = spec.add_texture(
        name="grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.82, 0.84, 0.86],
        rgb2=[0.7, 0.72, 0.75],
        width=512,
        height=512,
    )
    material = spec.add_material(name="grid")
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
    material.texrepeat = [10, 10]
    floor = world.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [8, 8, 0.1]
    floor.pos = [0, 0, floor_z]
    floor.material = "grid"
    top = floor_z + _TABLE_TOP
    table = world.add_geom()
    table.type = mujoco.mjtGeom.mjGEOM_BOX
    table.size = [_TABLE_HALF[0], _TABLE_HALF[1], _TABLE_TOP / 2]
    table.pos = [_TABLE_X, 0, floor_z + _TABLE_TOP / 2]
    table.rgba = [0.62, 0.47, 0.33, 1]
    for dx, y, rgba in _BLOCKS:
        block = world.add_geom()
        block.type = mujoco.mjtGeom.mjGEOM_BOX
        block.size = [0.025, 0.025, 0.025]
        block.pos = [_TABLE_X + dx, y, top + 0.025]
        block.rgba = list(rgba)
    for pos, direction, level in (
        ([0.6, 0.0, 2.5], [0, 0, -1], 0.6),
        ([-1.0, 1.5, 2.0], [0.4, -0.6, -1], 0.35),
    ):
        light = world.add_light()
        light.pos = pos
        light.dir = direction
        light.diffuse = [level] * 3
        light.castshadow = False
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    # The offscreen buffer bounds the frame size (MuJoCo's default is 640x480).
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    if "head" not in streams:
        # No head camera mount: look over the shoulders at the table instead.
        from scipy.spatial.transform import Rotation

        tilt = np.radians(35.0)
        # Columns: image right (-y), image up, and back along the view.
        axes = np.array(
            [
                [0.0, np.sin(tilt), -np.cos(tilt)],
                [-1.0, 0.0, 0.0],
                [0.0, np.cos(tilt), np.sin(tilt)],
            ]
        )
        x, y, z, w = Rotation.from_matrix(axes).as_quat()
        camera = world.add_camera()
        camera.name = "head"
        camera.pos = [-0.35, 0.0, 1.05]
        camera.quat = [w, x, y, z]
        camera.fovy = 70.0
        streams.insert(0, "head")
    model = spec.compile()
    return model, mujoco.MjData(model), tuple(streams)


def pose_scene(model: Any, data: Any, state: SimState) -> None:
    """Set every joint the sim state describes, then update the kinematics."""
    import mujoco

    def put(name: str, value: float) -> None:
        try:
            data.joint(name).qpos[0] = value
        except KeyError:
            pass

    arms = state.get("arms")
    if arms is not None:
        for side, offset in (("left", 0), ("right", 7)):
            for i, joint in enumerate(_ARM_JOINTS):
                put(f"{side}_{joint}", float(arms[offset + i]))
    grip = state.get("grip")
    if grip is not None:
        for side, value in zip(("left", "right"), grip, strict=False):
            opening = min(max(float(value), 0.0), 1.0) * _FINGER_OPEN_M
            put(f"{side}_finger_1_joint", opening)
            put(f"{side}_finger_2_joint", opening)
    lift = state.get("lift")
    if lift is not None:
        try:
            travel = float(model.jnt_range[model.joint("lift").id][1])
        except KeyError:
            travel = 0.0
        value = (1.0 - min(max(float(lift), 0.0), 1.0)) * travel
        put("lift", value)
        put("lift_2", value)
    base = state.get("base")
    if base is not None:
        put("base_x", float(base[0]))
        put("base_y", float(base[1]))
        put("base_yaw", float(base[2]))
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)


def _worker(conn: Any, urdf: str) -> None:
    """Subprocess: build the scene, then answer render requests."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    try:
        import mujoco

        from .cameras import encode_jpeg

        model, data, streams = build_scene(urdf)
    except Exception as exc:  # noqa: BLE001 - reported to the parent
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        return
    conn.send(("ready", streams))
    renderers: dict[tuple[int, int], Any] = {}
    while True:
        try:
            request = conn.recv()
        except (EOFError, OSError):
            return
        if request is None:
            return
        request_id, wanted, width, quality, state = request
        try:
            pose_scene(model, data, state)
            height = max(2, round(width * 9 / 16))
            renderer = renderers.get((width, height))
            if renderer is None:
                renderer = renderers[(width, height)] = mujoco.Renderer(
                    model, height, width
                )
            frames = {}
            for stream in wanted:
                if stream not in streams:
                    continue
                renderer.update_scene(data, stream)
                frames[stream] = encode_jpeg(renderer.render(), width, quality)
            conn.send(("frames", request_id, frames))
        except Exception as exc:  # noqa: BLE001 - reported to the parent
            conn.send(("error", request_id, f"{type(exc).__name__}: {exc}"))


class SimCameraSource:
    """Render the sim's cameras in a subprocess from the live sim state."""

    name = "sim"

    def __init__(self, urdf_path: str, state: Callable[[], SimState]) -> None:
        self._urdf_path = urdf_path
        self._state = state
        self._conn: Any = None
        self._proc: multiprocessing.process.BaseProcess | None = None
        self._lock = threading.Lock()
        self._request = 0
        self.available: tuple[str, ...] = ()
        self.error: str | None = None

    def start(self) -> None:
        with open(self._urdf_path, encoding="utf-8") as handle:
            urdf = handle.read()
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe()
        proc = ctx.Process(
            target=_worker,
            args=(child, urdf),
            name="marionette-sim-cameras",
            daemon=True,
        )
        proc.start()
        child.close()
        self._conn, self._proc = parent, proc
        if not parent.poll(60.0):
            self.stop()
            raise RuntimeError("the sim camera renderer did not start")
        kind, payload = parent.recv()
        if kind != "ready":
            self.stop()
            raise RuntimeError(f"the sim camera renderer failed: {payload}")
        self.available = tuple(payload)

    def stop(self) -> None:
        conn, proc = self._conn, self._proc
        self._conn = self._proc = None
        if conn is not None:
            try:
                conn.send(None)
            except (OSError, ValueError):
                pass
            conn.close()
        if proc is not None:
            proc.join(timeout=3.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=3.0)

    def grab(
        self, streams: Iterable[str], width: int, quality: int
    ) -> dict[str, bytes]:
        conn = self._conn
        if conn is None:
            return {}
        with self._lock:
            self._request += 1
            request = self._request
            conn.send((request, list(streams), int(width), int(quality), self._state()))
            while True:
                if not conn.poll(5.0):
                    raise RuntimeError("the sim camera renderer stopped answering")
                kind, answered, payload = conn.recv()
                if answered == request:
                    break  # an older, late answer is dropped
        if kind != "frames":
            raise RuntimeError(str(payload))
        return payload


def sim_state(
    arms: np.ndarray | None,
    grip: tuple[float, float],
    lift: float | None,
    base: tuple[float, float, float] | None,
) -> SimState:
    """The state :func:`pose_scene` takes, from plain values."""
    return {
        "arms": None if arms is None else [float(v) for v in arms],
        "grip": [float(grip[0]), float(grip[1])],
        "lift": None if lift is None else float(lift),
        "base": None if base is None else [float(v) for v in base],
    }
