"""Render the robot and a scene from any camera, through the ZED lens models.

:class:`SceneRenderer` builds a MuJoCo model from the URDF's visual meshes
(every link a mocap body, posed from our own forward kinematics) plus simple
props: a room, a table, objects, a calibration board. It renders pinhole
images from a camera pose given as an optical frame (+z out of the lens,
+x right, +y down).

:class:`LensImager` turns pinhole renders into the raw image a real camera
would give, with its factory intrinsics and distortion: the head camera's
``plumb_bob`` / ``rational_polynomial`` lens from one wide pinhole render,
the wrist cameras' ``equidistant`` fisheye (wider than 180 degrees across
the sensor) from five cube faces.

MuJoCo renders offscreen through EGL, or OSMesa where there is no GPU
(set ``MUJOCO_GL``). Importing this module does not open a context.
"""

from __future__ import annotations

import math
import os
import tempfile
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Cameras in MuJoCo look along -z with +y up; an optical frame looks along
# +z with +y down.
_OPTICAL_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])


def _quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    import mujoco

    out = np.empty(4)
    mujoco.mju_mat2Quat(out, np.asarray(rotation, dtype=np.float64).reshape(9))
    return out


def _rpy_matrix(rpy: Iterable[float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


@dataclass
class Prop:
    """Something in the scene besides the robot.

    ``kind`` is ``box`` (``size`` = half extents), ``cylinder`` (radius,
    half height) or ``board`` (a thin box whose top face shows ``texture``;
    ``size`` = half extents, the texture spans x and y).
    """

    name: str
    kind: str
    size: tuple[float, ...]
    rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    texture: np.ndarray | None = None  # HxWx3 uint8, for a board
    movable: bool = False


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


class SceneRenderer:
    """A MuJoCo scene of the robot's visual meshes and some props."""

    def __init__(
        self,
        urdf_path: Path,
        props: Iterable[Prop] = (),
        *,
        max_size: tuple[int, int] = (2048, 2048),
        samples: int = 4,
        shadows: bool = True,
        room: bool = True,
        floor_z: float = 0.0,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.props = {p.name: p for p in props}
        self._tmp = tempfile.TemporaryDirectory(prefix="axol-render-")
        xml, self.links = self._mjcf(
            Path(urdf_path), max_size, samples, shadows, room, floor_z
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._mocap: dict[str, int] = {}
        for name in [*self.links, *self.props, "_eye"]:
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body >= 0 and self.model.body_mocapid[body] >= 0:
                self._mocap[name] = int(self.model.body_mocapid[body])
        self._camera = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")
        self._renderers: dict[tuple[int, int], Any] = {}
        for prop in self.props.values():
            self.set_pose(prop.name, prop.pose)

    # -- model -------------------------------------------------------------

    def _mjcf(
        self,
        urdf_path: Path,
        max_size: tuple[int, int],
        samples: int,
        shadows: bool,
        room: bool,
        floor_z: float,
    ) -> tuple[str, list[str]]:
        urdf = ElementTree.parse(urdf_path).getroot()
        root = ElementTree.Element("mujoco", model="axol_scene")
        ElementTree.SubElement(root, "compiler", angle="radian")
        ElementTree.SubElement(root, "option", gravity="0 0 0")
        visual = ElementTree.SubElement(root, "visual")
        ElementTree.SubElement(
            visual, "global", offwidth=str(max_size[0]), offheight=str(max_size[1])
        )
        ElementTree.SubElement(
            visual,
            "quality",
            shadowsize="4096" if shadows else "0",
            offsamples=str(samples),
        )
        # No headlight beyond ambient: it moves with the camera, so the cube
        # faces of a fisheye would each be lit differently.
        ElementTree.SubElement(
            visual,
            "headlight",
            ambient="0.42 0.42 0.42",
            diffuse="0 0 0",
            specular="0 0 0",
        )
        ElementTree.SubElement(visual, "map", znear="0.002", zfar="50")
        default = ElementTree.SubElement(root, "default")
        ElementTree.SubElement(
            default, "geom", contype="0", conaffinity="0", density="0"
        )
        asset = ElementTree.SubElement(root, "asset")
        ElementTree.SubElement(
            asset,
            "texture",
            name="floor",
            type="2d",
            builtin="checker",
            rgb1="0.55 0.55 0.52",
            rgb2="0.45 0.45 0.43",
            width="512",
            height="512",
            mark="edge",
            markrgb="0.35 0.35 0.33",
        )
        ElementTree.SubElement(
            asset, "material", name="floor", texture="floor", texrepeat="8 8"
        )
        ElementTree.SubElement(
            asset,
            "texture",
            name="wall",
            type="2d",
            builtin="gradient",
            rgb1="0.86 0.85 0.8",
            rgb2="0.7 0.72 0.75",
            width="256",
            height="256",
        )
        ElementTree.SubElement(asset, "material", name="wall", texture="wall")
        world = ElementTree.SubElement(root, "worldbody")
        if room:
            # The floor is the lowest point of the robot (its wheels).
            ElementTree.SubElement(
                world,
                "geom",
                type="plane",
                size="6 6 0.1",
                pos=_fmt((0, 0, floor_z)),
                material="floor",
            )
            # Walls and ceiling are thin boxes: a plane seen from behind
            # renders see-through.
            height = 2.8
            mid = floor_z + height / 2
            for name, pos, size in (
                ("wall_front", (3.0, 0, mid), (0.02, 3.0, height / 2)),
                ("wall_back", (-3.0, 0, mid), (0.02, 3.0, height / 2)),
                ("wall_left", (0, 3.0, mid), (3.0, 0.02, height / 2)),
                ("wall_right", (0, -3.0, mid), (3.0, 0.02, height / 2)),
                ("ceiling", (0, 0, floor_z + height), (3.0, 3.0, 0.02)),
            ):
                ElementTree.SubElement(
                    world,
                    "geom",
                    name=name,
                    type="box",
                    size=_fmt(size),
                    pos=_fmt(pos),
                    material="wall",
                )
        for i, (xyz, cast) in enumerate(
            (
                ((1.2, 1.0, 2.5), True),
                ((-0.5, -1.5, 2.5), False),
                ((0.5, 0, 2.6), False),
            )
        ):
            light = np.array([xyz[0], xyz[1], floor_z + xyz[2]])
            ElementTree.SubElement(
                world,
                "light",
                name=f"light{i}",
                pos=_fmt(light),
                dir=_fmt(np.array([0.7, 0.0, floor_z + 0.75]) - light),
                diffuse="0.6 0.6 0.57" if i == 0 else "0.35 0.35 0.35",
                specular="0.1 0.1 0.1",
                castshadow="true" if (cast and shadows) else "false",
                directional="false",
                attenuation="1 0 0",
                cutoff="80",
            )

        links: list[str] = []
        materials: dict[str, str] = {}
        for link in urdf.findall("link"):
            visuals = link.findall("visual")
            meshes = [v for v in visuals if v.find("geometry/mesh") is not None]
            if not meshes:
                continue
            name = link.get("name") or ""
            links.append(name)
            body = ElementTree.SubElement(world, "body", name=name, mocap="true")
            for i, vis in enumerate(meshes):
                mesh = vis.find("geometry/mesh")
                assert mesh is not None
                path = urdf_path.parent / (mesh.get("filename") or "")
                mesh_name = f"{name}_{i}"
                scale = mesh.get("scale") or "1 1 1"
                ElementTree.SubElement(
                    asset,
                    "mesh",
                    name=mesh_name,
                    file=str(path.resolve()),
                    scale=scale,
                    inertia="shell",
                )
                color = vis.find("material/color")
                rgba = color.get("rgba") if color is not None else "0.7 0.7 0.7 1"
                rgba = rgba or "0.7 0.7 0.7 1"
                if rgba not in materials:
                    materials[rgba] = f"mat{len(materials)}"
                    ElementTree.SubElement(
                        asset,
                        "material",
                        name=materials[rgba],
                        rgba=rgba,
                        specular="0.3",
                        shininess="0.4",
                    )
                origin = vis.find("origin")
                xyz = [0.0, 0.0, 0.0]
                rpy = [0.0, 0.0, 0.0]
                if origin is not None:
                    xyz = [float(v) for v in (origin.get("xyz") or "0 0 0").split()]
                    rpy = [float(v) for v in (origin.get("rpy") or "0 0 0").split()]
                ElementTree.SubElement(
                    body,
                    "geom",
                    type="mesh",
                    mesh=mesh_name,
                    material=materials[rgba],
                    pos=_fmt(xyz),
                    quat=_fmt(_quat_wxyz(_rpy_matrix(rpy))),
                )
        for prop in self.props.values():
            self._add_prop(asset, world, prop)
        eye = ElementTree.SubElement(world, "body", name="_eye", mocap="true")
        ElementTree.SubElement(eye, "camera", name="eye", fovy="60")
        return ElementTree.tostring(root, encoding="unicode"), links

    def _add_prop(
        self, asset: ElementTree.Element, world: ElementTree.Element, prop: Prop
    ) -> None:
        body = ElementTree.SubElement(world, "body", name=prop.name, mocap="true")
        if prop.kind == "board":
            assert prop.texture is not None
            from PIL import Image

            path = Path(self._tmp.name) / f"{prop.name}.png"
            # MuJoCo puts the image's first row at the plane's +y edge and
            # its first column at -x: the image reads the right way round,
            # seen from above the plane.
            Image.fromarray(np.ascontiguousarray(prop.texture)).save(path)
            ElementTree.SubElement(
                asset, "texture", name=f"{prop.name}_tex", type="2d", file=str(path)
            )
            ElementTree.SubElement(
                asset,
                "material",
                name=f"{prop.name}_mat",
                texture=f"{prop.name}_tex",
                texuniform="false",
                texrepeat="1 1",
                specular="0.05",
                shininess="0.1",
                rgba="1 1 1 1",
            )
            sx, sy, sz = prop.size
            # The texture face, a hair above the backing.
            ElementTree.SubElement(
                body,
                "geom",
                type="plane",
                size=_fmt((sx, sy, 0.01)),
                pos=_fmt((0, 0, 0.0002)),
                material=f"{prop.name}_mat",
            )
            ElementTree.SubElement(
                body,
                "geom",
                type="box",
                size=_fmt((sx, sy, sz)),
                pos=_fmt((0, 0, -sz)),
                rgba="0.92 0.92 0.92 1",
            )
            return
        ElementTree.SubElement(
            body, "geom", type=prop.kind, size=_fmt(prop.size), rgba=_fmt(prop.rgba)
        )

    # -- posing ------------------------------------------------------------

    def set_pose(self, body: str, pose: np.ndarray) -> None:
        """Place a link, a prop or the eye (a 4x4 pose in the world)."""
        index = self._mocap.get(body)
        if index is None:
            return
        self.data.mocap_pos[index] = pose[:3, 3]
        self.data.mocap_quat[index] = _quat_wxyz(pose[:3, :3])

    def set_robot(self, link_poses: Mapping[str, np.ndarray]) -> None:
        for name, pose in link_poses.items():
            self.set_pose(name, pose)

    # -- rendering ---------------------------------------------------------

    def _renderer(self, width: int, height: int) -> Any:
        key = (width, height)
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = self._mujoco.Renderer(self.model, height=height, width=width)
            self._renderers[key] = renderer
        return renderer

    def render(
        self,
        optical_pose: np.ndarray,
        width: int,
        height: int,
        focal_px: float,
        hide: Iterable[str] = (),
    ) -> np.ndarray:
        """A pinhole image (H, W, 3) from an optical frame's world pose.

        The principal point is the image centre and pixels are square, with
        ``focal_px`` the focal length in pixels. ``hide`` names links left out
        of this image (a camera's own housing, which surrounds its lens).
        """
        mujoco = self._mujoco
        moved = []
        for name in hide:
            index = self._mocap.get(name)
            if index is not None:
                moved.append((index, self.data.mocap_pos[index].copy()))
                self.data.mocap_pos[index] = (1e3, 1e3, -1e3)
        eye = np.eye(4)
        eye[:3, :3] = optical_pose[:3, :3] @ _OPTICAL_TO_MUJOCO
        eye[:3, 3] = optical_pose[:3, 3]
        self.set_pose("_eye", eye)
        self.model.cam_fovy[self._camera] = math.degrees(
            2.0 * math.atan(0.5 * height / focal_px)
        )
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        renderer = self._renderer(width, height)
        renderer.update_scene(self.data, camera=self._camera)
        image = renderer.render()
        for index, position in moved:
            self.data.mocap_pos[index] = position
        return image

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        self._tmp.cleanup()


# ---------------------------------------------------------------------------
# Lens models


@dataclass(frozen=True)
class LensSpec:
    """A camera's factory intrinsics at the recorded size."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str  # plumb_bob, rational_polynomial or equidistant
    d: tuple[float, ...]

    @classmethod
    def from_intrinsics(cls, intrinsics: Any) -> LensSpec:
        return cls(
            width=int(intrinsics.width),
            height=int(intrinsics.height),
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            cx=float(intrinsics.cx),
            cy=float(intrinsics.cy),
            model=str(intrinsics.distortion_model),
            d=tuple(float(v) for v in intrinsics.d),
        )


def _fisheye_theta(theta_d: np.ndarray, k: tuple[float, ...]) -> np.ndarray:
    """Invert theta_d = theta (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)."""
    k1, k2, k3, k4 = (list(k) + [0.0] * 4)[:4]
    theta = theta_d.copy()
    for _ in range(20):
        t2 = theta * theta
        poly = 1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))
        dpoly = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        step = (theta * poly - theta_d) / np.where(np.abs(dpoly) < 1e-9, 1e-9, dpoly)
        theta = theta - step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return theta


def lens_rays(lens: LensSpec) -> np.ndarray:
    """Unit ray (H, W, 3) in the optical frame for every pixel centre."""
    import cv2

    u, v = np.meshgrid(
        np.arange(lens.width, dtype=np.float64),
        np.arange(lens.height, dtype=np.float64),
    )
    if lens.model == "equidistant":
        xd = (u - lens.cx) / lens.fx
        yd = (v - lens.cy) / lens.fy
        theta_d = np.hypot(xd, yd)
        theta = _fisheye_theta(theta_d, lens.d)
        scale = np.where(theta_d > 1e-12, np.sin(theta) / np.maximum(theta_d, 1e-12), 1)
        rays = np.stack([xd * scale, yd * scale, np.cos(theta)], axis=-1)
        return rays
    k = np.array([[lens.fx, 0, lens.cx], [0, lens.fy, lens.cy], [0, 0, 1.0]])
    pts = np.stack([u, v], axis=-1).reshape(-1, 1, 2)
    norm = cv2.undistortPointsIter(
        pts,
        k,
        np.asarray(lens.d, dtype=np.float64),
        None,
        None,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10),
    ).reshape(lens.height, lens.width, 2)
    rays = np.concatenate([norm, np.ones((lens.height, lens.width, 1))], axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


# The five cube faces a fisheye is drawn from: (name, rotation from the face's
# optical frame to the camera's). The face behind the lens is never needed
# for lenses up to 270 degrees across.
_FACES = (
    ("front", np.eye(3)),
    ("right", np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)),
    ("left", np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)),
    ("down", np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)),
    ("up", np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)),
)


class LensImager:
    """Renders what one camera sees, through its lens, at its recorded size."""

    def __init__(
        self, lens: LensSpec, *, supersample: float = 1.25, max_side: int = 2048
    ) -> None:
        import cv2

        self.lens = lens
        rays = lens_rays(lens)
        valid = rays[..., 2] > 1e-6
        if lens.model == "equidistant":
            # Cube faces of 90 degrees, matching the lens's resolution at its
            # centre (fx px per radian is f/2 per unit of tan on a face).
            n = int(min(max_side, round(2 * lens.fx * supersample)))
            self.face = n
            self.passes = [(name, rot) for name, rot in _FACES]
            atlas_x = np.full(rays.shape[:2], -1.0, dtype=np.float32)
            atlas_y = np.full(rays.shape[:2], -1.0, dtype=np.float32)
            best = np.zeros(rays.shape[:2], dtype=bool)
            for i, (_name, rot) in enumerate(self.passes):
                local = rays @ rot  # rays in the face's optical frame
                z = local[..., 2]
                on = (
                    (z > 1e-9)
                    & (np.abs(local[..., 0]) <= z)
                    & (np.abs(local[..., 1]) <= z)
                    & ~best
                )
                f = n / 2.0
                x = f * local[..., 0] / np.where(on, z, 1) + n / 2.0 - 0.5
                y = f * local[..., 1] / np.where(on, z, 1) + n / 2.0 - 0.5
                atlas_x[on] = x[on]
                atlas_y[on] = (y + i * n)[on]
                best |= on
            self.map_x, self.map_y = atlas_x, atlas_y
            self.focal = n / 2.0
            self.size = (n, n)
        else:
            norm_x = rays[..., 0] / np.where(valid, rays[..., 2], 1)
            norm_y = rays[..., 1] / np.where(valid, rays[..., 2], 1)
            f = max(lens.fx, lens.fy) * supersample
            half_w = float(np.max(np.abs(norm_x[valid]))) * f + 2
            half_h = float(np.max(np.abs(norm_y[valid]))) * f + 2
            scale = min(1.0, max_side / (2 * half_w), max_side / (2 * half_h))
            f *= scale
            width = int(math.ceil(2 * half_w * scale))
            height = int(math.ceil(2 * half_h * scale))
            self.passes = [("pinhole", np.eye(3))]
            self.map_x = (f * norm_x + width / 2.0 - 0.5).astype(np.float32)
            self.map_y = (f * norm_y + height / 2.0 - 0.5).astype(np.float32)
            self.focal = f
            self.size = (width, height)
        self.map_x, self.map_y = cv2.convertMaps(
            self.map_x, self.map_y, cv2.CV_16SC2, nninterpolation=False
        )

    def render(
        self, scene: SceneRenderer, optical_pose: np.ndarray, hide: Iterable[str] = ()
    ) -> np.ndarray:
        """The raw image (H, W, 3, RGB) the camera at ``optical_pose`` sees."""
        import cv2

        width, height = self.size
        tiles = []
        for _name, rot in self.passes:
            pose = optical_pose.copy()
            pose[:3, :3] = optical_pose[:3, :3] @ rot
            tiles.append(scene.render(pose, width, height, self.focal, hide))
        atlas = tiles[0] if len(tiles) == 1 else np.concatenate(tiles, axis=0)
        return cv2.remap(
            atlas,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )


def offscreen_backend() -> str:
    """Pick MuJoCo's offscreen OpenGL backend (EGL) unless one is set.

    Call it before MuJoCo is imported, in this process or the workers it
    starts; ``MUJOCO_GL=osmesa`` picks software rendering instead.
    """
    if not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "egl"
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return os.environ["MUJOCO_GL"]
