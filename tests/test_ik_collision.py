"""The fitted IK collision model of every Axol version.

Each URDF's ``<ik_collision>`` are boxes and capsules fitted to each part's
hull (``scripts/build_urdfs.py``). They must enclose every part (IK's
clearance is never more than the real one) while staying close to it, the
capsule-box distance must be exact, and the fingers must follow the gripper
opening.
"""

from __future__ import annotations

import logging
import threading
import unittest
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import trimesh
import yourdfpy
from pyroki.collision import Box, Capsule

from almond_axol.constants import URDF_DIR, AxolModel, urdf_path
from almond_axol.kinematics.collision import capsule_box_distance
from almond_axol.kinematics.config import KinematicsConfig
from almond_axol.kinematics.model import shared_robot, shared_robot_collision
from almond_axol.kinematics.solver import (
    KinematicsSolver,
    body_collision_distance,
    shift_links,
)
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore

_GRIPPERS = ("parallel", "parcel")
# Every bundled URDF: (version, gripper).
_MODELS = (
    *((AxolModel.MOBILE, g) for g in _GRIPPERS),
    *((m, "parallel") for m in AxolModel if m is not AxolModel.MOBILE),
)


def _brute_capsule_box(p0, p1, radius, half, n: int = 4001) -> float:
    """Box-frame segment p0-p1: the box SDF's minimum over dense samples."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    p = p0 + (p1 - p0) * t
    q = np.abs(p) - half
    sdf = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
    return float(sdf.min() - radius)


class CapsuleBoxDistanceTest(unittest.TestCase):
    def test_matches_a_dense_search(self) -> None:
        rng = np.random.default_rng(0)
        n = 400
        half = rng.uniform(0.01, 0.3, (n, 3))
        rot = jaxlie.SO3.sample_uniform(jax.random.PRNGKey(0), batch_axes=(n,))
        box_pos = rng.uniform(-0.2, 0.2, (n, 3))
        # Segments near the box: across faces, edges and corners, some inside.
        mid = box_pos + rng.uniform(-0.5, 0.5, (n, 3))
        axis = rng.normal(size=(n, 3))
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        length = rng.uniform(0.0, 0.5, n)
        radius = rng.uniform(0.01, 0.08, n)
        cap_rot = jaxlie.SO3.from_matrix(
            np.stack(
                [trimesh.geometry.align_vectors([0, 0, 1], a)[:3, :3] for a in axis]
            )
        )
        capsule = Capsule.from_radius_height(
            radius=radius, height=length, position=mid, wxyz=cap_rot.wxyz
        )
        box = Box.from_extent(extent=2 * half, position=box_pos, wxyz=rot.wxyz)
        got = np.asarray(capsule_box_distance(capsule, box))
        to_box = np.asarray(rot.inverse().as_matrix())
        for k in range(n):
            p0 = to_box[k] @ (mid[k] - axis[k] * length[k] / 2 - box_pos[k])
            p1 = to_box[k] @ (mid[k] + axis[k] * length[k] / 2 - box_pos[k])
            want = _brute_capsule_box(p0, p1, radius[k], half[k])
            self.assertAlmostEqual(got[k], want, delta=2e-4, msg=k)

    def test_gradient_is_finite_touching_a_face(self) -> None:
        # A segment lying on a face: the SDF is flat along it.
        def d(z):
            capsule = Capsule.from_radius_height(
                radius=0.02, height=0.2, position=jnp.array([0.0, 0.0, z])
            )
            box = Box.from_extent(extent=jnp.array([0.4, 0.4, 0.2]))
            return capsule_box_distance(capsule, box)

        grad = jax.grad(d)(0.1)
        self.assertTrue(np.isfinite(grad))
        self.assertAlmostEqual(float(grad), 1.0, places=4)


def _ik_shapes(link) -> list[tuple[str, np.ndarray, tuple[float, ...]]]:
    """A link's IK shapes: (kind, pose in the link frame, size) where size
    is (radius, length) for a capsule and the extents for a box."""
    out = []
    for el in link.findall("ik_collision"):
        o = el.find("origin")
        pose = trimesh.transformations.euler_matrix(
            *(float(v) for v in o.get("rpy").split()), axes="sxyz"
        )
        pose[:3, 3] = [float(v) for v in o.get("xyz").split()]
        cyl, box = el.find("geometry/cylinder"), el.find("geometry/box")
        if cyl is not None:
            out.append(
                ("capsule", pose, (float(cyl.get("radius")), float(cyl.get("length"))))
            )
        else:
            out.append(("box", pose, tuple(float(v) for v in box.get("size").split())))
    return out


def _encloses(kind: str, pose: np.ndarray, size, pts: np.ndarray) -> bool:
    local = trimesh.transform_points(pts, np.linalg.inv(pose))
    if kind == "box":
        return bool(np.all(np.abs(local) <= np.asarray(size) / 2 + 1e-6))
    radius, length = size
    t = np.clip(local[:, 2], -length / 2, length / 2)
    dist = np.linalg.norm(local - np.c_[np.zeros((len(t), 2)), t], axis=1)
    return bool(dist.max() <= radius)


def _volume(kind: str, size) -> float:
    if kind == "box":
        return float(np.prod(size))
    radius, length = size
    return float(np.pi * radius**2 * length + 4 / 3 * np.pi * radius**3)


def _mesh_path(path, filename: str):
    # The classic URDF's meshes are package paths; the others' are relative.
    prefix = "package://assembly/"
    if filename.startswith(prefix):
        return URDF_DIR / filename[len(prefix) :]
    return path.parent / filename


class FittedShapesTest(unittest.TestCase):
    """Every part IK should see is inside an IK shape, and each shape is
    close to the part it is fitted to (everything in the world frame, every
    joint at zero)."""

    def _check(self, model: AxolModel, gripper: str) -> None:
        from xml.etree import ElementTree

        path = urdf_path(model, gripper=gripper)
        root = ElementTree.parse(path).getroot()
        full = yourdfpy.URDF.load(
            str(path), build_collision_scene_graph=False, load_meshes=False
        )
        full.update_cfg({n: 0.0 for n in full.actuated_joint_names})
        shapes, hulls = [], []
        for link in root.findall("link"):
            name = link.get("name")
            world = full.get_transform(name)
            for kind, pose, size in _ik_shapes(link):
                shapes.append((name, kind, world @ pose, size))
            for col in link.findall("collision"):
                ref = col.find("geometry/mesh").get("filename")
                # Out of the arms' reach: the wheels and the Ox cart's caster
                # frame (IK holds the lifts fully raised).
                if name.startswith("wheel_") or "cart_frame" in ref:
                    continue
                hull = trimesh.load(_mesh_path(path, ref), force="mesh").convex_hull
                o = col.find("origin")
                pose = np.eye(4)
                if o is not None:
                    pose = trimesh.transformations.euler_matrix(
                        *(float(v) for v in o.get("rpy").split()), axes="sxyz"
                    )
                    pose[:3, 3] = [float(v) for v in o.get("xyz").split()]
                hulls.append((ref, hull.apply_transform(world @ pose)))
        for ref, hull in hulls:
            with self.subTest(model=model.value, gripper=gripper, part=ref):
                self.assertTrue(
                    any(_encloses(k, p, sz, hull.vertices) for _, k, p, sz in shapes)
                )
        for name, kind, pose, size in shapes:
            # The parcel plate's sweep covers its whole motion.
            if name.endswith("_fingers"):
                continue
            with self.subTest(model=model.value, gripper=gripper, link=name):
                inside = [
                    h for _, h in hulls if _encloses(kind, pose, size, h.vertices)
                ]
                self.assertTrue(inside)
                # Tight: one capsule per link was 2-24 times its part's volume.
                # The Owl foot is a thin plate with a sloping rib: its hull is
                # a wedge, a third of its box.
                limit = 2.0 if kind == "box" else 5.0
                if name == "foot":
                    limit = 3.0
                ratio = _volume(kind, size) / max(h.volume for h in inside)
                self.assertLess(ratio, limit)

    def test_every_version(self) -> None:
        for model, gripper in _MODELS:
            self._check(model, gripper)


class FingerSlideTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = KinematicsSolver(
            KinematicsConfig(robot_model="mobile", gripper="parallel")
        )

    def tearDown(self) -> None:
        self.solver.set_gripper(1.0, 1.0)

    def _clearance(self, finger: str, body: str = "jelly") -> float:
        s = self.solver
        rc = s.robot_coll
        d = np.asarray(
            body_collision_distance(
                s.robot, rc, jnp.zeros(s.robot.joints.num_actuated_joints), s.link_shift
            )
        )
        return min(
            float(d[k])
            for k, (i, j) in enumerate(zip(rc.active_idx_i, rc.active_idx_j))
            if (rc.link_names[int(i)], rc.link_names[int(j)]) == (finger, body)
        )

    def test_opening_brings_the_inner_finger_to_the_column(self) -> None:
        s = self.solver
        s.set_gripper(0.0, 0.0)
        closed = self._clearance("left_finger_1")
        s.set_gripper(1.0, 0.0)
        # Each finger travels 58.8 mm; the inner one toward the column.
        self.assertLess(self._clearance("left_finger_1"), closed - 0.03)
        # The other gripper stays where it was.
        s.set_gripper(0.0, 0.0)
        right = self._clearance("right_finger_2")
        s.set_gripper(0.0, 1.0)
        self.assertLess(self._clearance("right_finger_2"), right - 0.03)
        s.set_gripper(1.0, 0.0)
        self.assertAlmostEqual(self._clearance("right_finger_2"), right, places=5)

    def test_starts_open(self) -> None:
        self.assertEqual(self.solver.grip, (1.0, 1.0))

    def test_parcel_gripper_has_no_sliding_fingers(self) -> None:
        s = KinematicsSolver(KinematicsConfig(robot_model="mobile", gripper="parcel"))
        before = np.asarray(s.link_shift)
        s.set_gripper(0.0, 0.0)
        np.testing.assert_array_equal(np.asarray(s.link_shift), before)

    def test_fingers_follow_the_urdf_on_every_parallel_gripper(self) -> None:
        # The exports freeze the fingers at different points of their
        # stroke (closed on Jelly, open or part way on the others): the
        # shift must put them where the full URDF does.
        expected = {
            AxolModel.MOBILE: (0.0, 0.0588),
            AxolModel.STANDALONE: (0.0, -0.044),
            AxolModel.OX_CART: (0.0, -0.044),
            AxolModel.OWL: (0.011, -0.044),
        }
        for model, (closed_q, stroke) in expected.items():
            with self.subTest(model=model.value):
                s = KinematicsSolver(KinematicsConfig(robot_model=model.value))
                names = list(s.robot.links.names)
                full = yourdfpy.URDF.load(
                    str(urdf_path(model)),
                    build_collision_scene_graph=False,
                    load_meshes=False,
                )
                poses = s.robot.forward_kinematics(
                    jnp.zeros(s.robot.joints.num_actuated_joints)
                )
                for grip in (0.0, 0.4, 1.0):
                    s.set_gripper(grip, 1.0 - grip)
                    shifted = np.asarray(shift_links(poses, s.link_shift))
                    cfg = {n: 0.0 for n in full.actuated_joint_names}
                    cfg["left_finger_1_joint"] = closed_q + grip * stroke
                    cfg["right_finger_1_joint"] = closed_q + (1.0 - grip) * stroke
                    full.update_cfg(cfg)
                    for side in ("left", "right"):
                        for k in (1, 2):
                            link = f"{side}_finger_{k}"
                            want = full.get_transform(link, "root")[:3, 3]
                            got = shifted[names.index(link), 4:]
                            np.testing.assert_allclose(got, want, atol=2e-4)

    def test_classic_gripper_is_rigid(self) -> None:
        s = KinematicsSolver(KinematicsConfig(robot_model="classic"))
        before = np.asarray(s.link_shift)
        s.set_gripper(0.0, 0.0)
        np.testing.assert_array_equal(np.asarray(s.link_shift), before)


class ShapesTest(unittest.TestCase):
    def test_every_version_collides_boxes_and_capsules(self) -> None:
        for model, gripper in _MODELS:
            with self.subTest(model=model.value, gripper=gripper):
                rc = shared_robot_collision(model, gripper=gripper)
                # The shoulder bar at least is a box.
                self.assertIn("s1", {rc.link_names[k] for k in rc.box_link})
                # Every arm shape is a capsule: no box is on an arm link.
                for k in rc.box_link:
                    self.assertFalse(rc.link_names[k].startswith(("left_", "right_")))
                # Each arm shape mirrors the other side's (bar the fingers):
                # both arms see the body alike.
                names = [rc.link_names[k] for k in rc.capsule_link]
                arm = [n for n in names if n.startswith("left_")]
                self.assertEqual(
                    sorted(n.replace("left_", "right_") for n in arm),
                    sorted(n for n in names if n.startswith("right_")),
                )

    def test_robot_link_order(self) -> None:
        for model, gripper in _MODELS:
            with self.subTest(model=model.value, gripper=gripper):
                rc = shared_robot_collision(model, gripper=gripper)
                robot = shared_robot(model, gripper=gripper)
                self.assertEqual(rc.link_names, tuple(robot.links.names))


class WorkerForwardTest(unittest.TestCase):
    def _worker(self, mink: bool):
        from almond_axol.teleop.worker import IKWorker

        worker = IKWorker.__new__(IKWorker)
        worker._mink_backend = mink
        # The Mink solver has neither method.
        worker._solver = MagicMock(spec=[]) if mink else MagicMock()
        return worker

    def test_pyroki_solver_gets_the_openings_and_height(self) -> None:
        worker = self._worker(mink=False)
        worker.set_gripper(0.25, 1.0)
        worker.set_lift(0.5)
        worker._solver.set_gripper.assert_called_once_with(0.25, 1.0)
        worker._solver.set_lift.assert_called_once_with(0.5)

    def test_mink_solver_ignores_them(self) -> None:
        worker = self._worker(mink=True)
        worker.set_gripper(0.25, 1.0)
        worker.set_lift(0.5)


class _Conn:
    def __init__(self) -> None:
        self.sent: list = []

    def send(self, msg) -> None:
        self.sent.append(msg)

    def poll(self, _timeout=None) -> bool:
        return True

    def recv(self):
        return (np.zeros(16, np.float32), None)


class ForwardGripTest(unittest.TestCase):
    def test_sends_the_openings_when_they_move(self) -> None:
        core = VRTeleopCore(
            VRTeleopConfig(ik_frequency=1000.0),
            logging.getLogger("test"),
            broadcast_tracking=lambda _enabled: None,
        )
        core.set_target(np.zeros(16, np.float32))
        grips = iter([(0.0, 0.0), (0.0, 0.0), (0.005, 0.0), (0.5, 0.0), (0.5, 1.0)])
        stop = threading.Event()

        def get_lift():
            # Called once per tick, before the grip is read: step the grips.
            try:
                core.l_grip, core.r_grip = next(grips)
            except StopIteration:
                stop.set()
            return None

        conn = _Conn()
        core.run_ik_loop(
            conn,
            MagicMock(return_value=None),
            stop,
            lambda: True,
            lambda _t: None,
            get_lift=get_lift,
        )
        sent = [m[1:] for m in conn.sent if isinstance(m, tuple) and m[0] == "grip"]
        # Sub-percent moves are dropped.
        self.assertEqual(sent, [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0)])


if __name__ == "__main__":
    unittest.main()
