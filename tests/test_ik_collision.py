"""The fitted IK collision model of Axol on Jelly.

The Jelly URDFs' ``<ik_collision>`` are boxes and capsules fitted to each
part's hull (``scripts/build_urdfs.py``, ``ik_primitives``). They must enclose
every part (IK's clearance is never more than the real one) while staying
close to it, the capsule-box distance must be exact, and the fingers must
follow the gripper opening.
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

from almond_axol.constants import AxolModel, urdf_path
from almond_axol.kinematics.collision import capsule_box_distance
from almond_axol.kinematics.config import KinematicsConfig
from almond_axol.kinematics.model import shared_robot_collision
from almond_axol.kinematics.solver import KinematicsSolver, body_collision_distance
from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore

_GRIPPERS = ("parallel", "parcel")


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


class FittedShapesTest(unittest.TestCase):
    """Each IK shape encloses its part's hull, and not by much."""

    def _check(self, gripper: str) -> None:
        from xml.etree import ElementTree

        path = urdf_path(AxolModel.MOBILE, gripper=gripper)
        root = ElementTree.parse(path).getroot()
        full = yourdfpy.URDF.load(
            str(path), build_collision_scene_graph=False, load_meshes=False
        )
        full.update_cfg({n: 0.0 for n in full.actuated_joint_names})
        links = {ln.get("name"): ln for ln in root.findall("link")}
        checked = 0
        for name, link in links.items():
            shapes = _ik_shapes(link)
            # The parcel plate's sweep is fitted over its whole motion.
            if not shapes or name.endswith("_fingers"):
                continue
            # The part's hulls; the deck's is on the Jelly base link.
            owner = "jelly" if name == "deck" else name
            hulls = []
            for col in links[owner].findall("collision"):
                ref = col.find("geometry/mesh").get("filename")
                if (name == "deck") != ("deck" in ref):
                    continue
                hull = trimesh.load(path.parent / ref, force="mesh")
                to_link = np.linalg.inv(full.get_transform(name)) @ full.get_transform(
                    owner
                )
                hulls.append(hull.apply_transform(to_link))
            pts = np.vstack([h.vertices for h in hulls])
            hull = trimesh.util.concatenate(hulls).convex_hull
            with self.subTest(gripper=gripper, link=name):
                ((kind, pose, size),) = shapes
                self.assertTrue(_encloses(kind, pose, size, pts))
                # Tight: the old fits were 2-24 times their part's volume.
                ratio = _volume(kind, size) / hull.volume
                self.assertLess(ratio, 2.0 if kind == "box" else 5.0)
                checked += 1
        self.assertGreater(checked, 20)

    def test_parallel_gripper(self) -> None:
        self._check("parallel")

    def test_parcel_gripper(self) -> None:
        self._check("parcel")


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


class ShapesTest(unittest.TestCase):
    def test_jelly_collides_boxes_and_capsules(self) -> None:
        for gripper in _GRIPPERS:
            with self.subTest(gripper=gripper):
                rc = shared_robot_collision(AxolModel.MOBILE, gripper=gripper)
                self.assertGreater(len(rc.box_link), 0)
                # Every arm shape is a capsule: no box is on an arm link.
                for k in rc.box_link:
                    self.assertFalse(rc.link_names[k].startswith(("left_", "right_")))

    def test_other_versions_keep_one_capsule_per_link(self) -> None:
        for model in (AxolModel.CLASSIC, AxolModel.OWL):
            with self.subTest(model=model.value):
                rc = shared_robot_collision(model)
                self.assertEqual(rc.box_link, ())
                self.assertEqual(len(rc.capsule_link), len(rc.link_names))


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
