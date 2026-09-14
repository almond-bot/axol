import os
import unittest
from unittest.mock import Mock, patch

import numpy as np

from almond_axol.kinematics.config import KinematicsConfig
from almond_axol.teleop import worker as worker_module
from almond_axol.teleop.config import VRTeleopConfig
from tests.test_ik_freeze_clutch import _frame


class IKWorkerStartupTests(unittest.TestCase):
    def test_constructor_loads_configured_tcp_transform(self) -> None:
        solver = Mock(num_joints=14)
        solver.left_indices = list(range(7))
        solver.right_indices = list(range(7, 14))
        q = np.zeros(14, dtype=np.float32)
        half = 2**-0.5
        config = VRTeleopConfig(
            tcp_transform_left=[0.1, 0.2, 0.3, 0.0, 0.0, half, half]
        )
        with (
            patch.object(worker_module, "KinematicsSolver", return_value=solver),
            patch.object(worker_module.IKWorker, "_settle_rest_pose", return_value=q),
        ):
            worker = worker_module.IKWorker(config, KinematicsConfig())

        pos = np.array([1.0, 2.0, 3.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        mapped_pos, mapped_quat = worker._apply_tcp_transform("left", pos, quat)
        np.testing.assert_allclose(mapped_pos, [1.1, 2.2, 3.3])
        np.testing.assert_allclose(mapped_quat, [0.0, 0.0, half, half])
        np.testing.assert_allclose(worker._last_mapped_quat["left"], mapped_quat)
        right_pos, right_quat = worker._apply_tcp_transform("right", pos, quat)
        np.testing.assert_array_equal(right_pos, pos)
        np.testing.assert_array_equal(right_quat, quat)

    def test_first_frame_reply_before_reset(self) -> None:
        for absolute, engaged in ((True, False), (True, True), (False, False)):
            with self.subTest(absolute=absolute, engaged=engaged):
                solver = Mock()
                solver.num_joints = 14
                solver.left_indices = list(range(7))
                solver.right_indices = list(range(7, 14))
                solver.fk.return_value = (
                    (np.array([0.4, 0.2, 0.3]), np.eye(3)),
                    (np.array([0.4, -0.2, 0.3]), np.eye(3)),
                )
                solver.ik.side_effect = lambda q, **_kwargs: q.copy()
                q = np.zeros(14, dtype=np.float32)
                conn = Mock()
                conn.recv.side_effect = [
                    _frame(left_forward=0.4, t_ms=1, l_lock=engaged, r_lock=engaged),
                    None,
                ]
                with (
                    patch.object(
                        worker_module, "KinematicsSolver", return_value=solver
                    ),
                    patch.object(
                        worker_module.IKWorker, "_settle_rest_pose", return_value=q
                    ),
                    patch.object(
                        worker_module.IKWorker,
                        "compute_reset_trajectory",
                        return_value=[],
                    ),
                    patch.object(worker_module.signal, "signal"),
                    patch.object(worker_module.os, "nice"),
                    patch.dict(os.environ),
                    patch("almond_axol.utils.affinity.pin_ik_startup"),
                    patch("almond_axol.utils.affinity.pin_ik"),
                ):
                    worker_module.run_ik_worker(
                        conn,
                        VRTeleopConfig(absolute_mode=absolute),
                        KinematicsConfig(),
                    )

                self.assertEqual(conn.send.call_count, 2)
                self.assertEqual(conn.send.call_args_list[0].args[0][0], "ready")
                reply = conn.send.call_args.args[0]
                if absolute:
                    kind, actual_q, base, tcp = reply
                    self.assertEqual(kind, "q")
                    np.testing.assert_array_equal(actual_q, q)
                    self.assertEqual(base is not None, engaged)
                    for side, pose in zip(("left", "right"), solver.fk.return_value):
                        np.testing.assert_allclose(tcp[side][:3], pose[0])
                        self.assertEqual(len(tcp[side]), 7)
                else:
                    np.testing.assert_array_equal(reply, q)
