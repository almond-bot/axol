"""Mink dispatch, continuity and startup guards with a fake solver and core."""

from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from almond_axol.cli import run_policy
from almond_axol.lerobot.robot.config_axol import AxolRobotConfig
from almond_axol.lerobot.robot.robot_axol import AxolRobot


class MinkRobotControllerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mink-robot-test-")
        self.addCleanup(self.temporary.cleanup)
        self.robot = AxolRobot(
            AxolRobotConfig(
                cameras={},
                calibration_dir=Path(self.temporary.name),
                observe_cartesian=False,
                action_space="cartesian",
                cartesian_controller="mink",
            )
        )
        self.dispatches = []

        async def motion_control(*, left, right):
            self.dispatches.append((threading.get_ident(), left.copy(), right.copy()))

        self.robot._axol = SimpleNamespace(
            left=SimpleNamespace(positions=np.zeros(8, dtype=np.float32)),
            right=SimpleNamespace(positions=np.zeros(8, dtype=np.float32)),
            motion_control=mock.AsyncMock(side_effect=motion_control),
        )
        self.solver = mock.Mock(
            spec=["solve", "set_rest_posture", "reset_tracking_state"]
        )
        self.solver.solve.return_value = np.linspace(-0.1, 0.1, 14).astype(np.float32)
        self.robot._ik = self.solver
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()

        def run_loop():
            asyncio.set_event_loop(self.loop)
            self.ready.set()
            self.loop.run_forever()

        self.worker = threading.Thread(target=run_loop, name="fake-core-loop")
        self.worker.start()
        self.assertTrue(self.ready.wait(timeout=2))
        self.robot._loop = self.loop
        self.addCleanup(self.stop_loop)
        self.action = dict.fromkeys(self.robot.action_features, 0.0)
        self.action["left_ee.z"] = 0.3
        self.action["right_ee.z"] = 0.3
        self.action["left_gripper.pos"] = 0.2
        self.action["right_gripper.pos"] = 0.7
        self.guards = ExitStack()
        self.addCleanup(self.guards.close)
        for target in (
            "almond_axol.lerobot.robot.robot_axol.TrapezoidalFilter",
            "almond_axol.lerobot.robot.robot_axol.AxolRobot.connect",
        ):
            self.guards.enter_context(
                mock.patch(target, side_effect=AssertionError(f"forbidden {target}"))
            )

    def stop_loop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.worker.join(timeout=2)
        self.assertFalse(self.worker.is_alive())
        self.robot._axol = None
        self.robot._loop = None
        self.loop.close()

    def test_sync_solve_stays_on_caller_and_only_core_dispatch_uses_loop(self):
        solver_thread = []
        output = self.solver.solve.return_value.copy()

        def solve(*args):
            solver_thread.append(threading.get_ident())
            return output

        self.solver.solve.side_effect = solve
        sent = self.robot.send_action(self.action)
        self.assertEqual(solver_thread, [threading.get_ident()])
        self.assertEqual(len(self.dispatches), 1)
        thread_id, left, right = self.dispatches[0]
        self.assertEqual(thread_id, self.worker.ident)
        np.testing.assert_array_equal(left[:7], output[:7])
        np.testing.assert_array_equal(right[:7], output[7:])
        np.testing.assert_allclose([left[7], right[7]], [0.2, 0.7])
        self.assertNotIn("left_ee.x", sent)
        self.assertEqual(set(sent), set(self.robot.observation_features))
        self.assertIsNone(self.robot._cartesian_shapers)
        np.testing.assert_array_equal(
            self.robot._last_joint_command, np.concatenate((left, right))
        )

    def test_handover_transform_runs_after_ik_and_returns_actual_joint_send(self):
        original = self.action.copy()
        seen = []

        def transition(joints):
            seen.append(dict(joints))
            self.assertEqual(set(joints), set(self.robot.observation_features))
            return {key: value / 2 for key, value in joints.items()}

        sent = self.robot.send_action(self.action, joint_transform=transition)
        self.assertEqual(self.action, original)
        self.assertEqual(len(self.dispatches), 1)
        _, left, right = self.dispatches[0]
        np.testing.assert_allclose(left[:7], self.solver.solve.return_value[:7] / 2)
        np.testing.assert_allclose(right[:7], self.solver.solve.return_value[7:] / 2)
        self.assertEqual(sent, {key: value / 2 for key, value in seen[0].items()})
        self.assertAlmostEqual(left[7], 0.1)
        self.assertAlmostEqual(right[7], 0.35)
        np.testing.assert_array_equal(
            self.robot._last_joint_command, np.r_[left, right]
        )

    def test_takeover_rejection_after_ik_never_submits_motor_send(self):
        def revoke(joints):
            raise RuntimeError("operator took over during IK")

        with self.assertRaisesRegex(RuntimeError, "operator took over"):
            self.robot.send_action(self.action, joint_transform=revoke)
        self.solver.solve.assert_called_once()
        self.assertEqual(self.dispatches, [])
        self.assertIsNone(self.robot._last_joint_command)
        # Rejection relinquishes operation ownership; a joint hold can send.
        hold = dict.fromkeys(self.robot.observation_features, 0.0)
        self.assertEqual(self.robot.send_action(hold), hold)
        self.assertEqual(len(self.dispatches), 1)

    def test_seed_uses_previous_command_only_below_strict_resync_threshold(self):
        for distance, use_previous in (
            (0.349, True),
            (0.35, False),
            (0.351, False),
        ):
            with self.subTest(distance=distance):
                previous = np.zeros(14, dtype=np.float64)
                previous[0] = distance
                self.robot._last_ik_q = previous.copy()
                self.solver.solve.reset_mock()
                self.robot.send_action(self.action)
                seed = self.solver.solve.call_args.args[0]
                np.testing.assert_array_equal(
                    seed, previous if use_previous else np.zeros(14)
                )

    def test_current_frame_hold_reaches_solver_in_model_frame(self):
        from scipy.spatial.transform import Rotation

        fixture = Path(__file__).with_name("data") / "mink_ik_reference" / "stream.npz"
        with np.load(fixture, allow_pickle=False) as saved:
            joints = saved["fk_joints"][-1].copy()
            positions = saved["fk_positions"][-1].copy()
            rotations = saved["fk_rotations"][-1].copy()
        self.robot._axol.left.positions[:7] = joints[:7]
        self.robot._axol.right.positions[:7] = joints[7:]
        self.solver.solve.return_value = joints.copy()
        quarter_turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        root = np.array([0, 0, 0.86])
        for index, side in enumerate(("left", "right")):
            # Build generic wire poses independently from recorded model-frame FK.
            position = root + quarter_turn @ (positions[index] - root)
            rotation = Rotation.from_matrix(quarter_turn @ rotations[index]).as_rotvec()
            for axis, value in zip(
                ("x", "y", "z", "rx", "ry", "rz"),
                np.concatenate((position, rotation)),
                strict=True,
            ):
                self.action[f"{side}_ee.{axis}"] = float(value)
        self.robot.send_action(self.action)
        seed, left, right = self.solver.solve.call_args.args
        np.testing.assert_array_equal(seed, joints.astype(np.float32))
        for index, (position, rotation) in enumerate((left, right)):
            np.testing.assert_allclose(position, positions[index], atol=1e-6)
            np.testing.assert_allclose(rotation, rotations[index], atol=1e-6)

    def test_first_solve_uses_measured_joints_then_reuses_previous_solution(self):
        self.robot._axol.left.positions[:7] = 0.04
        self.robot._axol.right.positions[:7] = -0.03
        self.robot.send_action(self.action)
        np.testing.assert_array_equal(
            self.solver.solve.call_args.args[0],
            np.concatenate(
                (
                    self.robot._axol.left.positions[:7],
                    self.robot._axol.right.positions[:7],
                )
            ),
        )
        previous = self.solver.solve.return_value.copy()
        self.robot.send_action(self.action)
        np.testing.assert_array_equal(self.solver.solve.call_args.args[0], previous)

    def test_nonfinite_input_never_solves_or_sends_and_preserves_seed(self):
        seed = np.full(14, 0.1, dtype=np.float32)
        for source in ("pose", "gripper", "measurement"):
            with self.subTest(source=source):
                self.robot._last_ik_q = seed.copy()
                action = self.action.copy()
                if source == "pose":
                    action["left_ee.x"] = np.nan
                elif source == "gripper":
                    action["right_gripper.pos"] = np.inf
                else:
                    self.robot._axol.left.positions[0] = np.nan
                try:
                    with self.assertRaisesRegex(ValueError, "finite"):
                        self.robot.send_action(action)
                finally:
                    self.robot._axol.left.positions[0] = 0
                np.testing.assert_array_equal(self.robot._last_ik_q, seed)
                self.solver.solve.assert_not_called()
                self.robot._axol.motion_control.assert_not_awaited()
                self.assertIsNone(self.robot._last_joint_command)

    def test_invalid_solver_output_never_sends_or_poisons_seed(self):
        seed = np.full(14, 0.1, dtype=np.float32)
        for output in (np.full(14, np.nan), np.full(14, np.inf), np.zeros(13)):
            with self.subTest(output=repr(output)):
                self.robot._last_ik_q = seed.copy()
                self.solver.solve.return_value = output
                with self.assertRaisesRegex(ValueError, "invalid joint command"):
                    self.robot.send_action(self.action)
                np.testing.assert_array_equal(self.robot._last_ik_q, seed)
                self.robot._axol.motion_control.assert_not_awaited()
                self.assertIsNone(self.robot._last_joint_command)

    def test_episode_seed_tracks_last_joint_reset_waypoint(self):
        self.robot.send_action(self.action)
        first = self.solver.solve.return_value.copy()
        self.robot.reset_cartesian_seed()
        np.testing.assert_array_equal(self.robot._last_ik_q, first)
        waypoint = {
            key: value
            for key, value in zip(
                self.robot.observation_features,
                np.linspace(-0.05, 0.05, 16),
                strict=True,
            )
        }
        self.robot.send_action(waypoint)
        # A reset waypoint changes the episode anchor even though no IK ran.
        self.robot.reset_cartesian_seed()
        expected = self.robot._last_joint_command.copy()
        expected = np.concatenate((expected[:7], expected[8:15]))
        np.testing.assert_array_equal(self.robot._last_ik_q, expected)
        self.robot.send_action(self.action)
        np.testing.assert_array_equal(self.solver.solve.call_args.args[0], expected)
        self.assertEqual(self.solver.reset_tracking_state.call_count, 2)

    def test_async_solve_does_not_block_core_loop(self):
        entered = threading.Event()
        release = threading.Event()
        solver_threads = []
        output = self.solver.solve.return_value.copy()

        def solve(*args):
            solver_threads.append(threading.get_ident())
            entered.set()
            if not release.wait(timeout=3):
                raise AssertionError("test did not release solver")
            return output

        async def heartbeat():
            return threading.get_ident()

        self.solver.solve.side_effect = solve
        pending = asyncio.run_coroutine_threadsafe(
            self.robot.send_action_async(self.action), self.loop
        )
        try:
            self.assertTrue(entered.wait(timeout=2))
            beat = asyncio.run_coroutine_threadsafe(heartbeat(), self.loop)
            self.assertEqual(beat.result(timeout=1), self.worker.ident)
            self.assertFalse(pending.done())
            self.assertEqual(self.dispatches, [])
        finally:
            release.set()
            sent = pending.result(timeout=2)
        self.assertNotEqual(solver_threads[0], self.worker.ident)
        self.assertEqual(self.dispatches[0][0], self.worker.ident)
        self.assertNotIn("left_ee.x", sent)
        self.assertIsNone(self.robot._cartesian_shapers)

    def test_generic_episode_reset_calls_robot_seed_reset(self):
        from almond_axol.lerobot.inference_patch import (
            import_robot_client_preserving_logging,
        )

        import_robot_client_preserving_logging()
        config = SimpleNamespace(
            fps=30,
            environment_dt=1 / 30,
            server_address="unused",
            policy_type="custom",
            pretrained_name_or_path="custom",
            actions_per_chunk=30,
            policy_device="cpu",
            client_device="cpu",
            task="local recording label",
            aggregate_fn=None,
        )
        client = run_policy._build_axol_robot_client(
            config=config,
            robot=self.robot,
            publisher=None,
            custom_policy_url="ws://unused",
        )
        self.addCleanup(client.stop)
        client._policy_client.reset = mock.Mock()
        previous = np.linspace(-0.1, 0.1, 16).astype(np.float32)
        self.robot._last_joint_command = previous.copy()
        with mock.patch.object(
            self.robot, "_joints_to_cartesian", return_value=self.action.copy()
        ):
            client.reset_episode_state()
        np.testing.assert_array_equal(
            self.robot._last_ik_q, np.concatenate((previous[:7], previous[8:15]))
        )
        self.solver.reset_tracking_state.assert_called_once_with()

    def _recording_client(self):
        from almond_axol.lerobot.inference_patch import (
            import_robot_client_preserving_logging,
        )
        from almond_axol.lerobot.rollout import ActionPublisher

        import_robot_client_preserving_logging()
        publisher = ActionPublisher()
        config = SimpleNamespace(
            fps=30,
            environment_dt=1 / 30,
            server_address="unused",
            policy_type="custom",
            pretrained_name_or_path="custom",
            actions_per_chunk=30,
            policy_device="cpu",
            client_device="cpu",
            task="recording regression",
            aggregate_fn=None,
        )
        client = run_policy._build_axol_robot_client(
            config=config,
            robot=self.robot,
            publisher=publisher,
            custom_policy_url="ws://unused",
        )
        self.addCleanup(client.stop)
        client._action_schema_confirmed = True
        self.robot._axol.torque_residuals = lambda: (np.zeros(7), np.zeros(7))
        return client, publisher

    def test_policy_recording_maps_dispatched_mink_joints_to_cartesian_schema(self):
        from lerobot.utils.feature_utils import build_dataset_frame

        from almond_axol.kinematics.mujoco_fk import AxolForwardKinematics
        from almond_axol.recording.datasets import dataset_features_for_robot

        self.robot._fk = AxolForwardKinematics()
        client, publisher = self._recording_client()
        performed = client._shape_and_send(np.array(list(self.action.values())))
        recorded = publisher.latest()
        self.assertEqual(set(performed), set(self.robot.observation_features))
        self.assertEqual(set(recorded), set(self.robot.action_features))
        _, left, right = self.dispatches[-1]
        left_pose, right_pose = self.robot._fk.ee_poses(left, right)
        np.testing.assert_allclose(
            list(recorded.values()), np.r_[left_pose, left[7], right_pose, right[7]]
        )
        self.assertNotEqual(recorded["left_ee.z"], self.action["left_ee.z"])
        features = dataset_features_for_robot(self.robot)
        frame = build_dataset_frame(features, recorded, prefix="action")
        self.assertEqual(frame["action"].shape, (14,))
        self.assertTrue(np.isfinite(frame["action"]).all())

    def test_policy_recording_preserves_actions_already_in_dataset_schema(self):
        client, publisher = self._recording_client()
        with (
            mock.patch.object(self.robot, "send_action", return_value=self.action),
            mock.patch.object(
                self.robot,
                "action_to_dataset",
                side_effect=AssertionError("already in Cartesian schema"),
            ),
        ):
            performed = client._shape_and_send(np.array(list(self.action.values())))
        self.assertIs(performed, self.action)
        self.assertEqual(publisher.latest(), self.action)

    def test_overlapping_mink_sends_are_refused_before_second_solve(self):
        entered = threading.Event()
        release = threading.Event()
        output = self.solver.solve.return_value.copy()

        def solve(*args):
            entered.set()
            if not release.wait(timeout=3):
                raise AssertionError("test did not release solver")
            return output

        self.solver.solve.side_effect = solve
        pending = asyncio.run_coroutine_threadsafe(
            self.robot.send_action_async(self.action), self.loop
        )
        try:
            self.assertTrue(entered.wait(timeout=2))
            with self.assertRaisesRegex(
                RuntimeError, "already has an action in flight"
            ):
                self.robot.send_action(self.action)
            overlapping = asyncio.run_coroutine_threadsafe(
                self.robot.send_action_async(self.action), self.loop
            )
            with self.assertRaisesRegex(
                RuntimeError, "already has an action in flight"
            ):
                overlapping.result(timeout=1)
            self.solver.solve.assert_called_once()
            self.assertEqual(self.dispatches, [])
        finally:
            release.set()
            pending.result(timeout=2)
        self.assertEqual(len(self.dispatches), 1)

    def test_cancelled_async_solve_drains_before_releasing_ownership(self):
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        output = self.solver.solve.return_value.copy()

        def solve(*args):
            entered.set()
            if not release.wait(timeout=3):
                raise AssertionError("test did not release solver")
            finished.set()
            return output

        self.solver.solve.side_effect = solve

        async def scenario():
            task = asyncio.create_task(self.robot.send_action_async(self.action))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                self.assertTrue(self.robot._mink_send_lock.locked())
                with self.assertRaisesRegex(RuntimeError, "action in flight"):
                    await self.robot.send_action_async(self.action)
                self.assertFalse(finished.is_set())
                self.assertEqual(self.dispatches, [])
            finally:
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            self.assertTrue(finished.is_set())
            self.assertFalse(self.robot._mink_send_lock.locked())
            np.testing.assert_array_equal(self.robot._last_ik_q, output)
            self.assertIsNone(self.robot._last_joint_command)
            self.assertEqual(self.dispatches, [])
            await self.robot.send_action_async(self.action)
            self.assertEqual(len(self.dispatches), 1)

        pending = asyncio.run_coroutine_threadsafe(scenario(), self.loop)
        try:
            pending.result(timeout=5)
        finally:
            release.set()

    def test_sync_timeout_cancels_queued_dispatch_before_loop_resumes(self):
        entered = threading.Event()
        release = threading.Event()

        def block_loop():
            entered.set()
            if not release.wait(timeout=3):
                raise AssertionError("test did not release event loop")

        self.loop.call_soon_threadsafe(block_loop)
        self.assertTrue(entered.wait(timeout=1))
        submit = asyncio.run_coroutine_threadsafe
        futures = []

        def submit_and_release_after_cancel(coroutine, loop):
            future = submit(coroutine, loop)
            futures.append(future)
            if len(futures) == 1:
                future.add_done_callback(
                    lambda done: release.set() if done.cancelled() else None
                )
            return future

        try:
            with (
                mock.patch(
                    "asyncio.run_coroutine_threadsafe",
                    side_effect=submit_and_release_after_cancel,
                ),
                self.assertRaises(TimeoutError),
            ):
                self.robot.send_action(self.action)
        finally:
            release.set()
        self.assertTrue(futures[0].cancelled())
        self.assertEqual(self.dispatches, [])
        self.assertFalse(self.robot._dispatch_untrusted)
        self.assertFalse(self.robot._mink_send_lock.locked())
        self.robot.send_action(self.action)
        self.assertEqual(len(self.dispatches), 1)

    def test_undrained_sync_timeout_blocks_future_sends(self):
        from almond_axol.robot.base import HardwareCleanupError

        entered = threading.Event()
        release = threading.Event()

        def block_loop():
            entered.set()
            if not release.wait(timeout=4):
                raise AssertionError("test did not release event loop")

        self.loop.call_soon_threadsafe(block_loop)
        self.assertTrue(entered.wait(timeout=1))
        try:
            with self.assertRaisesRegex(HardwareCleanupError, "did not drain"):
                self.robot.send_action(self.action)
            self.assertTrue(self.robot._dispatch_untrusted)
            with self.assertRaisesRegex(
                HardwareCleanupError, "Previous action dispatch"
            ):
                self.robot.send_action(self.action)
        finally:
            release.set()
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0), self.loop).result(
                timeout=1
            )
        self.assertEqual(self.dispatches, [])

    def test_sync_timeout_waits_for_motion_cancellation_cleanup(self):
        cleanup_entered = threading.Event()
        coroutine_exited = threading.Event()
        sender_exited = threading.Event()
        cleanup_release = asyncio.Event()
        failures = []

        async def motion_with_cleanup(**kwargs):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_entered.set()
                await cleanup_release.wait()
                raise
            finally:
                coroutine_exited.set()

        def send():
            try:
                self.robot.send_action(self.action)
            except BaseException as error:
                failures.append(error)
            finally:
                sender_exited.set()

        self.robot._axol.motion_control.side_effect = motion_with_cleanup
        sender = threading.Thread(target=send, name="fake-policy-dispatch")
        sender.start()
        try:
            self.assertTrue(cleanup_entered.wait(timeout=2))
            self.assertFalse(sender_exited.wait(timeout=0.1))
            self.assertTrue(self.robot._mink_send_lock.locked())
            self.assertFalse(coroutine_exited.is_set())
        finally:
            self.loop.call_soon_threadsafe(cleanup_release.set)
            sender.join(timeout=2)
        self.assertFalse(sender.is_alive())
        self.assertTrue(coroutine_exited.is_set())
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], TimeoutError)
        self.assertFalse(self.robot._mink_send_lock.locked())
        self.assertFalse(self.robot._dispatch_untrusted)
        self.assertIsNone(self.robot._last_joint_command)


class MinkStartupTest(unittest.TestCase):
    def test_cli_prepares_mink_and_selects_rest_posture_before_connect(self):
        from almond_axol.teleop.config import VRTeleopConfig

        for explicit, fps in ((True, 15), (False, 30), (True, 60)):
            with self.subTest(explicit=explicit, fps=fps):
                defaults = VRTeleopConfig()
                cfg = run_policy.RunPolicyConfig(
                    policy_type="custom",
                    task="test",
                    actions_per_chunk=30,
                    fps=fps,
                    robot_config=AxolRobotConfig(
                        action_space="cartesian", cartesian_controller="mink"
                    ),
                    rest_pose_left=[0.1] * 7 if explicit else None,
                    rest_pose_right=[-0.2] * 7 if explicit else None,
                )
                expected_left = (
                    cfg.rest_pose_left if explicit else defaults.rest_pose_left
                )
                expected_right = (
                    cfg.rest_pose_right if explicit else defaults.rest_pose_right
                )
                result = self.run_until_connect(cfg)
                result.constructor.assert_called_once_with(
                    cfg.robot_config, mink_solve_hz=fps
                )
                self.assertEqual(result.events, ["prepare", "posture", "connect"])
                left, right = result.robot.set_cartesian_posture.call_args.args
                np.testing.assert_allclose(left, expected_left)
                np.testing.assert_allclose(right, expected_right)

    def test_solver_preparation_failure_prevents_connect(self):
        cfg = run_policy.RunPolicyConfig(
            policy_type="custom",
            task="test",
            actions_per_chunk=30,
            robot_config=AxolRobotConfig(
                action_space="cartesian", cartesian_controller="mink"
            ),
        )
        failure = ValueError("solver cannot prepare")
        result = self.run_until_connect(cfg, prepare_failure=failure)
        self.assertIs(result.failure, failure)
        result.robot.connect.assert_not_called()
        result.robot.set_cartesian_posture.assert_not_called()

    def run_until_connect(self, cfg, *, prepare_failure=None):
        from almond_axol.lerobot.camera.configuration_zed import ZedCameraConfig

        events = []
        cfg.robot_config.cameras = {"wrist": ZedCameraConfig(serial=12345)}
        stop = RuntimeError("guarded before hardware connect")
        robot = mock.Mock(config=cfg.robot_config)

        def prepare():
            events.append("prepare")
            if prepare_failure is not None:
                raise prepare_failure

        def connect():
            events.append("connect")
            raise stop

        robot.prepare_cartesian_actions.side_effect = prepare
        robot.set_cartesian_posture.side_effect = lambda *args: events.append("posture")
        robot.connect.side_effect = connect
        client = mock.Mock()
        client.start.return_value = True
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch("almond_axol.zed.stereo_serials", return_value=set())
            )
            for target, value in (
                ("almond_axol.lerobot.robot.robot_axol.AxolRobot", robot),
                ("lerobot.processor.make_default_processors", (None, None, None)),
                ("lerobot.async_inference.configs.RobotClientConfig", object()),
            ):
                patched = stack.enter_context(mock.patch(target, return_value=value))
                if target.endswith(".AxolRobot"):
                    constructor = patched
            stack.enter_context(mock.patch.object(run_policy, "IKResetController"))
            stack.enter_context(mock.patch.object(run_policy, "ActionPublisher"))
            stack.enter_context(
                mock.patch.object(
                    run_policy, "_build_axol_robot_client", return_value=client
                )
            )
            stack.enter_context(mock.patch("signal.signal"))
            with self.assertRaises(
                RuntimeError if prepare_failure is None else ValueError
            ) as raised:
                run_policy._run(cfg, stop_event=threading.Event(), control=mock.Mock())
        return SimpleNamespace(
            robot=robot,
            events=events,
            failure=raised.exception,
            constructor=constructor,
        )


if __name__ == "__main__":
    unittest.main()
