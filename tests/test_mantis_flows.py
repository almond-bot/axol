from __future__ import annotations

import json
import logging
import math
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from almond_axol.cli import collect_data, teleop
from almond_axol.cli.config import TeleopCmdConfig
from almond_axol.cli.collect_data import _validate_mantis_calibration
from almond_axol.lerobot.robot.config_mantis import MantisRobotConfig
from almond_axol.lerobot.teleop.config_vr import AxolVRTeleopConfig
from almond_axol.mantis.calibration import (
    LEGACY_TRACKER_KEY,
    ULTIMATE_POSE_CONVENTION_FIELD,
    VIVE_TRACKER_CAD_ORIGINS_MM,
    candidate_transform_for,
    design_transform_for,
    load_tcp_transforms,
    validate_tcp_transform,
)
from almond_axol.mantis.relative import quat_xyzw_to_matrix
from almond_axol.teleop.config import VRTeleopConfig, apply_mantis_teleop_profile
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.tracker.base import TrackerPose
from almond_axol.tracker.bridge import StopEventControls, TrackerBridge
from almond_axol.tracker.config import (
    TrackerConfig,
    load_tracker_config,
    save_tracker_config,
    select_tracker_backend,
)
from almond_axol.tracker.trigger import decode_trigger_payload, parse_trigger_frame


class _Source:
    def __init__(self) -> None:
        self.tracking = True

    def poses(self) -> dict[str, TrackerPose]:
        now = time.perf_counter()
        return {
            side: TrackerPose(
                pos=np.array([offset, 1.0, -0.4]),
                quat=np.array([0.0, 0.0, 0.0, 1.0]),
                t=now,
                tracking=self.tracking,
            )
            for side, offset in (("left", 0.2), ("right", -0.2))
        }


class _Trigger:
    def __init__(self, grip: float = 1.0) -> None:
        self.value = grip

    def grip(self) -> float:
        return self.value

    def is_stale(self) -> bool:
        return False


class _SkewedSource:
    def __init__(self, skew_s: float) -> None:
        self.skew_s = skew_s
        self.calls = 0

    def poses(self) -> dict[str, TrackerPose]:
        self.calls += 1
        now = time.perf_counter()
        return {
            "left": TrackerPose(
                pos=np.array([0.2, 1.0, -0.4]),
                quat=np.array([0.0, 0.0, 0.0, 1.0]),
                t=now - self.skew_s,
            ),
            "right": TrackerPose(
                pos=np.array([-0.2, 1.0, -0.4]),
                quat=np.array([0.0, 0.0, 0.0, 1.0]),
                t=now,
            ),
        }


class MantisFlowTest(unittest.TestCase):
    def test_direct_mantis_teleop_disables_inherited_powered_cart(self) -> None:
        cfg = TeleopCmdConfig(mantis=True)
        cfg.cart.enabled = True
        with (
            mock.patch(
                "almond_axol.teleop.config.apply_mantis_teleop_profile"
            ) as apply_teleop,
            mock.patch(
                "almond_axol.kinematics.config.apply_mantis_kinematics_profile"
            ) as apply_kinematics,
        ):
            teleop._prepare_mantis_teleop(cfg)

        self.assertFalse(cfg.cart.enabled)
        apply_teleop.assert_called_once_with(
            cfg.teleop, tracker_source=cfg.mantis_source
        )
        apply_kinematics.assert_called_once_with(cfg.kinematics)

    def test_mantis_collection_disables_inherited_powered_cart(self) -> None:
        cfg = collect_data.CollectDataConfig(repo_id="test/repo", task="test")
        self.assertIsInstance(cfg.teleop_config, AxolVRTeleopConfig)
        assert isinstance(cfg.teleop_config, AxolVRTeleopConfig)
        cfg.teleop_config.cart.enabled = True
        with (
            mock.patch(
                "almond_axol.teleop.config.apply_mantis_teleop_profile"
            ) as apply_teleop,
            mock.patch(
                "almond_axol.kinematics.config.apply_mantis_kinematics_profile"
            ) as apply_kinematics,
        ):
            collect_data._apply_mantis_profile(cfg)

        self.assertFalse(cfg.teleop_config.cart.enabled)
        apply_teleop.assert_called_once()
        apply_kinematics.assert_called_once()

    def test_mantis_collection_restores_required_gripper_schema(self) -> None:
        cfg = collect_data.CollectDataConfig(repo_id="test/repo", task="test")
        cfg.robot_config.axol_config.has_gripper = False
        self.assertIsInstance(cfg.teleop_config, AxolVRTeleopConfig)
        assert isinstance(cfg.teleop_config, AxolVRTeleopConfig)
        cfg.teleop_config.has_gripper = False

        with (
            mock.patch("almond_axol.teleop.config.apply_mantis_teleop_profile"),
            mock.patch("almond_axol.kinematics.config.apply_mantis_kinematics_profile"),
        ):
            collect_data._apply_mantis_profile(cfg)

        self.assertIsInstance(cfg.robot_config, MantisRobotConfig)
        self.assertTrue(cfg.robot_config.axol_config.has_gripper)
        self.assertTrue(cfg.teleop_config.has_gripper)

    def test_explicit_mantis_config_rejects_gripperless_hardware_schema(self) -> None:
        cfg = collect_data._default_robot_config()
        cfg.axol_config.has_gripper = False
        with self.assertRaisesRegex(ValueError, "Mantis always has two"):
            MantisRobotConfig(axol_config=cfg.axol_config)

    def test_direct_collection_rejects_camera_config_before_tracker_bridge(
        self,
    ) -> None:
        cfg = SimpleNamespace(
            mantis=True,
            mantis_source="lighthouse",
            log_level="INFO",
        )
        with (
            mock.patch.object(collect_data, "parse", return_value=cfg),
            mock.patch.object(collect_data, "_prepare_mantis_collection"),
            mock.patch.object(
                collect_data,
                "_prepare_recording_cameras",
                side_effect=ValueError("no recording camera"),
            ),
            mock.patch("almond_axol.cli.mantis_bridge.managed_mantis_bridge") as bridge,
            self.assertRaisesRegex(ValueError, "no recording camera"),
        ):
            collect_data.main([])

        bridge.assert_not_called()

    def test_failed_collection_setup_disconnects_robot_and_preserves_error(
        self,
    ) -> None:
        dataset_dir = tempfile.TemporaryDirectory()
        self.addCleanup(dataset_dir.cleanup)
        robot = mock.Mock()
        robot.action_features = {}
        robot.observation_features = {}
        robot.get_joint_observation.return_value = {}
        robot.positions = (np.zeros(8), np.zeros(8))
        robot.cameras = {}
        robot.name = "test-robot"
        robot.disconnect.side_effect = RuntimeError("cleanup sentinel")

        teleop = mock.Mock()
        teleop.cart = None
        teleop.connect.side_effect = ValueError("setup sentinel")
        cfg = SimpleNamespace(
            mantis=False,
            repo_id="test/repo",
            task="test",
            fps=60,
            teleop_hz=120,
            vcodec="auto",
            root=dataset_dir.name,
            push_to_hub=False,
            rerun_ip=None,
            rerun_port=9876,
            dataset_resolution="SVGA",
            log_level="INFO",
            robot_config=SimpleNamespace(observation_cameras=lambda: {}),
            teleop_config=SimpleNamespace(),
        )
        original_affinity = {2, 3, 4, 5}

        with (
            mock.patch.object(collect_data, "_prepare_recording_cameras"),
            mock.patch.object(collect_data.affinity, "pin_realtime"),
            mock.patch.object(
                collect_data.os,
                "sched_getaffinity",
                return_value=original_affinity,
            ),
            mock.patch.object(collect_data.os, "sched_setaffinity") as restore,
            mock.patch.object(collect_data, "_start_video_relay", return_value=None),
            mock.patch(
                "almond_axol.lerobot.robot.robot_axol.AxolRobot",
                return_value=robot,
            ),
            mock.patch(
                "almond_axol.lerobot.teleop.teleop_vr.AxolVRTeleop",
                return_value=teleop,
            ),
            mock.patch("almond_axol.utils.network.local_ip", return_value="127.0.0.1"),
            self.assertRaisesRegex(ValueError, "setup sentinel") as raised,
        ):
            collect_data._run(cfg)

        robot.connect.assert_called_once_with()
        teleop.disconnect.assert_called_once_with()
        robot.disconnect.assert_called_once_with()
        restore.assert_called_once_with(0, original_affinity)
        self.assertTrue(
            getattr(raised.exception, "_axol_hardware_cleanup_uncertain", False)
        )

    def test_trigger_decoder_accepts_both_firmware_lengths(self) -> None:
        core = struct.pack("<fH", 0.25, 1234)
        self.assertEqual(decode_trigger_payload(core), (0.25, 1234))
        self.assertEqual(decode_trigger_payload(core + b"\xa5"), (0.25, 1234))
        self.assertIsNone(decode_trigger_payload(core + b"\x00\x00"))
        self.assertIsNone(decode_trigger_payload(struct.pack("<fH", math.nan, 0)))
        self.assertAlmostEqual(parse_trigger_frame(core + b"\x00").position, 0.25)

    def test_managed_engage_requires_release_then_both_squeeze(self) -> None:
        source = _Source()
        left = _Trigger()
        right = _Trigger()
        bridge = TrackerBridge(
            source,
            left="left",
            right="right",
            controls=StopEventControls(threading.Event()),
            left_trigger=left,
            right_trigger=right,
            auto_engage=True,
            confirm_auto_engage=True,
        )

        released = bridge.compose_frame()
        self.assertFalse(released["l_lock"])
        self.assertTrue(released["l_tracked"])
        left.value = 0.0
        self.assertFalse(bridge.compose_frame()["l_lock"])
        right.value = 0.0
        confirmed = bridge.compose_frame()
        self.assertTrue(confirmed["l_lock"] and confirmed["r_lock"])
        for recognizer in bridge._gesture.values():
            self.assertTrue(recognizer._pressed)
            self.assertEqual(recognizer._presses, 0)

    def test_managed_bridge_uses_operation_pose_source_token(self) -> None:
        bridge = TrackerBridge(
            _Source(),
            left="left",
            right="right",
            controls=StopEventControls(threading.Event()),
            pose_source_id="managed-ultimate-run",
        )

        self.assertEqual(
            bridge.compose_frame()["pose_source_id"], "managed-ultimate-run"
        )

    def test_bridge_never_marks_malformed_custom_source_pose_live(self) -> None:
        source = _Source()
        bridge = TrackerBridge(
            source,
            left="left",
            right="right",
            controls=StopEventControls(threading.Event()),
        )
        self.assertTrue(bridge.compose_frame()["l_tracked"])

        original_poses = source.poses

        def malformed() -> dict[str, TrackerPose]:
            poses = original_poses()
            poses["left"].pos[0] = math.nan
            return poses

        source.poses = malformed  # type: ignore[method-assign]
        frame = bridge.compose_frame()
        self.assertFalse(frame["l_tracked"])
        self.assertTrue(frame["r_tracked"])
        self.assertTrue(math.isfinite(frame["l_ee"]["position"]["x"]))

    def test_confirmation_must_restart_after_tracker_dropout(self) -> None:
        source = _Source()
        left = _Trigger()
        right = _Trigger()
        bridge = TrackerBridge(
            source,
            left="left",
            right="right",
            controls=StopEventControls(threading.Event()),
            left_trigger=left,
            right_trigger=right,
            auto_engage=True,
            confirm_auto_engage=True,
        )

        bridge.compose_frame()  # both released: arm confirmation
        source.tracking = False
        left.value = right.value = 0.0
        self.assertFalse(bridge.compose_frame()["l_lock"])
        source.tracking = True
        self.assertFalse(bridge.compose_frame()["l_lock"])
        left.value = right.value = 1.0
        self.assertFalse(bridge.compose_frame()["l_lock"])
        left.value = right.value = 0.0
        self.assertTrue(bridge.compose_frame()["l_lock"])

    def test_managed_confirmation_requires_two_distinct_inputs(self) -> None:
        source = _Source()
        with self.assertRaisesRegex(ValueError, "two distinct"):
            TrackerBridge(source, left="left", right="left")
        with self.assertRaisesRegex(ValueError, "left and right trigger"):
            TrackerBridge(
                source,
                left="left",
                right="right",
                left_trigger=_Trigger(),
                auto_engage=True,
                confirm_auto_engage=True,
            )

    def test_absolute_core_requires_release_after_tracking_loss(self) -> None:
        broadcasts: list[bool] = []
        core = VRTeleopCore(
            VRTeleopConfig(absolute_mode=True),
            logging.getLogger(__name__),
            broadcasts.append,
        )

        def frame(*, tracked: bool, left: bool, right: bool) -> SimpleNamespace:
            return SimpleNamespace(
                l_tracked=tracked,
                r_tracked=tracked,
                l_lock=left,
                r_lock=right,
                l_grip=0.5,
                r_grip=0.5,
                lock_release_id=None,
            )

        engaged = frame(tracked=True, left=True, right=True)
        released = frame(tracked=True, left=False, right=False)
        self.assertTrue(core._accept_tracking_frame(engaged))
        core.update_engage(engaged)
        self.assertTrue(core.teleop_enabled)

        # A total WebXR stream gap has no frame carrying tracked=False; the
        # stale-stream path still funnels through the same forced-disengage
        # gate and must reject a held-over squeeze on recovery.
        core._disengage_all()
        self.assertFalse(core._accept_tracking_frame(engaged))
        self.assertTrue(core._accept_tracking_frame(released))
        core.update_engage(released)
        self.assertTrue(core._accept_tracking_frame(engaged))
        core.update_engage(engaged)
        self.assertTrue(core.teleop_enabled)

        self.assertFalse(
            core._accept_tracking_frame(frame(tracked=False, left=True, right=True))
        )
        self.assertFalse(core.teleop_enabled)
        self.assertFalse(core._accept_tracking_frame(engaged))
        self.assertFalse(
            core._accept_tracking_frame(frame(tracked=True, left=False, right=True))
        )

        self.assertTrue(core._accept_tracking_frame(released))
        core.update_engage(released)
        self.assertFalse(core.teleop_enabled)
        self.assertTrue(core._accept_tracking_frame(engaged))
        core.update_engage(engaged)
        self.assertTrue(core.teleop_enabled)

    def test_quest_calibration_rejects_wrong_profile_or_pose_space(self) -> None:
        core = VRTeleopCore(
            VRTeleopConfig(
                absolute_mode=True,
                quest_controller_profile="oculus-touch-v3",
                quest_pose_space="grip",
            ),
            logging.getLogger(__name__),
            lambda _enabled: None,
        )

        def frame(profile: str, pose_space: str) -> SimpleNamespace:
            return SimpleNamespace(
                l_tracked=True,
                r_tracked=True,
                l_pose_profile=profile,
                r_pose_profile=profile,
                l_pose_space=pose_space,
                r_pose_space=pose_space,
            )

        self.assertEqual(
            core._validated_tracking_flags(frame("oculus-touch-v3", "grip")),
            {"left": True, "right": True},
        )
        self.assertEqual(
            core._validated_tracking_flags(frame("oculus-touch-v3", "target-ray")),
            {"left": False, "right": False},
        )
        self.assertEqual(
            core._validated_tracking_flags(frame("other-controller", "grip")),
            {"left": False, "right": False},
        )

    def test_lost_tracking_is_carried_per_side(self) -> None:
        source = _Source()
        bridge = TrackerBridge(
            source,
            left="left",
            right="right",
            controls=StopEventControls(threading.Event()),
            auto_engage=True,
        )
        self.assertTrue(bridge.compose_frame()["l_tracked"])
        source.tracking = False
        lost = bridge.compose_frame()
        self.assertFalse(lost["l_tracked"])
        self.assertFalse(lost["r_tracked"])

    def test_tracker_frame_uses_one_snapshot_and_rejects_side_skew(self) -> None:
        source = _SkewedSource(0.08)
        bridge = TrackerBridge(
            source,
            left="left",
            right="right",
            controls=StopEventControls(threading.Event()),
            auto_engage=True,
        )
        frame = bridge.compose_frame()
        self.assertEqual(source.calls, 1)
        self.assertFalse(frame["l_tracked"])
        self.assertTrue(frame["r_tracked"])

    def test_tracker_frame_timestamp_is_oldest_valid_side(self) -> None:
        source = _SkewedSource(0.02)
        bridge = TrackerBridge(
            source,
            left="left",
            right="right",
            controls=StopEventControls(threading.Event()),
        )
        frame = bridge.compose_frame()
        self.assertTrue(frame["l_tracked"] and frame["r_tracked"])
        left_capture_ms = bridge._held["left"].t * 1000.0
        self.assertAlmostEqual(frame["t"], left_capture_ms, places=3)

    def test_source_binding_restore_is_backend_specific(self) -> None:
        config = TrackerConfig(
            backend="survive",
            bindings={
                "survive": {"left": "T20", "right": "T21"},
                "ultimate": {"left": "a:b:c:d:e:f", "right": "1:2:3:4:5:6"},
            },
        )
        select_tracker_backend(config, "survive")
        self.assertEqual((config.left, config.right), ("T20", "T21"))
        select_tracker_backend(config, "ultimate")
        self.assertEqual((config.left, config.right), ("a:b:c:d:e:f", "1:2:3:4:5:6"))

    def test_invalid_config_and_transform_files_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            config_path.write_text("[]")
            self.assertEqual(load_tracker_config(config_path), TrackerConfig())

            transform_path = root / "tcp.json"
            transform_path.write_text(
                json.dumps(
                    {
                        "left": {
                            "quest": {
                                "pos": [0, 0, 0],
                                "quat": [0, 0, 0, 0],
                            }
                        },
                        "right": {
                            "quest": {
                                "pos": [0, 0, 0],
                                "quat": [0, 0, 0, 2],
                            }
                        },
                    }
                )
            )
            loaded = load_tcp_transforms(transform_path)
            self.assertNotIn("left", loaded)
            self.assertEqual(loaded["right"]["quest"][3:], [0.0, 0.0, 0.0, 1.0])

    def test_ultimate_transform_requires_matching_pose_convention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "tracker.json"
            transform_path = root / "tcp.json"
            config = TrackerConfig(
                backend="ultimate",
                left="a:b:c:d:e:f",
                right="1:2:3:4:5:6",
                ultimate_quat_order="wxyz",
                ultimate_up_axis="z",
            )
            save_tracker_config(config, config_path)
            entry = {
                "pos": [0.0, 0.0465, -0.092],
                "quat": [0.7071068, 0.0, 0.0, 0.7071068],
            }
            transform_path.write_text(
                json.dumps(
                    {
                        "left": {
                            "survive:T20": dict(entry),
                            "ultimate:a:b:c:d:e:f": dict(entry),
                        },
                        "right": {
                            "ultimate:1:2:3:4:5:6": {
                                **entry,
                                ULTIMATE_POSE_CONVENTION_FIELD: {
                                    "quat_order": "wxyz",
                                    "up_axis": "z",
                                },
                            }
                        },
                    }
                )
            )

            loaded = load_tcp_transforms(
                transform_path,
                tracker_config_path=config_path,
            )
            self.assertIn("survive:T20", loaded["left"])
            self.assertNotIn("ultimate:a:b:c:d:e:f", loaded["left"])
            self.assertIn("ultimate:1:2:3:4:5:6", loaded["right"])

            config.ultimate_up_axis = "y"
            save_tracker_config(config, config_path)
            loaded = load_tcp_transforms(
                transform_path,
                tracker_config_path=config_path,
            )
            self.assertNotIn("ultimate:1:2:3:4:5:6", loaded.get("right", {}))

            config.ultimate_up_axis = "z"
            config.ultimate_quat_order = "xyzw"
            save_tracker_config(config, config_path)
            loaded = load_tcp_transforms(
                transform_path,
                tracker_config_path=config_path,
            )
            self.assertNotIn("ultimate:1:2:3:4:5:6", loaded.get("right", {}))

            # Malformed convention metadata is stale, not an exception that
            # could turn a readiness inspection into an availability failure.
            document = json.loads(transform_path.read_text())
            document["left"]["ultimate:a:b:c:d:e:f"][ULTIMATE_POSE_CONVENTION_FIELD] = {
                "quat_order": [],
                "up_axis": "z",
            }
            transform_path.write_text(json.dumps(document))
            self.assertNotIn(
                "ultimate:a:b:c:d:e:f",
                load_tcp_transforms(
                    transform_path,
                    tracker_config_path=config_path,
                ).get("left", {}),
            )

    def test_tcp_transform_validator_rejects_unsafe_values(self) -> None:
        unsafe = {
            "wrong length": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "non-finite position": [math.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "non-finite quaternion": [0.0, 0.0, 0.0, 0.0, math.inf, 0.0, 1.0],
            "overflowing numeric": [10**1000, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "zero quaternion": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "millimetres entered as metres": [47, 0, 35, 0, 0, 0, 1],
        }
        for name, transform in unsafe.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_tcp_transform(transform)

        self.assertEqual(
            validate_tcp_transform([0, 0, 0, 0, 0, 0, 2]),
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        )

    def test_explicit_tcp_transforms_are_validated_before_teleop(self) -> None:
        identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        unsafe = (
            identity[:-1],
            [math.nan, *identity[1:]],
            [*identity[:3], 0.0, 0.0, 0.0, 0.0],
        )
        for transform in unsafe:
            with self.subTest(transform=transform):
                config = VRTeleopConfig(
                    tcp_transform_left=transform,
                    tcp_transform_right=identity,
                )
                with self.assertRaisesRegex(ValueError, "Mantis left.*invalid"):
                    apply_mantis_teleop_profile(config, tracker_source="lighthouse")

        non_unit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]
        config = VRTeleopConfig(
            tcp_transform_left=non_unit,
            tcp_transform_right=non_unit,
        )
        apply_mantis_teleop_profile(config, tracker_source="lighthouse")
        self.assertEqual(config.tcp_transform_left, identity)
        self.assertEqual(config.tcp_transform_right, identity)

    def test_collection_preflight_rejects_unsafe_explicit_transform(self) -> None:
        identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        def command(transform: list[float], *, allow: bool = False) -> SimpleNamespace:
            return SimpleNamespace(
                mantis_allow_uncalibrated=allow,
                mantis_source="lighthouse",
                teleop_config=SimpleNamespace(
                    vr_teleop_config=VRTeleopConfig(
                        tcp_transform_left=transform,
                        tcp_transform_right=identity,
                    )
                ),
            )

        for transform in (
            identity[:-1],
            [math.inf, *identity[1:]],
            [*identity[:3], 0.0, 0.0, 0.0, 0.0],
        ):
            with self.subTest(transform=transform):
                with self.assertRaisesRegex(ValueError, "Mantis left.*invalid"):
                    _validate_mantis_calibration(command(transform))

        # The uncalibrated bring-up flag allows absence, never malformed input.
        with self.assertRaisesRegex(ValueError, "Mantis left.*invalid"):
            _validate_mantis_calibration(command(identity[:-1], allow=True))

    def test_collection_requires_active_tracker_keyed_transform(self) -> None:
        identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        config = SimpleNamespace(
            mantis_allow_uncalibrated=False,
            mantis_source="lighthouse",
            teleop_config=SimpleNamespace(
                vr_teleop_config=VRTeleopConfig(
                    tcp_transform_left=identity,
                    tcp_transform_right=identity,
                )
            ),
        )
        keys = {"left": "survive:T20", "right": "survive:T21"}

        def active_key(side: str, **_kwargs: object) -> tuple[str, str]:
            return keys[side], "test binding"

        with (
            mock.patch(
                "almond_axol.mantis.calibration.tracker_key_for_side",
                side_effect=active_key,
            ),
            mock.patch(
                "almond_axol.mantis.calibration.load_tcp_transforms",
                return_value={},
            ),
            self.assertRaisesRegex(ValueError, "left \\(survive:T20\\)"),
        ):
            _validate_mantis_calibration(config)

        measured = {
            "left": {"survive:T20": identity},
            "right": {"survive:T21": identity},
        }
        with (
            mock.patch(
                "almond_axol.mantis.calibration.tracker_key_for_side",
                side_effect=active_key,
            ),
            mock.patch(
                "almond_axol.mantis.calibration.load_tcp_transforms",
                return_value=measured,
            ),
        ):
            _validate_mantis_calibration(config)

        config.teleop_config.vr_teleop_config.tcp_transform_left = [
            0.01,
            *identity[1:],
        ]
        with (
            mock.patch(
                "almond_axol.mantis.calibration.tracker_key_for_side",
                side_effect=active_key,
            ),
            mock.patch(
                "almond_axol.mantis.calibration.load_tcp_transforms",
                return_value=measured,
            ),
            self.assertRaisesRegex(ValueError, "unproven transform override"),
        ):
            _validate_mantis_calibration(config)

    def test_pending_cad_transform_does_not_authorize_collection(self) -> None:
        self.assertEqual(VIVE_TRACKER_CAD_ORIGINS_MM["survive"], (47.0, 0.0, 35.0))
        self.assertEqual(VIVE_TRACKER_CAD_ORIGINS_MM["ultimate"], (47.0, 0.0, 46.0))
        survive = candidate_transform_for("left", "survive:T20")
        self.assertIsNotNone(survive)
        assert survive is not None
        self.assertIsNone(design_transform_for("left", "survive:T20"))
        ultimate = [0.0, 0.0465, -0.092, 0.7071068, 0.0, 0.0, 0.7071068]
        self.assertEqual(
            candidate_transform_for("left", "ultimate:aa:bb:cc:dd:ee:ff"),
            ultimate,
        )
        self.assertEqual(
            candidate_transform_for("right", "ultimate:11:22:33:44:55:66"),
            ultimate,
        )
        # Origins are expressed in the shared gripper/CAD frame G, while the
        # stored translation is the TCP origin in tracker frame T. Therefore
        # delta_p_TG = -R_TG @ delta_O; Rx(+90 deg) turns the +11 mm CAD-z
        # origin shift into +11 mm of tracker-y TCP translation.
        origin_delta_m = (
            np.asarray(VIVE_TRACKER_CAD_ORIGINS_MM["ultimate"])
            - np.asarray(VIVE_TRACKER_CAD_ORIGINS_MM["survive"])
        ) / 1000.0
        rotation_tg = quat_xyzw_to_matrix(np.asarray(survive[3:]))
        np.testing.assert_allclose(
            np.asarray(ultimate[:3]) - np.asarray(survive[:3]),
            -(rotation_tg @ origin_delta_m),
            atol=1e-9,
        )
        self.assertIsNone(design_transform_for("left", "ultimate:aa:bb:cc:dd:ee:ff"))
        config = VRTeleopConfig(tracker_key="survive:T20")
        with mock.patch(
            "almond_axol.mantis.calibration.load_tcp_transforms", return_value={}
        ):
            apply_mantis_teleop_profile(config, tracker_source="lighthouse")
        self.assertIsNone(config.tcp_transform_left)
        self.assertIsNone(config.tcp_transform_right)

        ultimate_config = VRTeleopConfig(tracker_key="ultimate:aa:bb:cc:dd:ee:ff")
        with mock.patch(
            "almond_axol.mantis.calibration.load_tcp_transforms", return_value={}
        ):
            apply_mantis_teleop_profile(ultimate_config, tracker_source="ultimate")
        self.assertIsNone(ultimate_config.tcp_transform_left)
        self.assertIsNone(ultimate_config.tcp_transform_right)

        collection = SimpleNamespace(
            mantis_allow_uncalibrated=False,
            mantis_source="ultimate",
            teleop_config=SimpleNamespace(vr_teleop_config=ultimate_config),
        )
        keys = {
            "left": "ultimate:aa:bb:cc:dd:ee:ff",
            "right": "ultimate:11:22:33:44:55:66",
        }
        with (
            mock.patch(
                "almond_axol.mantis.calibration.tracker_key_for_side",
                side_effect=lambda side, **_kwargs: (keys[side], "test binding"),
            ),
            mock.patch(
                "almond_axol.mantis.calibration.load_tcp_transforms",
                return_value={},
            ),
            self.assertRaisesRegex(ValueError, "no verified.*Ultimate|no verified"),
        ):
            _validate_mantis_calibration(collection)

    def test_ultimate_convention_change_invalidates_production_transform(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "tracker.json"
            transform_path = root / "tcp.json"
            tracker_config = TrackerConfig(
                backend="ultimate",
                left="a:b:c:d:e:f",
                right="1:2:3:4:5:6",
                ultimate_quat_order="wxyz",
                ultimate_up_axis="z",
            )
            save_tracker_config(tracker_config, config_path)
            transform = {
                "pos": [0.0, 0.0465, -0.092],
                "quat": [0.7071068, 0.0, 0.0, 0.7071068],
                ULTIMATE_POSE_CONVENTION_FIELD: {
                    "quat_order": "wxyz",
                    "up_axis": "z",
                },
            }
            transform_path.write_text(
                json.dumps(
                    {
                        "left": {"ultimate:a:b:c:d:e:f": transform},
                        "right": {"ultimate:1:2:3:4:5:6": transform},
                    }
                )
            )

            vrt = VRTeleopConfig()
            collection = SimpleNamespace(
                mantis_allow_uncalibrated=False,
                mantis_source="ultimate",
                teleop_config=SimpleNamespace(vr_teleop_config=vrt),
            )
            with (
                mock.patch(
                    "almond_axol.mantis.calibration.MANTIS_TCP_TRANSFORM_FILE",
                    transform_path,
                ),
                mock.patch(
                    "almond_axol.tracker.config.TRACKER_CONFIG_FILE",
                    config_path,
                ),
            ):
                apply_mantis_teleop_profile(vrt, tracker_source="ultimate")
                _validate_mantis_calibration(collection)

                tracker_config.ultimate_up_axis = "y"
                save_tracker_config(tracker_config, config_path)
                with self.assertRaisesRegex(ValueError, "no verified"):
                    _validate_mantis_calibration(collection)

    def test_active_source_never_uses_unknown_legacy_transform(self) -> None:
        identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        saved = {
            "left": {LEGACY_TRACKER_KEY: identity},
            "right": {LEGACY_TRACKER_KEY: identity},
        }
        config = VRTeleopConfig(tracker_key="ultimate:aa")
        with mock.patch(
            "almond_axol.mantis.calibration.load_tcp_transforms", return_value=saved
        ):
            apply_mantis_teleop_profile(config, tracker_source="ultimate")
        self.assertIsNone(config.tcp_transform_left)
        self.assertIsNone(config.tcp_transform_right)

    def test_keyed_measured_transform_is_applied(self) -> None:
        identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        saved = {
            "left": {"ultimate:aa": identity},
            "right": {"ultimate:aa": identity},
        }
        config = VRTeleopConfig(tracker_key="ultimate:aa")
        with mock.patch(
            "almond_axol.mantis.calibration.load_tcp_transforms", return_value=saved
        ):
            apply_mantis_teleop_profile(config, tracker_source="ultimate")
        self.assertEqual(config.tcp_transform_left, identity)
        self.assertEqual(config.tcp_transform_right, identity)

    def test_profile_scoped_quest_transform_is_selected_and_validated(self) -> None:
        identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        key = "quest:oculus-touch-v3:grip"
        saved = {"left": {key: identity}, "right": {key: identity}}
        config = VRTeleopConfig()
        with mock.patch(
            "almond_axol.mantis.calibration.load_tcp_transforms", return_value=saved
        ):
            apply_mantis_teleop_profile(config, tracker_source="quest")
        self.assertEqual(config.tracker_key, key)
        self.assertEqual(config.quest_controller_profile, "oculus-touch-v3")
        self.assertEqual(config.quest_pose_space, "grip")
        self.assertEqual(config.tcp_transform_left, identity)
        self.assertEqual(config.tcp_transform_right, identity)
        self.assertTrue(config.urdf_viewer_world_aligned)

    def test_external_tracker_world_hides_unregistered_quest_overlay(self) -> None:
        config = VRTeleopConfig()
        with mock.patch(
            "almond_axol.mantis.calibration.load_tcp_transforms", return_value={}
        ):
            apply_mantis_teleop_profile(config, tracker_source="lighthouse")
        self.assertFalse(config.urdf_viewer_world_aligned)

        explicitly_registered = VRTeleopConfig(urdf_viewer_world_aligned=True)
        with mock.patch(
            "almond_axol.mantis.calibration.load_tcp_transforms", return_value={}
        ):
            apply_mantis_teleop_profile(
                explicitly_registered, tracker_source="ultimate"
            )
        self.assertTrue(explicitly_registered.urdf_viewer_world_aligned)

    def test_quest_collection_gate_requires_scoped_grip_datum(self) -> None:
        identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        def command(config: VRTeleopConfig) -> SimpleNamespace:
            return SimpleNamespace(
                mantis_allow_uncalibrated=False,
                mantis_source="quest",
                teleop_config=SimpleNamespace(vr_teleop_config=config),
            )

        for config in (
            VRTeleopConfig(
                tcp_transform_left=identity,
                tcp_transform_right=identity,
                tracker_key="quest",
            ),
            VRTeleopConfig(
                tcp_transform_left=identity,
                tcp_transform_right=identity,
                tracker_key="quest:oculus-touch-v3:target-ray",
                quest_controller_profile="oculus-touch-v3",
                quest_pose_space="target-ray",
            ),
            VRTeleopConfig(
                tcp_transform_left=identity,
                tcp_transform_right=identity,
            ),
        ):
            with self.subTest(key=config.tracker_key):
                with self.assertRaisesRegex(ValueError, "profile-scoped"):
                    _validate_mantis_calibration(command(config))

        good = VRTeleopConfig(
            tcp_transform_left=identity,
            tcp_transform_right=identity,
            tracker_key="quest:oculus-touch-v3:grip",
            quest_controller_profile="oculus-touch-v3",
            quest_pose_space="grip",
        )
        key = "quest:oculus-touch-v3:grip"
        saved = {"left": {key: identity}, "right": {key: identity}}
        with mock.patch(
            "almond_axol.mantis.calibration.load_tcp_transforms",
            return_value=saved,
        ):
            _validate_mantis_calibration(command(good))

    def test_quest_scoped_key_rejects_conflicting_expected_metadata(self) -> None:
        config = VRTeleopConfig(
            tracker_key="quest:oculus-touch-v3:grip",
            quest_controller_profile="wrong-profile",
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            apply_mantis_teleop_profile(config, tracker_source="quest")


if __name__ == "__main__":
    unittest.main()
