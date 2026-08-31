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

from almond_axol.cli import collect_data
from almond_axol.cli.collect_data import _validate_mantis_calibration
from almond_axol.mantis.calibration import (
    LEGACY_TRACKER_KEY,
    candidate_transform_for,
    design_transform_for,
    load_tcp_transforms,
    validate_tcp_transform,
)
from almond_axol.teleop.config import VRTeleopConfig, apply_mantis_teleop_profile
from almond_axol.teleop.core import VRTeleopCore
from almond_axol.tracker.base import TrackerPose
from almond_axol.tracker.bridge import StopEventControls, TrackerBridge
from almond_axol.tracker.config import (
    TrackerConfig,
    load_tracker_config,
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

    def test_tcp_transform_validator_rejects_unsafe_values(self) -> None:
        unsafe = {
            "wrong length": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "non-finite position": [math.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "non-finite quaternion": [0.0, 0.0, 0.0, 0.0, math.inf, 0.0, 1.0],
            "overflowing numeric": [10**1000, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "zero quaternion": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
        self.assertIsNotNone(candidate_transform_for("left", "survive:T20"))
        self.assertIsNone(design_transform_for("left", "survive:T20"))
        self.assertIsNone(candidate_transform_for("left", "ultimate:aa:bb:cc:dd:ee:ff"))
        config = VRTeleopConfig(tracker_key="survive:T20")
        with mock.patch(
            "almond_axol.mantis.calibration.load_tcp_transforms", return_value={}
        ):
            apply_mantis_teleop_profile(config, tracker_source="lighthouse")
        self.assertIsNone(config.tcp_transform_left)
        self.assertIsNone(config.tcp_transform_right)

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
