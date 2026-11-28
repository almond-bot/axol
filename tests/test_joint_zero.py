"""The joint zeroing record: every zeroing is kept, and episodes and camera
calibrations name the zeroing each arm is on."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import unittest
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

from almond_axol.robot import joint_zero as jz

ALL = {j: {"motor_id": i + 1} for i, j in enumerate(jz.ARM_JOINT_NAMES)}
DAY = datetime(2026, 12, 1, 18, 0, tzinfo=UTC)


class RecordTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "joint_zero.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _record(self, side: str, joints=ALL, serial: str = "HUB1", **kw) -> dict:
        return jz.record(
            side,
            joints,
            jz.METHOD_GUIDED,
            robot_serial=serial,
            path=self.path,
            now=kw.pop("now", DAY),
            **kw,
        )

    def test_ids_count_up_per_robot_and_day(self) -> None:
        self.assertEqual(self._record("left")["zero_id"], "HUB1-20261201-zero-01")
        self.assertEqual(self._record("right")["zero_id"], "HUB1-20261201-zero-02")
        self.assertEqual(
            self._record("left", serial="HUB2")["zero_id"], "HUB2-20261201-zero-01"
        )
        later = DAY.replace(day=2)
        self.assertEqual(
            self._record("left", now=later)["zero_id"], "HUB1-20261202-zero-01"
        )
        events = jz.load(self.path)
        self.assertEqual(len(events), 4)
        self.assertEqual(events[0]["zeroed_at"], "2026-12-01T18:00:00Z")
        self.assertEqual(events[0]["joints"]["shoulder_1"], {"motor_id": 1})

    def test_state_follows_the_latest_zeroing_of_each_joint(self) -> None:
        first = self._record("left")
        self._record("left", serial="OTHER")  # another robot's arm: ignored
        state = jz.state("HUB1", self.path)
        self.assertEqual(state.status, "partly recorded")  # right arm unrecorded
        self.assertEqual(state.ids(), {"left": first["zero_id"], "right": ""})
        self.assertEqual(state.unrecorded("right"), list(jz.ARM_JOINT_NAMES))

        second = self._record("left", {"wrist_2": {}})
        self._record("right")
        state = jz.state("HUB1", self.path)
        self.assertEqual(state.status, "recorded")
        # One joint re-zeroed moves the whole arm to the new zeroing's ID.
        self.assertEqual(state.arm_id("left"), second["zero_id"])
        self.assertEqual(
            state.joints["left"]["shoulder_1"]["zero_id"], first["zero_id"]
        )

    def test_no_file_and_a_bad_file(self) -> None:
        state = jz.state("HUB1", self.path)
        self.assertEqual(state.status, "no record")
        self.assertIn("no joint zeroing record", state.problem)
        self.path.write_text("not json")
        self.assertIn("not valid JSON", jz.state("HUB1", self.path).problem)
        # A new zeroing keeps the unreadable file beside the new record.
        self._record("left")
        kept = list(self.path.parent.glob("joint_zero.json.unreadable-*"))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].read_text(), "not json")
        self.assertEqual(len(jz.load(self.path)), 1)

    def test_calibration_match(self) -> None:
        left, right = self._record("left"), self._record("right")
        ids = {"left": left["zero_id"], "right": right["zero_id"]}
        state = jz.state("HUB1", self.path)
        self.assertEqual(jz.calibration_match(state, None), "")
        self.assertEqual(jz.calibration_match(state, ids), "same")
        self.assertEqual(jz.calibration_match(state, {}), "unknown")
        again = self._record("right", {"elbow": {}})
        match = jz.calibration_match(jz.state("HUB1", self.path), ids)
        self.assertEqual(
            match,
            f"changed: right arm re-zeroed on 2026-12-01T18:00:00Z ({again['zero_id']})",
        )

    def test_episode_metadata(self) -> None:
        left = self._record("left")
        meta = jz.episode_metadata(jz.state("HUB1", self.path), None)
        self.assertEqual(meta["status"], "partly recorded")
        self.assertEqual(meta["left.zero_id"], left["zero_id"])
        self.assertEqual(meta["left.last_zeroed_at"], "2026-12-01T18:00:00Z")
        self.assertEqual(meta["right.zero_id"], "")
        self.assertEqual(meta["file"], "joint_zero.json")
        self.assertEqual(len(meta["file_sha256"]), 64)
        self.assertEqual(meta["camera_calibration"], "")
        joints = json.loads(meta["joints"])
        self.assertEqual(joints["left_elbow"]["zero_id"], left["zero_id"])
        self.assertNotIn("right_elbow", joints)

    def test_record_quietly_never_raises(self) -> None:
        with mock.patch.object(jz, "record", side_effect=OSError("disk full")):
            self.assertIsNone(jz.record_quietly("left", ALL, jz.METHOD_SDK))
        self.assertIsNone(jz.record_quietly("left", {}, jz.METHOD_SDK))


class _FakeMotor:
    def __init__(self, positions: list[float], motor_id: int = 4):
        self._positions = list(positions)
        self.motor_id = motor_id
        self._driver = object()

    async def get_position(self) -> float:
        return self._positions.pop(0)

    async def set_zero_position(self) -> None:
        pass

    async def set_direction_flag(self) -> bool:
        return False


class ZeroingCommandTest(unittest.IsolatedAsyncioTestCase):
    """``axol motor.set-zero-pos`` records each zero it saves."""

    async def test_guided_records_the_zeroed_joints(self) -> None:
        from almond_axol.cli.motor import set_zero_pos

        motor = _FakeMotor([0.2, 0.2 + math.radians(140)])

        @asynccontextmanager
        async def fake_bus(channel: str):  # type: ignore[no-untyped-def]
            yield object()

        args = argparse.Namespace(
            l=True, r=False, channel=None, joints="elbow", web_prompts=False
        )
        with (
            mock.patch.object(set_zero_pos, "CanBus", fake_bus),
            mock.patch.object(set_zero_pos, "Motor", lambda bus, joint: motor),
            mock.patch.object(set_zero_pos, "_prompt", lambda *a: True),
            mock.patch.object(jz, "_hub_serial", return_value="HUB1"),
        ):
            await set_zero_pos._run_guided(args)
        (event,) = jz.load()
        self.assertEqual(event["side"], "left")
        self.assertEqual(event["method"], jz.METHOD_GUIDED)
        self.assertEqual(event["robot_serial"], "HUB1")
        self.assertEqual(
            event["joints"],
            {
                "elbow": {
                    "motor_id": 4,
                    "end_stop_deg": 150.0,
                    "reading_before_rad": round(0.2 + math.radians(140), 6),
                }
            },
        )

    async def test_single_motor_on_an_arm_is_recorded(self) -> None:
        from almond_axol.cli.motor import set_zero_pos

        motor = _FakeMotor([0.031, 0.0])

        @asynccontextmanager
        async def fake_bus(channel: str):  # type: ignore[no-untyped-def]
            yield object()

        for args, side in (
            (
                argparse.Namespace(l=False, r=True, channel=None, id=6, type=None),
                "right",
            ),
            # A motor on another bus (a test stand) is not an arm's zero.
            (
                argparse.Namespace(l=False, r=False, channel="can0", id=6, type=None),
                None,
            ),
        ):
            motor._positions = [0.031, 0.0]
            with (
                mock.patch.object(set_zero_pos, "CanBus", fake_bus),
                mock.patch.object(set_zero_pos, "make_driver", lambda *a, **k: motor),
                mock.patch.object(jz, "_hub_serial", return_value="HUB1"),
            ):
                await set_zero_pos._run_single(args)
            events = jz.load()
            if side is None:
                self.assertEqual(len(events), 1)
                continue
            self.assertEqual(events[-1]["side"], side)
            self.assertEqual(events[-1]["method"], jz.METHOD_SINGLE)
            self.assertEqual(
                events[-1]["joints"],
                {
                    "wrist_2": {
                        "motor_id": 6,
                        "end_stop_deg": None,
                        "reading_before_rad": 0.031,
                    }
                },
            )


class SdkZeroingTest(unittest.IsolatedAsyncioTestCase):
    async def test_arm_set_zero_position_is_recorded(self) -> None:
        from almond_axol.constants import Joint
        from almond_axol.robot.axol import AxolArm

        arm = object.__new__(AxolArm)
        arm._is_left = False
        arm.motors = {Joint.ELBOW: _FakeMotor([0.5], motor_id=4)}
        arm._joint_offsets = [0.0] * len(Joint)
        arm._unresolved_offsets = set()
        arm._unverified_zeros = set()
        with mock.patch.object(jz, "_hub_serial", return_value="HUB1"):
            await arm.set_zero_position([Joint.ELBOW])
        (event,) = jz.load()
        self.assertEqual(event["side"], "right")
        self.assertEqual(event["method"], jz.METHOD_SDK)
        self.assertEqual(
            event["joints"]["elbow"],
            {"motor_id": 4, "end_stop_deg": None, "reading_before_rad": 0.5},
        )


class CalibrationFileTest(unittest.TestCase):
    def test_joint_zero_is_parsed(self) -> None:
        import numpy as np

        from almond_axol.camera_calibration.file import (
            document,
            parse,
            transform_entry,
        )

        doc = document(
            robot_serial="HUB1",
            calibration_id="c1",
            calibrated_at="2026-12-01T18:00:00Z",
            status="pass",
            transforms=[transform_entry("root", "cam", np.eye(4))],
            joint_zero={"left": "HUB1-20261201-zero-01", "right": 3, "x": "y"},
        )
        parsed = parse(json.dumps(doc).encode(), Path("camera_calibration.json"))
        self.assertEqual(parsed.joint_zero, {"left": "HUB1-20261201-zero-01"})
        del doc["joint_zero"]
        parsed = parse(json.dumps(doc).encode(), Path("camera_calibration.json"))
        self.assertEqual(parsed.joint_zero, {})


class ProcedureTest(unittest.TestCase):
    def test_the_calibration_names_recording_a_zeroing(self) -> None:
        from types import SimpleNamespace

        from almond_axol.camera_calibration.procedure import joint_zero_ids

        rec = SimpleNamespace(
            metadata={"joint_zero": {"left.zero_id": "Z1", "right.zero_id": ""}}
        )
        self.assertEqual(joint_zero_ids(rec), {"left": "Z1", "right": ""})
        self.assertEqual(
            joint_zero_ids(SimpleNamespace(metadata={})), {"left": "", "right": ""}
        )


if __name__ == "__main__":
    unittest.main()
