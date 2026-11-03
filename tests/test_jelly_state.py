"""Record Jelly: the lift/wheel dataset column and its readout."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from almond_axol.recording.jelly_state import (
    JELLY_FEATURE,
    JELLY_KEYS,
    JellyStateReader,
    jelly_config_problem,
    jelly_features,
)
from almond_axol.robot.jelly import JellyConfig


class _FakeJelly:
    def __init__(self) -> None:
        self.lift_raised: float | None = None
        self.wheel_positions: list[float] | None = None


class JellyConfigProblemTest(unittest.TestCase):
    def test_no_jelly(self) -> None:
        problem = jelly_config_problem(None)
        assert problem is not None
        self.assertIn("no Jelly is attached", problem)

    def test_wheels_and_lift_needed(self) -> None:
        self.assertIsNone(jelly_config_problem(JellyConfig()))
        problem = jelly_config_problem(JellyConfig(channel=None))
        assert problem is not None
        self.assertIn("Jelly's wheels are off", problem)
        problem = jelly_config_problem(JellyConfig(lift=False))
        assert problem is not None
        self.assertIn("Jelly's lift is off", problem)
        problem = jelly_config_problem(JellyConfig(channel=None, lift=False))
        assert problem is not None
        self.assertIn("wheels and lift are off", problem)


class JellyStateReaderTest(unittest.TestCase):
    def test_waits_for_both_parts(self) -> None:
        jelly = _FakeJelly()
        reader = JellyStateReader(jelly)
        self.assertEqual(reader.missing(), ["lift", "wheels"])
        jelly.lift_raised = 0.5
        with self.assertRaisesRegex(RuntimeError, "wheels aren't reporting"):
            reader.wait_ready(timeout_s=0.0)
        jelly.wheel_positions = [0.0, 1.0, 2.0, 3.0]
        reader.wait_ready(timeout_s=0.0)

    def test_holds_the_last_lift_reading(self) -> None:
        jelly = _FakeJelly()
        jelly.lift_raised = 0.25
        jelly.wheel_positions = [0.1, -0.2, 0.3, -0.4]
        reader = JellyStateReader(jelly)
        reader.wait_ready(timeout_s=0.0)
        # A late lift status frame reads as unknown for a moment.
        jelly.lift_raised = None
        values = reader.read()
        self.assertEqual(list(values), list(JELLY_KEYS))
        self.assertEqual(values["jelly_lift.pos"], 0.25)
        self.assertEqual(values["jelly_wheel_4.pos"], -0.4)

    def test_dataset_frame_fills_the_column(self) -> None:
        from lerobot.utils.feature_utils import build_dataset_frame

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["left_shoulder_1.pos"],
            },
            **jelly_features(),
        }
        jelly = SimpleNamespace(lift_raised=1.0, wheel_positions=[1.0, 2.0, 3.0, 4.0])
        reader = JellyStateReader(jelly)
        reader.wait_ready(timeout_s=0.0)
        frame = build_dataset_frame(
            features,
            {"left_shoulder_1.pos": 0.5, **reader.read()},
            prefix="observation",
        )
        np.testing.assert_array_equal(frame["observation.state"], [0.5])
        np.testing.assert_array_equal(frame[JELLY_FEATURE], [1.0, 1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
