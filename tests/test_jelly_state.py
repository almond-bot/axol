"""Record Jelly: the lift/wheel dataset column and its readout."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from almond_axol.recording.jelly_state import (
    JELLY_FEATURE,
    JELLY_KEYS,
    MAX_EXTRAPOLATION_S,
    JellyStateReader,
    jelly_config_problem,
    jelly_features,
    zero_episode_wheels,
)
from almond_axol.robot.jelly import JellyConfig, WheelSample


def _sample(positions, velocities=(0.0, 0.0, 0.0, 0.0), at=0.0) -> WheelSample:
    return WheelSample(
        positions=tuple(float(p) for p in positions),  # type: ignore[arg-type]
        velocities=tuple(float(v) for v in velocities),  # type: ignore[arg-type]
        at=at,
    )


class _FakeJelly:
    def __init__(self) -> None:
        self.lift_raised: float | None = None
        self.wheel_sample: WheelSample | None = None


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
        jelly.wheel_sample = _sample([0.0, 1.0, 2.0, 3.0])
        reader.wait_ready(timeout_s=0.0)

    def test_holds_the_last_lift_reading(self) -> None:
        jelly = _FakeJelly()
        jelly.lift_raised = 0.25
        jelly.wheel_sample = _sample([0.1, -0.2, 0.3, -0.4])
        reader = JellyStateReader(jelly)
        reader.wait_ready(timeout_s=0.0)
        # A late lift status frame reads as unknown for a moment.
        jelly.lift_raised = None
        values = reader.read()
        self.assertEqual(list(values), list(JELLY_KEYS))
        self.assertEqual(values["jelly_lift.pos"], 0.25)
        self.assertEqual(values["jelly_wheel_4.pos"], -0.4)

    def test_each_episode_starts_the_wheels_at_zero(self) -> None:
        jelly = _FakeJelly()
        jelly.lift_raised = 0.75
        jelly.wheel_sample = _sample([10.0, -20.0, 30.0, -40.0])
        reader = JellyStateReader(jelly)
        reader.wait_ready(timeout_s=0.0)
        reader.zero_wheels()
        values = reader.read()
        self.assertEqual([values[k] for k in JELLY_KEYS[1:]], [0.0] * 4)
        # The lift stays absolute.
        self.assertEqual(values["jelly_lift.pos"], 0.75)

        jelly.wheel_sample = _sample([11.0, -21.5, 30.0, -38.0])
        values = reader.read()
        self.assertEqual([values[k] for k in JELLY_KEYS[1:]], [1.0, -1.5, 0.0, 2.0])

        # Driving between takes doesn't carry into the next episode.
        jelly.wheel_sample = _sample([50.0, -60.0, 70.0, -80.0])
        reader.zero_wheels()
        values = reader.read()
        self.assertEqual([values[k] for k in JELLY_KEYS[1:]], [0.0] * 4)

    def test_carries_the_sample_to_the_read_time(self) -> None:
        jelly = _FakeJelly()
        jelly.lift_raised = 0.5
        jelly.wheel_sample = _sample(
            [1.0, 2.0, 3.0, 4.0], velocities=[10.0, -10.0, 0.0, 1.0], at=100.0
        )
        reader = JellyStateReader(jelly)
        reader.wait_ready(timeout_s=0.0)
        values = reader.read(now=100.02)
        for got, want in zip(
            [values[k] for k in JELLY_KEYS[1:]], [1.2, 1.8, 3.0, 4.02], strict=True
        ):
            self.assertAlmostEqual(got, want)
        # A sample too old to trust is reported as it was, not run forward.
        values = reader.read(now=100.0 + MAX_EXTRAPOLATION_S + 0.05)
        self.assertEqual([values[k] for k in JELLY_KEYS[1:]], [1.0, 2.0, 3.0, 4.0])
        # Zeroing uses the same estimate, so the episode starts at zero.
        reader.zero_wheels(now=100.01)
        values = reader.read(now=100.01)
        for k in JELLY_KEYS[1:]:
            self.assertAlmostEqual(values[k], 0.0)

    def test_steady_driving_gives_even_steps_at_30_fps(self) -> None:
        # Wheels at 12 rad/s report every 20 ms; each report reaches Python
        # 0-15 ms late. Sampling the latest report at 30 fps alone gives
        # steps of 0, 1 or 2 reports; carried forward, every step is 12/30.
        speed, period = 12.0, 0.02
        rng = np.random.default_rng(0)
        delays = rng.uniform(0.0, 0.015, size=400)
        jelly = _FakeJelly()
        jelly.lift_raised = 1.0
        reader = JellyStateReader(jelly)
        out = []
        for frame in range(150):
            now = 1.0 + frame / 30.0
            k = int(now / period)
            while k > 0 and k * period + delays[k] > now:
                k -= 1  # that report hasn't arrived yet
            at = k * period
            jelly.wheel_sample = _sample(
                [speed * at] * 4, velocities=[speed] * 4, at=at
            )
            out.append(reader.read(now=now)[JELLY_KEYS[1]])
        steps = np.diff(out)
        np.testing.assert_allclose(steps, speed / 30.0, atol=1e-9)

    def test_saved_episode_starts_the_wheels_at_zero(self) -> None:
        rows = [
            np.array([0.5, 1.2, -1.25, 1.23, -1.25], dtype=np.float32),
            np.array([0.5, 1.4, -1.45, 1.43, -1.45], dtype=np.float32),
            np.array([0.6, 2.2, -2.25, 2.23, -2.25], dtype=np.float32),
        ]
        buffer = {"size": 3, JELLY_FEATURE: rows, "observation.state": [None] * 3}
        zero_episode_wheels(buffer)
        np.testing.assert_allclose(rows[0], [0.5, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(rows[1], [0.5, 0.2, -0.2, 0.2, -0.2], atol=1e-6)
        # The lift stays absolute.
        np.testing.assert_allclose(rows[2], [0.6, 1.0, -1.0, 1.0, -1.0], atol=1e-6)
        self.assertEqual(rows[2].dtype, np.float32)
        # A session without the column, or an empty episode, is left alone.
        zero_episode_wheels({"size": 0})
        zero_episode_wheels({"size": 0, JELLY_FEATURE: []})
        zero_episode_wheels(None)

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
        jelly = SimpleNamespace(
            lift_raised=1.0, wheel_sample=_sample([1.0, 2.0, 3.0, 4.0])
        )
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
