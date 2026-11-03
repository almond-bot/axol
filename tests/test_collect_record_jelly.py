"""collect-data's Record Jelly switch: off by default, checked before hardware opens."""

from __future__ import annotations

import unittest
from unittest import mock

from almond_axol.cli import collect_data
from almond_axol.cli.config import parse


def _parse(*extra: str) -> collect_data.CollectDataConfig:
    return parse(
        collect_data.CollectDataConfig,
        ["--repo_id", "me/test", "--task", "t", *extra],
    )


class RecordJellyConfigTest(unittest.TestCase):
    def test_off_by_default(self) -> None:
        self.assertFalse(_parse().record_jelly)

    def test_flag_turns_it_on(self) -> None:
        self.assertTrue(_parse("--record_jelly", "true").record_jelly)


class RecordJellyPreflightTest(unittest.TestCase):
    def test_refuses_without_jelly_before_opening_the_robot(self) -> None:
        cfg = _parse("--record_jelly", "true")
        robot_cls = mock.Mock(side_effect=AssertionError("robot opened"))
        with (
            mock.patch.object(collect_data, "_prepare_recording_cameras"),
            mock.patch("almond_axol.zed.stereo_serials", return_value=set()),
            mock.patch("almond_axol.robot.jelly.detect_jelly", return_value=None),
            mock.patch("almond_axol.lerobot.robot.robot_axol.AxolRobot", robot_cls),
        ):
            with self.assertRaisesRegex(ValueError, "no Jelly is attached"):
                collect_data._run_session(cfg)
        robot_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
