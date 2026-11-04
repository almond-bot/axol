"""collect-data's Record Jelly switch: off by default, checked before hardware opens."""

from __future__ import annotations

import ast
import inspect
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


class RecordJellyEpisodeZeroTest(unittest.TestCase):
    def test_wheels_are_zeroed_before_capture_opens(self) -> None:
        session = ast.parse(inspect.getsource(collect_data._run_session))
        start = next(
            node
            for node in ast.walk(session)
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_start_recording"
        )
        zero_line = next(
            node.lineno
            for node in ast.walk(start)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "jelly_state.zero_wheels"
        )
        capture_line = next(
            node.lineno
            for node in ast.walk(start)
            if isinstance(node, ast.Name) and node.id == "_start_capture"
        )
        self.assertLess(zero_line, capture_line)


if __name__ == "__main__":
    unittest.main()
