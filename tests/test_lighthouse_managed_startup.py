"""The hosted bridge must allow the complete native identity acquisition."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from almond_axol.serve.manager import Session
from almond_axol.serve.runner import OperationRunner
from almond_axol.tracker.config import TrackerConfig
from almond_axol.tracker.lighthouse_validation import LIGHTHOUSE_ACQUISITION_TIMEOUT_S


class ManagedAcquisitionBudgetTest(unittest.TestCase):
    def test_host_allows_slow_lighthouse_identity(self) -> None:
        runner = OperationRunner()
        parent_conn = Mock()
        # Model a healthy child that reports ready after 60 seconds.
        parent_conn.poll.side_effect = lambda seconds: seconds >= 60.0
        parent_conn.recv.return_value = {"ok": True}
        context = Mock()
        context.Pipe.return_value = (parent_conn, Mock())
        config = TrackerConfig(left="LHR-LEFT", right="LHR-RIGHT")
        cfg = SimpleNamespace(
            mantis_source="lighthouse",
            left_channel="can_mantis_l",
            right_channel="can_mantis_r",
            vr_server=SimpleNamespace(
                port=8000, expected_pose_source_id="managed-token"
            ),
        )
        with (
            patch("almond_axol.tracker.load_tracker_config", return_value=config),
            patch(
                "almond_axol.serve.runner.multiprocessing.get_context",
                return_value=context,
            ),
            patch("almond_axol.serve.runner.threading.Thread"),
            patch.object(runner, "_stop_tracker_bridge") as stop,
        ):
            runner._start_tracker_bridge(
                Session("collect-data", {}), cfg, "collect-data"
            )
        parent_conn.recv.assert_called_once_with()
        stop.assert_not_called()
        parent_conn.poll.assert_called_once_with(
            LIGHTHOUSE_ACQUISITION_TIMEOUT_S + 15.0
        )
        parent_conn.close.assert_called_once_with()

    def test_panel_bridge_follows_the_policy_collect_data_resolved(self) -> None:
        # The saved setting may change between the run's gates and bridge
        # startup; the bridge must police what the gates assumed.
        runner = OperationRunner()
        parent_conn = Mock()
        parent_conn.poll.return_value = True
        parent_conn.recv.return_value = {"ok": True}
        context = Mock()
        context.Pipe.return_value = (parent_conn, Mock())
        config = TrackerConfig(left="LHR-LEFT", right="LHR-RIGHT")
        cfg = SimpleNamespace(
            mantis_source="lighthouse",
            left_channel="can_mantis_l",
            right_channel="can_mantis_r",
            vr_server=SimpleNamespace(port=8000, expected_pose_source_id="token"),
            _lighthouse_quality="strict",
        )
        with (
            patch("almond_axol.tracker.load_tracker_config", return_value=config),
            patch(
                "almond_axol.serve.runner.multiprocessing.get_context",
                return_value=context,
            ),
            patch("almond_axol.serve.runner.threading.Thread"),
        ):
            runner._start_tracker_bridge(
                Session("collect-data", {}), cfg, "collect-data"
            )
        self.assertEqual(config.lighthouse_quality, "strict")


if __name__ == "__main__":
    unittest.main()
