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
    def test_host_allows_slow_lighthouse_identity_without_extending_ultimate(
        self,
    ) -> None:
        for source, budget, should_succeed in (
            ("lighthouse", LIGHTHOUSE_ACQUISITION_TIMEOUT_S + 15.0, True),
            ("ultimate", 35.0, False),
        ):
            with self.subTest(source=source):
                runner = OperationRunner()
                parent_conn = Mock()
                # Model a healthy child that reports ready after 60 seconds.
                parent_conn.poll.side_effect = lambda seconds: seconds >= 60.0
                parent_conn.recv.return_value = {"ok": True}
                child_conn = Mock()
                context = Mock()
                context.Pipe.return_value = (parent_conn, child_conn)
                config = TrackerConfig(left="LHR-LEFT", right="LHR-RIGHT")
                config.bindings["ultimate"] = {"left": "left", "right": "right"}
                cfg = SimpleNamespace(
                    mantis_source=source,
                    left_channel="can_mantis_l",
                    right_channel="can_mantis_r",
                    vr_server=SimpleNamespace(
                        port=8000, expected_pose_source_id="managed-token"
                    ),
                )
                with (
                    patch(
                        "almond_axol.tracker.load_tracker_config", return_value=config
                    ),
                    patch(
                        "almond_axol.serve.runner.multiprocessing.get_context",
                        return_value=context,
                    ),
                    patch("almond_axol.serve.runner.threading.Thread"),
                    patch.object(runner, "_stop_tracker_bridge") as stop,
                ):
                    if should_succeed:
                        runner._start_tracker_bridge(
                            Session("collect-data", {}), cfg, "collect-data"
                        )
                        parent_conn.recv.assert_called_once_with()
                        stop.assert_not_called()
                    else:
                        with self.assertRaisesRegex(RuntimeError, "within 35s"):
                            runner._start_tracker_bridge(
                                Session("collect-data", {}), cfg, "collect-data"
                            )
                        stop.assert_called_once()
                parent_conn.poll.assert_called_once_with(budget)
                parent_conn.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
