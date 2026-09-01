from __future__ import annotations

import unittest
from unittest.mock import patch

from almond_axol.cli.collect_data import _resolve_control_trace_prefix


class CollectDataControlTraceTest(unittest.TestCase):
    def test_default_none_keeps_flight_recorder_disabled(self) -> None:
        with patch("almond_axol.cli.collect_data.resolve_prefix") as resolve:
            self.assertIsNone(_resolve_control_trace_prefix(None))

        resolve.assert_not_called()

    def test_explicit_prefix_is_resolved(self) -> None:
        with patch(
            "almond_axol.cli.collect_data.resolve_prefix",
            return_value="/tmp/resolved-trace",
        ) as resolve:
            self.assertEqual(
                _resolve_control_trace_prefix("operator-trace"),
                "/tmp/resolved-trace",
            )

        resolve.assert_called_once_with("operator-trace")


if __name__ == "__main__":
    unittest.main()
