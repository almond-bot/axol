"""run-policy captures through the same camera backend as collect-data.

The gst and SDK paths deliver visibly different frames; a policy run through
the other path sees images its demonstrations never contained.
"""

from __future__ import annotations

import unittest

from almond_axol.cli import collect_data, run_policy


class CameraBackendDefaultTest(unittest.TestCase):
    def test_run_policy_matches_collect_data(self) -> None:
        self.assertEqual(
            run_policy._default_robot_config().video_backend,
            collect_data._default_robot_config().video_backend,
        )


if __name__ == "__main__":
    unittest.main()
