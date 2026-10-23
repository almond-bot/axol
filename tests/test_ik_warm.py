"""``axol ik.warm``: precompile the IK exactly as a session on this robot would.

The compiled programs are only reused when they match what the session asks
for, so the warm-up must resolve the same robot model and gripper and set
the IK worker's XLA flags (which the compilation cache keys on).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from almond_axol.serve.settings import SettingsStore
from almond_axol.teleop import warm


class WarmIkTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.store = SettingsStore(Path(self._dir.name) / "settings.json", strict=True)

    def _warm(self, jelly_attached: bool) -> MagicMock:
        worker = MagicMock()
        worker.get_rest_q.return_value = np.zeros(14, np.float32)
        with (
            patch("almond_axol.teleop.worker.IKWorker", return_value=worker) as cls,
            patch("almond_axol.robot.jelly._iface_exists", return_value=jelly_attached),
            patch.dict(os.environ, {"XLA_FLAGS": ""}),
        ):
            warm.warm_ik(self.store)
            self.flags = os.environ["XLA_FLAGS"]
        # The worker's startup: rest pose, then the zero -> rest trajectory.
        worker.compute_path_trajectory.assert_called_once()
        self.assertEqual(
            worker.compute_path_trajectory.call_args.args[1], ("zero", "rest")
        )
        return cls

    def test_uses_the_saved_mount_without_jelly(self) -> None:
        self.store.update(values={"kinematics.mount": "owl"})
        cls = self._warm(jelly_attached=False)
        kinematics = cls.call_args.args[1]
        self.assertEqual(kinematics.robot_model, "owl")
        # The worker's XLA flags, which the compilation cache keys on.
        self.assertIn("--xla_cpu_multi_thread_eigen=false", self.flags)

    def test_jelly_with_the_parcel_gripper(self) -> None:
        self.store.update(values={"kinematics.mount": "owl", "axol.gripper": "parcel"})
        cls = self._warm(jelly_attached=True)
        config, kinematics = cls.call_args.args
        self.assertEqual(kinematics.robot_model, "mobile")
        self.assertEqual(kinematics.gripper, "parcel")
        self.assertEqual(config.gripper, "parcel")


if __name__ == "__main__":
    unittest.main()
