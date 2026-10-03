"""calibration.push: the local file, merged per joint over the cloud copy."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from almond_axol.cli import calibration as cli
from almond_axol.robot.calibration import update_joint_calibration


class PushTest(unittest.TestCase):
    def test_merge_keeps_cloud_only_joints_and_replaces_local_ones(self) -> None:
        cloud = {
            "version": 1,
            "hub_serial": "h",
            "left": {"elbow": {"com": [0, 0, 0]}},
            "right": {
                "shoulder_1": {"friction": {"fc": 9.0}},
                "wrist_1": {"com": [1, 1, 1]},
            },
        }
        local = {"left": {}, "right": {"shoulder_1": {"friction": {"fc": 0.4}}}}
        merged = cli.merge_documents(cloud, local, "h")
        self.assertEqual(merged["right"]["shoulder_1"], {"friction": {"fc": 0.4}})
        self.assertEqual(merged["right"]["wrist_1"], {"com": [1, 1, 1]})
        self.assertEqual(merged["left"]["elbow"], {"com": [0, 0, 0]})
        self.assertEqual(cli.merge_documents(None, local, "h")["hub_serial"], "h")

    def test_push_uploads_the_sanitized_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "calibration.json"
            update_joint_calibration(
                "right",
                "shoulder_1",
                friction={"fc": 0.4, "k": 100.0, "fv": 0.0, "fo": -0.3, "fl": 0.05},
                stribeck={"stribeck_gain": 0.8, "stribeck_dfs": 0.7},
                hub_serial="hub1",
                path=path,
            )
            raw = json.loads(path.read_text())
            raw["right"]["shoulder_1"]["junk"] = "not a calibration field"
            path.write_text(json.dumps(raw))
            pushed = {}
            args = argparse.Namespace(hub_serial="hub1", dry_run=False)
            with (
                mock.patch.object(cli, "CALIBRATION_PATH", path),
                mock.patch.object(cli, "fetch_calibration", lambda s: None),
                mock.patch.object(cli, "supabase_credentials", lambda: ("u", "k")),
                mock.patch.object(
                    cli, "push_calibration", lambda c, s, doc: pushed.update(doc)
                ),
                redirect_stdout(io.StringIO()) as out,
            ):
                cli.run_push(args)
            entry = pushed["right"]["shoulder_1"]
            self.assertEqual(entry["friction"]["fl"], 0.05)
            self.assertEqual(entry["stribeck_gain"], 0.8)
            self.assertNotIn("junk", entry)
            self.assertIn("friction+stribeck", out.getvalue())
            # --dry-run uploads nothing, and no key is needed for it.
            pushed.clear()
            with (
                mock.patch.object(cli, "CALIBRATION_PATH", path),
                mock.patch.object(cli, "fetch_calibration", lambda s: None),
                mock.patch.object(cli, "supabase_credentials", lambda: None),
                mock.patch.object(
                    cli, "push_calibration", lambda *a: pushed.update(x=1)
                ),
                redirect_stdout(io.StringIO()),
            ):
                cli.run_push(argparse.Namespace(hub_serial="hub1", dry_run=True))
            self.assertEqual(pushed, {})


if __name__ == "__main__":
    unittest.main()
