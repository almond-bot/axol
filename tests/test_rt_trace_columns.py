"""The Rust trace CSV grows ``cogging_ff`` / ``tf_pct``; the compactor takes
the new layout and still the one before it."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from almond_axol.teleop import recorder


def _write(path: Path, columns: tuple[str, ...], rows: int) -> None:
    lines = [",".join(columns)]
    for k in range(rows):
        lines.append(",".join(str(k + i * 0.5) for i in range(len(columns))))
    path.write_text("\n".join(lines) + "\n")


class CompactTest(unittest.TestCase):
    def test_new_and_legacy_layouts_compact_side_by_side(self) -> None:
        self.assertEqual(recorder._RT_TRACE_COLUMNS[-2:], ("cogging_ff", "tf_pct"))
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "run")
            _write(Path(f"{prefix}_rt-left.csv"), recorder._RT_TRACE_COLUMNS, 3)
            _write(Path(f"{prefix}_rt-right.csv"), recorder._RT_TRACE_COLUMNS_V1, 2)
            out = recorder.compact_rt_trace(prefix)
            assert out is not None
            with np.load(out) as z:
                self.assertEqual(len(z["t"]), 5)
                cog = z["cogging_ff"]
                self.assertTrue(np.all(np.isfinite(cog[:3])))
                self.assertTrue(np.all(np.isnan(cog[3:])))
                self.assertEqual(list(z["side"]), [0, 0, 0, 1, 1])

    def test_an_unknown_layout_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            prefix = str(Path(d) / "run")
            _write(Path(f"{prefix}_rt-left.csv"), ("tick", "bogus"), 1)
            with self.assertRaisesRegex(ValueError, "schema"):
                recorder.compact_rt_trace(prefix)


if __name__ == "__main__":
    unittest.main()
