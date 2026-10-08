"""The robot-tuning playbook's tools (.claude/skills/axol-robot-tuning): the
bus-recording verdict, the creep motions, the run scorer and the A/B queue."""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


can_trace = _load("can_trace")
creep_motions = _load("creep_motions")
tuning_queue = _load("tuning_queue")

# ------------------------------------------------------------ bus frames


def _f2u(x: float, lo: float, hi: float, bits: int) -> int:
    x = min(max(x, lo), hi)
    return int(round((x - lo) / (hi - lo) * ((1 << bits) - 1)))


def _mit_cmd(p: float, kp: float = 250.0, kd: float = 3.5) -> bytes:
    pm, tm = can_trace._P_V44, can_trace._T_V44[1]
    pi = _f2u(p, -pm, pm, 16)
    vi = _f2u(0.0, -45.0, 45.0, 12)
    kpi = _f2u(kp, 0, 500.0, 12)
    kdi = _f2u(kd, 0, 5.0, 12)
    ti = _f2u(0.0, -tm, tm, 12)
    return bytes(
        [
            pi >> 8,
            pi & 0xFF,
            vi >> 4,
            ((vi & 0xF) << 4) | (kpi >> 8),
            kpi & 0xFF,
            kdi >> 4,
            ((kdi & 0xF) << 4) | (ti >> 8),
            ti & 0xFF,
        ]
    )


def _mit_fb(p: float) -> bytes:
    pm, tm = can_trace._P_V44, can_trace._T_V44[1]
    pi = _f2u(p, -pm, pm, 16)
    vi = _f2u(0.0, -45.0, 45.0, 12)
    ti = _f2u(0.0, -tm, tm, 12)
    return bytes(
        [1, pi >> 8, pi & 0xFF, vi >> 4, ((vi & 0xF) << 4) | (ti >> 8), ti & 0xFF, 0, 0]
    )


def _log(
    cmd_fn, meas_fn, seconds: float = 10.0, extra: list[str] | None = None
) -> Path:
    """A recording of left shoulder_1 at 240 Hz: command then reply."""
    lines = ["# time_s,iface,arb_id,data_hex\n"]
    for k in range(int(seconds * 240)):
        t = 100.0 + k / 240.0
        c = cmd_fn(k / 240.0)
        lines.append(f"{t:.6f},can_alm_axol_l,401,{_mit_cmd(c).hex()}\n")
        lines.append(
            f"{t + 0.0005:.6f},can_alm_axol_l,501,{_mit_fb(meas_fn(k / 240.0, c)).hex()}\n"
        )
    lines += extra or []
    path = Path(tempfile.mkdtemp()) / "trace.log"
    path.write_text("".join(lines))
    return path


def _verdict(path: Path) -> dict:
    rows = can_trace.parse_log(path, {1, 2})
    return can_trace.joint_summary(rows[("left", 1)])


DEG = math.radians(1.0)


def slow(t: float) -> float:  # deliberate operator motion, 0.3 Hz
    return 0.3 + 10 * DEG * math.sin(2 * math.pi * 0.3 * t)


class SummaryVerdictTest(unittest.TestCase):
    def test_quiet_tracking(self) -> None:
        s = _verdict(_log(slow, lambda t, c: c + 0.01 * DEG * math.sin(40 * t)))
        self.assertEqual(s["verdict"], "quiet")

    def test_joint_ringing_around_a_smooth_command(self) -> None:
        s = _verdict(
            _log(slow, lambda t, c: c + 0.6 * DEG * math.sin(2 * math.pi * 2.5 * t))
        )
        self.assertIn("control ringing", s["verdict"])
        self.assertTrue(s["verdict"].startswith("STRONG"))
        self.assertAlmostEqual(s["err_peak_hz"], 2.5, delta=0.3)

    def test_command_jitter_the_arm_follows(self) -> None:
        def jittery(t: float) -> float:
            return slow(t) + 0.4 * DEG * math.sin(2 * math.pi * 3.0 * t)

        s = _verdict(_log(jittery, lambda t, c: c))
        self.assertIn("command jitter", s["verdict"])
        self.assertAlmostEqual(s["cmd_peak_hz"], 3.0, delta=0.3)

    def test_a_single_jump_is_a_transient_not_jitter(self) -> None:
        # Ctrl-C mid-sweep: the target jumps 5.7° once, the joint rings ~2 Hz and decays.
        def cmd(t: float) -> float:
            return 0.3 + (-5.7 * DEG if t >= 5.0 else 0.0)

        def meas(t: float, c: float) -> float:
            if t < 5.0:
                return c
            tau = t - 5.0
            return c + 5.7 * DEG * math.exp(-2.5 * tau) * math.cos(
                2 * math.pi * 2.0 * tau
            )

        s = _verdict(_log(cmd, meas))
        self.assertIn("transient", s["verdict"])
        self.assertTrue(s["command_jumps_s"])
        self.assertNotIn("jitter", s["verdict"])

    def test_firmware_loop_frames_are_called_out(self) -> None:
        a4 = [
            f"200.0,can_alm_axol_l,141,{bytes([0xA4, 0, 14, 0, 0x28, 0x23, 0, 0]).hex()}\n"
        ]
        s = _verdict(_log(slow, lambda t, c: c, extra=a4))
        self.assertIn("0xA4", s["verdict"])


# --------------------------------------------------------- creep motions


class CreepMotionsTest(unittest.TestCase):
    def test_every_creep_is_safe_and_moves_one_joint(self) -> None:
        from almond_axol.motor import Joint
        from almond_axol.robot.axol import arm_limits

        pose = creep_motions.start_pose()
        for joint in creep_motions.KEYS:
            for arm in ("right", "left"):
                q, meta = creep_motions.build(joint, arm, pose)
                col = list(creep_motions.KEYS).index(joint) + (
                    0 if arm == "left" else 7
                )
                moving = np.flatnonzero(np.ptp(q, axis=0) > 1e-9)
                self.assertEqual(moving.tolist(), [col], (joint, arm))
                lo, hi = arm_limits(Joint(joint), arm == "left")
                self.assertTrue(lo < q[:, col].min() and q[:, col].max() < hi)
                v = np.degrees(np.abs(np.gradient(q[:, col], 1 / creep_motions.RATE)))
                self.assertLessEqual(v.max(), 6.01)
                cruise3 = np.abs(v - 3.0) < 0.05
                self.assertGreater(cruise3.sum() / creep_motions.RATE, 2.0)
                if joint == "wrist_2":  # outboard half only
                    sign = 1.0 if arm == "left" else -1.0
                    self.assertGreaterEqual((q[:, col] * sign).min(), -1e-4)

    def test_left_creep_is_the_right_one_mirrored(self) -> None:
        pose = creep_motions.start_pose()
        right, _ = creep_motions.build("shoulder_3", "right", pose)
        left, _ = creep_motions.build("shoulder_3", "left", pose)
        np.testing.assert_allclose(left[:, 2], -right[:, 9], atol=1e-4)


# ------------------------------------------------------------- the scorer


class ScoreTest(unittest.TestCase):
    def test_creep_ripple_is_measured_on_the_moving_joint(self) -> None:
        from almond_axol.tuning.creep_score import score_series

        q, _ = creep_motions.build("shoulder_3", "right")
        t = np.arange(len(q)) / creep_motions.RATE
        amp = math.radians(0.030)  # 30 mdeg ripple at 5 Hz
        actual = q.copy()
        actual[:, 9] += amp * np.sin(2 * np.pi * 5.0 * t)
        s = score_series(t, q, actual, "right")
        self.assertEqual(s["moving"], "shoulder_3")
        self.assertAlmostEqual(s["rip3_rms_mdeg"], 30 / math.sqrt(2), delta=3.0)
        self.assertLess(s["band_mdeg"]["elbow"], 1.0)


# ---------------------------------------------------------- the A/B queue


def _rec(
    variant: str, rnd: int, rip: float, baseline: bool = False, trips=(), imu=None
) -> dict:
    score = {"rip3_rms_mdeg": rip}
    if imu is not None:
        score["imu"] = {"low_mm": imu}
    return {
        "item": {
            "label": f"m {variant} r{rnd}",
            "variant": variant,
            "round": rnd,
            "baseline": baseline,
        },
        "trips": list(trips),
        "scores": [score, score],
    }


class QueueTest(unittest.TestCase):
    def test_plan_interleaves_rounds_baseline_first(self) -> None:
        v = [
            tuning_queue.parse_variant("base"),
            tuning_queue.parse_variant("g=--gain left.shoulder_3.stribeck_gain=0.4"),
            tuning_queue.parse_variant("old@calib=/tmp/old.json"),
        ]
        items = tuning_queue.plan_items("left", "s3_creep", v, rounds=2)
        self.assertEqual(
            [i["label"] for i in items[:3]],
            ["s3_creep base r1", "s3_creep g r1", "s3_creep old r1"],
        )
        self.assertEqual(len(items), 6)
        self.assertTrue(items[0]["baseline"] and not items[1]["baseline"])
        self.assertIn("--guard-dev-deg", items[1]["args"])
        self.assertEqual(
            items[1]["args"][-2:], ["--gain", "left.shoulder_3.stribeck_gain=0.4"]
        )
        self.assertEqual(items[2]["calib"], "/tmp/old.json")
        with self.assertRaises(ValueError):
            tuning_queue.plan_items(
                "left", "m", [tuning_queue.parse_variant("x=--no-guard")], 1
            )

    def test_verdicts(self) -> None:
        recs = []
        for r, (b, good, mixed) in enumerate(
            [(30, 20, 25), (31, 21, 35), (29, 19, 28)], 1
        ):
            recs += [
                _rec("base", r, b, baseline=True, imu=0.5),
                _rec("good", r, good, imu=0.4),
                _rec("mixed", r, mixed, imu=0.5),
                _rec("trippy", r, 10, trips=["GUARD TRIP"] if r == 2 else []),
            ]
        s = tuning_queue.summarize(recs)
        self.assertEqual(s["baseline"], "base")
        v = tuning_queue.verdict(s)
        self.assertTrue(v["good"].startswith("better"), v["good"])
        self.assertTrue(v["mixed"].startswith("inconclusive"), v["mixed"])
        self.assertTrue(v["trippy"].startswith("rejected"), v["trippy"])
        self.assertEqual(
            s["variants"]["good"]["metrics"]["rip3_rms_mdeg"]["rounds_better"], "3/3"
        )

    def test_halt_rules(self) -> None:
        base = {"label": "b", "baseline": True}
        var = {"label": "v", "baseline": False}
        self.assertIsNone(tuning_queue.halt_reason(var, 0, "", [], ["id"], 0))
        self.assertIn(
            "baseline",
            tuning_queue.halt_reason(base, 3, "", ["GUARD TRIP x"], ["id"], 1),
        )
        self.assertIsNone(
            tuning_queue.halt_reason(var, 3, "", ["GUARD TRIP x"], ["id"], 1)
        )
        self.assertIn(
            "two guard trips",
            tuning_queue.halt_reason(var, 3, "", ["GUARD TRIP"], ["id"], 2),
        )
        self.assertIn("holding", tuning_queue.halt_reason(var, 4, "", [], ["id"], 0))
        self.assertIn(
            "no saved run", tuning_queue.halt_reason(var, 1, "boom", [], [], 0)
        )


class SkillDiscoveryTest(unittest.TestCase):
    def test_claude_and_codex_both_find_the_playbook(self) -> None:
        root = SCRIPTS.parent
        claude = root / ".claude/skills/axol-robot-tuning"
        codex = root / ".agents/skills/axol-robot-tuning"
        self.assertEqual(codex.resolve(), claude.resolve())
        text = (codex / "SKILL.md").read_text()
        head = text.split("---")[1]
        self.assertIn("name: axol-robot-tuning", head)
        self.assertIn("description:", head)
        for ref in ("reference/known-issues.md", "reference/numbers.md"):
            self.assertIn(ref, text)
            self.assertTrue((codex / ref).is_file(), ref)


if __name__ == "__main__":
    unittest.main()
