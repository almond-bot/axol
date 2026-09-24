"""Wrist-camera IMU shake scoring and the recorder's subprocess plumbing
(with a fake camera worker — no ZED SDK needed)."""

from __future__ import annotations

import math
import time
import unittest
from typing import Any

import numpy as np

from almond_axol.tuning.wrist_imu import WristImu, format_imu, shake_metrics

_W = 2.0 * math.pi * 5.0  # a 5 Hz shake


def _record(seconds: float = 10.0, fs: float = 400.0, amp_m: float = 0.5e-3):
    t = np.arange(0.0, seconds, 1.0 / fs)
    acc = np.zeros((len(t), 3))
    acc[:, 2] = 9.81 - amp_m * _W * _W * np.sin(_W * t)  # vertical, gravity on z
    gyro = np.zeros((len(t), 3))
    gyro[:, 0] = 2.0 * np.sin(_W * t)
    return t, acc, gyro


def fake_worker(
    serial: int,
    out_path: str,
    ready: Any,
    stop: Any,
    dump: Any,
    dumped: Any,
    errors: Any,
) -> None:
    """A camera that streams the 5 Hz shake on the perf_counter clock."""
    from almond_axol.zed.imu_worker import write_samples as _write_samples

    ts, acc, gyro = [], [], []
    ready.set()
    while not stop.is_set():
        now = time.perf_counter()
        ts.append(now)
        acc.append((0.0, 0.0, 9.81 - 0.5e-3 * _W * _W * math.sin(_W * now)))
        gyro.append((0.0, 0.0, 0.0))
        if dump.is_set():
            dump.clear()
            _write_samples(out_path, ts, acc, gyro)
            dumped.set()
        time.sleep(0.0025)
    _write_samples(out_path, ts, acc, gyro)


def broken_worker(
    serial: int,
    out_path: str,
    ready: Any,
    stop: Any,
    dump: Any,
    dumped: Any,
    errors: Any,
) -> None:
    errors.put(f"camera {serial} did not open: CAMERA NOT DETECTED")


class ShakeMetricsTest(unittest.TestCase):
    def test_a_half_millimetre_5_hz_shake_scores_1_mm_peak_to_peak(self) -> None:
        m = shake_metrics(*_record())
        self.assertAlmostEqual(m["shake_mm"], 1.0, places=2)
        self.assertAlmostEqual(m["vertical_mm"], 1.0, places=2)
        self.assertAlmostEqual(m["high_mm"], 1.0, places=2)
        self.assertLess(m["low_mm"], 0.02)
        self.assertAlmostEqual(m["peak_hz"], 5.0, delta=0.2)
        self.assertAlmostEqual(m["acc_rms"], 0.5e-3 * _W * _W / math.sqrt(2), places=2)
        self.assertAlmostEqual(m["gyro_rms"], 2.0 / math.sqrt(2), delta=0.02)

    def test_a_2_hz_sway_counts_and_lands_in_the_low_band(self) -> None:
        fs = 400.0
        t = np.arange(0.0, 12.0, 1.0 / fs)
        w = 2 * math.pi * 2.0
        acc = np.zeros((len(t), 3))
        acc[:, 2] = 9.81 - 1e-3 * w * w * np.sin(w * t)  # 1 mm amplitude
        m = shake_metrics(t, acc)
        self.assertAlmostEqual(m["vertical_mm"], 2.0, delta=0.05)
        self.assertAlmostEqual(m["low_mm"], 2.0, delta=0.05)
        self.assertLess(m["high_mm"], 0.05)
        self.assertAlmostEqual(m["peak_hz"], 2.0, delta=0.1)

    def test_slow_arm_motion_and_gravity_do_not_count(self) -> None:
        t, acc, gyro = _record()
        acc[:, 1] += 0.05 * np.sin(2 * math.pi * 0.3 * t)  # the motion itself
        self.assertAlmostEqual(shake_metrics(t, acc, gyro)["shake_mm"], 1.0, places=2)
        still = np.tile([0.0, 0.0, 9.81], (len(t), 1))
        self.assertLess(shake_metrics(t, still)["shake_mm"], 1e-6)

    def test_vertical_follows_gravity_whatever_the_camera_orientation(self) -> None:
        t, acc, _ = _record()
        # Tilt the camera 90°: gravity (and the vertical shake) now on x.
        tilted = acc[:, [2, 1, 0]]
        m = shake_metrics(t, tilted)
        self.assertAlmostEqual(m["vertical_mm"], 1.0, places=2)

    def test_too_short_is_empty(self) -> None:
        t, acc, gyro = _record(seconds=1.5)
        self.assertEqual(shake_metrics(t, acc, gyro), {})

    def test_format_names_each_side(self) -> None:
        lines = format_imu({"right": shake_metrics(*_record())})
        self.assertEqual(len(lines), 1)
        self.assertIn("wrist IMU (right)", lines[0])
        self.assertIn("1.00 mm", lines[0])


class RecorderTest(unittest.TestCase):
    def test_records_windows_on_the_run_clock_and_flushes_mid_session(self) -> None:
        imu = WristImu(
            ["right"], serial_of=lambda side: 1234, worker=f"{__name__}:fake_worker"
        )
        imu.start()
        try:
            self.assertEqual(imu.sides, ["right"])
            t0 = time.perf_counter()
            time.sleep(5.0)
            t1 = time.perf_counter()
            imu.flush()
            metrics, series = imu.run_blocks(t0, t1, origin=t0 - 1.0)
            self.assertAlmostEqual(metrics["right"]["shake_mm"], 1.0, delta=0.1)
            # t relative to the given origin: the window starts ~1 s in.
            self.assertAlmostEqual(float(series["imu_right_t"][0]), 1.0, delta=0.05)
            self.assertEqual(series["imu_right_acc"].shape[1], 3)
        finally:
            imu.stop()
        imu.stop()  # idempotent

    def test_no_camera_or_a_camera_that_fails_leaves_no_data(self) -> None:
        with WristImu(["left"], serial_of=lambda side: None) as imu:
            self.assertEqual(imu.sides, [])
        self.assertEqual(imu.run_blocks(0.0, 1e9), ({}, {}))
        with WristImu(
            ["left"], serial_of=lambda side: 1, worker=f"{__name__}:broken_worker"
        ) as imu:
            self.assertEqual(imu.sides, [])
        disabled = WristImu(["left"], enabled=False, serial_of=lambda side: 1)
        disabled.start()
        self.assertEqual(disabled.sides, [])
        disabled.stop()


if __name__ == "__main__":
    unittest.main()
