"""Subprocess that records one ZED X One camera's IMU.

Run by :class:`almond_axol.tuning.wrist_imu.WristImu` as ``python -m
almond_axol.zed.imu_worker --serial N --out PATH``, in a process of its own so
a wedged ZED SDK can never stall — or crash — the tuning process streaming
motor commands. Deliberately light to import (numpy only; ``pyzed`` inside
:func:`record`): a spawn that re-imported the ``axol`` CLI took seconds of the
camera-open budget.

Protocol, one line each way on stdin/stdout: the worker prints ``ready`` once
the camera is open, ``dumped`` after each ``dump`` request (the samples so far
written to ``PATH``), ``error <text>`` for anything that goes wrong; ``stop``
(or stdin closing) ends it after a final write.

``PATH`` is an ``.npz`` of ``t`` (``time.perf_counter`` seconds —
``CLOCK_MONOTONIC``, shared across processes, the clock the tuning logs use),
``acc`` (m/s², gravity included) and ``gyro`` (deg/s).

What the SDK (5.4.1, ``pyzed/sl.pyi``) specifies and the right wrist camera (a
ZED X One GS, serial 308393615) showed on 2026-09-23:

- ``CameraOne.get_sensors_data(data, TIME_REFERENCE.CURRENT)`` returns the
  latest sample received (the SDK's advice: poll at 800 Hz in a thread to get
  them all). It needs no ``grab()`` — an open, never-grabbed camera delivers.
  800 Hz is not enough here: a 1 ms poll (~940 calls/s) phase-locked with the
  SDK's update and caught only 128 of the ~200 samples a second; 0.5 ms lost
  2%; 0.2–0.3 ms caught all of them for ~5% of a core, hence ``_POLL_S``.
- The IMU is specified at 400 Hz (``sensors_configuration``) but delivers
  ~200 Hz (``IMUData.effective_rate`` 202): 5 ms apart, the newest ~5 ms old.
  Nyquist 100 Hz — the 3–15 Hz shake band is well inside it.
- ``get_linear_acceleration()`` is m/s² and ``get_angular_velocity()`` deg/s,
  both calibrated (bias, scale, misalignment); ranges ±78.5 m/s², ±1000 deg/s.
- ``IMUData.timestamp`` is the acquisition time in UNIX nanoseconds (the
  wall clock), mapped here onto ``perf_counter``.
- ``get_sensors_data_batch`` (every sample of the last grabbed frame) is
  lossless too, but only behind a ``grab()`` loop — the 1080p/30 fps image
  pipeline running for nothing — so it is not used.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import threading
import time
from typing import Any

import numpy as np

# Poll period (s): 0.25 ms catches every ~200 Hz sample (1 ms caught 64%).
_POLL_S = 0.00025
# An open camera that has delivered no IMU sample for this long is reported.
_SILENT_S = 1.0


def write_samples(
    path: str,
    ts: list[float],
    acc: list[tuple[float, float, float]],
    gyro: list[tuple[float, float, float]],
) -> None:
    """Write the samples atomically (tmp + rename)."""
    tmp = path + ".tmp.npz"
    np.savez(
        tmp,
        t=np.asarray(ts, dtype=np.float64),
        acc=np.asarray(acc, dtype=np.float32).reshape(-1, 3),
        gyro=np.asarray(gyro, dtype=np.float32).reshape(-1, 3),
    )
    os.replace(tmp, path)


def record(
    serial: int,
    out_path: str,
    ready: Any,
    stop: Any,
    dump: Any,
    dumped: Any,
    errors: Any,
) -> None:
    """Open camera ``serial`` and record its IMU until ``stop`` is set.

    ``ready`` / ``dumped`` are set, ``stop`` / ``dump`` polled (Event-like),
    ``errors.put(text)`` reports a failure.
    """
    try:
        import pyzed.sl as sl
    except ImportError as exc:
        errors.put(f"pyzed not importable ({exc}) — run `axol zed.install`")
        return
    zed = sl.CameraOne()
    init = sl.InitParametersOne()
    init.set_from_serial_number(serial)
    if hasattr(init, "sdk_verbose"):
        init.sdk_verbose = 0
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        errors.put(f"camera {serial} did not open: {err}")
        return
    ts: list[float] = []
    acc: list[tuple[float, float, float]] = []
    gyro: list[tuple[float, float, float]] = []
    try:
        sensors = sl.SensorsData()
        # IMU stamps are UNIX nanoseconds; map them to CLOCK_MONOTONIC.
        wall_minus_perf = time.time() - time.perf_counter()
        last = None
        last_new = time.perf_counter()
        silent_reported = False
        ready.set()
        while not stop.is_set():
            if (
                zed.get_sensors_data(sensors, sl.TIME_REFERENCE.CURRENT)
                == sl.ERROR_CODE.SUCCESS
            ):
                imu = sensors.get_imu_data()
                stamp = imu.timestamp.get_nanoseconds()
                if stamp and stamp != last:
                    last = stamp
                    last_new = time.perf_counter()
                    ts.append(stamp * 1e-9 - wall_minus_perf)
                    acc.append(tuple(imu.get_linear_acceleration()))
                    gyro.append(tuple(imu.get_angular_velocity()))
            if not silent_reported and time.perf_counter() - last_new > _SILENT_S:
                silent_reported = True
                errors.put(f"camera {serial}: no IMU sample for {_SILENT_S:g} s")
            if dump.is_set():
                dump.clear()
                write_samples(out_path, ts, acc, gyro)
                dumped.set()
            time.sleep(_POLL_S)
    except Exception as exc:  # noqa: BLE001 - reported to the parent
        errors.put(f"camera {serial}: {exc}")
    finally:
        zed.close()
        write_samples(out_path, ts, acc, gyro)


class _Line:
    """``set()`` prints a protocol line (``ready`` / ``dumped``)."""

    def __init__(self, word: str) -> None:
        self._word = word

    def set(self) -> None:
        print(self._word, flush=True)


class _Errors:
    def put(self, text: str) -> None:
        print("error " + str(text).replace("\n", " "), flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Record a ZED X One IMU (see module docs)")
    p.add_argument("--serial", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--worker",
        default=f"{__name__}:record",
        help="module:function to run in place of the camera (tests)",
    )
    args = p.parse_args(argv)
    module, _, name = args.worker.partition(":")
    worker = getattr(importlib.import_module(module), name)
    stop, dump = threading.Event(), threading.Event()

    def _commands() -> None:
        for line in sys.stdin:
            word = line.strip()
            if word == "dump":
                dump.set()
            elif word == "stop":
                break
        stop.set()

    threading.Thread(target=_commands, daemon=True).start()
    worker(
        args.serial, args.out, _Line("ready"), stop, dump, _Line("dumped"), _Errors()
    )


if __name__ == "__main__":
    main()
