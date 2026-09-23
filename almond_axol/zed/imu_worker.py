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

# No new IMU sample for this long after opening → drive grab() as well (a
# camera whose SDK only refreshes sensors on grab).
_GRAB_FALLBACK_S = 1.0
_POLL_S = 0.001


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
    grab_thread: threading.Thread | None = None
    grab_stop = threading.Event()

    def _grab() -> None:
        while not grab_stop.is_set():
            zed.grab()

    try:
        sensors = sl.SensorsData()
        # The SDK stamps sensors on the wall clock; map to CLOCK_MONOTONIC.
        wall_minus_perf = time.time() - time.perf_counter()
        last = None
        opened = time.perf_counter()
        ready.set()
        while not stop.is_set():
            if (
                grab_thread is None
                and not ts
                and time.perf_counter() - opened > _GRAB_FALLBACK_S
            ):
                grab_thread = threading.Thread(target=_grab, daemon=True)
                grab_thread.start()
            if (
                zed.get_sensors_data(sensors, sl.TIME_REFERENCE.CURRENT)
                == sl.ERROR_CODE.SUCCESS
            ):
                imu = sensors.get_imu_data()
                stamp = imu.timestamp.get_nanoseconds()
                if stamp and stamp != last:
                    last = stamp
                    ts.append(stamp * 1e-9 - wall_minus_perf)
                    acc.append(tuple(imu.get_linear_acceleration()))
                    gyro.append(tuple(imu.get_angular_velocity()))
            if dump.is_set():
                dump.clear()
                write_samples(out_path, ts, acc, gyro)
                dumped.set()
            time.sleep(_POLL_S)
    except Exception as exc:  # noqa: BLE001 - reported to the parent
        errors.put(f"camera {serial}: {exc}")
    finally:
        grab_stop.set()
        if grab_thread is not None:
            grab_thread.join(timeout=2.0)
        # Closing while grab() is in flight segfaults the SDK: only close a
        # camera whose grab thread (if any) has exited.
        if grab_thread is None or not grab_thread.is_alive():
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
