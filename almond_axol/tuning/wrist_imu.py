"""Record the wrist cameras' IMUs during a tuning run and score the shake.

The joint encoders sit on the motor side of the gearboxes: they cannot see
backlash, link flex or anything past the last joint, and on 2026-09-22 they
put the right arm's slow-motion tool-tip shake at ~0.6 mm peak-to-peak while
it looked like a couple of millimetres. The ZED X One on each wrist carries an
IMU that measures the gripper's actual motion, so every tuning tool that
moves an arm (``tune.motion``, ``tune.a4``, ``tune.pid``) records it and adds
a ``imu`` block to the run's metrics plus the raw samples to its series.

Each camera is opened in its own subprocess (:mod:`almond_axol.zed.imu_worker`,
light to import): the ZED SDK can wedge its process, and a stuck SDK must
never stall — or crash — a process that is streaming motor commands. The
worker polls the latest IMU sample, drops repeats by timestamp and maps the
SDK's wall-clock stamps onto ``time.perf_counter`` (``CLOCK_MONOTONIC``,
shared by every process), the clock the tuning logs use. The cameras are exclusive: nothing else may hold
them for the run (the tools declare ``uses_cameras`` so the dashboard blocks
previews meanwhile), and a camera that cannot be opened just means no IMU
metrics — the run itself goes ahead.

The metric (:func:`shake_metrics`): acceleration band-passed to the shake
band (1–15 Hz — above the motion, below the structure's buzz), integrated
twice in the frequency domain to displacement, and reported as the median and
90th-percentile 2 s peak-to-peak excursion in millimetres — overall (3-D) and
along gravity (vertical, the direction the tool tip was seen to bounce), the
vertical split into 1–3 Hz (the impedance sway) and 3–15 Hz — plus the band's
acceleration and angular-rate RMS and its dominant frequency.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)

#: The shake band (Hz). Under impedance the visible wobble is at 1–3 Hz —
#: ~2 mm in each of 1–2 and 2–3 Hz on the jelly robot's right arm, against
#: ~1 mm in 3–6 and 0.3 mm in 6–15 (2026-09-23) — so the band starts at 1 Hz;
#: ``slow_osc``'s own commanded motion is below 1 Hz (0.3 mm left in 1–2 Hz).
#: A faster motion leaks more of itself in: the score does not subtract the
#: commanded motion.
SHAKE_BAND = (1.0, 15.0)
#: Sub-bands reported alongside: the impedance sway and the faster shake.
LOW_BAND = (1.0, 3.0)
HIGH_BAND = (3.0, 15.0)

_OPEN_TIMEOUT_S = 12.0
_STOP_TIMEOUT_S = 6.0


def imu_serial(side: str) -> int | None:
    """The ``{side}_arm`` wrist camera's serial from the shared settings, or
    ``None`` when unassigned or the settings cannot be read."""
    try:
        from ..settings import load_store

        cameras = load_store(strict=False).cameras() or {}
    except Exception as exc:  # noqa: BLE001 - no settings → no IMU
        _logger.debug("wrist IMU: no camera settings (%s)", exc)
        return None
    serial = (cameras.get("serials") or {}).get(f"{side}_arm")
    try:
        serial = int(serial)
    except (TypeError, ValueError):
        return None
    return serial or None


@dataclass
class _Recorder:
    side: str
    serial: int
    path: str
    proc: subprocess.Popen[str]
    ready: threading.Event = field(default_factory=threading.Event)
    dumped: threading.Event = field(default_factory=threading.Event)
    errors: list[str] = field(default_factory=list)
    data: dict[str, np.ndarray] = field(default_factory=dict)
    stopped: bool = False

    def send(self, word: str) -> None:
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(word + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError):
            pass

    def close(self) -> None:
        """Close the worker's stdin once it is done (it may have exited)."""
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except (OSError, ValueError):
            pass

    def listen(self) -> None:
        """Reader thread: the worker's protocol lines → events / errors."""
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            word, _, rest = line.strip().partition(" ")
            if word == "ready":
                self.ready.set()
            elif word == "dumped":
                self.dumped.set()
            elif word == "error":
                self.errors.append(rest)


class WristImu:
    """Record the wrist IMUs of ``sides`` for the length of a ``with`` block.

    >>> with WristImu(["right"]) as imu:
    ...     t0 = time.perf_counter(); run(); t1 = time.perf_counter()
    >>> imu.metrics("right", t0, t1)

    Every failure — no pyzed, no camera assigned, a camera that will not
    open — is logged once and leaves that side without data; it never
    raises into the run.
    """

    def __init__(
        self,
        sides: list[str],
        *,
        enabled: bool = True,
        serial_of: Any = None,
        worker: str | None = None,
    ) -> None:
        """``serial_of`` replaces :func:`imu_serial`; ``worker`` (a
        ``module:function`` the subprocess runs in place of
        :func:`almond_axol.zed.imu_worker.record`) replaces the camera (tests)."""
        self._sides = [s for s in sides if s in ("left", "right")]
        self._enabled = enabled
        self._serial_of = serial_of or imu_serial
        self._worker = worker
        self._recorders: dict[str, _Recorder] = {}
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> "WristImu":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def sides(self) -> list[str]:
        """Sides that are recording (or have recorded) IMU data."""
        return list(self._recorders)

    def start(self) -> None:
        if not self._enabled:
            return
        self._tmp = tempfile.TemporaryDirectory(prefix="axol-imu-")
        for side in self._sides:
            serial = self._serial_of(side)
            if serial is None:
                _logger.info(
                    "wrist IMU: no %s_arm camera assigned (Settings → cameras) — "
                    "no IMU metrics for the %s arm",
                    side,
                    side,
                )
                continue
            path = str(Path(self._tmp.name) / f"{side}.npz")
            cmd = [
                sys.executable,
                "-m",
                "almond_axol.zed.imu_worker",
                "--serial",
                str(serial),
                "--out",
                path,
            ]
            if self._worker:
                cmd += ["--worker", self._worker]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                    # The parent's import path: the worker (and a test's fake)
                    # resolve exactly as they do here.
                    env={
                        **os.environ,
                        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
                    },
                )
            except OSError as exc:
                _logger.warning("wrist IMU: could not start the recorder: %s", exc)
                continue
            rec = _Recorder(side, serial, path, proc)
            threading.Thread(target=rec.listen, daemon=True).start()
            # Ready, or the worker gave up (no pyzed, camera missing): no
            # need to sit out the whole timeout for a worker that has exited.
            deadline = time.perf_counter() + _OPEN_TIMEOUT_S
            while (
                not rec.ready.wait(0.1)
                and proc.poll() is None
                and time.perf_counter() < deadline
            ):
                pass
            if not rec.ready.is_set():
                rec.ready.wait(0.2)  # a last line racing the exit
            if not rec.ready.is_set():
                reason = "; ".join(rec.errors) or "camera did not open in time"
                _logger.warning(
                    "wrist IMU: %s camera %d unavailable (%s) — no IMU metrics",
                    side,
                    serial,
                    reason,
                )
                rec.send("stop")
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                rec.close()
                continue
            _logger.info("wrist IMU: recording the %s camera (%d)", side, serial)
            self._recorders[side] = rec

    def flush(self, timeout: float = 2.0) -> None:
        """Load the samples recorded so far without stopping — for tools that
        save a run per candidate mid-session (``tune.pid``)."""
        for rec in self._recorders.values():
            if rec.stopped:
                continue
            rec.dumped.clear()
            rec.send("dump")
            if rec.dumped.wait(timeout):
                self._load(rec)

    def stop(self) -> None:
        """Stop every recorder and load its samples (idempotent)."""
        for rec in self._recorders.values():
            if rec.stopped:
                continue
            rec.stopped = True
            rec.send("stop")
            try:
                rec.proc.wait(timeout=_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                rec.proc.kill()
                _logger.warning(
                    "wrist IMU: the %s recorder did not stop — its last samples "
                    "are lost",
                    rec.side,
                )
            rec.close()
            if rec.errors:
                _logger.warning("wrist IMU: %s: %s", rec.side, "; ".join(rec.errors))
            self._load(rec)
            if not len(rec.data["t"]):
                _logger.warning(
                    "wrist IMU: the %s camera returned no IMU samples", rec.side
                )
        if self._tmp is not None and all(r.stopped for r in self._recorders.values()):
            self._tmp.cleanup()
            self._tmp = None

    @staticmethod
    def _load(rec: _Recorder) -> None:
        try:
            with np.load(rec.path) as z:
                rec.data = {k: z[k] for k in ("t", "acc", "gyro")}
        except (OSError, KeyError, ValueError):
            if not rec.data:
                rec.data = {
                    "t": np.empty(0),
                    "acc": np.empty((0, 3)),
                    "gyro": np.empty((0, 3)),
                }

    def window(
        self, side: str, t0: float, t1: float, origin: float | None = None
    ) -> dict[str, np.ndarray] | None:
        """``{t, acc, gyro}`` of ``side`` between perf_counter ``t0`` and ``t1``,
        ``t`` relative to ``origin`` (default ``t0``) — pass the run log's own
        time origin to put the IMU on its time axis; ``None`` without data."""
        rec = self._recorders.get(side)
        if rec is None or not rec.data or not len(rec.data["t"]):
            return None
        t = rec.data["t"]
        sel = (t >= t0) & (t <= t1)
        if sel.sum() < 2:
            return None
        return {
            "t": t[sel] - (t0 if origin is None else origin),
            "acc": rec.data["acc"][sel],
            "gyro": rec.data["gyro"][sel],
        }

    def metrics(self, side: str, t0: float, t1: float) -> dict[str, float] | None:
        """:func:`shake_metrics` over a window, or ``None`` without data."""
        w = self.window(side, t0, t1)
        if w is None:
            return None
        return shake_metrics(w["t"], w["acc"], w["gyro"])

    def run_blocks(
        self, t0: float, t1: float, origin: float | None = None
    ) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
        """The ``imu`` metrics block (per side) and the series entries
        (``imu_{side}_t/acc/gyro``, ``t`` relative to ``origin``) for a run
        window — both empty without data, so callers can merge them
        unconditionally."""
        metrics: dict[str, dict[str, float]] = {}
        series: dict[str, np.ndarray] = {}
        for side in self.sides:
            w = self.window(side, t0, t1, origin)
            if w is None:
                continue
            m = shake_metrics(w["t"], w["acc"], w["gyro"])
            if m:
                metrics[side] = m
            series[f"imu_{side}_t"] = w["t"].astype(np.float64)
            series[f"imu_{side}_acc"] = w["acc"].astype(np.float32)
            series[f"imu_{side}_gyro"] = w["gyro"].astype(np.float32)
        return metrics, series


def format_imu(metrics: dict[str, dict[str, float]]) -> list[str]:
    """One scorecard line per side of an ``imu`` metrics block."""
    return [
        f"  wrist IMU ({side}): shake {m['shake_mm']:.2f} mm p2p "
        f"(p90 {m['shake_mm_p90']:.2f}; vertical {m['vertical_mm']:.2f} = "
        f"{LOW_BAND[0]:g}-{LOW_BAND[1]:g} Hz {m.get('low_mm', math.nan):.2f} + "
        f"{HIGH_BAND[0]:g}-{HIGH_BAND[1]:g} Hz {m.get('high_mm', math.nan):.2f}), "
        f"accel {m['acc_rms']:.3f} m/s², gyro {m['gyro_rms']:.2f} °/s, "
        f"peak {m['peak_hz']:.1f} Hz"
        for side, m in metrics.items()
    ]


def _band_integrate(x: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """Displacement from acceleration within ``band``: ``X(f) / -(2πf)²``,
    zero outside it (per column)."""
    n = len(x)
    spec = np.fft.rfft(x - x.mean(axis=0), axis=0)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    gain = np.zeros_like(f)
    inside = (f >= band[0]) & (f <= band[1])
    gain[inside] = -1.0 / (2.0 * math.pi * f[inside]) ** 2
    return np.fft.irfft(spec * gain[:, None], n, axis=0)


def shake_metrics(
    t: np.ndarray,
    acc: np.ndarray,
    gyro: np.ndarray | None = None,
    *,
    band: tuple[float, float] = SHAKE_BAND,
    window_s: float = 2.0,
) -> dict[str, float]:
    """Score the shake in an IMU record.

    Args:
        t: Sample times (s), increasing.
        acc: ``(N, 3)`` linear acceleration (m/s², gravity included — it is
            the band-pass's DC and gives the vertical).
        gyro: ``(N, 3)`` angular rate (deg/s), optional.
        band: Shake band (Hz).
        window_s: Peak-to-peak window — 2 s holds two cycles of the 1 Hz
            band edge.

    Returns:
        ``{"shake_mm", "shake_mm_p90", "vertical_mm", "vertical_mm_p90",
        "low_mm", "high_mm", "acc_rms", "gyro_rms", "peak_hz", "rate_hz",
        "seconds"}`` — the windowed peak-to-peak displacement over ``band``
        (3-D: twice the largest excursion from the window's mean; vertical:
        along the record's mean acceleration, i.e. gravity), the vertical
        one again within :data:`LOW_BAND` and :data:`HIGH_BAND`, band
        acceleration RMS (m/s²), band angular-rate RMS (deg/s), the band's
        dominant frequency. Empty when the record is too short (under two
        windows or 50 samples).
    """
    from scipy.signal import butter, sosfiltfilt

    t = np.asarray(t, dtype=float)
    acc = np.asarray(acc, dtype=float).reshape(-1, 3)
    ok = np.isfinite(t) & np.all(np.isfinite(acc), axis=1)
    t, acc = t[ok], acc[ok]
    if len(t) < 50 or t[-1] - t[0] < 2.0 * window_s:
        return {}
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return {}
    fs = 1.0 / dt
    hi = min(band[1], 0.45 * fs)
    if hi <= band[0]:
        return {}
    grid = np.arange(t[0], t[-1], dt)
    a = np.stack([np.interp(grid, t, acc[:, i]) for i in range(3)], axis=1)
    up = a.mean(axis=0)
    up = up / (np.linalg.norm(up) or 1.0)
    sos = butter(4, [band[0], hi], btype="band", fs=fs, output="sos")
    a_bp = sosfiltfilt(sos, a, axis=0)
    disp = _band_integrate(a, fs, (band[0], hi))
    vert = disp @ up
    low_v = _band_integrate(a, fs, (LOW_BAND[0], min(LOW_BAND[1], hi))) @ up
    high_v = (
        _band_integrate(a, fs, (HIGH_BAND[0], min(HIGH_BAND[1], hi))) @ up
        if hi > HIGH_BAND[0]
        else np.zeros(len(grid))
    )
    w = max(2, int(round(window_s * fs)))
    edge = min(w // 2, len(grid) // 4)
    starts = range(edge, len(grid) - w - edge + 1, max(1, w // 2))
    p2p_3d, p2p_v, p2p_low, p2p_high = [], [], [], []
    for s in starts:
        seg = disp[s : s + w]
        p2p_3d.append(
            2.0 * float(np.max(np.linalg.norm(seg - seg.mean(axis=0), axis=1)))
        )
        p2p_v.append(float(np.ptp(vert[s : s + w])))
        p2p_low.append(float(np.ptp(low_v[s : s + w])))
        p2p_high.append(float(np.ptp(high_v[s : s + w])))
    if not p2p_3d:
        return {}
    # Power summed over the axes: a magnitude would rectify each axis and
    # double its frequency.
    win = np.hanning(len(a_bp))[:, None]
    spec = np.sum(np.abs(np.fft.rfft(a_bp * win, axis=0)) ** 2, axis=1)
    freq = np.fft.rfftfreq(len(a_bp), dt)
    inband = (freq >= band[0]) & (freq <= hi)
    out = {
        "shake_mm": 1e3 * float(np.median(p2p_3d)),
        "shake_mm_p90": 1e3 * float(np.percentile(p2p_3d, 90)),
        "vertical_mm": 1e3 * float(np.median(p2p_v)),
        "vertical_mm_p90": 1e3 * float(np.percentile(p2p_v, 90)),
        "low_mm": 1e3 * float(np.median(p2p_low)),
        "high_mm": 1e3 * float(np.median(p2p_high)),
        "acc_rms": float(np.sqrt(np.mean(np.sum(a_bp**2, axis=1)))),
        "gyro_rms": math.nan,
        "peak_hz": float(freq[inband][np.argmax(spec[inband])])
        if inband.any()
        else math.nan,
        "rate_hz": fs,
        "seconds": float(t[-1] - t[0]),
    }
    if gyro is not None:
        g = np.asarray(gyro, dtype=float).reshape(-1, 3)[ok]
        if len(g) == len(t) and np.all(np.isfinite(g)):
            gg = np.stack([np.interp(grid, t, g[:, i]) for i in range(3)], axis=1)
            g_bp = sosfiltfilt(sos, gg, axis=0)
            out["gyro_rms"] = float(np.sqrt(np.mean(np.sum(g_bp**2, axis=1))))
    return out
