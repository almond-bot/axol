"""
axol umi.latency

Pose↔image latency validation for the Mantis UMI (exUMI method).

UMI training pairs each wrist-camera frame with the tracker pose captured at
the same instant, and the recorder already does nearest-timestamp matching on
the shared ``perf_counter`` timeline — but that is only as good as the two
timestamps themselves (camera exposure time from the ZED SDK, tracker capture
time propagated as ``t_host``). This command measures the *residual* skew
end to end:

1. Print (or display) an ArUco marker flat on the table.
2. Hold the rig so the marker stays in the wrist camera's view and sweep it
   side to side (~1 Hz, ~20 s).
3. The marker's pixel trajectory in the camera stream and the tracker's
   position trajectory are each projected onto their dominant motion axis and
   cross-correlated; the correlation peak's offset is the residual latency
   between the two timelines.

A positive result means the tracker poses are stamped *earlier* than the
camera frames showing the same motion (pose leads image); negative means the
poses are stamped late. |residual| under ~10 ms is fine for 60 fps data —
anything larger indicates a broken timestamp path (e.g. a bridge stamping
compose time, or camera capture timestamps falling back to receive time).

One-time per setup (per tracking backend + camera pipeline), not per episode.
Requires the tracking source connected and streaming (VR app or ``axol
tracker.bridge``) and the wrist ZED attached. ``--selftest`` verifies the
estimator on synthetic signals (no hardware).
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import numpy as np

# Uniform resample rate for the cross-correlation grid. Well above both
# sample rates (camera 60 Hz, trackers 72-120 Hz); with parabolic peak
# refinement the estimator resolves well below one grid step.
_GRID_HZ = 250.0
# Largest |lag| searched. Real pipelines are within tens of ms; a full half
# second of headroom also catches a grossly broken timestamp path.
_MAX_LAG_S = 0.5
# Minimum correlation peak to trust the estimate — below this the two
# trajectories don't share a dominant motion (marker lost, rig not moving).
_MIN_PEAK_CORR = 0.8


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``umi.latency`` subcommand."""
    p = subparsers.add_parser(
        "umi.latency",
        help="Measure residual pose↔image latency by sweeping over an ArUco marker.",
    )
    p.add_argument(
        "--side",
        choices=("left", "right"),
        default="left",
        help="Which rig side's tracker to correlate against (default: left).",
    )
    p.add_argument(
        "--serial",
        type=int,
        default=0,
        help="ZED serial of the rig's wrist camera (default: auto when exactly "
        "one camera is attached).",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Sweep recording duration in seconds (default: 20).",
    )
    p.add_argument(
        "--selftest",
        action="store_true",
        help="Verify the lag estimator on synthetic data (no hardware); "
        "exits nonzero on failure.",
    )
    p.set_defaults(func=run)


def _local_ip() -> str:
    """Best-effort LAN IP of this machine.

    The UDP-connect trick needs a route toward the probe address (no packet
    is sent) and raises ``OSError`` on a robot LAN with no internet route, so
    fall back to resolving the hostname, then ``0.0.0.0`` with a warning.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if not ip.startswith("127."):
            return ip
    except OSError:
        pass
    print(
        "WARNING: could not determine the local IP (no default route?) — "
        "use the hostname, or find the address with `ip addr`."
    )
    return "0.0.0.0"


def _principal_signal(samples: np.ndarray) -> np.ndarray:
    """Project a (N, D) trajectory onto its dominant motion axis (PCA-1)."""
    x = np.asarray(samples, dtype=np.float64)
    x = x - x.mean(axis=0)
    # Right singular vector of the largest singular value = principal axis.
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[0]


def estimate_lag(
    t_a: np.ndarray,
    x_a: np.ndarray,
    t_b: np.ndarray,
    x_b: np.ndarray,
    max_lag_s: float = _MAX_LAG_S,
) -> tuple[float, float]:
    """Lag ``d`` (seconds) such that ``a(t) ~ ±b(t - d)``, and the peak |corr|.

    Both signals are resampled onto a uniform grid over their overlapping
    time range, mean-removed, and Pearson-correlated at every integer-step
    shift within ``±max_lag_s``; the winning shift is refined by parabolic
    interpolation of the |corr| peak. The sign of the underlying correlation
    is irrelevant (the two 1-D projections have an arbitrary relative sign).

    ``d > 0`` means events appear in ``a``'s timeline *after* they appear in
    ``b``'s — i.e. ``b`` leads ``a`` by ``d``.
    """
    t0 = max(float(t_a[0]), float(t_b[0]))
    t1 = min(float(t_a[-1]), float(t_b[-1]))
    if t1 - t0 < 2.0 * max_lag_s:
        raise ValueError(
            f"only {t1 - t0:.2f}s of overlapping signal — record longer sweeps"
        )
    dt = 1.0 / _GRID_HZ
    grid = np.arange(t0, t1, dt)
    a = np.interp(grid, t_a, x_a)
    b = np.interp(grid, t_b, x_b)
    a = a - a.mean()
    b = b - b.mean()

    n = len(grid)
    k_max = int(round(max_lag_s / dt))
    corrs = np.zeros(2 * k_max + 1)
    for idx, k in enumerate(range(-k_max, k_max + 1)):
        # Compare a[i] with b[i - k] over the valid overlap.
        lo, hi = max(0, k), n + min(0, k)
        seg_a = a[lo:hi]
        seg_b = b[lo - k : hi - k]
        na = float(np.linalg.norm(seg_a))
        nb = float(np.linalg.norm(seg_b))
        if na < 1e-12 or nb < 1e-12:
            continue
        corrs[idx] = float(np.dot(seg_a, seg_b)) / (na * nb)

    mag = np.abs(corrs)
    peak = int(np.argmax(mag))
    # Parabolic sub-step refinement on the |corr| peak.
    frac = 0.0
    if 0 < peak < len(mag) - 1:
        y0, y1, y2 = mag[peak - 1], mag[peak], mag[peak + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            frac = 0.5 * (y0 - y2) / denom
    lag = ((peak - k_max) + frac) * dt
    return lag, float(mag[peak])


def _detect_marker_center(gray: np.ndarray, detector) -> np.ndarray | None:
    """Pixel center (u, v) of the first detected ArUco marker, or ``None``."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return None
    return np.asarray(corners[0], dtype=np.float64).reshape(4, 2).mean(axis=0)


def _make_aruco_detector():
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)


def _camera_capture_thread(
    cam,
    detector,
    out: list[tuple[float, float, float]],
    stop: threading.Event,
    stats: dict[str, int],
) -> None:
    """Detect the marker in each fresh camera frame; append (cap_ts, u, v)."""
    import cv2

    last_cap = -1.0
    while not stop.is_set():
        try:
            frame, cap_ts, _recv_ts = cam.read_latest_with_ts()
        except RuntimeError:
            time.sleep(0.005)
            continue
        if cap_ts == last_cap:
            time.sleep(0.002)
            continue
        last_cap = cap_ts
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        stats["frames"] += 1
        center = _detect_marker_center(gray, detector)
        if center is not None:
            stats["detections"] += 1
            out.append((cap_ts, float(center[0]), float(center[1])))


def _report(lag: float, peak: float) -> None:
    print(
        f"\nresidual pose↔image latency: {lag * 1e3:+.1f} ms (peak |corr| {peak:.3f})"
    )
    if peak < _MIN_PEAK_CORR:
        print(
            "  WARNING: weak correlation — the marker and tracker trajectories "
            "don't share a clear dominant motion. Keep the marker in view and "
            "sweep more decisively, then re-run."
        )
        return
    if lag > 0:
        print(
            "  tracker poses are stamped earlier than the camera frames showing "
            "the same motion (pose leads image)."
        )
    else:
        print(
            "  tracker poses are stamped later than the camera frames showing "
            "the same motion (pose trails image)."
        )
    if abs(lag) <= 0.010:
        print("  |residual| <= 10 ms: timestamp path is sound for 60 fps data.")
    else:
        print(
            "  |residual| > 10 ms: check the timestamp path (tracker bridge "
            "capture-time stamping, camera capture_perf_ts source) before "
            "collecting data."
        )


async def _run(side: str, serial: int, duration: float) -> None:
    from ..lerobot.camera import ZedCamera, ZedCameraConfig
    from ..vr.server import VRServer

    if serial <= 0:
        found = ZedCamera.find_cameras()
        if len(found) != 1:
            raise SystemExit(
                f"--serial required: {len(found)} ZED cameras detected "
                f"({', '.join(str(c.get('serial')) for c in found) or 'none'})."
            )
        serial = int(found[0]["serial"])
        print(f"Using the only attached ZED (serial {serial}).")

    detector = _make_aruco_detector()

    hostname = socket.gethostname()
    local_ip = _local_ip()
    print("Connect the tracking source (VR app or `axol tracker.bridge`):")
    print(f"  Hostname : {hostname}.local")
    print(f"  IP       : {local_ip}")
    print()

    tracker_samples: list[tuple[float, float, float, float]] = []

    def on_frame(frame) -> None:
        ee = frame.l_ee if side == "left" else frame.r_ee
        t = frame.t_host if frame.t_host is not None else time.perf_counter()
        tracker_samples.append((t, ee.position.x, ee.position.y, ee.position.z))

    async with VRServer() as server:
        while server.get_frame() is None:
            await asyncio.sleep(0.2)
        print("Tracking source connected.\n")

        cam = ZedCamera(ZedCameraConfig(serial=serial))
        cam.connect()
        try:
            await asyncio.to_thread(
                input,
                f"[{side.upper()}] Place an ArUco marker (4x4 dictionary) flat "
                f"on the table, hold the rig so the {side} wrist camera sees "
                f"it, then press Enter and sweep side to side (~1 Hz) for "
                f"{duration:.0f}s... ",
            )
            marker_track: list[tuple[float, float, float]] = []
            stats = {"frames": 0, "detections": 0}
            stop = threading.Event()
            server.set_on_frame(on_frame)
            cam_thread = threading.Thread(
                target=_camera_capture_thread,
                args=(cam, detector, marker_track, stop, stats),
                name="umi-latency-cam",
                daemon=True,
            )
            cam_thread.start()
            await asyncio.sleep(duration)
            stop.set()
            server.set_on_frame(None)
            cam_thread.join(timeout=5.0)
        finally:
            cam.disconnect()

    print(
        f"\ncaptured {stats['detections']}/{stats['frames']} frames with the "
        f"marker, {len(tracker_samples)} tracker samples"
    )
    if stats["detections"] < 0.5 * max(stats["frames"], 1):
        print(
            "  WARNING: marker detected in under half the frames — keep it in "
            "view for a cleaner estimate."
        )
    if len(marker_track) < 100 or len(tracker_samples) < 100:
        raise SystemExit("not enough samples — re-run with the marker in view.")

    cam_arr = np.asarray(marker_track)
    trk_arr = np.asarray(tracker_samples)
    cam_sig = _principal_signal(cam_arr[:, 1:])
    trk_sig = _principal_signal(trk_arr[:, 1:])
    # a = camera, b = tracker: d > 0 <=> tracker leads the camera.
    lag, peak = estimate_lag(cam_arr[:, 0], cam_sig, trk_arr[:, 0], trk_sig)
    _report(lag, peak)


def _selftest() -> None:
    """Verify the estimator recovers a known lag from realistic signals.

    Synthesizes a smooth multi-sine 3-D rig trajectory, samples it as the
    tracker stream (110 Hz, 3-D positions, mm noise) and as the camera stream
    (60 Hz, 2-D projection through an arbitrary axis pair, px noise) with a
    known timestamp offset between the two, and asserts recovery to within
    2 ms across several offsets — including zero and sub-grid-step values.
    """
    rng = np.random.default_rng(11)

    def traj(t: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                0.20 * np.sin(2 * np.pi * 0.9 * t)
                + 0.05 * np.sin(2 * np.pi * 0.23 * t),
                0.04 * np.sin(2 * np.pi * 0.9 * t + 1.1),
                0.02 * np.sin(2 * np.pi * 0.5 * t + 0.4),
            ],
            axis=1,
        )

    duration = 20.0
    # Arbitrary camera projection: two non-orthogonal image axes (px / m).
    proj = np.array([[900.0, 120.0, -40.0], [80.0, -700.0, 200.0]])

    for true_lag in (0.0, 0.0333, -0.0517, 0.1201, 0.0021):
        t_trk = np.sort(rng.uniform(0.0, duration, int(duration * 110)))
        t_cam = np.sort(rng.uniform(0.0, duration, int(duration * 60)))
        # Tracker leads by true_lag: the pose stamped t shows the motion that
        # the camera stamps at t + true_lag.
        trk = traj(t_trk) + rng.normal(0.0, 0.001, (len(t_trk), 3))
        cam = traj(t_cam - true_lag) @ proj.T + rng.normal(0.0, 0.5, (len(t_cam), 2))
        lag, peak = estimate_lag(
            t_cam, _principal_signal(cam), t_trk, _principal_signal(trk)
        )
        err_ms = abs(lag - true_lag) * 1e3
        print(
            f"selftest: true {true_lag * 1e3:+8.1f} ms -> est {lag * 1e3:+8.1f} ms "
            f"(err {err_ms:.2f} ms, peak |corr| {peak:.3f})"
        )
        assert peak > _MIN_PEAK_CORR, f"weak peak {peak:.3f}"
        assert err_ms < 2.0, f"lag not recovered: {err_ms:.2f} ms error"
    print("selftest PASSED")


def run(args) -> None:
    """Run the latency validation (or the synthetic selftest)."""
    if args.selftest:
        _selftest()
        return
    asyncio.run(_run(args.side, args.serial, args.duration))
