"""
axol umi.calibrate

Robot-sweep hand-eye calibration of the UMI tracker→gripper (TCP) transform.

The rig's tracker (Quest controller / Vive tracker) is rigidly mounted to the
handheld gripper, but the tracking system only knows the tracker's own origin
— the full 6-DoF transform from that origin to the gripper frame is what lets
absolute-mode teleop and UMI data collection place the *gripper* (not the
tracker) at the recorded pose, independent of how the tracker happens to be
mounted. This command measures it the way RDT2 does:

1. Bolt the rig onto the robot's wrist so the rig gripper coincides with the
   robot's gripper frame (same mount both sides use in normal operation).
2. The arm sweeps slowly through a multi-sine elbow/wrist trajectory while
   robot FK gripper poses and tracker poses are recorded together.
3. The rigid transform is solved from the paired relative motions (AX = XB,
   Park–Martin least squares) — world-frame conventions cancel, so any
   tracking backend that streams VRFrames works unchanged.

The two timestamp streams are not trusted to agree: tracker backends stamp
poses at driver-callback time (not capture time) and FK samples are stamped
after a CAN round trip, so a constant offset of tens of ms between the
timelines is normal — and it biases the solved transform in a
motion-correlated way that stays *under* the residual gate. The solve
therefore searches over a tracker↔FK time offset (±150 ms, coarse-to-fine),
keeps the offset minimizing the position RMS residual, and reports it (a
useful per-tracker latency measurement in its own right).

Transforms are saved per side *and per tracker identity* (Quest / Vive
backend + device, ``--tracker`` overrides the derived key) to
``~/.almond/umi/tcp_transform.json`` and picked up automatically by ``axol
teleop --umi`` / ``axol collect-data --umi`` for the matching tracker.
Requires the tracking source connected and streaming poses (VR web app or
``axol tracker.bridge``); no engage needed.

``--selftest`` runs the full pairing + solver path on synthetic data (FK
sweep pushed through a known ground-truth transform, plus streams offset by
a known time skew) and asserts recovery — no hardware needed.
"""

from __future__ import annotations

import asyncio
import math
import socket
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..umi.handeye import HandEyeResult

_CONTROL_HZ = 50.0
# Multi-sine sweep: per-joint (frequency Hz, amplitude scale) on the elbow +
# wrist joints (Joint enum indices 3..6). Distinct frequencies keep the wrist
# rotation axes varying, which the hand-eye fit needs (two parallel-axis
# rotations leave one translation component unobservable).
_SWEEP_JOINTS = (3, 4, 5, 6)
_SWEEP_FREQS = (0.050, 0.080, 0.110, 0.140)
_SWEEP_SCALES = (0.6, 1.0, 1.0, 1.0)
_RAMP_S = 3.0

# Fit acceptance thresholds.
_MAX_POS_RMS = 0.010  # m
_MAX_ORI_RMS_DEG = 1.5
_MIN_AXIS_SPREAD = 0.15
# A residual above this fraction of its gate passes but is flagged as
# marginal (pos_rms in the top third of the gate, ori likewise).
_MARGINAL_FRACTION = 2.0 / 3.0
# Sweeps per side before giving up (each retry re-runs the full sweep).
_MAX_ATTEMPTS = 3
# Below this average tracker frame rate over the sweep, dropouts are likely
# corrupting the interpolation (and thus the solve).
_MIN_TRACKER_HZ = 30.0
# Half-width of the tracker↔FK time-offset search.
_OFFSET_SPAN_S = 0.150


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``umi.calibrate`` subcommand."""
    p = subparsers.add_parser(
        "umi.calibrate",
        help="Hand-eye calibrate the UMI tracker→gripper transforms (robot sweep).",
    )
    p.add_argument(
        "--side",
        choices=("left", "right", "both"),
        default="both",
        help="Which rig side to calibrate (default: both, one after the other).",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Sweep duration per side in seconds (default: 60).",
    )
    p.add_argument(
        "--amplitude",
        type=float,
        default=0.35,
        help="Peak sweep amplitude in radians on the wrist joints (default: 0.35).",
    )
    p.add_argument(
        "--tracker",
        default=None,
        help="Tracker identity key to save the calibration under (e.g. "
        "'quest', 'survive:T20', 'ultimate:<mac>'). Default: derived from "
        "~/.almond/tracker/config.json when present, else 'quest'.",
    )
    p.add_argument(
        "--selftest",
        action="store_true",
        help="Verify the solver on synthetic data (no hardware); exits nonzero on failure.",
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


def _sweep_offsets(t: float, duration: float, amplitude: float) -> np.ndarray:
    """Joint offsets (rad, shape (8,)) of the sweep at time ``t``."""
    ramp = min(1.0, t / _RAMP_S, max(0.0, (duration - t) / _RAMP_S))
    out = np.zeros(8)
    for j, f, s in zip(_SWEEP_JOINTS, _SWEEP_FREQS, _SWEEP_SCALES):
        out[j] = amplitude * s * ramp * math.sin(2.0 * math.pi * f * t)
    return out


def _slerp(qa: np.ndarray, qb: np.ndarray, u: float) -> np.ndarray:
    """Spherical interpolation between two xyzw quaternions."""
    d = float(np.dot(qa, qb))
    if d < 0.0:
        qb = -qb
        d = -d
    if d > 0.9995:
        q = qa + u * (qb - qa)
        return q / np.linalg.norm(q)
    theta = math.acos(min(1.0, d))
    return (math.sin((1.0 - u) * theta) * qa + math.sin(u * theta) * qb) / math.sin(
        theta
    )


def _align_tracker_to_fk(
    fk_times: list[float],
    tracker_samples: list[tuple[float, np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Interpolate tracker poses at each FK sample time.

    ``tracker_samples`` are ``(t, pos_3, quat_xyzw)`` sorted by time. Returns
    one ``(R, t)`` per FK time, or ``None`` where the FK time falls outside
    the tracker stream (or across a gap > 0.2 s).
    """
    from ..teleop.worker import _quat_xyzw_to_matrix

    out: list[tuple[np.ndarray, np.ndarray] | None] = []
    times = np.array([s[0] for s in tracker_samples])
    for t in fk_times:
        j = int(np.searchsorted(times, t))
        if j <= 0 or j >= len(times):
            out.append(None)
            continue
        t0, p0, q0 = tracker_samples[j - 1]
        t1, p1, q1 = tracker_samples[j]
        if t1 - t0 > 0.2:
            out.append(None)
            continue
        u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        pos = p0 + u * (p1 - p0)
        quat = _slerp(q0, q1, u)
        rot = _quat_xyzw_to_matrix(*quat).astype(np.float64)
        out.append((rot, pos.astype(np.float64)))
    return out


def _solve_at_offset(
    offset: float,
    fk_times: list[float],
    fk_poses: list[tuple[np.ndarray, np.ndarray]],
    tracker_samples: list[tuple[float, np.ndarray, np.ndarray]],
    catch: bool = True,
):
    """Hand-eye solve with the tracker timeline shifted by ``offset`` seconds.

    ``offset`` is added to the tracker timestamps (equivalently, the tracker
    stream is interpolated at ``fk_time - offset``), so a positive value
    means the tracker stamps its poses *early* relative to the FK timeline.

    Returns ``(result, n_aligned)``, or ``None`` when the shifted alignment
    yields too few samples / degenerate motion (``catch=False`` lets the
    underlying ``ValueError`` propagate instead, for error reporting).
    """
    from ..umi.handeye import solve_hand_eye

    aligned = _align_tracker_to_fk([t - offset for t in fk_times], tracker_samples)
    fk_sub = []
    trk_sub = []
    for pose, tp in zip(fk_poses, aligned):
        if tp is None:
            continue
        fk_sub.append(pose)
        trk_sub.append(tp)
    try:
        return solve_hand_eye(fk_sub, trk_sub), len(fk_sub)
    except ValueError:
        if not catch:
            raise
        return None


def _search_time_offset(
    fk_times: list[float],
    fk_poses: list[tuple[np.ndarray, np.ndarray]],
    tracker_samples: list[tuple[float, np.ndarray, np.ndarray]],
    span: float = _OFFSET_SPAN_S,
) -> tuple[float, HandEyeResult, int, bool]:
    """Find the tracker↔FK time offset minimizing the position RMS residual.

    Neither timestamp stream is capture-time accurate (driver-callback
    stamping on the tracker side, a CAN round trip on the FK side); the
    constant skew between them biases the solved transform in a
    motion-correlated way while staying under the residual gate. Scanning
    the offset and keeping the residual minimum removes the bias and
    measures the skew as a by-product.

    The score minimized is the gate-normalized sum ``pos_rms / _MAX_POS_RMS
    + ori_rms_deg / _MAX_ORI_RMS_DEG``: the multi-sine sweep is
    wrist-rotation dominant, so a timestamp skew shows up far more sharply
    in the orientation residual than in the (lever-arm-limited, partially
    absorbed into the solved transforms) position residual — position alone
    localizes the minimum poorly.

    Coarse-to-fine grid scan: 21 candidates across ``±span``, then two
    refinement passes around the running best (final resolution ~0.4 ms).
    Candidates where the solve fails (too few aligned samples at extreme
    shifts) are skipped; if *every* candidate fails, the zero-offset solve's
    ``ValueError`` is re-raised so the caller sees the real reason.

    Returns:
        ``(offset_s, result, n_aligned, at_boundary)`` — the winning offset
        (positive: tracker stamps early vs the FK timeline), its
        :class:`~almond_axol.umi.handeye.HandEyeResult`, the aligned sample
        count it used, and whether the coarse scan peaked at the boundary
        (the true offset may lie outside the scanned range).
    """
    best: tuple[float, HandEyeResult, int] | None = None
    best_score = math.inf

    def consider(offset: float) -> None:
        nonlocal best, best_score
        out = _solve_at_offset(offset, fk_times, fk_poses, tracker_samples)
        if out is None:
            return
        res, n_aligned = out
        score = res.pos_rms / _MAX_POS_RMS + res.ori_rms_deg / _MAX_ORI_RMS_DEG
        if score < best_score:
            best = (offset, res, n_aligned)
            best_score = score

    coarse = np.linspace(-span, span, 21)
    for d in coarse:
        consider(float(d))
    if best is None:
        # Every candidate failed — surface the underlying error.
        _solve_at_offset(0.0, fk_times, fk_poses, tracker_samples, catch=False)
        raise ValueError("hand-eye solve failed at every candidate time offset")
    at_boundary = abs(best[0]) >= span - 1e-9

    step = float(coarse[1] - coarse[0])
    for _ in range(2):
        center = best[0]
        for d in np.linspace(center - step, center + step, 13):
            consider(float(d))
        step /= 6.0  # the fine grid's spacing becomes the next half-width

    return best[0], best[1], best[2], at_boundary


def _solve_and_report(
    side: str,
    fk_times: list[float],
    fk_poses: list[tuple[np.ndarray, np.ndarray]],
    tracker_samples: list[tuple[float, np.ndarray, np.ndarray]],
):
    """Offset-search + hand-eye solve, with the fit report printed.

    Returns ``(result, time_offset_s)``.
    """
    offset, res, n_aligned, at_boundary = _search_time_offset(
        fk_times, fk_poses, tracker_samples
    )
    stamp_word = "early" if offset >= 0 else "late"
    print(
        f"  [{side}] tracker↔FK time offset {offset * 1e3:+.1f} ms "
        f"(tracker stamps its poses ~{abs(offset) * 1e3:.0f} ms {stamp_word} "
        f"relative to the FK timeline)"
    )
    if at_boundary:
        print(
            f"  [{side}] WARNING: the offset search peaked at the "
            f"±{_OFFSET_SPAN_S * 1e3:.0f} ms scan boundary — the real skew "
            "may be larger; the timestamp path looks broken."
        )
    off = res.translation
    print(
        f"  [{side}] {n_aligned} aligned samples, {res.n_pairs} motion pairs, "
        f"axis spread {res.axis_spread:.2f}\n"
        f"  [{side}] tracker→gripper offset "
        f"[{off[0]:+.4f} {off[1]:+.4f} {off[2]:+.4f}] m (|{np.linalg.norm(off):.3f}| m), "
        f"residual {res.pos_rms * 1e3:.1f} mm / {res.ori_rms_deg:.2f} deg RMS"
    )
    return res, offset


async def _sweep_side(robot, server, side: str, duration: float, amplitude: float):
    """Run the sweep on one arm while recording FK + tracker pose streams.

    Returns the raw streams ``(fk_times, fk_poses, tracker_samples)`` —
    time alignment is done later, per candidate time offset, by the solve.
    """
    from ..umi.fk import ArmFK

    fk = ArmFK()

    tracker_samples: list[tuple[float, np.ndarray, np.ndarray]] = []

    def on_frame(frame) -> None:
        ee = frame.l_ee if side == "left" else frame.r_ee
        t = frame.t_host if frame.t_host is not None else time.perf_counter()
        tracker_samples.append(
            (
                t,
                np.array([ee.position.x, ee.position.y, ee.position.z]),
                np.array(
                    [
                        ee.quaternion.x,
                        ee.quaternion.y,
                        ee.quaternion.z,
                        ee.quaternion.w,
                    ]
                ),
            )
        )

    server.set_on_frame(on_frame)

    left0, right0 = await robot.get_positions()
    q0 = left0 if side == "left" else right0
    if q0 is None:
        raise RuntimeError(f"the {side} arm reported no positions — is it enabled?")

    fk_times: list[float] = []
    q_samples: list[np.ndarray] = []

    interval = 1.0 / _CONTROL_HZ
    start = time.perf_counter()
    tick = 0
    while True:
        t = time.perf_counter() - start
        if t >= duration:
            break
        target = q0 + _sweep_offsets(t, duration, amplitude)
        if side == "left":
            await robot.motion_control(left=target.astype(np.float32))
        else:
            await robot.motion_control(right=target.astype(np.float32))

        # Sample measured joints (not the command) a few times a second — the
        # impedance controller's tracking error would otherwise bias FK.
        if tick % 3 == 0:
            left, right = await robot.get_positions()
            q = left if side == "left" else right
            if q is not None:
                fk_times.append(time.perf_counter())
                q_samples.append(np.asarray(q, dtype=np.float64))
        tick += 1
        await asyncio.sleep(max(0.0, start + (tick * interval) - time.perf_counter()))

    server.set_on_frame(None)

    if len(tracker_samples) < 50:
        raise RuntimeError(
            f"only {len(tracker_samples)} tracker frames arrived during the "
            "sweep — is the tracking source streaming?"
        )
    tracker_hz = len(tracker_samples) / duration
    if tracker_hz < _MIN_TRACKER_HZ:
        print(
            f"  [{side}] WARNING: tracker delivered only {tracker_hz:.0f} "
            f"samples/s during the sweep (< {_MIN_TRACKER_HZ:.0f}/s) — "
            "dropouts corrupt the solve; check the tracking link (occlusion, "
            "Wi-Fi, bridge load) before trusting this fit."
        )

    fk_poses = [fk.gripper_pose(side, q) for q in q_samples]
    return fk_times, fk_poses, tracker_samples


async def _run(
    side_arg: str, duration: float, amplitude: float, tracker_arg: str | None
) -> None:
    from ..robot import Axol
    from ..umi.calibration import (
        UMI_TCP_TRANSFORM_FILE,
        save_tcp_transforms,
        tracker_key_for_side,
    )
    from ..vr.server import VRServer

    hostname = socket.gethostname()
    local_ip = _local_ip()
    print("Connect the tracking source (VR app or `axol tracker.bridge`):")
    print(f"  Hostname : {hostname}.local")
    print(f"  IP       : {local_ip}")
    print()

    sides = ("left", "right") if side_arg == "both" else (side_arg,)
    tracker_keys: dict[str, str] = {}
    for side in sides:
        key, reason = tracker_key_for_side(side, override=tracker_arg)
        tracker_keys[side] = key
        print(
            f"[{side}] calibration will be saved for tracker '{key}' "
            f"({reason}); pass --tracker to override."
        )
    print()

    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    time_offsets: dict[str, float] = {}

    def _save() -> None:
        if not results:
            return
        save_tcp_transforms(results, tracker_keys, time_offsets)
        print(f"\nSaved {'/'.join(results)} to {UMI_TCP_TRANSFORM_FILE}")
        print(
            "teleop --umi / collect-data --umi will pick these up "
            "automatically for the matching tracker."
        )

    async with VRServer() as server:
        while server.get_frame() is None:
            await asyncio.sleep(0.2)
        print("Tracking source connected.\n")

        async with Axol() as robot:
            for side in sides:
                await asyncio.to_thread(
                    input,
                    f"[{side.upper()}] Bolt the {side} rig onto the {side} "
                    f"wrist mount (rig gripper aligned with the robot "
                    f"gripper frame), clear the workspace, then press Enter "
                    f"to start the {duration:.0f}s sweep... ",
                )
                for attempt in range(1, _MAX_ATTEMPTS + 1):
                    if attempt > 1:
                        print(f"  [{side}] sweep attempt {attempt}/{_MAX_ATTEMPTS}")
                    fk_times, fk_poses, tracker_samples = await _sweep_side(
                        robot, server, side, duration, amplitude
                    )
                    try:
                        res, offset = _solve_and_report(
                            side, fk_times, fk_poses, tracker_samples
                        )
                    except ValueError as exc:
                        print(f"  fit failed: {exc} — retrying the sweep.")
                        continue
                    if res.axis_spread < _MIN_AXIS_SPREAD:
                        print(
                            "  sweep rotation axes too parallel for a stable "
                            "fit — increase --amplitude and retry."
                        )
                        continue
                    if res.pos_rms > _MAX_POS_RMS:
                        print(
                            f"  position residual {res.pos_rms * 1e3:.1f} mm "
                            f"exceeds the {_MAX_POS_RMS * 1e3:.0f} mm gate — "
                            "check the rig is rigidly bolted and tracking is "
                            "clean, then retry."
                        )
                        continue
                    if res.ori_rms_deg > _MAX_ORI_RMS_DEG:
                        print(
                            f"  orientation residual {res.ori_rms_deg:.2f} deg "
                            f"exceeds the {_MAX_ORI_RMS_DEG:.1f} deg gate — "
                            "check the mount for flex and tracking for "
                            "jitter, then retry."
                        )
                        continue
                    if (
                        res.pos_rms > _MARGINAL_FRACTION * _MAX_POS_RMS
                        or res.ori_rms_deg > _MARGINAL_FRACTION * _MAX_ORI_RMS_DEG
                    ):
                        print(
                            f"  [{side}] PASSED but marginal (pos "
                            f"{res.pos_rms * 1e3:.1f}/{_MAX_POS_RMS * 1e3:.0f} mm, "
                            f"ori {res.ori_rms_deg:.2f}/{_MAX_ORI_RMS_DEG:.1f} deg) "
                            "— consider re-running for a cleaner fit."
                        )
                    results[side] = (res.rotation, res.translation)
                    time_offsets[side] = offset
                    break
                else:
                    _save()
                    raise SystemExit(
                        f"[{side}] no acceptable fit after {_MAX_ATTEMPTS} "
                        f"sweeps — aborting. Fix the mount/tracking (or raise "
                        f"--amplitude) and re-run `axol umi.calibrate "
                        f"--side {side}`."
                    )

    _save()


def _selftest() -> None:
    """End-to-end synthetic check: FK sweep through a known transform.

    Part 1 generates the same multi-sine joint sweep the hardware path
    drives, computes FK gripper poses, pushes them through a known
    ground-truth tracker→gripper transform (plus tracking noise) to
    synthesize already-aligned tracker poses, and asserts the solver
    recovers the transform.

    Part 2 exercises the time-offset search: the tracker stream is
    re-synthesized as a timestamped 110 Hz stream whose stamps carry a known
    skew against the FK timeline, and the search must recover both the skew
    (within a few ms) and the transform.
    """
    from ..teleop.worker import _quat_xyzw_to_matrix
    from ..umi.calibration import _quat_xyzw
    from ..umi.fk import ArmFK
    from ..umi.handeye import solve_hand_eye

    rng = np.random.default_rng(7)
    fk = ArmFK()

    def _rot(quat: np.ndarray) -> np.ndarray:
        q = np.asarray(quat, dtype=np.float64)
        q = q / np.linalg.norm(q)
        return _quat_xyzw_to_matrix(*q).astype(np.float64)

    def _rodrigues(v: np.ndarray) -> np.ndarray:
        theta = float(np.linalg.norm(v))
        if theta < 1e-12:
            return np.eye(3)
        k = v / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)

    # Ground truth: a deliberately awkward mount (big lever arm, big tilt).
    R_x = _rot(np.array([0.24, -0.13, 0.4, 0.876]))
    t_x = np.array([0.03, -0.11, 0.07])
    R_y = _rot(np.array([-0.5, 0.5, -0.5, 0.5]))
    t_y = np.array([0.4, 1.1, -0.6])

    q0 = np.array([-0.16, 0.0, 0.0, 0.31, 0.0, 0.0, -0.16, 0.0])
    duration, amplitude, hz = 60.0, 0.35, 15.0

    def fk_pose(t: float) -> tuple[np.ndarray, np.ndarray]:
        q = q0 + _sweep_offsets(t, duration, amplitude)
        return fk.gripper_pose("left", q)

    def tracker_pose(t: float) -> tuple[np.ndarray, np.ndarray]:
        """Noisy tracker pose of the sweep at time ``t``.

        B = Y * A * X^-1  (T^w_c = T^w_b * T^b_g * (T^c_g)^-1), plus
        ~1 mm position / ~0.15 deg orientation tracking noise.
        """
        R_a, t_a = fk_pose(t)
        R_wg = R_y @ R_a
        t_wg = R_y @ t_a + t_y
        R_b = R_wg @ R_x.T
        t_b = t_wg - R_b @ t_x
        t_b = t_b + rng.normal(0.0, 0.001, 3)
        R_n = _rodrigues(rng.normal(0.0, math.radians(0.15), 3))
        return R_b @ R_n, t_b

    def x_errors(res) -> tuple[float, float]:
        pos_err = float(np.linalg.norm(res.translation - t_x))
        ori_err = math.degrees(
            math.acos(
                max(
                    -1.0,
                    min(1.0, (float(np.trace(res.rotation.T @ R_x)) - 1.0) * 0.5),
                )
            )
        )
        return pos_err, ori_err

    # --- Part 1: aligned streams straight into the solver. ---
    n = int(duration * hz)
    fk_times = [i / hz for i in range(n)]
    fk_poses = [fk_pose(t) for t in fk_times]
    tracker_poses = [tracker_pose(t) for t in fk_times]

    res = solve_hand_eye(fk_poses, tracker_poses)
    pos_err, ori_err = x_errors(res)
    y_pos_err = float(np.linalg.norm(res.world_base[1] - t_y))
    print(
        f"selftest: X pos err {pos_err * 1e3:.2f} mm, ori err {ori_err:.3f} deg; "
        f"Y pos err {y_pos_err * 1e3:.2f} mm; fit residual "
        f"{res.pos_rms * 1e3:.2f} mm / {res.ori_rms_deg:.3f} deg RMS "
        f"({res.n_pairs} pairs, axis spread {res.axis_spread:.2f})"
    )
    assert pos_err < 0.005, f"translation not recovered: {pos_err * 1e3:.1f} mm off"
    assert ori_err < 0.5, f"rotation not recovered: {ori_err:.2f} deg off"

    # --- Part 2: time-offset search on a skewed tracker timestamp stream. ---
    # A tracker sample of the motion at time t is *stamped* t + skew (skew >
    # 0 is the usual late driver-callback stamp). The search reports the
    # shift to add to tracker stamps to land on the FK timeline, i.e. -skew.
    tracker_hz = 110.0
    for skew_ms in (42.0, -27.5, 0.0):
        skew = skew_ms * 1e-3
        tracker_samples = []
        for i in range(int(duration * tracker_hz)):
            t = i / tracker_hz
            R_b, t_b = tracker_pose(t)
            tracker_samples.append(
                (t + skew, t_b, np.asarray(_quat_xyzw(R_b), dtype=np.float64))
            )
        offset, res, n_aligned, at_boundary = _search_time_offset(
            fk_times, fk_poses, tracker_samples
        )
        pos_err, ori_err = x_errors(res)
        err_ms = abs(offset + skew) * 1e3
        print(
            f"selftest: stamp skew {skew_ms:+8.1f} ms -> offset "
            f"{offset * 1e3:+8.1f} ms (err {err_ms:.2f} ms); X pos err "
            f"{pos_err * 1e3:.2f} mm, ori err {ori_err:.3f} deg, residual "
            f"{res.pos_rms * 1e3:.2f} mm RMS ({n_aligned} aligned samples)"
        )
        assert not at_boundary, "offset search pinned at the scan boundary"
        assert err_ms < 3.0, f"time offset not recovered: {err_ms:.2f} ms error"
        assert pos_err < 0.005, f"translation not recovered: {pos_err * 1e3:.1f} mm"
        assert ori_err < 0.5, f"rotation not recovered: {ori_err:.2f} deg"
    print("selftest PASSED")


def run(args) -> None:
    """Run the hand-eye calibration (or the synthetic selftest)."""
    if args.selftest:
        _selftest()
        return
    asyncio.run(_run(args.side, args.duration, args.amplitude, args.tracker))
