"""
axol umi.calibrate

Pivot-calibrates the controller→gripper-TCP offset of each handheld UMI rig.

The Quest controller is rigidly mounted to the gripper, but the tracking
system only knows the controller's own origin — the physical lever arm from
that origin to the gripper jaws (the TCP) is what makes the URDF overlay (and
recorded data) sit on the controller body instead of the real gripper. This
command measures it with the classic pivot/tip calibration: hold the gripper's
closed jaws pinned on a fixed point and slowly rotate the wrist through a wide
range of orientations. Every sampled controller pose ``(R_i, t_i)`` satisfies
``R_i @ p + t_i = c`` for the unknown offset ``p`` (controller frame) and
pivot point ``c`` (world), which is a linear least-squares problem.

Offsets are saved to ``~/.almond/umi/tcp_offset.json`` and picked up
automatically by ``axol teleop --umi`` and ``axol collect-data --umi``.
Requires the VR web app connected and streaming poses (no engage needed).
"""

from __future__ import annotations

import asyncio
import json
import socket
import time

import numpy as np

from ..teleop.config import UMI_TCP_OFFSET_FILE

_CAPTURE_SECONDS = 12.0
# Worst-case residual (m) above which the fit is rejected — the jaws slipped
# off the pivot point or tracking glitched.
_MAX_RMS = 0.012
# Minimum orientation spread (deg) across samples for a well-conditioned fit.
_MIN_SPREAD_DEG = 35.0


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``umi.calibrate`` subcommand."""
    subparsers.add_parser(
        "umi.calibrate",
        help="Pivot-calibrate the UMI controller→gripper-TCP offsets.",
    ).set_defaults(func=run)


def _solve_pivot(
    poses: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, float]:
    """Least-squares pivot fit. Returns ``(p_offset, rms_residual)``.

    ``poses`` are ``(R_3x3, t_3)`` controller poses sampled while the TCP was
    held on a fixed point: ``R_i @ p + t_i = c`` → stack ``[R_i | -I] [p; c] =
    -t_i`` and solve the 6-unknown linear system.
    """
    n = len(poses)
    a = np.zeros((3 * n, 6))
    b = np.zeros(3 * n)
    for i, (rot, pos) in enumerate(poses):
        a[3 * i : 3 * i + 3, :3] = rot
        a[3 * i : 3 * i + 3, 3:] = -np.eye(3)
        b[3 * i : 3 * i + 3] = -pos
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    residuals = a @ x - b
    rms = float(np.sqrt(np.mean(np.sum(residuals.reshape(-1, 3) ** 2, axis=1))))
    return x[:3], rms


def _orientation_spread_deg(poses: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Largest geodesic angle (deg) between any sampled orientation and the first."""
    if len(poses) < 2:
        return 0.0
    r0 = poses[0][0]
    worst = 0.0
    for rot, _ in poses[1:]:
        cos = (float(np.trace(r0.T @ rot)) - 1.0) * 0.5
        worst = max(worst, float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))))
    return worst


async def _capture_side(server, side: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sample ``(R, t)`` controller poses for one side for the capture window."""
    from ..teleop.worker import _quat_xyzw_to_matrix

    poses: list[tuple[np.ndarray, np.ndarray]] = []
    last = None
    deadline = time.perf_counter() + _CAPTURE_SECONDS
    while time.perf_counter() < deadline:
        frame = server.get_frame()
        if frame is not None and frame is not last:
            last = frame
            ee = frame.l_ee if side == "left" else frame.r_ee
            rot = _quat_xyzw_to_matrix(
                ee.quaternion.x, ee.quaternion.y, ee.quaternion.z, ee.quaternion.w
            ).astype(np.float64)
            pos = np.array([ee.position.x, ee.position.y, ee.position.z])
            poses.append((rot, pos))
        await asyncio.sleep(0.01)
    return poses


async def _hold_grippers_closed(umi) -> None:
    """Keep both grippers firmly closed for the duration of the session.

    The pivot fit needs the jaw tip rigid relative to the controller; limp
    (unpowered) jaws would wander. Transient CAN errors are swallowed — a
    dead bus was already reported at enable time.
    """
    hold = np.zeros(8, dtype=np.float32)
    while True:
        try:
            await umi.motion_control(left=hold, right=hold)
        except Exception:  # noqa: BLE001 - keep holding through transients
            pass
        await asyncio.sleep(0.05)


async def _run() -> None:
    from ..vr.server import VRServer

    hostname = socket.gethostname()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    print("Connect the VR app to this machine and keep the headset on:")
    print(f"  Hostname : {hostname}.local")
    print(f"  IP       : {local_ip}")
    print()

    # Enable the grippers and hold them closed throughout — the jaws are the
    # calibration reference, so they must be rigid. Falls back to a warning
    # when the CAN buses aren't reachable (jaws must then be held closed some
    # other way, e.g. a rubber band).
    umi = None
    hold_task: asyncio.Task | None = None
    try:
        from ..robot import Umi

        umi = Umi()
        print("Enabling grippers (they will sweep open to calibrate, then close)...")
        await umi.enable()
        hold_task = asyncio.create_task(_hold_grippers_closed(umi))
        print("Grippers holding closed.\n")
    except Exception as exc:  # noqa: BLE001 - CAN optional for calibration
        print(
            f"WARNING: could not enable the grippers over CAN ({exc}).\n"
            "Proceeding anyway — keep the jaws fully closed by hand (rubber "
            "band / tape) so the tip stays rigid.\n"
        )
        umi = None

    results: dict[str, list[float]] = {}
    async with VRServer() as server:
        while server.get_frame() is None:
            await asyncio.sleep(0.2)
        print("Headset connected.\n")

        for side in ("left", "right"):
            while True:
                await asyncio.to_thread(
                    input,
                    f"[{side.upper()}] Pinch the gripper's closed jaws onto a fixed "
                    f"point (table corner, dot mark), then press Enter and slowly "
                    f"roll/tilt the wrist through a wide range for "
                    f"{_CAPTURE_SECONDS:.0f}s without letting the jaws move... ",
                )
                poses = await _capture_side(server, side)
                spread = _orientation_spread_deg(poses)
                if len(poses) < 100:
                    print(
                        f"  only {len(poses)} samples — is the headset streaming? retrying."
                    )
                    continue
                offset, rms = _solve_pivot(poses)
                print(
                    f"  {len(poses)} samples, orientation spread {spread:.0f} deg, "
                    f"residual {rms * 1e3:.1f} mm, offset [{offset[0]:+.4f} "
                    f"{offset[1]:+.4f} {offset[2]:+.4f}] m (|{np.linalg.norm(offset):.3f}| m)"
                )
                if spread < _MIN_SPREAD_DEG:
                    print(
                        "  too little wrist rotation for a stable fit — try again, "
                        "roll and tilt further."
                    )
                    continue
                if rms > _MAX_RMS:
                    print(
                        "  residual too high — the jaws moved off the point. Try again."
                    )
                    continue
                results[side] = [float(v) for v in offset]
                break

    if hold_task is not None:
        hold_task.cancel()
    if umi is not None:
        try:
            await umi.disable()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass

    UMI_TCP_OFFSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    UMI_TCP_OFFSET_FILE.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved to {UMI_TCP_OFFSET_FILE}")
    print("teleop --umi / collect-data --umi will pick these up automatically.")


def run(_args: object = None) -> None:
    """Run the interactive pivot calibration."""
    asyncio.run(_run())
