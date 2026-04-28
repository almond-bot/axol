"""Dance sequence — simulation only.

A choreographed routine shown in the browser.

Run:
    python -m almond_axol.test.dance

Open http://localhost:8080 in a browser, then press Enter to start.
"""

import asyncio
import math
import time

import numpy as np

from ..robot.axol import (
    LIMITS,
    SHOULDER_2_LEFT_LIMITS,
    SHOULDER_2_RIGHT_LIMITS,
)
from ..robot.sim import Sim
from ..shared import Joint

_SPEED = 3.0  # rad/s
_RATE_HZ = 100.0
_BEAT = 0.25  # seconds to hold each pose

_ELBOW = math.pi * 2 / 3  # ~120° — safe elbow bend
_S1_HI = LIMITS[Joint.SHOULDER_1][1]  # +1.5708
_S1_LO = LIMITS[Joint.SHOULDER_1][0]  # -1.5708
_S3_HI = LIMITS[Joint.SHOULDER_3][1]
_S3_LO = LIMITS[Joint.SHOULDER_3][0]
_W1_HI = LIMITS[Joint.WRIST_1][1]
_W1_LO = LIMITS[Joint.WRIST_1][0]
_W2_HI = LIMITS[Joint.WRIST_2][1]
_W2_LO = LIMITS[Joint.WRIST_2][0]
_W3_HI = LIMITS[Joint.WRIST_3][1]
_W3_LO = LIMITS[Joint.WRIST_3][0]
_S2L_LO, _S2L_HI = SHOULDER_2_LEFT_LIMITS
_S2R_LO, _S2R_HI = SHOULDER_2_RIGHT_LIMITS

_IDX: dict[Joint, int] = {j: i for i, j in enumerate(Joint)}
_N = len(list(Joint))


def _home() -> np.ndarray:
    return np.zeros(_N, dtype=np.float32)


def _q(**kw: float) -> np.ndarray:
    """Build a joint position array from keyword args (e.g. shoulder_1=1.0)."""
    q = _home()
    for name, val in kw.items():
        q[_IDX[Joint[name.upper()]]] = val
    return q


async def _move(
    sim: Sim,
    q_l: np.ndarray,
    q_r: np.ndarray,
    tgt_l: np.ndarray,
    tgt_r: np.ndarray,
    hold: float = _BEAT,
) -> tuple[np.ndarray, np.ndarray]:
    dist = max(float(np.max(np.abs(tgt_l - q_l))), float(np.max(np.abs(tgt_r - q_r))))
    dur = max(dist / _SPEED, 0.05)
    dt = 1.0 / _RATE_HZ
    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        a = min(t / dur, 1.0)
        s = a * a * (3.0 - 2.0 * a)
        await sim.motion_control(
            left=(q_l * (1 - s) + tgt_l * s).astype(np.float32),
            right=(q_r * (1 - s) + tgt_r * s).astype(np.float32),
        )
        if a >= 1.0:
            break
        await asyncio.sleep(dt)
    await asyncio.sleep(hold)
    return tgt_l.copy(), tgt_r.copy()


async def _run() -> None:
    sim = Sim()
    await sim.enable()
    try:
        print("=== DANCE — simulation ===")
        print("Viser server running at  http://localhost:8080")
        print("Open that URL in a browser, then come back here.\n")
        await asyncio.to_thread(input, "Press Enter to start the dance ...")

        q_l = _home()
        q_r = _home()
        await sim.motion_control(left=q_l, right=q_r)
        await asyncio.sleep(0.5)
        print("\nDancing ...\n")

        # ── Move 1: Raise the roof (x2) ──────────────────────────────────────
        print("Move 1: Raise the roof")
        for _ in range(2):
            # Arms forward, elbows bent
            q_l, q_r = await _move(
                sim,
                q_l,
                q_r,
                _q(shoulder_1=_S1_LO * 0.6, elbow=+_ELBOW),
                _q(shoulder_1=_S1_LO * 0.6, elbow=-_ELBOW),
            )
            # Push up
            q_l, q_r = await _move(
                sim,
                q_l,
                q_r,
                _q(shoulder_1=_S1_HI * 0.8, elbow=+_ELBOW),
                _q(shoulder_1=_S1_HI * 0.8, elbow=-_ELBOW),
            )
            # Back down
            q_l, q_r = await _move(
                sim,
                q_l,
                q_r,
                _q(shoulder_1=_S1_LO * 0.6, elbow=+_ELBOW),
                _q(shoulder_1=_S1_LO * 0.6, elbow=-_ELBOW),
            )
        q_l, q_r = await _move(sim, q_l, q_r, _home(), _home(), hold=0.4)

        # ── Move 2: Spread wings ──────────────────────────────────────────────
        print("Move 2: Spread wings")
        q_l, q_r = await _move(
            sim,
            q_l,
            q_r,
            _q(shoulder_2=_S2L_LO * 0.8),
            _q(shoulder_2=_S2R_HI * 0.8),
            hold=0.6,
        )
        q_l, q_r = await _move(sim, q_l, q_r, _home(), _home(), hold=0.4)

        # ── Move 3: Shimmy (alternating arms up/down, x3) ────────────────────
        print("Move 3: Shimmy")
        for _ in range(3):
            q_l, q_r = await _move(
                sim,
                q_l,
                q_r,
                _q(shoulder_1=_S1_HI * 0.7, shoulder_2=_S2L_LO * 0.4),
                _q(shoulder_1=_S1_LO * 0.7, shoulder_2=_S2R_HI * 0.4),
            )
            q_l, q_r = await _move(
                sim,
                q_l,
                q_r,
                _q(shoulder_1=_S1_LO * 0.7, shoulder_2=_S2L_LO * 0.4),
                _q(shoulder_1=_S1_HI * 0.7, shoulder_2=_S2R_HI * 0.4),
            )
        q_l, q_r = await _move(sim, q_l, q_r, _home(), _home(), hold=0.4)

        # ── Move 4: Wrist disco (elbows bent, wrists oscillate, x3) ──────────
        print("Move 4: Wrist disco")
        q_l, q_r = await _move(
            sim,
            q_l,
            q_r,
            _q(shoulder_1=_S1_LO * 0.5, elbow=+_ELBOW),
            _q(shoulder_1=_S1_LO * 0.5, elbow=-_ELBOW),
        )
        for _ in range(3):
            q_l, q_r = await _move(
                sim,
                q_l,
                q_r,
                _q(
                    shoulder_1=_S1_LO * 0.5,
                    elbow=+_ELBOW,
                    wrist_1=_W1_HI,
                    wrist_3=_W3_HI,
                ),
                _q(
                    shoulder_1=_S1_LO * 0.5,
                    elbow=-_ELBOW,
                    wrist_1=_W1_LO,
                    wrist_3=_W3_LO,
                ),
            )
            q_l, q_r = await _move(
                sim,
                q_l,
                q_r,
                _q(
                    shoulder_1=_S1_LO * 0.5,
                    elbow=+_ELBOW,
                    wrist_1=_W1_LO,
                    wrist_3=_W3_LO,
                ),
                _q(
                    shoulder_1=_S1_LO * 0.5,
                    elbow=-_ELBOW,
                    wrist_1=_W1_HI,
                    wrist_3=_W3_HI,
                ),
            )
        q_l, q_r = await _move(sim, q_l, q_r, _home(), _home(), hold=0.4)

        # ── Move 5: Shoulder roll (shoulder_3 sweep, x2) ─────────────────────
        print("Move 5: Shoulder roll")
        q_l, q_r = await _move(
            sim,
            q_l,
            q_r,
            _q(shoulder_1=_S1_LO * 0.5, elbow=+_ELBOW * 0.6),
            _q(shoulder_1=_S1_LO * 0.5, elbow=-_ELBOW * 0.6),
        )
        for _ in range(2):
            q_l, q_r = await _move(
                sim,
                q_l,
                q_r,
                _q(shoulder_1=_S1_LO * 0.5, shoulder_3=_S3_HI, elbow=+_ELBOW * 0.6),
                _q(shoulder_1=_S1_LO * 0.5, shoulder_3=_S3_LO, elbow=-_ELBOW * 0.6),
            )
            q_l, q_r = await _move(
                sim,
                q_l,
                q_r,
                _q(shoulder_1=_S1_LO * 0.5, shoulder_3=_S3_LO, elbow=+_ELBOW * 0.6),
                _q(shoulder_1=_S1_LO * 0.5, shoulder_3=_S3_HI, elbow=-_ELBOW * 0.6),
            )
        q_l, q_r = await _move(sim, q_l, q_r, _home(), _home(), hold=0.4)

        # ── Move 6: Clappers (gripper open-close, x4) ────────────────────────
        print("Move 6: Clappers")
        q_l, q_r = await _move(
            sim,
            q_l,
            q_r,
            _q(shoulder_1=_S1_LO * 0.5, elbow=+_ELBOW * 0.6),
            _q(shoulder_1=_S1_LO * 0.5, elbow=-_ELBOW * 0.6),
        )
        for _ in range(4):
            q_l, q_r = await _move(
                sim,
                q_l,
                q_r,
                _q(shoulder_1=_S1_LO * 0.5, elbow=+_ELBOW * 0.6, gripper=1.0),
                _q(shoulder_1=_S1_LO * 0.5, elbow=-_ELBOW * 0.6, gripper=1.0),
                hold=0.1,
            )
            q_l, q_r = await _move(
                sim,
                q_l,
                q_r,
                _q(shoulder_1=_S1_LO * 0.5, elbow=+_ELBOW * 0.6, gripper=0.0),
                _q(shoulder_1=_S1_LO * 0.5, elbow=-_ELBOW * 0.6, gripper=0.0),
                hold=0.1,
            )
        q_l, q_r = await _move(sim, q_l, q_r, _home(), _home(), hold=0.4)

        # ── Move 7: Finale — arms wide and up, grippers open ─────────────────
        print("Move 7: Finale")
        q_l, q_r = await _move(
            sim,
            q_l,
            q_r,
            _q(
                shoulder_1=_S1_HI * 0.8,
                shoulder_2=_S2L_LO * 0.7,
                elbow=+_ELBOW * 0.5,
                gripper=1.0,
            ),
            _q(
                shoulder_1=_S1_HI * 0.8,
                shoulder_2=_S2R_HI * 0.7,
                elbow=-_ELBOW * 0.5,
                gripper=1.0,
            ),
            hold=1.5,
        )
        q_l, q_r = await _move(sim, q_l, q_r, _home(), _home(), hold=0.5)

        print("\nDance complete!")
        print("Viser server still running — press Ctrl+C to exit.\n")
        await asyncio.Event().wait()
    finally:
        await sim.disable()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
