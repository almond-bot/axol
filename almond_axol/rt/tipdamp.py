"""The realtime core's tip damper: its settings, and what Python hands it.

``tune.motion --imu-damp`` closed the wrist-IMU damping loop from the Python
motion loop; on jelly that loop's 15-25 ms round trip capped the gain (the
damping's phase wraps near 7-8 Hz on shoulder_1, ~12 Hz on the elbow). The
core runs the same damper (``rust/axol-rt/src/tipdamp.rs``) on each bus
thread: the IMU worker's datagrams go straight to it, and its torque joins
the joints' feedforward on the tick it is computed.

Python's part is configuration:

- :class:`TipDampConfig` — the damper's settings for one arm (they mirror
  ``tune.motion``'s ``--imu-damp*`` flags);
- :func:`poe_chain` — the arm's forward kinematics as a product of
  exponentials (each joint's world axis and a point on it at zero angles,
  and the gripper mount's pose there), measured from the solver's own URDF
  model, so the core can compute the tool height and its Jacobian;
- :func:`config_lines` — the ``tipkin`` / ``tipmodel`` / ``tipdamp`` lines of
  the core's config. They need the resolved joint offsets (the core works in
  the motor frame), so they go on the second configure, with ``cogging``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..constants import ARM_JOINTS
from ..robot.config import TIP_REFERENCES as REFERENCES
from ..robot.config import TipDampConfig

__all__ = [
    "REFERENCES",
    "TipDampConfig",
    "chain_height",
    "check_chain",
    "config_lines",
    "poe_chain",
]


def poe_chain(solver: Any, side: str) -> dict[str, np.ndarray]:
    """The arm's FK as a product of exponentials, from ``solver``'s model.

    At zero joint angles (the other arm and any extra joints at zero too —
    they do not move during a pass), each joint's world axis ``w`` and a
    point ``r`` on it, from central differences of the mount's pose, and the
    mount's pose ``m``. Exact for a serial chain of revolute joints, which
    :func:`check_chain` verifies against the solver (to ~3 µm on jelly's
    model, the float32 FK's own resolution).
    """
    idx = solver.left_indices if side == "left" else solver.right_indices
    zero = np.zeros(solver.num_joints, dtype=np.float64)
    # The solver's FK is float32: a small step drowns in its rounding. A
    # central difference of a pure rotation is exact up to sin(h)/h, which
    # is divided out below.
    h = 0.02

    def pose(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(q, dtype=np.float32)[None]
        pl, pr = solver.ee_positions(rows)
        rl, rr = solver.ee_rotations(rows)
        if side == "left":
            return np.asarray(rl[0], float), np.asarray(pl[0], float)
        return np.asarray(rr[0], float), np.asarray(pr[0], float)

    r0, p0 = pose(zero)
    w = np.zeros((7, 3))
    r = np.zeros((7, 3))
    for i in range(7):
        qp, qm = zero.copy(), zero.copy()
        qp[idx[i]] += h
        qm[idx[i]] -= h
        rp, pp = pose(qp)
        rm, pm = pose(qm)
        d = rp @ rm.T  # ≈ exp([ω] 2h)
        omega = 0.5 * np.array(
            [d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]]
        )
        size = float(np.linalg.norm(omega))
        if size < 1e-6:
            raise ValueError(
                f"tip damper: {side} {ARM_JOINTS[i].value} does not rotate the "
                "gripper mount — not the serial revolute chain the core runs"
            )
        omega /= size
        v = (pp - pm) / (2 * math.sin(h))
        w[i] = omega
        # v = ω × (p − r): the point on the axis nearest p is p + ω × v.
        r[i] = p0 + np.cross(omega, v)
    m = np.eye(4)
    m[:3, :3] = r0
    m[:3, 3] = p0
    return {"w": w, "r": r, "m": m}


def _screw(w: np.ndarray, r: np.ndarray, th: float) -> np.ndarray:
    k = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    rot = np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * k @ k
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = r - rot @ r
    return t


def chain_height(chain: dict[str, np.ndarray], q: np.ndarray) -> float:
    """The mount's height (m) at the arm's joint-frame ``q`` — the core's
    `PoeChain::height`, for checking a chain in Python."""
    g = np.eye(4)
    for i in range(7):
        g = g @ _screw(chain["w"][i], chain["r"][i], float(q[i]))
    return float((g @ chain["m"])[2, 3])


def check_chain(
    solver: Any, side: str, chain: dict[str, np.ndarray], n: int = 20, seed: int = 0
) -> float:
    """Worst |height error| (m) of ``chain`` against the solver over ``n``
    random poses inside ±1 rad."""
    rng = np.random.default_rng(seed)
    idx = solver.left_indices if side == "left" else solver.right_indices
    worst = 0.0
    for _ in range(n):
        q = rng.uniform(-1.0, 1.0, 7)
        full = np.zeros(solver.num_joints, dtype=np.float32)
        full[idx] = q
        pl, pr = solver.ee_positions(full[None])
        z = float((pl if side == "left" else pr)[0, 2])
        worst = max(worst, abs(chain_height(chain, q) - z))
    return worst


def config_lines(
    side: int,
    iface: str,
    cfg: TipDampConfig,
    chain: dict[str, np.ndarray],
    offsets: np.ndarray,
) -> list[str]:
    """The core's config lines for one arm's tip damper.

    ``offsets`` are the resolved joint offsets (``joint = motor + offset``)
    of the seven arm joints, in :data:`ARM_JOINTS` order.
    """
    names = [j.value for j in ARM_JOINTS]
    if not np.all(np.isfinite(offsets[:7])):
        raise ValueError("tip damper: every arm joint's offset must be resolved")
    kin = " ".join(
        repr(float(x))
        for x in (*chain["w"].ravel(), *chain["r"].ravel(), *chain["m"].ravel())
    )
    lines = [
        f"tipkin {side} {iface} {kin}",
        f"tipoff {side} {iface} " + " ".join(repr(float(o)) for o in offsets[:7]),
    ]
    for name, model in cfg.tracking_models.items():
        wz = model.wz if math.isfinite(model.wz) else 0.0
        lines.append(
            f"tipmodel {side} {iface} {names.index(name)} "
            f"{model.wn!r} {model.zeta!r} {wz!r} {model.tau!r}"
        )
    cols = " ".join(
        f"{names.index(n)} {w!r} {float(cfg.joint_lp.get(n, 0.0))!r}"
        for n, w in cfg.joints.items()
    )
    lines.append(
        f"tipdamp {side} {iface} {cfg.imu_port} {cfg.gain!r} {cfg.hp_hz!r} "
        f"{cfg.lp_hz!r} {cfg.lead_hz!r} {cfg.notch_hz!r} {cfg.notch_q!r} "
        f"{cfg.max_torque!r} {REFERENCES[cfg.reference]} {cfg.delay_s!r} "
        f"{cfg.ramp_s!r} {cfg.stale_s!r} {cfg.trip_speed!r} {cfg.trip_s!r} "
        f"{len(cfg.joints)} {cols}"
    )
    return lines
