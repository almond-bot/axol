"""Fit a joint's full friction curve — the law the realtime core applies.

The core's friction feedforward (``rust/axol-rt/src/filter.rs``) is two terms:

- sliding friction on the *commanded* velocity,
  ``(fc + fl·|g|)·tanh(0.1·min(k, 100)·v) + fv·v + fo``;
- the Stribeck excess on the *measured* velocity,
  ``gain·(dfs + ls·|g|)·exp(−(v/vs)²)·tanh(v/0.02)``,

with ``g`` the joint's gravity torque. At a steady sweep speed the two
velocities agree, so the sum is one curve of speed and load, and a
bidirectional sweep measures it directly: the half-difference of the torque
going forward and backward over the same angle is the friction (gravity and
anything else position-dependent cancel), the average is gravity plus ``fo``.

:func:`fit_friction` fits that curve with the runtime's own shapes —
including the ``k ≤ 100`` cap, which keeps the Coulomb term ramping through
zero but leaves it at 17% of ``fc`` at 1°/s and 48% at 3°/s. Where the
measured friction is already fully developed at those speeds, the fit hands
the difference to the Stribeck term, whose ``tanh(v/0.02)`` is sharp: the
runtime then cancels what the joint really has at the speeds the slow shake
lives at, which a Coulomb-plus-viscous fit from 7-72°/s sweeps never did.

Inputs are raw cruise samples — ``tune.friction --raw-csv`` rows, or a
``tune.motion`` constant-speed run turned into the same rows — with the
gravity load at each sample.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

#: The runtime's cap on the Coulomb tanh steepness (``FRICTION_FF_K_MAX``).
K_MAX = 100.0
#: The runtime Stribeck term's zero-crossing speed (``STRIBECK_V0``, rad/s).
STRIBECK_V0 = 0.02
#: Angle grid the forward and backward passes are matched on (rad).
GRID_RAD = math.radians(0.5)


def coulomb_unit(v: np.ndarray, k: float) -> np.ndarray:
    return np.tanh(0.1 * min(k, K_MAX) * v)


def stribeck_shape(v: np.ndarray, vs: float) -> np.ndarray:
    return np.exp(-((v / vs) ** 2)) * np.tanh(v / STRIBECK_V0)


@dataclass(frozen=True)
class FrictionFit:
    """The fitted curve, in the runtime's parameters.

    ``halfdiff(v, load)`` is the friction magnitude at speed ``v`` (rad/s,
    positive) under gravity load ``load`` (Nm, magnitude).
    """

    fc: float
    fl: float
    k: float
    fv: float
    fo: float
    dfs: float
    ls: float
    vs: float
    rms: float  # residual RMS of the half-difference fit (Nm)
    r2: float
    n: int
    load_span: float  # Nm of gravity load the data covered
    speeds: tuple[float, ...]  # rad/s

    def halfdiff(self, v: np.ndarray, load: np.ndarray | float) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        load = np.abs(np.asarray(load, dtype=float))
        return (
            (self.fc + self.fl * load) * coulomb_unit(v, self.k)
            + self.fv * v
            + (self.dfs + self.ls * load) * stribeck_shape(v, self.vs)
        )

    def friction_params(self) -> dict[str, float]:
        """The ``friction`` calibration entry (``FrictionParams``)."""
        return {
            "fc": round(self.fc, 4),
            "k": round(self.k, 2),
            "fv": round(self.fv, 4),
            "fo": round(self.fo, 4),
            "fl": round(self.fl, 4),
        }

    def stribeck_params(self, gain: float) -> dict[str, float]:
        """The ``stribeck_*`` calibration fields, at cancellation ``gain``."""
        return {
            "stribeck_gain": round(gain, 3),
            "stribeck_dfs": round(self.dfs, 4),
            "stribeck_load_gain": round(self.ls, 4),
            "stribeck_vs": round(self.vs, 4),
        }

    def as_dict(self) -> dict[str, float]:
        d = asdict(self)
        d["speeds"] = list(self.speeds)
        return d


@dataclass(frozen=True)
class Sample:
    """One matched angle bin of one speed: the friction there."""

    speed: float  # rad/s
    q: float  # rad
    load: float  # Nm, |gravity|
    halfdiff: float  # Nm
    average: float  # Nm (gravity + fo + anything position-dependent)


def matched_samples(
    speed: np.ndarray,
    direction: np.ndarray,
    q: np.ndarray,
    tau: np.ndarray,
    load: Callable[[float], float] | np.ndarray,
    group: np.ndarray | None = None,
    grid: float = GRID_RAD,
) -> list[Sample]:
    """Half-difference and average per angle bin, per speed (and group).

    ``speed`` (rad/s, positive) and ``direction`` (``+1`` / ``-1``, or
    ``"+"`` / ``"-"``) label each raw sample; ``load`` is either a function
    of ``q`` (the sweep pose's gravity model) or a per-sample array (a run
    whose other joints moved). ``group`` keeps passes apart (e.g. separate
    sweeps at one speed, or different poses).
    """
    speed = np.round(np.asarray(speed, dtype=float), 5)
    direction = np.asarray(
        [
            1
            if d in ("+", 1, 1.0) or (isinstance(d, str) and d.startswith("+"))
            else -1
            for d in direction
        ]
    )
    q = np.asarray(q, dtype=float)
    tau = np.asarray(tau, dtype=float)
    per_sample_load = None if callable(load) else np.abs(np.asarray(load, dtype=float))
    group = np.zeros(len(q), dtype=int) if group is None else np.asarray(group)
    out: list[Sample] = []
    for key in sorted(set(zip(group.tolist(), speed.tolist()))):
        sel = (group == key[0]) & (speed == key[1])
        if sel.sum() < 10:
            continue
        idx = np.floor(q[sel] / grid).astype(int)
        fwd: dict[int, list[int]] = {}
        bwd: dict[int, list[int]] = {}
        where = np.where(sel)[0]
        for i, b, d in zip(where, idx, direction[sel]):
            (fwd if d > 0 else bwd).setdefault(int(b), []).append(i)
        for b in sorted(set(fwd) & set(bwd)):
            f, r = fwd[b], bwd[b]
            if len(f) < 2 or len(r) < 2:
                continue
            tf, tb = float(np.mean(tau[f])), float(np.mean(tau[r]))
            qc = (b + 0.5) * grid
            ld = (
                abs(float(load(qc)))  # type: ignore[operator]
                if per_sample_load is None
                else float(np.mean(per_sample_load[f + r]))
            )
            out.append(Sample(key[1], qc, ld, 0.5 * (tf - tb), 0.5 * (tf + tb)))
    return out


def fit_friction(
    samples: Iterable[Sample],
    *,
    gravity_residual: Callable[[Sample], float] | None = None,
    fit_load: bool = True,
    fit_k: bool = False,
) -> FrictionFit:
    """Least-squares fit of the runtime curve to matched samples.

    ``gravity_residual`` (optional) returns, per sample, the average torque
    minus the gravity model — its mean is ``fo``. Without it ``fo`` is 0.
    ``fit_load`` false (or data spanning < 3 Nm of load) pins ``fl`` and
    ``ls`` at 0: a narrow load range cannot separate them from ``fc``/``dfs``.
    ``k`` is pinned at the runtime cap (:data:`K_MAX`) unless ``fit_k``:
    real Coulomb friction is sharp, the cap is the only thing softening it,
    and a free ``k`` trades off against ``vs``/``dfs`` into unphysical
    corners (k ≈ 9 from three speeds of jelly shoulder_1 data — a Coulomb
    term still at half strength at 30°/s). Robust (soft-L1) loss: a bin at a
    gear-mesh bump should not steer it.
    """
    from scipy.optimize import least_squares

    samples = list(samples)
    if len(samples) < 12:
        raise ValueError(f"only {len(samples)} matched samples — too few to fit")
    v = np.array([s.speed for s in samples])
    load = np.array([s.load for s in samples])
    h = np.array([s.halfdiff for s in samples])
    speeds = tuple(sorted(set(v.tolist())))
    if len(speeds) < 3:
        raise ValueError(f"only {len(speeds)} sweep speeds — need 3 or more")
    span = float(np.ptp(load))
    use_load = fit_load and span >= 3.0

    # x = [fc, fl, k, fv, dfs, ls, log(vs)]
    def model(x: np.ndarray) -> np.ndarray:
        fc, fl, k, fv, dfs, ls, lvs = x
        vs = math.exp(lvs)
        return (
            (fc + fl * load) * coulomb_unit(v, k)
            + fv * v
            + (dfs + ls * load) * stribeck_shape(v, vs)
        )

    lo = [0.0, 0.0, 1.0 if fit_k else K_MAX - 1e-6, 0.0, 0.0, 0.0, math.log(0.005)]
    hi = [
        20.0,
        1.0 if use_load else 1e-9,
        K_MAX,
        20.0,
        20.0,
        1.0 if use_load else 1e-9,
        math.log(1.0),
    ]
    scale = float(np.median(np.abs(h))) or 0.1
    best = None
    for vs0 in (0.03, 0.08, 0.2):
        x0 = [scale, 0.0, K_MAX, 0.0, 0.3 * scale, 0.0, math.log(vs0)]
        x0 = np.clip(x0, lo, hi)
        res = least_squares(
            lambda x: model(x) - h,
            x0,
            bounds=(lo, hi),
            loss="soft_l1",
            f_scale=0.2 * scale,
        )
        if best is None or res.cost < best.cost:
            best = res
    assert best is not None
    fc, fl, k, fv, dfs, ls, lvs = (float(x) for x in best.x)
    if not fit_k:
        k = K_MAX  # pinned (the bound is K_MAX - 1e-6 to keep it feasible)
    resid = model(best.x) - h
    ss = float(np.sum((h - h.mean()) ** 2)) or 1e-12
    fo = 0.0
    if gravity_residual is not None:
        fo = float(np.mean([gravity_residual(s) for s in samples]))
    return FrictionFit(
        fc=fc,
        fl=fl if use_load else 0.0,
        k=k,
        fv=fv,
        fo=fo,
        dfs=dfs,
        ls=ls if use_load else 0.0,
        vs=math.exp(lvs),
        rms=float(np.sqrt(np.mean(resid**2))),
        r2=1.0 - float(np.sum(resid**2)) / ss,
        n=len(samples),
        load_span=span,
        speeds=speeds,
    )


def speed_table(fit: FrictionFit, samples: Iterable[Sample]) -> list[dict[str, float]]:
    """Per speed: measured mean friction, the fit, and the old runtime curve
    shape at the same load — the scorecard ``tune.friction`` prints."""
    rows = []
    samples = list(samples)
    for sp in sorted({s.speed for s in samples}):
        sel = [s for s in samples if s.speed == sp]
        load = np.array([s.load for s in sel])
        h = np.array([s.halfdiff for s in sel])
        rows.append(
            {
                "speed_deg_s": math.degrees(sp),
                "n": len(sel),
                "load_nm": float(load.mean()),
                "measured_nm": float(h.mean()),
                "fit_nm": float(fit.halfdiff(np.full(len(sel), sp), load).mean()),
            }
        )
    return rows


def read_raw_csv(paths: Iterable[Path]) -> dict[str, np.ndarray]:
    """Rows of one or more ``--raw-csv`` files (``joint, side, pass, v_rad_s,
    direction, q_rad, tau_nm`` and optionally ``load_nm``) as arrays; each
    file's passes get their own group."""
    cols: dict[str, list] = {
        "speed": [],
        "direction": [],
        "q": [],
        "tau": [],
        "group": [],
        "load": [],
    }
    joint = side = None
    for fi, path in enumerate(paths):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                joint = joint or row.get("joint")
                side = side or row.get("side")
                cols["speed"].append(abs(float(row["v_rad_s"])))
                cols["direction"].append(row["direction"])
                cols["q"].append(float(row["q_rad"]))
                cols["tau"].append(float(row["tau_nm"]))
                cols["group"].append(fi * 1000 + int(float(row.get("pass") or 0)))
                cols["load"].append(
                    float(row["load_nm"])
                    if row.get("load_nm") not in (None, "")
                    else math.nan
                )
    out = {k: np.asarray(v) for k, v in cols.items() if k != "direction"}
    out["direction"] = np.asarray(cols["direction"])
    out["joint"] = np.asarray(joint or "")
    out["side"] = np.asarray(side or "")
    return out
