"""What a ``tune.motion --learn`` correction says the feedforward is missing.

For each learned joint of a saved pass (its ``correction`` series), regress
the learned torque (kp × offset) on candidate feedforward terms of the
commanded motion — Coulomb, the Stribeck excess, viscous, inertia and
load-proportional Coulomb — band-limited like the learning was
(:func:`almond_axol.tuning.learning.explain_correction`). The coefficients
are what each runtime parameter would have to add; R² is how much of the
learned correction they account for — what a model-based feedforward can
take over from the motion-specific correction.

Usage:
    uv run python scripts/explain_learned.py RUN_ID [--runs-dir DIR] [--kp right.shoulder_1=450 ...]
"""

from __future__ import annotations

import argparse

import numpy as np

from almond_axol.constants import ARM_JOINTS
from almond_axol.robot.config import AxolConfig
from almond_axol.robot.gravity import GravityCompensator
from almond_axol.tuning.learning import LEARN_BAND, explain_correction
from almond_axol.tuning.runs import TUNING_RUNS_DIR, load_run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run")
    p.add_argument("--runs-dir", default=str(TUNING_RUNS_DIR))
    p.add_argument("--band", type=float, nargs=2, default=list(LEARN_BAND))
    p.add_argument(
        "--vs",
        type=float,
        default=0.1,
        help="Stribeck speed for the excess term (rad/s)",
    )
    args = p.parse_args()

    from pathlib import Path

    loaded = load_run(args.run, Path(args.runs_dir))
    if loaded is None:
        raise SystemExit(f"no run {args.run}")
    meta, series = loaded
    if "correction" not in series:
        raise SystemExit(f"run {args.run} has no learned correction")
    cols = meta["params"]["columns"]
    t = series["t"]
    fs = (len(t) - 1) / (t[-1] - t[0])
    corr = series["correction"].astype(float)
    ref = series["target"].astype(float)
    gains = meta.get("gains") or {}
    cfg = AxolConfig().resolved()
    gc = GravityCompensator()
    print(
        f"{args.run} ({meta.get('label')}): band {args.band[0]:g}-{args.band[1]:g} Hz"
    )
    for i, name in enumerate(cols):
        if not np.any(corr[:, i]):
            continue
        side, joint = name.split(".")
        kp = float(gains.get(f"{name}.kp", getattr(getattr(cfg, side), joint).kp))
        v = np.gradient(ref[:, i]) * fs
        a = np.gradient(v) * fs
        base = cols.index(f"{side}.{ARM_JOINTS[0].value}")
        arm = ref[:, base : base + 7].astype(np.float32)
        g = np.array(
            [gc.gravity_arm(q, is_left=side == "left")[i - base] for q in arm[::4]]
        )
        g = np.interp(np.arange(len(ref)), np.arange(0, len(ref), 4), g)
        coef, r2 = explain_correction(corr[:, i], kp, v, a, fs, tuple(args.band), g)
        rms = kp * float(np.std(corr[:, i]))
        print(
            f"\n  {name}: learned torque {rms:.3f} Nm RMS (kp {kp:g}), terms explain {r2 * 100:.0f}%"
        )
        for k, c in coef.items():
            print(f"    {k:24s} {c:+.4f}")


if __name__ == "__main__":
    main()
