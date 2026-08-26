"""Attribute teleop jitter to a pipeline stage from a flight-recorder capture.

Record one jittery session first::

    axol teleop --teleop.record /tmp/jit
    # reproduce the jitter, then exit teleop
    axol diag.teleop-jitter /tmp/jit

The recorder (see ``almond_axol/teleop/recorder.py``) writes three files
sharing one monotonic clock:

    <prefix>_ik.npz    raw VR pose -> filtered pose -> EE target -> IK output
    <prefix>_cmd.npz   segment target -> EMA -> final guarded command
    <prefix>_meas.npz  measured joint positions and torques

For every signal at every stage boundary this script resamples to a uniform
grid, splits the motion into frequency bands, and prints where the
high-frequency (jitter) content first appears or gets amplified — plus any
dominant spectral line (e.g. a 20 Hz beat from the 120 Hz IK / 100 Hz
control-rate seam would show up in ``cmd`` but not in ``ik``).

Analysis is restricted to the engaged window (from the ik file's engage
flags) unless ``--full`` is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ARM_JOINT_NAMES = (
    "shoulder_1",
    "shoulder_2",
    "shoulder_3",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)

# Band edges (Hz): below LOW is intentional motion / drift, LOW..HIGH is the
# structural band the arm responds to ("felt" jitter), above HIGH is noise
# the mechanics mostly filter out.
BAND_LOW = 3.0
BAND_HIGH = 15.0


def _resample(t: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray]:
    """Resample one channel onto a uniform grid at the stream's median rate."""
    dt = float(np.median(np.diff(t)))
    n = int((t[-1] - t[0]) / dt) + 1
    grid = t[0] + dt * np.arange(n)
    return 1.0 / dt, np.interp(grid, t, x)


def _band_rms(t: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float, float]:
    """RMS per band plus the dominant >BAND_LOW spectral peak (freq, amp).

    Returns ``(rms_low, rms_mid, rms_high, peak_freq, peak_rms)`` where the
    bands are [0, BAND_LOW), [BAND_LOW, BAND_HIGH), [BAND_HIGH, fs/2), all in
    the input's units.
    """
    mask = np.isfinite(x)
    if mask.sum() < 64:
        return (np.nan,) * 5
    fs, xr = _resample(t[mask], x[mask])
    xr = xr - xr.mean()
    win = np.hanning(len(xr))
    spec = np.fft.rfft(xr * win)
    freqs = np.fft.rfftfreq(len(xr), 1.0 / fs)
    # Power normalized so that summing bands reproduces the signal variance.
    power = (np.abs(spec) ** 2) / (win**2).sum() * 2.0 / len(xr)

    def band(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi)
        return float(np.sqrt(power[m].sum())) if m.any() else 0.0

    rms_low = band(0.0, BAND_LOW)
    rms_mid = band(BAND_LOW, BAND_HIGH)
    rms_high = band(BAND_HIGH, fs / 2)

    m = freqs >= BAND_LOW
    peak_freq, peak_rms = np.nan, np.nan
    if m.any() and power[m].max() > 0:
        # A "line" must stand clear of the local floor to be worth reporting.
        i = int(np.argmax(power[m]))
        floor = float(np.median(power[m]))
        if power[m][i] > 8.0 * max(floor, 1e-30):
            peak_freq = float(freqs[m][i])
            peak_rms = float(np.sqrt(power[m][i]))
    return rms_low, rms_mid, rms_high, peak_freq, peak_rms


def _print_table(
    title: str,
    t: np.ndarray,
    signals: dict[str, np.ndarray],
    unit_scale: float,
    unit: str,
) -> None:
    """Print band-RMS rows for a set of same-unit channels."""
    print(
        f"\n{title}  ({unit}, bands: <{BAND_LOW:g} / "
        f"{BAND_LOW:g}-{BAND_HIGH:g} / >{BAND_HIGH:g} Hz)"
    )
    print(f"  {'channel':<22} {'motion':>9} {'JITTER':>9} {'hf-noise':>9}   peak")
    for name, x in signals.items():
        lo, mid, hi, pf, pa = _band_rms(t, x)
        if not np.isfinite(lo):
            print(f"  {name:<22} {'—':>9}")
            continue
        peak = f"{pf:5.1f} Hz ({pa * unit_scale:.3g})" if np.isfinite(pf) else ""
        print(
            f"  {name:<22} {lo * unit_scale:9.3f} {mid * unit_scale:9.3f} "
            f"{hi * unit_scale:9.3f}   {peak}"
        )


def _load(prefix: str, stage: str) -> dict[str, np.ndarray] | None:
    p = Path(f"{prefix}_{stage}.npz")
    if not p.exists():
        print(f"  ({p} not found — skipping {stage} stage)")
        return None
    return dict(np.load(p))


def _engaged_window(ik: dict[str, np.ndarray] | None) -> tuple[float, float] | None:
    """Time span of the longest fully-engaged stretch in the ik capture."""
    if ik is None:
        return None
    engaged = ik["engaged"].max(axis=1) > 0.5
    if not engaged.any():
        return None
    # Longest contiguous run of engaged ticks.
    edges = np.diff(engaged.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if engaged[0]:
        starts.insert(0, 0)
    if engaged[-1]:
        ends.append(len(engaged))
    runs = sorted(zip(starts, ends), key=lambda se: se[1] - se[0])
    s, e = runs[-1]
    return float(ik["t"][s]), float(ik["t"][e - 1])


def _clip(data: dict[str, np.ndarray], span: tuple[float, float]) -> dict:
    m = (data["t"] >= span[0]) & (data["t"] <= span[1])
    return {k: v[m] for k, v in data.items()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="axol diag.teleop-jitter")
    ap.add_argument("prefix", help="the --teleop.record prefix used during capture")
    ap.add_argument(
        "--full",
        action="store_true",
        help="analyze the whole capture instead of the longest engaged window",
    )
    ap.add_argument(
        "--joints",
        nargs="*",
        default=None,
        help="restrict joint tables to these names (e.g. elbow wrist_2)",
    )
    args = ap.parse_args(argv)

    ik = _load(args.prefix, "ik")
    cmd = _load(args.prefix, "cmd")
    meas = _load(args.prefix, "meas")

    span = None if args.full else _engaged_window(ik)
    if span is not None:
        print(f"analysis window: engaged for {span[1] - span[0]:.1f} s")
        ik = _clip(ik, span) if ik is not None else None
        cmd = _clip(cmd, span) if cmd is not None else None
        meas = _clip(meas, span) if meas is not None else None
    else:
        print("analysis window: full capture")

    def joint_ok(name: str) -> bool:
        return args.joints is None or any(j in name for j in args.joints)

    # -- Cartesian stages (mm): raw VR -> filtered -> EE target ------------
    if ik is not None and len(ik["t"]) > 64:
        rate = 1.0 / np.median(np.diff(ik["t"]))
        print(f"\nIK stream: {len(ik['t'])} rows at ~{rate:.0f} Hz")
        for side in ("l", "r"):
            sigs = {}
            for stage in ("raw", "filt", "tgt"):
                for ai, ax in enumerate("xyz"):
                    sigs[f"{stage}_{side}.{ax}"] = ik[f"{stage}_{side}"][:, ai]
            _print_table(
                f"Cartesian stages — {'left' if side == 'l' else 'right'} hand",
                ik["t"],
                sigs,
                1e3,
                "mm RMS",
            )

    # -- Joint-space stages (mdeg): IK q -> tgt -> ema -> out -> measured --
    for arm_i, arm in enumerate(("left", "right")):
        sigs: dict[str, np.ndarray] = {}
        ts: dict[str, np.ndarray] = {}
        for ji, jn in enumerate(ARM_JOINT_NAMES):
            if not joint_ok(jn):
                continue
            if ik is not None:
                sigs[f"ik_q    {jn}"] = ik["q"][:, arm_i * 7 + ji]
                ts[f"ik_q    {jn}"] = ik["t"]
            if cmd is not None:
                for stage in ("tgt", "ema", "out"):
                    sigs[f"{stage:<7} {jn}"] = cmd[stage][:, arm_i * 7 + ji]
                    ts[f"{stage:<7} {jn}"] = cmd["t"]
            if meas is not None:
                sigs[f"meas    {jn}"] = meas["qm"][:, arm_i * 8 + ji]
                ts[f"meas    {jn}"] = meas["t"]
        if not sigs:
            continue
        print(f"\n===== {arm} arm, joint-space stages (mdeg RMS) =====")
        print(
            f"  {'stage/joint':<22} {'motion':>9} {'JITTER':>9} {'hf-noise':>9}   peak"
        )
        for name, x in sigs.items():
            lo, mid, hi, pf, pa = _band_rms(ts[name], np.degrees(x))
            if not np.isfinite(lo):
                print(f"  {name:<22} {'—':>9}")
                continue
            peak = f"{pf:5.1f} Hz ({pa * 1e3:.3g})" if np.isfinite(pf) else ""
            print(
                f"  {name:<22} {lo * 1e3:9.1f} {mid * 1e3:9.1f} "
                f"{hi * 1e3:9.1f}   {peak}"
            )

    # -- Measured torque (Nm) ----------------------------------------------
    if meas is not None and len(meas["t"]) > 64:
        for arm_i, arm in enumerate(("left", "right")):
            sigs = {
                jn: meas["tq"][:, arm_i * 8 + ji]
                for ji, jn in enumerate(ARM_JOINT_NAMES)
                if joint_ok(jn)
            }
            if sigs:
                _print_table(
                    f"Measured torque — {arm} arm", meas["t"], sigs, 1.0, "Nm RMS"
                )

    print(
        "\nReading the tables: the JITTER column is the 3-15 Hz band the arm"
        "\nphysically responds to. Walk each joint down its stage list — the"
        "\nfirst stage where JITTER jumps well above the previous stage is"
        "\nwhere the noise enters (raw->filt: VR/filter; filt->ik_q: solver;"
        "\ntgt->ema->out: smoothing; out->meas: the arm itself, i.e. control/"
        "\nPID territory). A sharp 'peak' line present in cmd but not ik"
        "\nsuggests the 120/100 Hz rate seam; one present only in meas is a"
        "\nstructural or control-loop resonance."
    )


if __name__ == "__main__":
    main()
