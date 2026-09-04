"""
axol tune.filter

Test the teleop filter stack by injecting noise into a clean motion — no
hardware, no VR headset, exactly reproducible.

The production pipeline is::

    VR stream -> pose low-pass -> IK solver -> EMA -> trapezoid
       ^ network noise enters here   ^ IK noise is created here

``--noise`` picks the source under test, and each source is injected at its
real entry point, which is what lets you test them independently:

* ``network`` — transport artifacts in front of the pose low-pass: white
  **jitter**, teleported **outlier** samples, and **stalls** where the
  stream freezes then jumps to catch up. The whole stack gets to clean them.
* ``ik`` — solver artifacts between the low-pass and the EMA (where the
  solver sits in production): band-limited 3-20 Hz per-joint **churn** and
  persistent **jumps** (redundancy flips). Only the EMA and the trapezoid
  can see these — the pose filter never gets a chance, exactly as on the
  real robot.
* ``combined`` (default) — both at once, each at its own injection point.

The output is scored against the *clean* reference, per joint: the stack
should track the intentional motion (low RMS error and lag) while removing
what was injected (jitter pass-through well below 1, peak error far below
the outlier magnitude, and output acceleration always inside teleop's
configured limit no matter how hard the corrupted input slams).

Everything is deterministic for a given ``--seed``, and the two noise
streams are seeded independently — the same seed gives the same network
noise with or without IK noise on top, so mode-to-mode comparisons are
apples to apples.

Examples:
    axol tune.filter --noise network --save-run
    axol tune.filter --noise ik --save-run
    axol tune.filter --save-run                          # combined
    axol tune.filter --noise network --stall-ms 300 --save-run
    axol tune.filter --noise ik --ik-churn 0.6 --save-run
    axol tune.filter --motion reach-and-place --save-run
    axol tune.filter --cutoff 1.5 --label "half cutoff" --save-run
"""

from __future__ import annotations

import argparse
import math

from ...tuning import save_run
from ...tuning.filtering import filter_noise_analysis


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``tune.filter`` subcommand."""
    p = subparsers.add_parser(
        "tune.filter",
        help="Inject stalls/outliers/jitter into a clean motion and score "
        "how much the teleop filter stack removes (offline, no hardware).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--motion",
        default=None,
        help="Committed reference motion to use as the clean signal "
        "(axol motion.list); default is a synthetic sine",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Sine mode: signal length in seconds (default: 10)",
    )
    p.add_argument(
        "--amp",
        type=float,
        default=15.0,
        metavar="DEG",
        help="Sine mode: amplitude in degrees (default: 15)",
    )
    p.add_argument(
        "--freq",
        type=float,
        default=0.5,
        help="Sine mode: frequency in Hz (default: 0.5)",
    )
    p.add_argument(
        "--noise",
        choices=("network", "ik", "combined"),
        default="combined",
        help="Noise source under test, injected at its real pipeline entry "
        "point: 'network' (jitter/outliers/stalls before the pose low-pass), "
        "'ik' (solver churn/jumps after it), or 'combined' (both; default)",
    )
    p.add_argument(
        "--jitter",
        type=float,
        default=0.3,
        metavar="DEG",
        help="Network: white-noise jitter RMS in degrees added to every "
        "sample (default: 0.3; 0 disables)",
    )
    p.add_argument(
        "--outlier-rate",
        type=float,
        default=0.5,
        help="Outlier samples injected per second (default: 0.5; 0 disables)",
    )
    p.add_argument(
        "--outlier-amp",
        type=float,
        default=10.0,
        metavar="DEG",
        help="Outlier magnitude in degrees — how far a glitched sample "
        "teleports (default: 10)",
    )
    p.add_argument(
        "--stall-rate",
        type=float,
        default=0.5,
        help="Stream stalls injected per second (default: 0.5; 0 disables)",
    )
    p.add_argument(
        "--stall-ms",
        type=float,
        default=150.0,
        help="Stall length in ms — the stream freezes on its last sample "
        "for this long, then jumps to catch up (default: 150)",
    )
    p.add_argument(
        "--ik-churn",
        type=float,
        default=0.2,
        metavar="DEG",
        help="IK: band-limited (3-20 Hz) solver churn RMS in degrees per "
        "joint (default: 0.2, the scale diag.offline kinematics measures "
        "on a healthy solve; 0 disables)",
    )
    p.add_argument(
        "--ik-jump-rate",
        type=float,
        default=0.2,
        help="IK: solution jumps injected per second — one joint steps and "
        "holds for 0.3-1 s, a redundancy flip (default: 0.2; 0 disables)",
    )
    p.add_argument(
        "--ik-jump-amp",
        type=float,
        default=3.0,
        metavar="DEG",
        help="IK: solution-jump magnitude in degrees (default: 3)",
    )
    p.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Pose low-pass pole frequency in Hz (default: the production "
        "pose_cutoff from VRTeleopConfig)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the injected noise — identical seed, identical "
        "corrupted stream (default: 0)",
    )
    p.add_argument(
        "--label",
        default=None,
        help="Free-form note stored on the run artifact (shows up in "
        "listings and the UI)",
    )
    p.add_argument(
        "--save-run",
        action="store_true",
        help="Persist the full time series and scores as a tuning-run "
        "artifact (~/.almond/diagnostics/tuning/) for the diagnostics UI",
    )
    p.set_defaults(func=run)


def _print_report(metrics: dict, params: dict) -> None:
    """Cleanup scorecard, one row per scored channel."""
    per_joint: dict[str, dict[str, float]] = metrics["per_joint"]
    parts = []
    if params["noise"] in ("network", "combined"):
        parts.append(
            f"network: jitter {math.degrees(params['jitter_rms']):.2f}° RMS, "
            f"{metrics['outliers']} outliers × "
            f"{math.degrees(params['outlier_amp']):.1f}°, "
            f"{metrics['stalls']} stalls × {params['stall_ms']:.0f} ms"
        )
    if params["noise"] in ("ik", "combined"):
        parts.append(
            f"ik: churn {math.degrees(params['ik_churn']):.2f}° RMS, "
            f"{metrics['ik_jumps']} jumps × "
            f"{math.degrees(params['ik_jump_amp']):.1f}°"
        )
    print(
        f"\nInjected [{params['noise']}]: " + "; ".join(parts) + " "
        f"(seed {params['seed']}, cutoff {params['cutoff']:.2f} Hz)"
    )
    print(f"{'═' * 78}")
    print(
        f"  {'channel':<18} {'in RMS °':>8} {'out RMS °':>9} {'lagfree °':>9} "
        f"{'lag ms':>7} {'jitter ×':>8} {'peak °':>7} {'accel':>7}"
    )
    for name, m in per_joint.items():
        jp = f"{m['jitter_passed']:.2f}" if math.isfinite(m["jitter_passed"]) else "-"
        lag = f"{m['lag_ms']:.0f}" if math.isfinite(m["lag_ms"]) else "-"
        print(
            f"  {name:<18} {math.degrees(m['input_rms']):>8.3f} "
            f"{math.degrees(m['rms_err']):>9.3f} "
            f"{math.degrees(m['rms_err_lagfree']):>9.3f} {lag:>7} {jp:>8} "
            f"{math.degrees(m['peak_err']):>7.3f} "
            f"{math.degrees(m['accel_peak']):>7.0f}"
        )
    print(f"{'═' * 78}")
    print(
        "  in/out RMS = error vs the clean reference before/after the stack;\n"
        "  lagfree = out RMS after removing the stack's delay (the residual\n"
        "  the noise actually left); jitter × = 3-15 Hz error passed through\n"
        "  (<1 = cleaned); peak = worst excursion; accel = peak output accel\n"
        f"  in °/s² (must stay under the {math.degrees(metrics['accel_limit']):.0f} "
        "teleop limit — outliers and\n"
        "  stall catch-ups can't slam the arm). Error during a stall is\n"
        "  missing data, not filter failure — the filter owns the catch-up."
    )


def run(args: argparse.Namespace) -> None:
    """Run the noise-injection filter test and print the cleanup scorecard."""
    try:
        # The CLI speaks degrees; the analysis library (like the rest of the
        # pipeline it replays) works in radians.
        series, metrics, params = filter_noise_analysis(
            motion=args.motion,
            duration=args.duration,
            amp=math.radians(args.amp),
            freq=args.freq,
            noise=args.noise,
            jitter_rms=math.radians(args.jitter),
            outlier_rate=args.outlier_rate,
            outlier_amp=math.radians(args.outlier_amp),
            stall_rate=args.stall_rate,
            stall_ms=args.stall_ms,
            ik_churn=math.radians(args.ik_churn),
            ik_jump_rate=args.ik_jump_rate,
            ik_jump_amp=math.radians(args.ik_jump_amp),
            cutoff=args.cutoff,
            seed=args.seed,
        )
    except FileNotFoundError as exc:
        # load_motion's message already lists the known motions.
        raise SystemExit(str(exc))

    print(
        f"Clean signal: {params['source']} "
        f"({params['duration']:.1f} s, {len(params['columns'])} channel(s))"
    )
    _print_report(metrics, params)

    if args.save_run:
        run_id = save_run("filter", series, metrics, params=params, label=args.label)
        print(f"\nSaved tuning run {run_id} (kind=filter)")
