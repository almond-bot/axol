"""
axol motion.build / motion.list

Build and inspect the committed reference motions used by ``axol
tune.motion``.

``motion.build`` postprocesses a teleop flight-recorder capture (``axol
teleop --teleop.jitter_record PREFIX``) into a reference motion: the final
guarded command stream is clipped to the engaged span, resampled onto a
uniform grid, zero-phase smoothed (keeping the operator's intent, dropping
tremor and network jitter), and projected waypoint-by-waypoint through the
collision-aware solver so the stored motion is joint-limit- and
self-collision-safe by construction. The result lands in the package's
``almond_axol/tuning/motions/`` directory — commit it so every robot can
replay the identical motion.

Examples:
    axol teleop --teleop.jitter_record /tmp/rec     # record a session first
    axol motion.build /tmp/rec --name reach-and-place
    axol motion.build /tmp/rec --name reach-slow --time-scale 2.0
    axol motion.list
"""

from __future__ import annotations

import argparse
import math


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``motion.build`` and ``motion.list`` subcommands."""
    b = subparsers.add_parser(
        "motion.build",
        help="Build a committed reference motion from a teleop flight recording.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    b.add_argument(
        "prefix",
        help="Flight-recorder prefix used with --teleop.jitter_record "
        "(reads <prefix>_cmd.npz and, if present, <prefix>_ik.npz)",
    )
    b.add_argument(
        "--name",
        required=True,
        help="Motion name; the file is written to the package's committed "
        "motions directory as <name>.npz (use --out for another location)",
    )
    b.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Write the motion to an explicit path instead of the committed "
        "motions directory",
    )
    b.add_argument(
        "--rate",
        type=float,
        default=100.0,
        help="Uniform playback rate in Hz (default: 100)",
    )
    b.add_argument(
        "--cutoff",
        type=float,
        default=6.0,
        metavar="HZ",
        help="Zero-phase low-pass cutoff (Hz) for the smoothing pass "
        "(default: 6.0 — keeps deliberate motion, drops tremor/jitter)",
    )
    b.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        metavar="X",
        help="Stretch playback time by this factor (2.0 = half speed; default: 1.0)",
    )
    b.add_argument(
        "--no-project",
        action="store_true",
        help="Skip the collision-aware waypoint projection (faster; only "
        "for captures known to stay clear of limits and the torso)",
    )
    b.add_argument(
        "--notes",
        default="",
        help="Free-form provenance note stored in the motion metadata",
    )
    b.set_defaults(func=run_build)

    ls = subparsers.add_parser(
        "motion.list",
        help="List the committed reference motions.",
    )
    ls.set_defaults(func=run_list)


def run_build(args: argparse.Namespace) -> None:
    """Build a reference motion from a flight-recorder capture."""
    from pathlib import Path

    from ..tuning.motion import build_motion, save_motion

    print(f"Building reference motion {args.name!r} from {args.prefix} ...")
    motion = build_motion(
        args.prefix,
        args.name,
        rate=args.rate,
        smooth_cutoff_hz=args.cutoff,
        time_scale=args.time_scale,
        collision_project=not args.no_project,
        notes=args.notes,
    )
    path = save_motion(motion, Path(args.out) if args.out else None)
    print(f"Wrote {path}")
    if args.out is None:
        print("Commit it so every robot can replay the identical motion.")


def run_list(args: argparse.Namespace) -> None:
    """List the committed reference motions."""
    from ..tuning.motion import list_motions

    motions = list_motions()
    if not motions:
        print("No committed reference motions (see axol motion.build --help).")
        return
    print(f"{'name':<24} {'dur':>6}  {'rate':>5}  {'peak vel':>8}  source")
    for m in motions:
        peak = math.degrees(float(m.peak_velocity().max()))
        print(
            f"{m.name:<24} {m.duration:>5.1f}s  {m.rate:>4.0f}Hz  "
            f"{peak:>6.0f}°/s  {m.meta.get('source', '?')}"
        )
