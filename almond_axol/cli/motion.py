"""
axol motion.build / motion.list

Build and inspect the committed reference motions used by ``axol
tune.motion``.

``motion.build`` postprocesses a flight-recorder capture into a reference
motion. The capture comes from either recorder: a teleoperated session
(``axol teleop --teleop.record PREFIX``, whose guarded command stream is
clipped to the engaged span) or a hand-guided gravity-comp session (``axol
gravity-comp --record PREFIX``, whose measured joint stream is trimmed of
its still lead-in/lead-out). Either way the stream is resampled onto a
uniform grid, zero-phase smoothed (keeping the operator's intent, dropping
tremor and network jitter), and projected waypoint-by-waypoint through the
collision-aware solver so the stored motion is joint-limit- and
self-collision-safe by construction. The result lands in the package's
``almond_axol/tuning/motions/`` directory — commit it so every robot can
replay the identical motion.

Examples:
    axol teleop --teleop.record rec1        # record via teleop, or ...
    axol gravity-comp --record rec1         # ... by hand-guiding the arms
    axol motion.build --name reach-and-place       # newest recording
    axol motion.build rec1 --name reach-slow --time-scale 2.0
    axol motion.list
"""

from __future__ import annotations

import argparse
import math


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``motion.build`` and ``motion.list`` subcommands."""
    b = subparsers.add_parser(
        "motion.build",
        help="Build a committed reference motion from a recorded session "
        "(teleop or gravity-comp).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    b.add_argument(
        "prefix",
        nargs="?",
        default=None,
        help="Flight-recorder prefix used with teleop's --teleop.record "
        "(reads <prefix>_cmd.npz) or gravity-comp's --record (reads "
        "<prefix>_gc.npz). A bare name resolves in the recordings "
        "directory (~/.almond/recordings/); omit it entirely to build from "
        "the newest recording there.",
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
        default=240.0,
        help="Uniform playback rate in Hz (default: 240 — the production "
        "control rate; tune.motion replays at the motion's stored rate)",
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


def _resolve_prefix(prefix: str | None) -> str:
    """Resolve the recording prefix: verbatim path, bare name, or newest.

    The newest recording may be from either recorder — teleop (``_cmd``)
    or gravity comp (``_gc``).
    """
    from ..teleop.recorder import resolve_or_latest

    return resolve_or_latest(prefix, stage=("cmd", "gc"))


def run_build(args: argparse.Namespace) -> None:
    """Build a reference motion from a flight-recorder capture."""
    from pathlib import Path

    from ..tuning.motion import build_motion, save_motion

    prefix = _resolve_prefix(args.prefix)
    print(f"Building reference motion {args.name!r} from {prefix} ...")
    motion, raw = build_motion(
        prefix,
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
    _save_build_run(args, prefix, motion, raw)


def _save_build_run(args: argparse.Namespace, prefix: str, motion, raw) -> None:
    """Persist a before/after tuning-run artifact for the diagnostics UI.

    The recorded (clipped raw command) stream and the built motion are stored
    side by side per joint so the smoothing + projection passes can be judged
    visually — the same per-joint charts (zoom / fullscreen) the other tuning
    runs get.
    """
    import math as _math

    import numpy as np

    from ..constants import ARM_JOINTS
    from ..tuning import save_run

    columns = [f"left.{j.value}" for j in ARM_JOINTS] + [
        f"right.{j.value}" for j in ARM_JOINTS
    ]
    t_built = motion.times()
    t_raw = np.asarray(raw["t"], dtype=float)
    q_raw = np.asarray(raw["q"], dtype=float)

    # Deviation of the built motion from the recording, evaluated on the raw
    # timestamps — what the smoothing + projection actually changed.
    per_joint: dict[str, dict[str, float]] = {}
    for i, name in enumerate(columns):
        if float(np.ptp(q_raw[:, i])) < _math.radians(1.0):
            continue
        built_at_raw = np.interp(t_raw, t_built, motion.q[:, i])
        dev = built_at_raw - q_raw[:, i]
        vel_raw = np.abs(np.diff(q_raw[:, i]) / np.maximum(np.diff(t_raw), 1e-9))
        vel_built = np.abs(np.diff(motion.q[:, i])) * motion.rate
        per_joint[name] = {
            "dev_rms_deg": _math.degrees(float(np.sqrt(np.mean(dev**2)))),
            "dev_max_deg": _math.degrees(float(np.max(np.abs(dev)))),
            "peak_vel_raw_dps": _math.degrees(float(vel_raw.max(initial=0.0))),
            "peak_vel_built_dps": _math.degrees(float(vel_built.max(initial=0.0))),
        }
    metrics = {
        "per_joint": per_joint,
        "waypoints": int(len(motion.q)),
        "duration_s": float(motion.duration),
        "peak_vel_built_dps": _math.degrees(float(motion.peak_velocity().max())),
        # The headline number: the largest single change the postprocessing
        # made to any moving joint.
        "dev_max_deg": max(
            (m["dev_max_deg"] for m in per_joint.values()), default=None
        ),
        "worst_joint": max(
            per_joint, key=lambda k: per_joint[k]["dev_max_deg"], default=None
        ),
    }
    run_id = save_run(
        "build",
        {
            "t": t_built,
            "built": np.asarray(motion.q, dtype=np.float32),
            "t_raw": t_raw,
            "raw": q_raw.astype(np.float32),
        },
        metrics,
        params={
            "name": motion.name,
            "prefix": str(prefix),
            "source_kind": motion.meta.get("source_kind"),
            "rate": float(motion.rate),
            "cutoff": float(args.cutoff),
            "time_scale": float(args.time_scale),
            "projected": not args.no_project,
            "columns": columns,
        },
        label=args.notes or None,
    )
    print(f"Saved tuning run {run_id} (kind=build) — before/after in the UI.")


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
