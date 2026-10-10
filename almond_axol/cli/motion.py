"""
axol motion.build / motion.list

Build and inspect the reference motions used by ``axol tune.motion``
(besides the built-in ``slow_osc``, ``fast_swing`` and ``hold``).

``motion.build`` postprocesses a flight-recorder capture into a reference
motion. The capture comes from either recorder: a teleoperated session
(``axol teleop --teleop.record PREFIX``, whose guarded command stream is
clipped to the engaged span) or a hand-guided gravity-comp session (``axol
gravity-comp --record PREFIX``, whose measured joint stream is trimmed of
its still lead-in/lead-out). Either way the stream is resampled onto a
uniform grid, zero-phase smoothed (keeping the operator's intent, dropping
tremor and network jitter), and projected waypoint-by-waypoint through the
collision-aware solver so the stored motion is joint-limit- and
self-collision-safe by construction. The result lands in
``~/.almond/motions/`` on the robot, where ``tune.motion`` finds it by name.

Examples:
    axol teleop --teleop.record rec1        # record via teleop, or ...
    axol gravity-comp --record rec1         # ... by hand-guiding the arms
    axol motion.build                       # newest recording, named after it
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
        help="Build a reference motion from a recorded session "
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
        default=None,
        help="Motion name; the file is written to ~/.almond/motions/ as "
        "<name>.npz (use --out for another location). "
        "Defaults to the recording's name.",
    )
    b.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Write the motion to an explicit path instead of ~/.almond/motions/",
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

    c = subparsers.add_parser(
        "motion.chirp",
        help="Build a one-joint sine-sweep motion for identifying its "
        "tracking dynamics (replay it with tune.motion, fit with tune.tf).",
    )
    c.add_argument("joint", metavar="SIDE.JOINT", help="e.g. right.shoulder_1")
    c.add_argument(
        "--base",
        default=None,
        help="Motion whose first pose the sweep is centred on (default: the "
        "built-in start pose, shoulder_1 and the elbow raised)",
    )
    c.add_argument(
        "--f0", type=float, default=0.3, help="Start frequency, Hz (default 0.3)"
    )
    c.add_argument(
        "--f1", type=float, default=8.0, help="End frequency, Hz (default 8)"
    )
    c.add_argument(
        "--duration", type=float, default=60.0, help="Sweep length, s (default 60)"
    )
    c.add_argument(
        "--amp-deg",
        type=float,
        default=1.5,
        help="Low-frequency amplitude, deg (default 1.5)",
    )
    c.add_argument(
        "--acc-max",
        type=float,
        default=200.0,
        metavar="DEG_S2",
        help="Amplitude cap by acceleration, deg/s² (default 200: 1.5° to "
        "1.8 Hz, 0.08° at 8 Hz)",
    )
    c.add_argument(
        "--carrier",
        type=float,
        default=0.0,
        metavar="DEG_S",
        help="Ride a triangle wave of this speed (over ±7.5°) so the joint "
        "slides one way for seconds at a time and friction linearises "
        "(default 0: a bare sweep)",
    )
    c.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Output path (default ~/.almond/motions/chirp_<side>_<joint>.npz)",
    )
    c.set_defaults(func=run_chirp)

    ls = subparsers.add_parser(
        "motion.list",
        help="List the reference motions: the built-ins and ~/.almond/motions/.",
    )
    ls.set_defaults(func=run_list)


def run_chirp(args: argparse.Namespace) -> None:
    """Write a one-joint chirp motion (see ``tuning.tracking_model``)."""
    import math
    from pathlib import Path

    from ..constants import ARM_JOINTS, Joint
    from ..robot.axol import arm_limits
    from ..tuning.motion import MOTIONS_DIR, load_motion, save_motion, start_pose
    from ..tuning.tracking_model import chirp_motion

    side, _, joint = args.joint.partition(".")
    names = [j.value for j in ARM_JOINTS]
    if side not in ("left", "right") or joint not in names:
        raise SystemExit(f"motion.chirp wants SIDE.JOINT, got {args.joint!r}")
    col = (0 if side == "left" else 7) + names.index(joint)
    base = load_motion(args.base).q[0].astype(float) if args.base else start_pose()
    motion = chirp_motion(
        base,
        col,
        f0=args.f0,
        f1=args.f1,
        duration=args.duration,
        amp_rad=math.radians(args.amp_deg),
        acc_max=math.radians(args.acc_max),
        carrier_rad_s=math.radians(args.carrier),
        name=f"chirp_{side}_{joint}",
    )
    lo, hi = arm_limits(Joint(joint), side == "left")
    x = motion.q[:, col]
    if x.min() < lo or x.max() > hi:
        raise SystemExit(
            f"motion.chirp: the sweep spans {math.degrees(x.min()):.1f}..."
            f"{math.degrees(x.max()):.1f}°, outside {joint}'s "
            f"{math.degrees(lo):.0f}..{math.degrees(hi):.0f}° — pick another --base"
        )
    out = Path(args.out) if args.out else MOTIONS_DIR / f"{motion.name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_motion(motion, out)
    print(
        f"Wrote {out}: {args.joint} sweep {args.f0:g}-{args.f1:g} Hz over "
        f"{args.duration:g} s, {math.degrees(x.min() - base[col]):+.2f}..."
        f"{math.degrees(x.max() - base[col]):+.2f}° about "
        f"{math.degrees(base[col]):+.1f}° ({motion.duration:.0f} s total)\n"
        f"Replay: axol tune.motion --motion {out} --arms {side} ... "
        f"--label chirp\nFit:    axol tune.tf {args.joint} <run_id>"
    )


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
    # The motion inherits the recording's name unless --name says otherwise,
    # so record -> build needs no naming step at all.
    name = args.name or Path(prefix).name
    print(f"Building reference motion {name!r} from {prefix} ...")
    motion, raw = build_motion(
        prefix,
        name,
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
    """List the reference motions: the built-ins and ``~/.almond/motions/``."""
    from ..tuning.motion import MOTIONS_DIR, list_motions

    motions = list_motions()
    if not motions:
        print(f"No reference motions in {MOTIONS_DIR} (see axol motion.build --help).")
        return
    print(f"{'name':<24} {'dur':>6}  {'rate':>5}  {'peak vel':>8}  source")
    for m in motions:
        peak = math.degrees(float(m.peak_velocity().max()))
        print(
            f"{m.name:<24} {m.duration:>5.1f}s  {m.rate:>4.0f}Hz  "
            f"{peak:>6.0f}°/s  {m.meta.get('source', '?')}"
        )
