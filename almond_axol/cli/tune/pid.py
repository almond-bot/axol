"""
axol tune.pid

Tune Kp/Kd for a single Axol joint at ~100 Hz.

Tests one or more (Kp, Kd) candidates — pass several values to either flag to
sweep the whole grid in one session — via sinusoidal or step-response
tracking, measures error (RMS, max, overshoot, settling), ranks the
candidates, and can persist the winner to this robot's calibration file
(``~/.almond/calibration.json``, loaded automatically by ``AxolConfig``).

The feedforward path mirrors the production ``motion_control`` loop —
``gravity + friction + j_eff·a_des`` — so gains found here transfer 1:1 to
teleop. Use ``--ff`` to tune against a reduced feedforward instead (e.g.
``--ff none`` for bare PD, ``--ff gravity`` for gravity-only).

``--pose-by-hand`` turns the session interactive: the arm runs in gravity
comp so you can drag it to any pose by hand, then press Enter to freeze and
step-probe the test joint right there (the other joints hold the pose with
production gains). Repeat at different poses to find where the ring frequency
drops and damping margins thin out — the worst-case pose that rest-pose
tuning never sees.

The runners, metrics, and safety geometry live in ``almond_axol.tuning`` —
this module is the CLI shell: argument parsing, session orchestration, and
result presentation.

Examples:
    axol tune.pid --l --joint elbow                       # config gains, production FF
    axol tune.pid --l --joint elbow --kp 20 30 45 --kd 0.5 1.0   # 6-way sweep
    axol tune.pid --r --joint shoulder_1 --mode step --ff none
    axol tune.pid --l --joint wrist_1 --kp 12 --kd 0.4 --save
    axol tune.pid --l --joint shoulder_1 --mode step --amp 3 --pose-by-hand
"""

import argparse
import asyncio
import csv
import math
import time
import uuid
from pathlib import Path

import numpy as np

from ...constants import ARM_JOINTS
from ...motor import CanBus, ControlMode, Joint, Motor
from ...robot.axol import arm_limits
from ...robot.calibration import CALIBRATION_PATH, update_joint_calibration
from ...robot.config import ArmConfig, AxolConfig, JointConfig
from ...robot.control import (
    DAMP_BP_Q,
    DAMP_BP_W0,
    VEL_CUTOFF_FREQ,
    BandPass,
    Differentiator,
)
from ...robot.gravity import GravityCompensator
from ...tuning import (
    FF_MODES,
    FeedForward,
    HolderMonitor,
    cached_meas,
    cached_torque,
    joint_frame_motors,
    log_to_series,
    make_target_noise,
    ramp_impedance,
    ramp_joints_to,
    ramp_others_to_zero,
    run_sine,
    run_step,
    save_run,
    sine_metrics,
    step_metrics,
)
from ..motor import add_side_and_channel_arguments, resolve_channel


def _print_chatter(m: dict) -> None:
    if m.get("torque_hf") is not None:
        print(f"  Trq chatter: {m['torque_hf']:.4f} Nm RMS (cycle-to-cycle)")
    if m.get("pos_ripple") is not None:
        print(
            f"  Pos ripple:  {math.degrees(m['pos_ripple']) * 1000:.2f} "
            f"millideg RMS (high-frequency)"
        )


def _print_holder_wobble(wobble: dict[str, float]) -> None:
    """One line on how still the non-test joints stayed during the probe.

    The holders run the firmware's own position servo, so under a clean test
    they should barely register above encoder noise (~0.1°). Anything past
    0.5° means the structure was genuinely moving under the reaction torque
    and part of the test joint's ring came from a compliant neighbour — worth
    knowing before blaming kp/kd.
    """
    if not wobble:
        return
    parts = ", ".join(f"{name} {v:.2f}°" for name, v in wobble.items())
    print(f"  Holder peak: {parts}")
    culprits = [name for name, v in wobble.items() if v > 0.5]
    if culprits:
        print(
            f"  ! holders moved >0.5° ({', '.join(culprits)}) — the structure "
            f"was flexing, some of the ring is not the test joint's fault"
        )


def _print_stats_sine(m: dict, n: int, kp: float, kd: float) -> None:
    print(f"\n{'─' * 40}")
    print(f"  Kp={kp}  Kd={kd}")
    print(f"  Samples:    {n}  ({m['hz']:.1f} Hz actual)")
    print(f"  RMS error:  {math.degrees(m['rms']):.3f}°")
    print(f"  Max error:  {math.degrees(m['max']):.3f}°")
    _print_chatter(m)
    print(f"{'─' * 40}")


def _print_stats_step(m: dict, n: int, kp: float, kd: float) -> None:
    settling = (
        f"{m['settling_s'] * 1000:.0f} ms"
        if m["settling_s"] is not None
        else ">hold time"
    )
    print(f"\n{'─' * 40}")
    print(f"  Kp={kp}  Kd={kd}")
    print(f"  Samples:    {n}  ({m['hz']:.1f} Hz actual)")
    print(f"  Settling:   {settling}  (5% threshold)")
    print(
        f"  Overshoot:  {math.degrees(m['overshoot']):.3f}°  "
        f"({m['overshoot_frac'] * 100:.1f}% of step)"
    )
    print(f"  SS RMS:     {math.degrees(m['ss_rms']):.3f}°")
    if m.get("ring_hz") is not None:
        print(f"  Ring:       {m['ring_hz']:.1f} Hz dominant oscillation")
    _print_chatter(m)
    print(f"{'─' * 40}")


def _print_ranking(mode: str, results: list[dict]) -> None:
    """Comparison table over all candidates, best first."""
    ranked = sorted(results, key=lambda r: r["metrics"]["score"])
    print(f"\n{'═' * 64}")
    print(f"  Candidate ranking ({mode}, lower score is better)")
    if mode == "sine":
        print(f"  {'Kp':>7}  {'Kd':>6}  {'RMS °':>8}  {'Max °':>8}  {'score':>8}")
        for r in ranked:
            m = r["metrics"]
            print(
                f"  {r['kp']:>7.1f}  {r['kd']:>6.2f}  "
                f"{math.degrees(m['rms']):>8.3f}  {math.degrees(m['max']):>8.3f}  "
                f"{m['score']:>8.5f}"
            )
    else:
        print(
            f"  {'Kp':>7}  {'Kd':>6}  {'settle':>8}  {'ovsh %':>7}  "
            f"{'SS °':>7}  {'score':>7}"
        )
        for r in ranked:
            m = r["metrics"]
            settling = (
                f"{m['settling_s'] * 1000:.0f}ms"
                if m["settling_s"] is not None
                else ">hold"
            )
            print(
                f"  {r['kp']:>7.1f}  {r['kd']:>6.2f}  {settling:>8}  "
                f"{m['overshoot_frac'] * 100:>7.1f}  "
                f"{math.degrees(m['ss_rms']):>7.3f}  {m['score']:>7.3f}"
            )
    best = ranked[0]
    print(f"\n  Best: Kp={best['kp']}  Kd={best['kd']}")
    print(f"{'═' * 64}")


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``tune.pid`` subcommand."""
    p = subparsers.add_parser(
        "tune.pid",
        help="Tune Kp/Kd for a single Axol joint (sweeps candidate gains).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    add_side_and_channel_arguments(p)
    p.add_argument(
        "--joint",
        required=True,
        choices=[j.value for j in ARM_JOINTS],
        metavar="JOINT",
        help=f"Joint to tune: {', '.join(j.value for j in ARM_JOINTS)}",
    )
    p.add_argument(
        "--kp",
        type=float,
        nargs="+",
        default=None,
        metavar="KP",
        help="Proportional gain candidate(s); several values sweep the grid "
        "(default: this joint's configured kp)",
    )
    p.add_argument(
        "--kd",
        type=float,
        nargs="+",
        default=None,
        metavar="KD",
        help="Derivative gain candidate(s); several values sweep the grid "
        "(default: this joint's configured kd)",
    )
    p.add_argument(
        "--ff",
        choices=list(FF_MODES),
        default="full",
        help="Feedforward during the test: full (gravity + friction + inertia "
        "— matches production; default), gravity, friction, or none (bare PD)",
    )
    p.add_argument(
        "--mode",
        choices=["sine", "step"],
        default="sine",
        help="sine (default): continuous tracking; step: step response",
    )
    p.add_argument(
        "--host-kd",
        type=float,
        default=None,
        metavar="KD",
        help="Host-side damping (Nm·s/rad) applied through t_ff from the "
        "frame-timestamped differentiated measured position — matches the "
        "production kd_host term, needed on the high-inertia shoulders "
        "where firmware kd underdelivers (default: this joint's configured "
        "kd_host)",
    )
    p.add_argument(
        "--host-kd-hz",
        type=float,
        default=None,
        metavar="HZ",
        help="Centre frequency (Hz) of the band-pass confining the host "
        "damping to the joint's resonance band — matches the production "
        "kd_host_hz. Aim it at the ring frequency the step test reports "
        f"(default: this joint's configured value, falling back to the "
        f"shared {DAMP_BP_W0 / (2 * math.pi):.1f} Hz shoulder default)",
    )
    p.add_argument(
        "--host-kd-q",
        type=float,
        default=None,
        metavar="Q",
        help="Quality factor of that band-pass (bandwidth = centre/q) — "
        f"matches the production kd_host_q (default: this joint's configured "
        f"value, falling back to the shared {DAMP_BP_Q}). At the {DAMP_BP_Q} "
        "default the band reaches an octave either side of the centre — into "
        "the <1.5 Hz intentional-motion band when the centre sits low, where "
        "the damping drags the final approach and the step never settles. "
        "When --host-kd-hz is pinned on a measured ring, q of 2-3 confines "
        "the damping to the ring and releases the approach",
    )
    p.add_argument(
        "--stiffness",
        type=float,
        default=None,
        metavar="S",
        help="Test the production gains at stiffness-slider position S in "
        "[0, 1] (kp/kd/j_eff/kd_host all taken from the blended config, "
        "exactly as the robot runs them — e.g. 1.0 matches diag.rom-enable, "
        "0.5 the teleop default). Explicit --kp/--kd/--host-kd still "
        "override. Incompatible with --save (saved gains must be midpoint "
        "anchors).",
    )
    p.add_argument(
        "--pose",
        action="append",
        default=None,
        metavar="JOINT=DEG",
        help="Hold another joint at this joint-frame angle (degrees) during "
        "the test, e.g. --pose shoulder_2=-90 --pose elbow=45. Repeatable. "
        "Gains that are stable at the hanging rest pose can be unstable "
        "where pose inertia and gravity load peak (e.g. the arm raised to "
        "the side), so tune at the worst-case pose. Posed joints are ramped "
        "back to their pre-test positions afterwards.",
    )
    p.add_argument(
        "--hold-host-kd",
        type=float,
        default=None,
        metavar="KD",
        help="[--pose-by-hand] Cap the kd_host used to *hold* the non-test "
        "joints at the frozen pose (default: each joint's config value). "
        "The hold runs production gains, so a config kd_host that is "
        "unstable at the frozen pose will jitter the holders even when the "
        "test joint itself is clean — cap it here to test a fix.",
    )
    p.add_argument(
        "--pose-by-hand",
        action="store_true",
        help="Interactive worst-pose hunting: the whole arm runs in gravity "
        "comp so you can drag it anywhere by hand; press Enter to freeze the "
        "pose, and the tool holds the other joints there (production gains + "
        "kd_host damping) while step-probing the test joint with every "
        "candidate — then drops back to gravity comp for the next pose. "
        "Compare the reported ring frequency and overshoot across poses: "
        "the pose with the lowest ring frequency has the highest reflected "
        "inertia and is where damping margins are thinnest. Requires "
        "--mode step; incompatible with --pose and --save.",
    )
    p.add_argument(
        "--target-noise",
        type=float,
        default=None,
        metavar="DEG",
        help="[sine] Add band-limited (~8 Hz) noise of this RMS amplitude "
        "(degrees) to the reference, simulating teleop hand-tracking jitter "
        "— stresses the differentiated feedforward terms (v_des, a_des, "
        "kd_host) the way production inputs do. Error is still measured "
        "against the clean underlying sine. Try 0.1-0.3.",
    )
    p.add_argument(
        "--amp",
        type=float,
        default=None,
        metavar="DEG",
        help="Motion amplitude in degrees (default: 10°, clamped to joint headroom)",
    )
    p.add_argument(
        "--center",
        type=float,
        default=None,
        metavar="DEG",
        help="Start/centre angle of the probe in joint-frame degrees "
        "(0 = rest). Step frames the step around it; sine centres the wave "
        "there (amplitude clamped to the remaining headroom). Gains that "
        "look fine hanging at rest can misbehave under gravity load, so "
        "probe 45, -45, … too. Default: step starts at the joint's current "
        "position, sine at the joint midpoint. wrist_2 must stay in its "
        "outboard (base-free) half; shoulder_2 additionally 10° outboard of "
        "0 — the base and the chest cameras sit inboard of that. "
        "Incompatible with --pose-by-hand (the probe runs at the hand-set "
        "pose).",
    )
    p.add_argument(
        "--freq", type=float, default=1.0, help="[sine] Frequency in Hz (default: 1.0)"
    )
    p.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="[sine] Duration in seconds (default: 5.0)",
    )
    p.add_argument(
        "--hold",
        type=float,
        default=2.0,
        help="[step] Hold time per phase in seconds (default: 2.0). Also the "
        "length of the unscored settle at the center before the step, so "
        "arrival/gain-change ringing dies out before anything is measured.",
    )
    p.add_argument(
        "--rate", type=float, default=100.0, help="Command rate in Hz (default: 100)"
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Save the best candidate's Kp/Kd to this robot's calibration file "
        f"({CALIBRATION_PATH}); it then overrides the shared defaults on this "
        "machine",
    )
    p.add_argument(
        "--save-run",
        action="store_true",
        help="Persist each candidate's full time series and metrics as a "
        "tuning-run artifact (~/.almond/diagnostics/tuning/) for charting "
        "and A/B comparison in the diagnostics UI",
    )
    p.add_argument(
        "--label",
        default=None,
        help="Free-form note stored on each saved run artifact (shows up in "
        "listings and the diagnostics UI)",
    )
    p.add_argument(
        "--dump-csv",
        nargs="?",
        const="__auto__",
        default=None,
        metavar="PATH",
        help="Write per-sample (kp, kd, t, target, actual, error) rows to a "
        "CSV for offline plotting. Pass without a value to auto-name as "
        "logs/pid_<side>_<joint>_<timestamp>.csv, or pass an explicit path.",
    )
    p.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    """Run the PID tuning session for the selected joint."""
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        # Cleanup (landing + disable) already ran in _run's finally; suppress
        # the traceback asyncio.run re-raises after a Ctrl-C.
        pass


async def _run(args: argparse.Namespace) -> None:
    joint = Joint(args.joint)
    is_left = args.l
    side_str = "left" if is_left else "right"
    lo, hi = arm_limits(joint, is_left)

    if args.stiffness is not None:
        if args.save:
            raise SystemExit(
                "--save cannot be combined with --stiffness: calibration "
                "stores the tuned s=0.5 midpoint anchors, not blended values."
            )
        if not 0.0 <= args.stiffness <= 1.0:
            raise SystemExit("--stiffness must be in [0, 1]")
        cfg = AxolConfig(
            left_stiffness=args.stiffness, right_stiffness=args.stiffness
        ).resolved()
    else:
        cfg = AxolConfig()
    arm_cfg: ArmConfig = cfg.left if is_left else cfg.right
    jc: JointConfig = getattr(arm_cfg, joint.value)
    kps: list[float] = args.kp if args.kp else [jc.kp]
    kds: list[float] = args.kd if args.kd else [jc.kd]
    candidates = [(kp, kd) for kp in kps for kd in kds]

    pose: dict[Joint, float] = {}
    for spec in args.pose or []:
        name, _, deg = spec.partition("=")
        try:
            pj = Joint(name)
        except ValueError:
            raise SystemExit(f"--pose: unknown joint {name!r}")
        if pj == joint:
            raise SystemExit(f"--pose: {name} is the test joint")
        if pj not in ARM_JOINTS:
            raise SystemExit(f"--pose: {name} is not an arm joint")
        try:
            rad = math.radians(float(deg))
        except ValueError:
            raise SystemExit(f"--pose: bad angle in {spec!r} (want JOINT=DEG)")
        p_lo, p_hi = arm_limits(pj, is_left)
        if not (p_lo <= rad <= p_hi):
            raise SystemExit(
                f"--pose: {name}={deg}° is outside "
                f"[{math.degrees(p_lo):.0f}, {math.degrees(p_hi):.0f}]° "
                f"for the {side_str} arm"
            )
        pose[pj] = rad

    if args.pose_by_hand:
        if args.mode != "step":
            raise SystemExit(
                "--pose-by-hand supports --mode step only (the probe runs "
                "in place at the frozen pose; a sine would re-center the arm)"
            )
        if pose:
            raise SystemExit("--pose-by-hand and --pose are mutually exclusive")
        if args.center is not None:
            raise SystemExit(
                "--pose-by-hand and --center are mutually exclusive — the "
                "probe runs at the hand-set pose"
            )
        if args.save:
            raise SystemExit(
                "--pose-by-hand is for exploring pose margins; re-run the "
                "winning gains without it to --save them"
            )
        if args.ff not in ("full", "gravity"):
            raise SystemExit(
                "--pose-by-hand needs the gravity model to hand-guide the "
                "arm — use --ff full or --ff gravity"
            )

    use_gravity = args.ff in ("full", "gravity")
    use_friction = args.ff in ("full", "friction")
    f = jc.friction
    fric = (f.fc, f.k, f.fv, f.fo) if use_friction else (0.0, 0.0, 0.0, 0.0)
    j_eff = jc.j_eff if args.ff == "full" else 0.0
    host_kd = args.host_kd if args.host_kd is not None else jc.kd_host
    host_kd_hz = args.host_kd_hz if args.host_kd_hz is not None else jc.kd_host_hz
    host_kd_hz_eff = (
        host_kd_hz if host_kd_hz is not None else DAMP_BP_W0 / (2 * math.pi)
    )
    host_kd_q = args.host_kd_q if args.host_kd_q is not None else jc.kd_host_q
    host_kd_q_eff = host_kd_q if host_kd_q is not None else DAMP_BP_Q
    # User-facing angles are degrees; everything below the CLI runs radians.
    amp_rad = math.radians(args.amp) if args.amp is not None else None
    center_rad = math.radians(args.center) if args.center is not None else None

    gravity_comp = GravityCompensator() if use_gravity else None
    test_idx = ARM_JOINTS.index(joint)
    arm_q_buf = np.zeros(len(ARM_JOINTS), dtype=np.float32)

    def gravity_fn(q: float) -> float:
        if gravity_comp is None:
            return 0.0
        arm_q_buf[test_idx] = q
        return float(gravity_comp.gravity_arm(arm_q_buf, is_left=is_left)[test_idx])

    dump_csv: Path | None = None
    if args.dump_csv == "__auto__":
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        dump_csv = Path("logs") / f"pid_{side_str}_{joint.value}_{timestamp}.csv"
    elif args.dump_csv is not None:
        dump_csv = Path(args.dump_csv)
    if dump_csv is not None:
        dump_csv.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"\nAxol PID tuner — {side_str} {joint.value}  "
        f"limits=[{math.degrees(lo):.1f}, {math.degrees(hi):.1f}]°"
    )
    print(f"  mode={args.mode}  ff={args.ff}  candidates={len(candidates)}")
    if args.stiffness is not None:
        print(
            f"  stiffness s={args.stiffness}: blended production gains "
            f"kp={jc.kp:.1f} kd={jc.kd:.2f} kd_host={jc.kd_host:.1f} "
            f"j_eff={jc.j_eff:.3f}"
        )
    else:
        # The configured kp/kd are the tuned midpoint (s=0.5, the production
        # default); the stiffness slider softens/stiffens around them. Print
        # both so the operator knows what the tested values relate to.
        resolved_jc: JointConfig = getattr(
            AxolConfig().resolved().left if is_left else AxolConfig().resolved().right,
            joint.value,
        )
        print(
            f"  config kp={jc.kp:.1f} kd={jc.kd:.2f} (tuned midpoint)  |  "
            f"production at default stiffness: kp={resolved_jc.kp:.1f} kd={resolved_jc.kd:.2f}"
        )
    if use_friction:
        print(f"  friction  Fc={f.fc}  k={f.k}  Fv={f.fv}  Fo={f.fo}")
    if args.ff == "full":
        print(f"  inertia  j_eff={j_eff}")
    if host_kd:
        print(
            f"  host-kd  {host_kd} (host-side damping via t_ff, kd_host) "
            f"band-passed at {host_kd_hz_eff:.1f} Hz (kd_host_hz), "
            f"q={host_kd_q_eff:g} (kd_host_q — band "
            f"{host_kd_hz_eff * (math.sqrt(1 + 1 / (4 * host_kd_q_eff**2)) - 1 / (2 * host_kd_q_eff)):.1f}"
            f"-{host_kd_hz_eff * (math.sqrt(1 + 1 / (4 * host_kd_q_eff**2)) + 1 / (2 * host_kd_q_eff)):.1f} Hz)"
        )

    noise: list[float] | None = None
    if args.target_noise is not None:
        if args.mode != "sine":
            raise SystemExit("--target-noise only applies to --mode sine")
        noise = make_target_noise(
            math.radians(args.target_noise), args.rate, args.duration
        )
        print(
            f"  target-noise  {args.target_noise}° RMS band-limited "
            f"(~8 Hz) on the reference — teleop-jitter emulation"
        )

    channel = resolve_channel(args)

    csv_file = None
    csv_writer = None
    if dump_csv is not None:
        csv_file = dump_csv.open("w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["kp", "kd", "mode", "t", "target", "actual", "error", "torque"]
        )
        print(f"  dumping per-sample rows to {dump_csv}")

    # One shared group id per invocation links the sweep's runs for A/B.
    run_group = uuid.uuid4().hex[:8] if args.save_run else None

    def _persist_run(
        kp: float, kd: float, log: list[dict], metrics: dict, mode_label: str
    ) -> None:
        if not args.save_run or not log:
            return
        run_id = save_run(
            args.mode,
            log_to_series(log),
            metrics,
            side=side_str,
            joint=joint.value,
            gains=(
                {
                    "kp": kp,
                    "kd": kd,
                    "kd_host": host_kd,
                    "kd_host_hz": host_kd_hz_eff,
                    "kd_host_q": host_kd_q_eff,
                }
                if host_kd
                else {"kp": kp, "kd": kd, "kd_host": host_kd}
            ),
            params={
                "mode": mode_label,
                "ff": args.ff,
                "amp_deg": args.amp,
                "center_deg": args.center,
                "freq": args.freq,
                "duration": args.duration,
                "hold": args.hold,
                "rate": args.rate,
                "stiffness": args.stiffness,
                "target_noise_deg": args.target_noise,
            },
            label=args.label,
            group=run_group,
        )
        print(f"  saved tuning run {run_id}")

    results: list[dict] = []
    ref_kp, ref_kd = candidates[0]

    async with CanBus(channel) as bus:
        raw_motors = {j: Motor(bus, j) for j in ARM_JOINTS}
        await asyncio.gather(*[m.enable() for m in raw_motors.values()])
        # Motor encoders are zeroed at end stops; everything below (limits,
        # centers, gravity poses) is joint frame (0 = rest), so wrap the
        # motors in the frame conversion before any position I/O.
        motors = await joint_frame_motors(raw_motors, is_left)
        await asyncio.gather(
            *[
                motors[j].set_control_mode(
                    ControlMode.IMPEDANCE
                    if (j == joint or args.pose_by_hand)
                    else ControlMode.POSITION_VELOCITY
                )
                for j in motors
            ]
        )

        # ---- --pose-by-hand machinery -------------------------------- #
        # All joints stay in impedance mode for the whole session (no mode
        # switches — those reboot MyActuators mid-hold). One 100 Hz stream
        # covers both phases: joints in ``hold`` are pinned to a frozen
        # target with scale·(kp, kd) plus kd_host damping; the rest get
        # kp=kd=0 with gravity t_ff, i.e. hand-guidable gravity comp.
        hand_dt = 1.0 / args.rate
        jc_all: dict[Joint, JointConfig] = {
            j: getattr(arm_cfg, j.value) for j in ARM_JOINTS
        }
        q_now: dict[Joint, float] = {}
        # Holder damping mirrors production: fast measured-velocity estimate
        # band-passed around the shoulder resonance (targets are frozen, so
        # v_des ≡ 0 and the damper input is just -v_meas).
        hand_diff = Differentiator(len(ARM_JOINTS), cutoff=VEL_CUTOFF_FREQ)
        # Holders damp at their own configured band centre, like production.
        hand_bp = BandPass(
            len(ARM_JOINTS),
            [
                2 * math.pi * jc_all[j].kd_host_hz
                if jc_all[j].kd_host_hz is not None
                else DAMP_BP_W0
                for j in ARM_JOINTS
            ],
            q=[
                jc_all[j].kd_host_q if jc_all[j].kd_host_q is not None else DAMP_BP_Q
                for j in ARM_JOINTS
            ],
        )
        hand_q_buf = np.zeros(len(ARM_JOINTS), dtype=np.float32)
        hand_landed = False

        # kd_host used while *holding* joints at a frozen pose. A config
        # kd_host that is unstable at that pose jitters the holder even when
        # the probed joint is clean — --hold-host-kd caps it for testing.
        def _hold_kd_host(j: Joint) -> float:
            v = jc_all[j].kd_host
            if args.hold_host_kd is not None:
                v = min(v, args.hold_host_kd)
            return v

        hold_log: dict[Joint, list[tuple[float, float]]] = {}

        async def _hand_cycle(
            hold: dict[Joint, float],
            scale: float,
            skip: frozenset[Joint] = frozenset(),
            log_holders: bool = False,
        ) -> None:
            t0 = time.monotonic()
            qs: list[float] = []
            tss: list[float] = []
            for i, j in enumerate(ARM_JOINTS):
                meas = cached_meas(motors[j])
                if meas is not None:
                    q_now[j] = meas[0]
                qs.append(q_now[j])
                tss.append(meas[1] if meas is not None else 0.0)
                hand_q_buf[i] = q_now[j]
            grav = gravity_comp.gravity_arm(hand_q_buf, is_left=is_left)
            vels = hand_diff.differentiate(qs, tss)
            v_damp = hand_bp.update([-v for v in vels])
            cmds = []
            for i, j in enumerate(ARM_JOINTS):
                if j in skip:
                    continue
                if j in hold:
                    jc_j = jc_all[j]
                    t_ff = float(grav[i]) + scale * _hold_kd_host(j) * v_damp[i]
                    cmds.append(
                        motors[j].set_impedance(
                            hold[j], 0.0, scale * jc_j.kp, scale * jc_j.kd, t_ff
                        )
                    )
                    if log_holders:
                        hold_log.setdefault(j, []).append(
                            (q_now[j], cached_torque(motors[j]))
                        )
                else:
                    cmds.append(
                        motors[j].set_impedance(q_now[j], 0.0, 0.0, 0.0, float(grav[i]))
                    )
            await asyncio.gather(*cmds)
            spent = time.monotonic() - t0
            if spent < hand_dt:
                await asyncio.sleep(hand_dt - spent)

        def _print_holder_report() -> None:
            """Vibration metrics for the joints that held the pose during probes."""
            if not hold_log:
                return
            print("  holders during probe (ripple / torque chatter):")
            for j, samples in hold_log.items():
                if len(samples) < 12:
                    continue
                qh = [s[0] for s in samples]
                th = [s[1] for s in samples if not math.isnan(s[1])]
                dd = [qh[i + 2] - 2 * qh[i + 1] + qh[i] for i in range(len(qh) - 2)]
                rip = math.degrees(math.sqrt(sum(x * x for x in dd) / len(dd))) * 1000
                chat = 0.0
                if len(th) > 2:
                    dt_ = [th[i + 1] - th[i] for i in range(len(th) - 1)]
                    chat = math.sqrt(sum(x * x for x in dt_) / len(dt_))
                flag = "  <-- vibrating" if rip > 20.0 else ""
                print(
                    f"    {j.value:<12} {rip:7.1f} mdeg   {chat:6.3f} Nm"
                    f"  (kd_host={_hold_kd_host(j):.0f}){flag}"
                )
            hold_log.clear()

        async def _hand_drag(prompt: str) -> str:
            """Stream gravity comp on all joints until the operator answers."""
            print(prompt, flush=True)
            reader = asyncio.create_task(asyncio.to_thread(input))
            try:
                while not reader.done():
                    await _hand_cycle({}, 0.0)
            except BaseException:
                reader.cancel()
                raise
            try:
                return reader.result().strip().lower()
            except EOFError:
                return "q"

        async def _hand_land() -> None:
            nonlocal hand_landed
            if hand_landed or not q_now:
                return
            hand_landed = True
            await _hand_drag(
                "\n[hand] gravity comp — lower the arm to a safe rest, "
                "then press Enter to release"
            )

        async def _hand_session() -> None:
            positions = await asyncio.gather(
                *[motors[j].get_position() for j in ARM_JOINTS]
            )
            for j, p in zip(ARM_JOINTS, positions):
                q_now[j] = p
            ramp_n = max(int(1.5 * args.rate), 1)
            pose_n = 0
            while True:
                ans = await _hand_drag(
                    "\n[hand] gravity comp — drag the arm to a test pose; "
                    "Enter = probe here, q+Enter = finish"
                )
                if ans.startswith("q"):
                    break
                pose_n += 1
                frozen = dict(q_now)
                arm_q_buf[:] = [frozen[j] for j in ARM_JOINTS]
                print(
                    f"[hand] pose {pose_n}: "
                    + "  ".join(
                        f"{j.value}={math.degrees(frozen[j]):+.0f}°" for j in ARM_JOINTS
                    )
                )
                hold = {j: frozen[j] for j in ARM_JOINTS if j != joint}
                for k in range(ramp_n):
                    await _hand_cycle(hold, (k + 1) / ramp_n)

                stop = asyncio.Event()

                async def _hold_loop() -> None:
                    while not stop.is_set():
                        await _hand_cycle(
                            hold, 1.0, skip=frozenset({joint}), log_holders=True
                        )

                hold_task = asyncio.create_task(_hold_loop())
                pose_results: list[dict] = []
                try:
                    for i, (kp, kd) in enumerate(candidates):
                        if len(candidates) > 1:
                            print(
                                f"\n[{i + 1}/{len(candidates)}] pose {pose_n}: "
                                f"testing Kp={kp}  Kd={kd}"
                            )
                        else:
                            print(f"\n  pose {pose_n}: testing Kp={kp}  Kd={kd}")
                        ff = FeedForward(
                            gravity_fn,
                            *fric,
                            j_eff=j_eff,
                            differentiate_target=False,
                            host_kd=host_kd,
                            host_kd_hz=host_kd_hz,
                            host_kd_q=host_kd_q,
                        )
                        log, amp = await run_step(
                            motors,
                            joint,
                            kp,
                            kd,
                            amp_rad,
                            args.hold,
                            args.rate,
                            is_left,
                            ff,
                            relative=True,
                        )
                        metrics = step_metrics(log, amp, args.hold)
                        _print_stats_step(metrics, len(log), kp, kd)
                        pose_results.append({"kp": kp, "kd": kd, "metrics": metrics})
                        _persist_run(kp, kd, log, metrics, f"step@pose{pose_n}")
                        if csv_writer is not None:
                            for r in log:
                                csv_writer.writerow(
                                    [
                                        kp,
                                        kd,
                                        f"step@pose{pose_n}",
                                        f"{r['t']:.5f}",
                                        f"{r['target']:.6f}",
                                        f"{r['actual']:.6f}",
                                        f"{r['error']:.6f}",
                                        f"{r['torque']:.4f}",
                                    ]
                                )
                            csv_file.flush()
                finally:
                    stop.set()
                    await hold_task
                _print_holder_report()
                if len(pose_results) > 1:
                    _print_ranking("step", pose_results)
                results.extend(pose_results)
                # Release back to gravity comp (reverse ramp) for the next pose.
                for k in range(ramp_n):
                    await _hand_cycle(hold, 1.0 - (k + 1) / ramp_n)
            await _hand_land()

        pre_pose: dict[Joint, float] = {}
        try:
            if args.pose_by_hand:
                await _hand_session()
                return

            print("  ramping other joints to rest (joint-frame 0) ...")
            await ramp_others_to_zero(motors, joint, is_left)

            if pose:
                desc = ", ".join(
                    f"{j.value}={math.degrees(v):.0f}°" for j, v in pose.items()
                )
                print(f"  posing: {desc} ...")
                vals = await asyncio.gather(*[motors[j].get_position() for j in pose])
                pre_pose = dict(zip(pose, vals))
                await ramp_joints_to(motors, pose)

            # Fill the gravity pose with the *measured* positions of the
            # other joints: base-collision joints (shoulder_2, wrist_2) are
            # deliberately left off-zero, and assuming 0 there would bias
            # the gravity feedforward for every candidate.
            positions = await asyncio.gather(
                *[motors[j].get_position() for j in ARM_JOINTS]
            )
            arm_q_buf[:] = positions

            for i, (kp, kd) in enumerate(candidates):
                if len(candidates) > 1:
                    print(f"\n[{i + 1}/{len(candidates)}] testing Kp={kp}  Kd={kd}")
                else:
                    print(f"\n  testing Kp={kp}  Kd={kd}")
                ff = FeedForward(
                    gravity_fn,
                    *fric,
                    j_eff=j_eff,
                    differentiate_target=(args.mode == "sine"),
                    host_kd=host_kd,
                    host_kd_hz=host_kd_hz,
                    host_kd_q=host_kd_q,
                )
                # Verify the POSITION_VELOCITY holds actually stay put while
                # the test joint shakes the structure — a wobbling holder
                # contaminates the ring and the test joint's log can't show it.
                monitor = HolderMonitor(motors, exclude=joint)
                if args.mode == "sine":
                    log, amp = await run_sine(
                        motors,
                        joint,
                        kp,
                        kd,
                        args.freq,
                        amp_rad,
                        args.duration,
                        args.rate,
                        is_left,
                        ff,
                        noise=noise,
                        monitor=monitor,
                        center=center_rad,
                    )
                    metrics = sine_metrics(log)
                    _print_stats_sine(metrics, len(log), kp, kd)
                else:
                    log, amp = await run_step(
                        motors,
                        joint,
                        kp,
                        kd,
                        amp_rad,
                        args.hold,
                        args.rate,
                        is_left,
                        ff,
                        monitor=monitor,
                        center=center_rad,
                    )
                    metrics = step_metrics(log, amp, args.hold)
                    _print_stats_step(metrics, len(log), kp, kd)
                metrics["holder_wobble_deg"] = monitor.report()
                metrics["holder_peak_deg"] = max(
                    metrics["holder_wobble_deg"].values(), default=0.0
                )
                _print_holder_wobble(metrics["holder_wobble_deg"])

                results.append({"kp": kp, "kd": kd, "metrics": metrics})
                _persist_run(kp, kd, log, metrics, args.mode)
                if csv_writer is not None:
                    for r in log:
                        csv_writer.writerow(
                            [
                                kp,
                                kd,
                                args.mode,
                                f"{r['t']:.5f}",
                                f"{r['target']:.6f}",
                                f"{r['actual']:.6f}",
                                f"{r['error']:.6f}",
                                f"{r['torque']:.4f}",
                            ]
                        )
                    csv_file.flush()
                await asyncio.sleep(0.5)

            if len(results) > 1:
                _print_ranking(args.mode, results)

            if args.save and results:
                best = min(results, key=lambda r: r["metrics"]["score"])
                # Persist kd_host only when the operator set it explicitly:
                # the gains were validated with that damping active, so the
                # two must be saved (and later loaded) together.
                path = update_joint_calibration(
                    side_str,
                    joint.value,
                    kp=best["kp"],
                    kd=best["kd"],
                    kd_host=args.host_kd,
                )
                saved = f"Kp={best['kp']}  Kd={best['kd']}"
                if args.host_kd is not None:
                    saved += f"  kd_host={args.host_kd}"
                print(f"\n  Saved {saved} to {path}")
                print(
                    "  (loaded automatically by AxolConfig on this machine; "
                    "these are the tuned s=0.5 midpoint of the stiffness "
                    "blend, like the shared defaults they replace)"
                )

        except (KeyboardInterrupt, asyncio.CancelledError):
            # Ctrl-C reaches this task as a cancellation (asyncio.run cancels
            # the main task on SIGINT). Un-cancel so the cleanup below can
            # still await — otherwise the landing stream and the motor
            # disable are themselves cancelled and the arm is left powered.
            print("\n  interrupted")
            task = asyncio.current_task()
            if task is not None:
                while task.cancelling():
                    task.uncancel()
        finally:
            if csv_file is not None:
                csv_file.close()
            if args.pose_by_hand:
                # The arm may be frozen mid-air (or held by the operator):
                # drop to gravity comp and wait for them to lower it before
                # the disable below lets everything go limp.
                try:
                    await _hand_land()
                except Exception:
                    pass
            else:
                # Slow controlled ramp to rest — for shoulder_2 this is the
                # *safe* way to reach the base side: the danger was a fast
                # mid-step return-to-center, not the gentle approach at
                # _RAMP_SPEED.
                print("  returning to rest ...")
                try:
                    await ramp_impedance(
                        motors[joint], ref_kp, ref_kd, 0.0, gravity_fn, args.rate
                    )
                except Exception:
                    pass
                if pre_pose:
                    # Lower posed joints back to where they started (a safe
                    # hang) before the final disable lets everything go limp.
                    print("  returning posed joints ...")
                    try:
                        await ramp_joints_to(motors, pre_pose)
                    except Exception:
                        pass
                await asyncio.gather(
                    *[
                        m.set_control_mode(ControlMode.IMPEDANCE)
                        for m in motors.values()
                    ]
                )
            await asyncio.gather(*[m.disable() for m in motors.values()])
