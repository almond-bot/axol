"""Hardware tuning runners: sine / step references on a single joint.

Extracted from the ``tune.pid`` CLI so the serve backend and the
reference-motion replay share the same runners, safety geometry, and ramp
helpers. Everything here speaks the **joint frame** (0 = rest) through
:class:`~almond_axol.tuning.joint_frame.JointFrameMotor`.

Progress lines go to stdout: these runners narrate an operator-triggered
hardware session, and the serve session manager streams stdout to the UI.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time

import numpy as np

from ..constants import ARM_JOINTS
from ..motor import Joint, MotorError
from ..robot.axol import arm_limits
from .feedforward import FeedForward
from .joint_frame import JointFrameMotor

# Default sine amplitude / step size (rad). 0.175 rad ≈ 10° — well above
# the encoder noise floor and the ``5%`` settling threshold (≈0.5°), well
# clear of the friction-stiction breakaway condition (``kp · amp > Fc``)
# at all typical PID-tuning gains, and small enough to avoid hitting joint
# limits or driving any joint into its high-velocity saturation regime.
DEFAULT_AMP_RAD = 0.175
RAMP_SPEED = 0.25  # rad/s

# Joints whose motion can meet the robot base and are therefore never blindly
# commanded to 0 by ``ramp_others_to_zero`` (they are left in place, or moved
# by an explicit clearance target). shoulder_2's inboard half collides with
# the elbow straight; wrist_2's inboard half does too, *unless* the elbow is
# raised — which every wrist_2 probe/sweep now does (see
# ``probe_clearance_targets``), giving wrist_2 its full range.
BASE_COLLISION_JOINTS = frozenset({Joint.SHOULDER_2, Joint.WRIST_2})

# With the chest cameras mounted, probing these joints with the arm hanging
# at the rest pose swings the elbow/gripper right past the cameras. The
# tuning flows hold shoulder_2 out (``probe_clearance_targets``) so the arm
# runs clear of the torso for their sweeps.
CAMERA_CLEARANCE_JOINTS = frozenset({Joint.SHOULDER_3, Joint.WRIST_1})

# Clearance angle shoulder_2 is held at (outboard) while shoulder_3 / wrist_1
# probe: with the chest cameras mounted, those joints' probes swing the
# hanging arm right past them, and 10° of shoulder_2 clears it. shoulder_2's
# *own* motion may still start at 0 (the rest pose) — only the direction of
# travel is constrained, outboard.
SHOULDER_2_CLEARANCE = math.radians(10.0)


def safe_outboard_direction(joint: Joint, is_left: bool) -> int | None:
    """Step direction that swings away from the robot base, or ``None`` if the
    joint has no base-collision constraint.

    Only shoulder_2 remains constrained: wrist_2's probes and sweeps all run
    with the elbow raised (``probe_clearance_targets``), which keeps the
    gripper clear of the base through wrist_2's full range.
    """
    if joint == Joint.SHOULDER_2:
        return -1 if is_left else 1
    return None


def safe_limits(joint: Joint, is_left: bool) -> tuple[float, float]:
    """Joint limits with the base-collision boundary applied.

    For unconstrained joints these are just the arm limits; for the
    base-collision joints the inboard boundary is pinned at 0, so any motion
    planned inside these limits stays outboard of the base. 0 itself (the
    rest pose) is safe — it is crossing past it that collides.
    """
    lo, hi = arm_limits(joint, is_left)
    safe_dir = safe_outboard_direction(joint, is_left)
    if safe_dir is None:
        return lo, hi
    if safe_dir > 0:
        return max(lo, 0.0), hi
    return lo, min(hi, 0.0)


def probe_clearance_targets(test_joint: Joint, is_left: bool) -> dict[Joint, float]:
    """Hold targets the *other* joints take so ``test_joint`` can run safely.

    - shoulder_3 / wrist_1: shoulder_2 held 10° outboard — their probes
      rotate the hanging arm right where the chest cameras sit.
    - wrist_2: elbow raised to its midpoint — with the elbow bent the
      gripper clears the base through wrist_2's *full* range (with the
      elbow straight, the inboard half collides).

    Empty for joints whose probes are safe at rest.
    """
    if test_joint in CAMERA_CLEARANCE_JOINTS:
        direction = -1 if is_left else 1  # shoulder_2's outboard side
        return {Joint.SHOULDER_2: direction * SHOULDER_2_CLEARANCE}
    if test_joint == Joint.WRIST_2:
        elbow_lo, elbow_hi = arm_limits(Joint.ELBOW, is_left)
        return {Joint.ELBOW: (elbow_lo + elbow_hi) / 2.0}
    return {}


# Load poses for the friction/gravity sweeps. Gravity identification needs the
# test joint *loaded*: gravity has zero moment about a vertical axis no matter
# where the mass sits, so a joint whose axis hangs vertical at rest carries no
# CoM signal — its fit would chase torque-sensor noise (the failure that once
# moved a wrist CoM 90 mm and poisoned the elbow fit computed on top of it).
# Three joints hang axis-vertical and need other joints posed to tilt them.
# Clearances below were verified against the torso collision model; signal
# figures are the CAD gravity model's torque variation over the sweep.
SHOULDER_1_LOAD = math.radians(90.0)  # humerus horizontal for shoulder_3
WRIST_2_LOAD = math.radians(85.0)  # hand off wrist_1's axis (85°: limit is 90)
WRIST_1_LOAD = math.radians(90.0)  # hand off wrist_2's axis
# shoulder_3 / wrist_1 sweep cap at their loaded poses: ±90° keeps the bent
# forearm ≥ +60 mm clear of the torso capsules (≥ +199 mm for shoulder_3).
LOADED_SWEEP_CAP = math.radians(90.0)


def sweep_safety(
    joint: Joint, is_left: bool
) -> tuple[dict[Joint, float], float | None, float | None, list[str]]:
    """Clearance + load poses and range caps for a full-range sweep of ``joint``.

    Returns ``(clearance_targets, lo_cap, hi_cap, notes)``: joint-frame hold
    targets to ramp the *other* joints to before sweeping, optional overrides
    of the sweep's lower/upper limit, and human-readable notes explaining
    each measure. Shared by ``tune.friction``, ``tune.gravity`` and
    ``tune.factory`` (which derives both fits from one sweep, so the pose
    must serve both). Two kinds of rule are encoded:

    Physical safety:

    - wrist_2: elbow raised to its midpoint — with the elbow bent the
      gripper clears the base through wrist_2's *full* range, so the sweep
      is not capped. (The PID probes share this via
      ``probe_clearance_targets``.)
    - shoulder_2: sweep capped at 0 — starting at the rest pose is fine,
      but travel past it swings into the base.
    - shoulder_3 / wrist_1: shoulder_2 held 10° outboard so the arm
      swings clear of the chest cameras.

    Gravity load (see the module constants above): joints whose axis hangs
    vertical at rest are posed so gravity actually loads them —

    - shoulder_3: shoulder_1 raised 90° + elbow bent to midpoint; its axis
      goes horizontal with the bent forearm as the off-axis load (~2.9 Nm
      of sweep signal, ≥ +199 mm torso clearance over the ±90° sweep).
    - wrist_1: elbow bent + wrist_2 rotated 85° (~0.4 Nm, ≥ +60 mm).
    - wrist_2: elbow bent (base clearance, as before) + wrist_1 at 90°,
      restoring the load the elbow raise removed (~0.45 vs 0.23 Nm).

    Ramp the returned targets with :func:`ramp_stages` — proximal group
    first. Ramping everything at once from rest can rotate a wrist to its
    hold while the elbow is still straight, which is exactly the geometry
    that sweeps the gripper through the base.
    """
    notes: list[str] = []
    clearance = dict(probe_clearance_targets(joint, is_left))
    if Joint.SHOULDER_2 in clearance:
        notes.append(
            f"Holding shoulder_2 at "
            f"{math.degrees(clearance[Joint.SHOULDER_2]):+.0f}° to clear the "
            f"chest cameras for the {joint.value} sweep."
        )
    if Joint.ELBOW in clearance:
        notes.append(
            f"Moving elbow to {math.degrees(clearance[Joint.ELBOW]):.1f}° "
            "(midpoint of range) — with the elbow bent, wrist_2 sweeps its "
            "full range clear of the base."
        )
    lo_cap: float | None = None
    hi_cap: float | None = None
    elbow_lo, elbow_hi = arm_limits(Joint.ELBOW, is_left)
    elbow_mid = (elbow_lo + elbow_hi) / 2.0
    if joint == Joint.SHOULDER_2:
        lo_cap, hi_cap = safe_limits(joint, is_left)
        notes.append(
            f"Capping {joint.value} sweep at 0° — outboard travel only, "
            "the base is inboard."
        )
    elif joint == Joint.SHOULDER_3:
        clearance[Joint.SHOULDER_1] = SHOULDER_1_LOAD
        clearance[Joint.ELBOW] = elbow_mid
        lo_cap, hi_cap = -LOADED_SWEEP_CAP, LOADED_SWEEP_CAP
        notes.append(
            "Raising shoulder_1 to 90° and bending the elbow so gravity "
            "loads shoulder_3 (its axis is vertical at rest — zero gravity "
            "moment there); sweep capped at ±90° to stay clear of the torso."
        )
    elif joint == Joint.WRIST_1:
        clearance[Joint.ELBOW] = elbow_mid
        clearance[Joint.WRIST_2] = WRIST_2_LOAD
        lo_cap, hi_cap = -LOADED_SWEEP_CAP, LOADED_SWEEP_CAP
        notes.append(
            "Bending the elbow and rotating wrist_2 to 85° so gravity "
            "loads wrist_1 (its axis is vertical at rest); sweep capped "
            "at ±90°."
        )
    elif joint == Joint.WRIST_2:
        clearance[Joint.WRIST_1] = WRIST_1_LOAD
        notes.append(
            "Rotating wrist_1 to 90° so gravity loads wrist_2 with the "
            "elbow raised (the raise alone leaves it barely loaded)."
        )
    return clearance, lo_cap, hi_cap, notes


# Proximal group ramped first by ramp_stages. Shoulders and elbow reshape the
# arm's gross geometry; the wrists only spin mass near the gripper, which is
# safe *after* the elbow raise but not necessarily before it.
_PROXIMAL_JOINTS = frozenset(
    {Joint.SHOULDER_1, Joint.SHOULDER_2, Joint.SHOULDER_3, Joint.ELBOW}
)


def ramp_stages(targets: dict[Joint, float]) -> list[dict[Joint, float]]:
    """Split hold targets into safely ordered ramp groups (proximal first).

    Ramping every target simultaneously from rest can rotate wrist_2 toward
    its hold while the elbow is still straight — the exact inboard geometry
    that once swept the gripper into the base. Shoulders + elbow settle
    first (their combined path from rest was verified clear against the
    collision model), then the wrists rotate with the elbow already bent.
    The reverse order is the existing distal-first homing.
    """
    proximal = {j: v for j, v in targets.items() if j in _PROXIMAL_JOINTS}
    distal = {j: v for j, v in targets.items() if j not in _PROXIMAL_JOINTS}
    return [stage for stage in (proximal, distal) if stage]


def sine_center(joint: Joint, is_left: bool) -> float:
    # Midpoint of the *safe* range: the full range for most joints, the
    # outboard region for the base-collision joints.
    lo, hi = safe_limits(joint, is_left)
    return (lo + hi) / 2.0


def check_center(joint: Joint, is_left: bool, center: float) -> tuple[float, float]:
    """Validate an explicit probe centre; returns the effective limits.

    For the base-collision joints the inboard side physically hits the robot
    base (or the cameras, for shoulder_2), so the centre must sit outboard
    of the safe boundary and the returned limits pin it there — the swing
    can then never cross toward the base.
    """
    lo, hi = safe_limits(joint, is_left)
    safe_dir = safe_outboard_direction(joint, is_left)
    if safe_dir is not None and center * safe_dir < 0:
        side = "positive" if safe_dir > 0 else "negative"
        raise ValueError(
            f"{joint.value} center {math.degrees(center):.1f}° is in the "
            f"inboard half — the robot base is there. Use the {side} side "
            "(0, the rest pose, is fine)."
        )
    if not (lo <= center <= hi):
        raise ValueError(
            f"center {math.degrees(center):.1f}° is outside "
            f"[{math.degrees(lo):.1f}, {math.degrees(hi):.1f}]° for {joint.value}"
        )
    return lo, hi


def safe_amplitude(
    joint: Joint,
    is_left: bool,
    center: float,
    requested: float | None,
    limits: tuple[float, float] | None = None,
) -> float:
    lo, hi = limits if limits is not None else arm_limits(joint, is_left)
    if not (lo <= center <= hi):
        raise ValueError(
            f"Current position {center:.4f} rad is outside [{lo:.4f}, {hi:.4f}] for {joint.value}"
        )
    headroom = min(center - lo, hi - center)
    if headroom < 0.03:  # ~1.7°
        raise ValueError(
            f"{joint.value} center {center:.4f} rad is too close to a limit [{lo:.4f}, {hi:.4f}]. "
            f"Sine test centers on the joint midpoint ({sine_center(joint, is_left):.4f} rad) — "
            f"move there first, or use --mode step."
        )
    if requested is not None:
        amp = min(requested, headroom)
        if amp < requested:
            print(
                f"  ! requested amp {math.degrees(requested):.1f}° exceeds "
                f"headroom; clamped to {math.degrees(amp):.1f}°"
            )
    else:
        amp = min(DEFAULT_AMP_RAD, headroom)
    return amp


async def ramp_impedance(
    motor: JointFrameMotor,
    kp: float,
    kd: float,
    target: float,
    gravity_fn,
    rate_hz: float = 100.0,
    speed: float = RAMP_SPEED,
) -> None:
    """Ramp one impedance-mode joint to ``target`` (joint frame) at ``speed``, with gravity FF."""
    start = await motor.get_position()
    duration = max(abs(target - start) / speed, 0.5)
    dt = 1.0 / rate_hz
    t0 = time.monotonic()
    while True:
        t = time.monotonic() - t0
        alpha = min(t / duration, 1.0)
        q = start + alpha * (target - start)
        await motor.set_impedance(q, 0.0, kp, kd, gravity_fn(q))
        if alpha >= 1.0:
            break
        await asyncio.sleep(dt)


async def ramp_others_to_zero(
    motors: dict[Joint, JointFrameMotor],
    exclude: Joint,
    is_left: bool,
) -> None:
    """Send non-test joints to rest (joint-frame 0) and poll until arrival.

    Joints listed in ``BASE_COLLISION_JOINTS`` are also skipped: their inboard
    half can meet the robot base (the URDF limits don't capture this), so they
    are left where the operator posed them rather than blindly commanded.

    Probe clearances override the skip/zero defaults per test joint (see
    ``probe_clearance_targets``): shoulder_3 / wrist_1 hold shoulder_2 10°
    outboard to clear the chest cameras, and wrist_2 gets the elbow raised to
    its midpoint so its full range clears the base.
    """
    skip = {exclude} | BASE_COLLISION_JOINTS
    targets = {j: 0.0 for j in ARM_JOINTS if j not in skip}
    clearance = probe_clearance_targets(exclude, is_left)
    for j, v in clearance.items():
        print(
            f"  Holding {j.value} at {math.degrees(v):+.0f}° for the "
            f"{exclude.value} probe (base/camera clearance)."
        )
    targets.update(clearance)
    await ramp_joints_to(motors, targets)


async def ramp_joints_to(
    motors: dict[Joint, JointFrameMotor],
    targets: dict[Joint, float],
) -> None:
    """Ramp POSITION_VELOCITY-mode joints to joint-frame targets, poll until arrival."""
    joints = list(targets)
    if not joints:
        return
    pos_vals = await asyncio.gather(*[motors[j].get_position() for j in joints])
    max_dist = max((abs(p - targets[j]) for j, p in zip(joints, pos_vals)), default=0.0)
    await asyncio.gather(
        *[motors[j].set_position_velocity(targets[j], RAMP_SPEED) for j in joints]
    )
    timeout = max_dist / RAMP_SPEED + 2.0
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        await asyncio.sleep(0.1)
        positions = await asyncio.gather(*[motors[j].get_position() for j in joints])
        if all(abs(p - targets[j]) < 0.05 for j, p in zip(joints, positions)):
            break


def report_achieved_rate(log: list[dict], rate_hz: float) -> None:
    """Print the loop rate the run actually sustained.

    The loop runs open-throttle when a cycle's CAN round trips exceed the
    period, so the requested rate is a ceiling, not a guarantee — and gains
    tuned at a lower-than-production rate see more host-damping transport
    delay than teleop will. Warn when the shortfall is real (>10%).
    """
    if len(log) < 2:
        return
    span = log[-1]["t"] - log[0]["t"]
    if span <= 0:
        return
    achieved = (len(log) - 1) / span
    print(f"  achieved rate: {achieved:.0f} Hz (requested {rate_hz:.0f})")
    if achieved < 0.9 * rate_hz:
        print(
            "  ! loop saturated below the requested rate — CAN round trips "
            "exceed the cycle budget; the results reflect the achieved rate"
        )


def cached_torque(motor: JointFrameMotor) -> float:
    """Torque cached by the last impedance response (Nm), or NaN if absent."""
    try:
        return motor.torque
    except MotorError:
        return float("nan")


def cached_meas(motor: JointFrameMotor) -> tuple[float, float] | None:
    """(position, frame timestamp) from the last impedance response, if any."""
    try:
        return motor.position, motor.feedback_ts
    except MotorError:
        return None


class HolderMonitor:
    """Round-robin wobble sampler for the non-test joints during a probe.

    The holders sit in firmware POSITION_VELOCITY holds — the stiffest mode
    the motors offer — but stiff is not *proven quiet*: a holder wobbling at
    its own resonance feeds structure motion straight back into the test
    joint's ring, and the test joint's encoder alone can never show that.
    One extra position read per command cycle, rotating through the holders
    (~16 Hz per holder at a 100 Hz probe), is cheap enough to leave the
    probe rate intact while catching anything at the 2–3 Hz arm modes.

    Each holder's first sample (taken during the pre-step settle, before any
    reaction torque hits) is its baseline; ``peak`` accumulates the largest
    deviation seen since.
    """

    def __init__(self, motors: dict[Joint, JointFrameMotor], exclude: Joint) -> None:
        self._motors = {j: m for j, m in motors.items() if j != exclude}
        self._joints = list(self._motors)
        self._i = 0
        self._baseline: dict[Joint, float] = {}
        self._last: dict[Joint, float] = {}
        self.peak: dict[Joint, float] = dict.fromkeys(self._joints, 0.0)

    async def sample(self) -> None:
        """Read the next holder in the rotation and update its peak."""
        if not self._joints:
            return
        j = self._joints[self._i % len(self._joints)]
        self._i += 1
        pos = await self._motors[j].get_position()
        self._last[j] = pos
        base = self._baseline.setdefault(j, pos)
        dev = abs(pos - base)
        if dev > self.peak[j]:
            self.peak[j] = dev

    def rebase(self) -> None:
        """Freeze each holder's baseline at its latest reading, zero the peaks.

        Call at the boundary between warm-up and the scored phase: the ramp
        that parks the holders accepts arrival within 0.05 rad (~2.9°), so a
        holder can still be creeping to its target during the settle — motion
        that is ramp completion, not reaction-torque wobble. Rebasing after
        the settle makes the peaks count only motion during the test itself.
        """
        for j, pos in self._last.items():
            self._baseline[j] = pos
        self.peak = dict.fromkeys(self._joints, 0.0)

    def report(self) -> dict[str, float]:
        """Peak deviation per holder, in degrees, largest first."""
        return {
            j.value: math.degrees(v)
            for j, v in sorted(self.peak.items(), key=lambda kv: -kv[1])
        }


class LiveStream:
    """Streams probe samples to stdout live, for the workbench's live chart.

    The serve session manager already pipes every stdout line to the browser
    over the session WebSocket, so live charting needs no new transport: the
    runner prints ``@@live {…}`` lines, the workbench plots them as they
    arrive, and the console hides them. Only active under ``axol serve``
    (``AXOL_TUNE_LIVE=1``) — a terminal user just gets the normal narration.

    Samples are decimated to ~25 Hz and batched into ~4 lines/s: plenty for
    a chart that's watched, light enough that a run adds tens of log lines,
    not thousands. Each probe phase starts with a ``new`` marker so the
    chart clears between sweep candidates instead of overlaying their
    restarted time axes.
    """

    _RATE_HZ = 25.0
    _FLUSH_S = 0.25

    def __init__(self, mode: str, joint: Joint) -> None:
        self._on = os.environ.get("AXOL_TUNE_LIVE") == "1"
        self._buf: list[list[float]] = []
        self._last_t = -math.inf
        self._last_flush = time.monotonic()
        if self._on:
            print(f'@@live {{"new": {{"mode": "{mode}", "joint": "{joint.value}"}}}}')

    def add(self, t: float, target: float, actual: float) -> None:
        if not self._on or t - self._last_t < 1.0 / self._RATE_HZ:
            return
        self._last_t = t
        self._buf.append([round(t, 4), round(target, 5), round(actual, 5)])
        now = time.monotonic()
        if now - self._last_flush >= self._FLUSH_S:
            self.flush()
            self._last_flush = now

    def flush(self) -> None:
        if not self._on or not self._buf:
            return
        print(f'@@live {{"samples": {json.dumps(self._buf)}}}', flush=True)
        self._buf.clear()


def make_target_noise(
    rms: float, rate_hz: float, duration: float, cutoff_hz: float = 8.0
) -> list[float]:
    """Band-limited reference noise emulating teleop hand-tracking jitter.

    White noise through a one-pole low-pass at ``cutoff_hz`` (hand tremor +
    IK output sits mostly below ~10 Hz), normalized to the requested RMS.
    Fixed seed so A/B runs see the identical disturbance sequence.
    """
    rng = np.random.default_rng(0)
    n = int(duration * rate_hz) + 64
    alpha = 1.0 / (1.0 + rate_hz / (2.0 * math.pi * cutoff_hz))
    y = np.empty(n)
    acc = 0.0
    for i, w in enumerate(rng.standard_normal(n)):
        acc += alpha * (w - acc)
        y[i] = acc
    y -= y.mean()
    y *= rms / max(float(np.sqrt(np.mean(y**2))), 1e-12)
    return y.tolist()


async def run_sine(
    motors: dict[Joint, JointFrameMotor],
    joint: Joint,
    kp: float,
    kd: float,
    freq: float,
    requested_amp: float | None,
    duration: float,
    rate_hz: float,
    is_left: bool,
    ff: FeedForward,
    noise: list[float] | None = None,
    monitor: HolderMonitor | None = None,
    center: float | None = None,
) -> tuple[list[dict], float]:
    """Track a sine reference on ``joint`` and log target/actual error.

    ``center`` (rad, joint frame) overrides the default joint-midpoint
    centre — probe at 45°/-45°/… to see the gains under real gravity load.
    Amplitude is clamped to the headroom around it, and for base-collision
    joints the inboard boundary counts as a limit.

    ``noise`` (optional, pre-generated band-limited samples) is added to
    the commanded reference to emulate teleop hand-tracking jitter; the
    logged target/error stay relative to the clean sine — the operator's
    "intent" — so noise-induced motion and torque chatter show up as
    error rather than being normalized away.

    Returns the per-sample log and the amplitude actually used (clamped
    to the joint's headroom).
    """
    test_motor = motors[joint]
    if center is None:
        lo, hi = safe_limits(joint, is_left)
        center = sine_center(joint, is_left)
    else:
        lo, hi = check_center(joint, is_left, center)
    amp = safe_amplitude(joint, is_left, center, requested_amp, limits=(lo, hi))
    print(
        f"  limits=[{lo:.4f}, {hi:.4f}] rad  center={center:.4f} rad  "
        f"amp=±{amp:.4f} rad  freq={freq:.2f} Hz"
    )

    print("  moving to center ...")
    await ramp_impedance(test_motor, kp, kd, center, ff.gravity_fn, rate_hz)
    await asyncio.sleep(1.0)

    print(f"  running {duration:.1f} s at {rate_hz:.0f} Hz ...")
    dt = 1.0 / rate_hz
    log: list[dict] = []
    live = LiveStream("sine", joint)
    start = time.monotonic()
    k = 0

    while True:
        t = time.monotonic() - start
        if t >= duration:
            break
        loop_start = time.monotonic()

        target = center + amp * math.sin(2 * math.pi * freq * t)
        if noise is not None:
            target += noise[k % len(noise)]
        k += 1
        v_des, t_ff = ff.compute(target, cached_meas(test_motor))
        await test_motor.set_impedance(target, v_des, kp, kd, t_ff)
        # The impedance response frame already carried position feedback
        # (just cached) — reading it instead of a separate poll halves the
        # CAN round trips per cycle, which is what makes production-rate
        # (240 Hz) testing reachable. Production never polls separately.
        meas = cached_meas(test_motor)
        actual = meas[0] if meas is not None else await test_motor.get_position()
        if monitor is not None:
            await monitor.sample()
        t_read = time.monotonic() - start
        target_at_read = center + amp * math.sin(2 * math.pi * freq * t_read)
        log.append(
            {
                "t": round(t_read, 5),
                "target": target_at_read,
                "actual": actual,
                "error": actual - target_at_read,
                "torque": cached_torque(test_motor),
            }
        )
        live.add(t_read, target_at_read, actual)

        spent = time.monotonic() - loop_start
        if spent < dt:
            await asyncio.sleep(dt - spent)

    live.flush()
    report_achieved_rate(log, rate_hz)
    return log, amp


async def run_step(
    motors: dict[Joint, JointFrameMotor],
    joint: Joint,
    kp: float,
    kd: float,
    requested_amp: float | None,
    hold: float,
    rate_hz: float,
    is_left: bool,
    ff: FeedForward,
    relative: bool = False,
    monitor: HolderMonitor | None = None,
    center: float | None = None,
) -> tuple[list[dict], float]:
    """Drive a step on ``joint`` and log the step-response error.

    With ``relative=True`` the step is framed around the joint's *current*
    position even for the base-collision joints (stepping in their outboard
    direction) — used by ``--pose-by-hand``, where the whole point is to
    probe at the pose the operator set, not at a canned center.

    ``center`` (rad, joint frame) frames the step around an explicit start
    angle instead of the joint's current position — probe at 45°/-45°/… to
    see the gains under real gravity load. The step direction picks the
    side with more headroom (outboard, for base-collision joints).

    Returns the per-sample log and the amplitude actually used (clamped
    to the joint's safe headroom).
    """
    test_motor = motors[joint]
    current = await test_motor.get_position()
    if center is not None:
        lo, hi = check_center(joint, is_left, center)
    else:
        lo, hi = arm_limits(joint, is_left)

    safe_dir = safe_outboard_direction(joint, is_left)
    if joint == Joint.WRIST_2 and relative:
        # The standard flow raises the elbow before a wrist_2 probe
        # (``probe_clearance_targets``), freeing its full range — but the
        # hand-posed path probes wherever the operator left the arm. With
        # the elbow near straight, wrist_2's inboard half still meets the
        # base, so keep the old outboard-only rule in that case.
        elbow_pos = await motors[Joint.ELBOW].get_position()
        if abs(elbow_pos) < math.radians(30.0):
            safe_dir = 1 if is_left else -1
            print(
                "  elbow is near straight — stepping wrist_2 outboard only "
                "(the base is inboard)."
            )
    if safe_dir is not None and not relative and center is None:
        # The inboard side physically collides with the robot base; frame the
        # whole test in the safe half so that center *and* step_target stay
        # outboard of 0. amp goes from 0 → safe-limit/2 (room for a 2× swing).
        direction = safe_dir
        outboard_limit = lo if direction < 0 else hi
        max_safe_amp = abs(outboard_limit) / 2.0
        amp = min(
            requested_amp if requested_amp is not None else DEFAULT_AMP_RAD,
            max_safe_amp,
        )
        center = direction * amp
        step_target = direction * 2.0 * amp
        if requested_amp is not None and amp < requested_amp:
            print(
                f"  ! requested amp {requested_amp:.4f} rad would push past the safe half; clamped to {amp:.4f} rad"
            )
    else:
        if center is None:
            center = current
        headroom_up = hi - center
        headroom_down = center - lo
        if safe_dir is not None:
            # relative mode on a base-collision joint: only the outboard
            # direction is safe, regardless of which side has more headroom.
            direction = safe_dir
            headroom = headroom_up if direction > 0 else headroom_down
            if headroom < 0.03:
                raise ValueError(
                    f"{joint.value} at {center:.4f} rad has no outboard headroom "
                    f"within [{lo:.4f}, {hi:.4f}]."
                )
        elif headroom_up < 0.03 and headroom_down < 0.03:
            raise ValueError(
                f"{joint.value} at {center:.4f} rad has no headroom within [{lo:.4f}, {hi:.4f}]."
            )
        elif headroom_up >= headroom_down:
            direction, headroom = 1, headroom_up
        else:
            direction, headroom = -1, headroom_down

        if requested_amp is not None:
            amp = min(requested_amp, headroom)
            if amp < requested_amp:
                print(
                    f"  ! requested amp {math.degrees(requested_amp):.1f}° exceeds "
                    f"headroom; clamped to {math.degrees(amp):.1f}°"
                )
        else:
            amp = min(DEFAULT_AMP_RAD, headroom)
        step_target = center + direction * amp

    sign_str = f"+{amp:.4f}" if direction == 1 else f"-{amp:.4f}"
    print(
        f"  limits=[{lo:.4f}, {hi:.4f}] rad  center={center:.4f} rad  "
        f"step={sign_str} rad  hold={hold:.1f} s  rate={rate_hz:.0f} Hz"
    )

    if abs(current - center) > 0.01:
        print(f"  moving to step center ({center:.4f} rad) ...")
        await ramp_impedance(test_motor, kp, kd, center, ff.gravity_fn, rate_hz)

    dt = 1.0 / rate_hz

    # Settle at the center — same length as a step phase — running the test
    # gains and live feedforward, before the scored step. The ramp arrival
    # (or a gain change from the previous sweep candidate) leaves the joint
    # ringing, and the feedforward differentiators start cold; stepping
    # right away scores that leftover transient as if the step caused it.
    # This phase is commanded at full rate but not logged.
    print(f"  settling at center for {hold:.1f} s ...")
    settle_start = time.monotonic()
    while time.monotonic() - settle_start < hold:
        loop_start = time.monotonic()
        _, t_ff = ff.compute(center, cached_meas(test_motor))
        await test_motor.set_impedance(center, 0.0, kp, kd, t_ff)
        if monitor is not None:
            # Keep the holders' latest readings warm through the settle …
            await monitor.sample()
        spent = time.monotonic() - loop_start
        if spent < dt:
            await asyncio.sleep(dt - spent)
    if monitor is not None:
        # … then freeze the baselines here, so any holder still finishing its
        # park ramp during the settle doesn't score as reaction-torque wobble.
        monitor.rebase()

    log: list[dict] = []
    live = LiveStream("step", joint)
    start = time.monotonic()

    for phase_target in [step_target, center]:
        phase_start = time.monotonic()
        while time.monotonic() - phase_start < hold:
            loop_start = time.monotonic()
            t = time.monotonic() - start
            _, t_ff = ff.compute(phase_target, cached_meas(test_motor))
            await test_motor.set_impedance(phase_target, 0.0, kp, kd, t_ff)
            # Position from the impedance response just cached — no separate
            # poll (see run_sine).
            meas = cached_meas(test_motor)
            actual = meas[0] if meas is not None else await test_motor.get_position()
            if monitor is not None:
                await monitor.sample()
            log.append(
                {
                    "t": round(t, 5),
                    "target": phase_target,
                    "actual": actual,
                    "error": actual - phase_target,
                    "torque": cached_torque(test_motor),
                }
            )
            live.add(t, phase_target, actual)
            spent = time.monotonic() - loop_start
            if spent < dt:
                await asyncio.sleep(dt - spent)

    live.flush()
    report_achieved_rate(log, rate_hz)
    return log, amp
