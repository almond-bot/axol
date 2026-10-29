"""What Axol's phone screen means, over Marionette's phone records.

Marionette's :class:`~marionette.PhoneTracker` folds the gateway's messages
into one :class:`~marionette.Phone` per phone: its holds, drags, presses,
values, pose and Stop, with the rules a robot needs (a repeat never presses
again, a drag the gateway ended stays ended until let go, Stop acts from the
press, a lost gateway lets go of everything, a Stop survives a restarted
gateway). This module only says what Axol's screen (``screen.html``) means
by each action:

- Hold buttons, repeated every 200 ms while held: ``teleop.deadman`` (the
  hold-to-move pad), ``base.turnLeft`` / ``base.turnRight`` and ``lift.up``
  / ``lift.down``.
- ``safety.estop``: Stop, a toggle. ``arm.select`` (``left`` / ``right``),
  ``camera.view`` and ``teleop.frame`` (Frame: ``axol``, or a camera's
  frame: ``head``, ``left`` or ``right``): values.
- ``arm.resetLeft`` / ``arm.resetRight``: plain buttons in the More sheet
  that send an arm back to its rest pose.
- Live sliders, streamed while dragged: ``teleop.translationScale`` (the
  motion scale, for moves and turns alike), ``base.speedLimit`` (0 to 100 %)
  and ``gripper.set`` (how far the gripper should be closed, 0 open to 1
  closed).
- ``base.drive``: the drive stick, ``{"x": forward, "y": left}`` within the
  unit circle, streamed while touched.
"""

from __future__ import annotations

from marionette import Phone, PhoneTracker

DEADMAN = "teleop.deadman"
ESTOP = "safety.estop"
ARM = "arm.select"
VIEW = "camera.view"
FRAME = "teleop.frame"
FRAMES = ("axol", "head", "left", "right")
SCALE = "teleop.translationScale"
SPEED_LIMIT = "base.speedLimit"
GRIP = "gripper.set"
DRIVE = "base.drive"
TURN_LEFT = "base.turnLeft"
TURN_RIGHT = "base.turnRight"
LIFT_UP = "lift.up"
LIFT_DOWN = "lift.down"
RESETS = {"arm.resetLeft": "left", "arm.resetRight": "right"}

__all__ = ["Phone", "PhoneTracker", "tracker"]


def tracker(*, allow_mirror: bool) -> PhoneTracker:
    """A tracker for Axol's phones.

    Without ``allow_mirror`` (the robot) a desktop mirror of a phone's screen
    (the gateway's ``/screen``) can only turn Stop on: someone at a desk would
    otherwise drive an arm with a phone that nobody may be holding.
    """
    return PhoneTracker(mirror="allow" if allow_mirror else "stop-only")


def arm(phone: Phone) -> str:
    return phone.choice(ARM, ("left", "right"), "left")


def frame(phone: Phone) -> str:
    """Frame: ``axol``, or the camera whose frame the arm moves in."""
    return phone.choice(FRAME, FRAMES, "axol")


def scale(phone: Phone) -> float:
    return phone.number(SCALE, 1.0, 0.05, 4.0)


def speed(phone: Phone) -> float:
    """The drive speed limit as a fraction."""
    return phone.number(SPEED_LIMIT, 50.0, 0.0, 100.0) / 100.0


def view(phone: Phone) -> str:
    value = phone.value(VIEW)
    return value if isinstance(value, str) else "head"


def grip(phone: Phone) -> tuple[int, float | None]:
    """``(count, closure)``: how many grip requests the phone has sent (a
    resting finger's repeats don't count) and the latest, 0 open to 1
    closed. Act on a request when the count changes."""
    drag = phone.drag(GRIP)
    value = drag.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return drag.updates, None
    return drag.updates, min(max(float(value), 0.0), 1.0)


def base_command(
    phone: Phone, now: float, stick_timeout: float, hold_timeout: float
) -> tuple[float, float, float, int] | None:
    """``(vx, vy, wz, lift)`` from the stick and the turn and lift holds,
    before the speed limit; zeros when nothing is touched.

    None when a control is still touched but has gone quiet: the stick live
    with no sample for ``stick_timeout``, or a hold not repeated for
    ``hold_timeout``. That is a dead source, and the base stops at once.
    """
    stick = phone.stick(DRIVE, now, stick_timeout)
    if stick is None:
        return None
    axes = []
    for plus, minus in ((TURN_LEFT, TURN_RIGHT), (LIFT_UP, LIFT_DOWN)):
        value = 0
        for action, sign in ((plus, 1), (minus, -1)):
            hold = phone.hold(action)
            if hold.down:
                if not hold.live(now, hold_timeout):
                    return None
                value += sign
        axes.append(value)
    return stick[0], stick[1], float(axes[0]), int(axes[1])
