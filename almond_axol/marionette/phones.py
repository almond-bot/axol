"""What each Marionette phone is doing, from the gateway's messages.

The gateway relays, per phone: its control values and hold buttons
(``config`` with ``phone``), its pose in the operator frame (``pose``) and
every press verbatim (``input``). :class:`PhoneTracker` folds them into one
:class:`Phone` record each. It decides nothing; the session reads the
records every control tick.

The Axol phone screen (``consumers/axol/axol.html`` in the Marionette repo)
sends these actions:

- Hold buttons, repeated every 200 ms while held: ``teleop.deadman`` (the
  hold-to-move pad), ``base.turnLeft`` / ``base.turnRight`` and ``lift.up``
  / ``lift.down``.
- ``safety.estop``: Stop, a toggle. ``arm.select`` (``left`` / ``right``),
  ``camera.view`` and ``teleop.frame`` (Frame: ``axol``, or a camera's
  frame: ``head``, ``left`` or ``right``): values.
- ``arm.resetLeft`` / ``arm.resetRight``: plain buttons in the More sheet
  that send an arm back to its rest pose.
- Live sliders, streamed while dragged: ``teleop.translationScale`` (the
  motion scale, for moves and turns alike), ``base.speedLimit`` (0 to 100 %) and ``gripper.set`` (how far the gripper
  should be closed, 0 open to 1 closed).
- ``base.drive``: the drive stick, ``{"x": forward, "y": left}`` within the
  unit circle. It streams ``live`` samples while touched (repeated every
  50 ms while the finger rests) and a final sample at rest on let-go. If its
  samples stop for 400 ms the gateway ends the drag (``source: gateway``),
  and the drag stays ended until the operator lets go.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

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
HOLDS = (DEADMAN, TURN_LEFT, TURN_RIGHT, LIFT_UP, LIFT_DOWN)
RESETS = {"arm.resetLeft": "left", "arm.resetRight": "right"}

TRACKING_STATES = ("initializing", "normal", "limited", "relocalizing", "unavailable")


@dataclass(frozen=True)
class PhonePose:
    """One pose in the phone's operator frame (+x forward, +y left, +z up)."""

    position: np.ndarray  # (3,) metres
    rotation: np.ndarray  # (3, 3)
    tracking: str
    source: str
    heartbeat: bool
    calibration: str  # the gateway's calibrationVersion
    session: str
    received: float  # time.monotonic() on arrival
    stamp: float = 0.0  # the phone's own clock (monotonicTs), seconds


@dataclass
class Hold:
    """One hold button: pressed, and repeated while it stays pressed."""

    down: bool = False
    at: float = -math.inf  # last press or repeat
    presses: int = 0  # counts press edges

    def live(self, now: float, timeout: float) -> bool:
        """Pressed, and its repeats are still arriving."""
        return self.down and now - self.at <= timeout


@dataclass
class Stick:
    """The drive stick: where it is pushed while it is being dragged."""

    x: float = 0.0  # forward, -1..1
    y: float = 0.0  # left, -1..1
    live: bool = False  # a drag in progress
    at: float = -math.inf  # last sample or repeat
    dropped: bool = False  # the gateway ended the drag; wait for a let-go

    def rest(self) -> None:
        self.x = self.y = 0.0
        self.live = False


@dataclass
class Phone:
    """One phone as the gateway last described it."""

    id: str
    name: str
    color: str = "#6b6f73"
    connected: bool = False
    session: str | None = None
    values: dict[str, Any] = field(default_factory=dict)  # action -> value
    calibration: str = "uncalibrated"  # uncalibrated | capturing | captured
    epoch: int = 0
    pose: PhonePose | None = None
    holds: dict[str, Hold] = field(default_factory=lambda: {a: Hold() for a in HOLDS})
    stick: Stick = field(default_factory=Stick)
    grip: float | None = None  # requested closure, 0 open .. 1 closed
    grip_seq: int = 0
    # Presses of each arm's reset button, counted like the pad's.
    resets: dict[str, int] = field(default_factory=lambda: {"left": 0, "right": 0})

    def copy(self) -> Phone:
        return replace(
            self,
            values=dict(self.values),
            holds={a: replace(h) for a, h in self.holds.items()},
            stick=replace(self.stick),
            resets=dict(self.resets),
        )

    @property
    def number(self) -> int:
        """The gateway names phones in order: ``phone-2`` is 2."""
        try:
            return int(self.id.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            return 1_000_000

    @property
    def arm(self) -> str:
        value = self.values.get(ARM)
        return value if value in ("left", "right") else "left"

    @property
    def stop(self) -> bool:
        return self.values.get(ESTOP) is True

    @property
    def frame(self) -> str:
        """Frame: ``axol``, or the camera whose frame the arm moves in."""
        value = self.values.get(FRAME)
        return value if value in FRAMES else "axol"

    @property
    def scale(self) -> float:
        return _number(self.values.get(SCALE), 1.0, 0.05, 4.0)

    @property
    def speed(self) -> float:
        """The drive speed limit as a fraction."""
        return _number(self.values.get(SPEED_LIMIT), 50.0, 0.0, 100.0) / 100.0

    @property
    def view(self) -> str:
        value = self.values.get(VIEW)
        return value if isinstance(value, str) else "head"

    @property
    def deadman_down(self) -> bool:
        return self.holds[DEADMAN].down

    @property
    def deadman_presses(self) -> int:
        return self.holds[DEADMAN].presses

    def deadman_live(self, now: float, timeout: float) -> bool:
        """The pad is held and its renewals are still arriving."""
        return self.holds[DEADMAN].live(now, timeout)

    def pose_age(self, now: float) -> float | None:
        return None if self.pose is None else now - self.pose.received

    def base_command(
        self, now: float, stick_timeout: float, hold_timeout: float
    ) -> tuple[float, float, float, int] | None:
        """``(vx, vy, wz, lift)`` from the stick and the turn and lift holds,
        before the speed limit; zeros when nothing is touched.

        None when a control is still touched but has gone quiet: the stick
        live with no sample for ``stick_timeout``, or a hold not repeated for
        ``hold_timeout``. That is a dead source, and the base stops at once.
        """
        stick = self.stick
        vx = vy = 0.0
        if stick.live and not stick.dropped:
            if now - stick.at > stick_timeout:
                return None
            vx, vy = stick.x, stick.y
        axes = []
        for plus, minus in ((TURN_LEFT, TURN_RIGHT), (LIFT_UP, LIFT_DOWN)):
            value = 0
            for action, sign in ((plus, 1), (minus, -1)):
                hold = self.holds[action]
                if hold.down:
                    if now - hold.at > hold_timeout:
                        return None
                    value += sign
            axes.append(value)
        return vx, vy, float(axes[0]), int(axes[1])


def _number(value: Any, default: float, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    if not math.isfinite(value):
        return default
    return min(max(float(value), low), high)


def _vector(value: Any, keys: str) -> np.ndarray | None:
    if not isinstance(value, dict):
        return None
    try:
        out = np.array([float(value[k]) for k in keys], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    return out if np.all(np.isfinite(out)) else None


def _stamp(value: Any, fallback: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value):
            return float(value)
    return fallback


def parse_stick(value: Any) -> tuple[float, float] | None:
    """A stick sample ``{"x", "y"}``, kept inside the unit circle; None if
    malformed."""
    if not isinstance(value, dict):
        return None
    x, y = value.get("x"), value.get("y")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y)):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    length = math.hypot(x, y)
    if length > 1.0:
        x, y = x / length, y / length
    return float(x), float(y)


class PhoneTracker:
    """Phones by id, updated from gateway messages. Thread-safe.

    With ``allow_mirror`` off, presses from a desktop mirror of a phone's
    screen (the gateway's ``/screen``) cannot move anything: their pad,
    grip and drive inputs are ignored and only a Stop counts. Someone at a
    desk would otherwise drive an arm with a phone that nobody may be
    holding.
    """

    def __init__(
        self, clock: Callable[[], float] = time.monotonic, *, allow_mirror: bool = True
    ) -> None:
        self._clock = clock
        self._allow_mirror = allow_mirror
        self._lock = threading.Lock()
        self._phones: dict[str, Phone] = {}
        self._gateway: str | None = None  # the gateway's session id

    def snapshot(self) -> dict[str, Phone]:
        """A copy of every phone, safe to read without the lock."""
        with self._lock:
            return {pid: p.copy() for pid, p in self._phones.items()}

    def clear_offline_stops(self) -> None:
        """Forget Stop on phones that are offline (the desktop clears them)."""
        with self._lock:
            for pid, phone in list(self._phones.items()):
                if "@" in pid:
                    del self._phones[pid]  # kept from a restarted gateway for its Stop
                elif not phone.connected:
                    phone.values[ESTOP] = False

    def lost_gateway(self) -> None:
        """The gateway went away: nobody is connected or holding anything."""
        with self._lock:
            for phone in self._phones.values():
                phone.connected = False
                _let_go(phone)

    def apply(self, message: dict[str, Any]) -> None:
        """Fold one gateway message in. Unknown messages are ignored."""
        kind = message.get("type")
        with self._lock:
            if kind == "config":
                self._config(message)
            elif kind == "pose":
                self._pose(message)
            elif kind == "input":
                self._input(message)

    # -- messages (lock held) ------------------------------------------------

    def _phone(self, info: Any) -> Phone | None:
        if not isinstance(info, dict) or not isinstance(info.get("id"), str):
            return None
        phone = self._phones.get(info["id"])
        if phone is None:
            phone = self._phones[info["id"]] = Phone(info["id"], info["id"])
        if isinstance(info.get("name"), str):
            phone.name = info["name"]
        if isinstance(info.get("color"), str):
            phone.color = info["color"]
        connected = info.get("connected") is True
        if not connected:
            # A phone that drops releases everything it was holding.
            _let_go(phone)
        phone.connected = connected
        session = info.get("sessionId")
        phone.session = session if isinstance(session, str) else None
        return phone

    def _config(self, message: dict[str, Any]) -> None:
        gateway = message.get("sessionId")
        if isinstance(gateway, str) and gateway != self._gateway:
            if self._gateway is not None:
                # A restarted gateway names phones afresh. Forget the old
                # roster, except a phone that left Stop on: the robot stays
                # stopped until the desktop page clears it.
                self._phones = {
                    pid: p
                    for pid, p in self._phones.items()
                    if p.stop and not p.connected
                }
                for pid, phone in list(self._phones.items()):
                    del self._phones[pid]
                    stale = f"{pid}@{self._gateway}"
                    phone.id = stale
                    self._phones[stale] = phone
            self._gateway = gateway
        for info in message.get("phones") or []:
            self._phone(info)
        phone = self._phone(message.get("phone"))
        if phone is None:
            return
        for control in message.get("controls") or []:
            if not isinstance(control, dict):
                continue
            action = control.get("action")
            if not isinstance(action, str):
                continue
            kind = control.get("type")
            if kind == "button":
                if action in HOLDS and control.get("held") is False:
                    # The gateway's view wins: it ends holds whose renewals
                    # stopped, and a repeat never restarts them.
                    phone.holds[action].down = False
                continue
            if kind == "joystick":
                if action == DRIVE and control.get("live") is False:
                    phone.stick.rest()
                continue
            if "value" in control:
                phone.values[action] = control["value"]
        calibration = message.get("calibration")
        if isinstance(calibration, dict):
            status = calibration.get("status")
            if status in ("uncalibrated", "capturing", "captured"):
                phone.calibration = status
            epoch = calibration.get("epoch")
            if isinstance(epoch, int) and not isinstance(epoch, bool):
                phone.epoch = epoch

    def _pose(self, message: dict[str, Any]) -> None:
        phone = self._phones.get(message.get("phoneId"))  # type: ignore[arg-type]
        if phone is None:
            return
        position = _vector(message.get("position"), "xyz")
        quat = _vector(message.get("rotation"), "xyzw")
        tracking = message.get("trackingState")
        if (
            position is None
            or quat is None
            or tracking not in TRACKING_STATES
            or float(np.linalg.norm(quat)) < 1e-6
            or message.get("heartbeat") is not True
        ):
            phone.pose = None
            return
        phone.pose = PhonePose(
            position=position,
            rotation=Rotation.from_quat(quat / np.linalg.norm(quat)).as_matrix(),
            tracking=tracking,
            source=str(message.get("source")),
            heartbeat=True,
            calibration=str(message.get("calibrationVersion")),
            session=str(message.get("sessionId")),
            received=self._clock(),
            stamp=_stamp(message.get("monotonicTs"), self._clock()),
        )

    def _input(self, message: dict[str, Any]) -> None:
        phone = self._phones.get(message.get("phoneId"))  # type: ignore[arg-type]
        if phone is None:
            return
        action, value = message.get("action"), message.get("value")
        now = self._clock()
        renew = message.get("renew") is True
        live = message.get("live") is True
        if message.get("source") == "mirror" and not self._allow_mirror:
            if action == ESTOP and value is True:
                phone.values[ESTOP] = True
            return
        if action in HOLDS:
            hold = phone.holds[action]
            if value is True and renew:
                # A repeat only extends a hold this phone already has.
                if hold.down:
                    hold.at = now
            elif value is True:
                if not hold.down:
                    hold.presses += 1
                hold.down = True
                hold.at = now
            else:
                hold.down = False
        elif action == DRIVE:
            self._stick(phone, value, live, message.get("source") == "gateway", now)
        elif action == ESTOP and isinstance(value, bool):
            # Act on Stop from the press itself; the config follows.
            phone.values[ESTOP] = value
        elif action == ARM and value in ("left", "right"):
            phone.values[ARM] = value
        elif action == FRAME and value in FRAMES:
            phone.values[FRAME] = value
        elif action in (SCALE, SPEED_LIMIT):
            # Live slider samples; the gateway's config only follows a drag's
            # start and end.
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(value):
                    phone.values[action] = float(value)
        elif action == GRIP and not renew:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(value):
                    phone.grip = min(max(float(value), 0.0), 1.0)
                    phone.grip_seq += 1
        elif action in RESETS and not renew:
            phone.resets[RESETS[action]] += 1

    @staticmethod
    def _stick(phone: Phone, value: Any, live: bool, ended: bool, now: float) -> None:
        stick = phone.stick
        if ended:
            # The gateway stopped hearing the drag. It stays ended until the
            # operator lets go, however many samples still arrive.
            stick.dropped = stick.live or stick.dropped
            stick.rest()
            return
        sample = parse_stick(value)
        if not live:
            # Let go: the final sample, at rest.
            stick.rest()
            stick.dropped = False
            return
        if stick.dropped or sample is None:
            return
        stick.x, stick.y = sample
        stick.live = True
        stick.at = now


def _let_go(phone: Phone) -> None:
    """A phone that is gone holds nothing and touches nothing."""
    phone.pose = None
    for hold in phone.holds.values():
        hold.down = False
    phone.stick.rest()
    phone.stick.dropped = False
