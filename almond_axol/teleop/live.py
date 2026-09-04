"""Live session settings: the knobs an operator can turn *while* teleop runs.

Launch-time configuration lives in :class:`~almond_axol.teleop.config.VRTeleopConfig`
and the control panel's saved settings (:mod:`almond_axol.serve.settings`).
This module is the small subset of it — plus a couple of robot parameters —
that is worth changing mid-session without restarting: the arm-control mode,
how a grip re-engages, how hard the grippers squeeze, reach scaling, speed.

The set is declared once, here, as :data:`LIVE_SETTINGS`; the schema (type,
range, label, help) is published to every client so the in-headset HUD and
the web control panel render the same controls generically and a new knob
is a one-line addition.

Wire protocol (over the VR server's WebSocket, any client):

- client → server ``{"type": "set", "key": "<key>", "value": <value>}``
  (a boolean key also takes ``"value": "toggle"``, flipped server-side)
- server → clients ``{"type": "settings", "value": {"schema": [...],
  "values": {"<key>": <value>, ...}}}`` — on connect (via
  ``VRServer.set_announce``) and after every change, so all clients converge
  on the server's state and a rejected request simply leaves the last
  announced value in place.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .core import VRTeleopCore

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveSettingDef:
    """One live-adjustable setting, as published to clients.

    Attributes:
        key: Wire key; for ``source="core"`` also the :meth:`VRTeleopCore.set_live`
            key. ``gripper_torque`` is the one robot-side setting.
        label: Short control label (HUD space is tight: keep it to ~2 words).
        type: ``"boolean"`` toggle, ``"select"`` from ``options``, or a
            ``"number"`` stepped between ``min`` and ``max``.
        help: One sentence for the control panel's tooltip / HUD hint.
        options: Choices for ``"select"``.
        min / max / step: Range and increment for ``"number"``.
        unit: Display unit for ``"number"`` (``"Nm"``, ``"×"``, ``"rev/s"``…).
        source: ``"core"`` (applied through :meth:`VRTeleopCore.set_live`) or
            ``"robot"`` (applied to the robot object by :class:`LiveSettings`).
    """

    key: str
    label: str
    type: Literal["boolean", "select", "number"]
    help: str
    options: tuple[str, ...] = ()
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str = ""
    source: Literal["core", "robot"] = "core"


LIVE_SETTINGS: tuple[LiveSettingDef, ...] = (
    LiveSettingDef(
        key="box_mode",
        label="Box mode",
        type="boolean",
        help=(
            "One controller drives both arms as a parallel-gripper pair; the "
            "thumbsticks jog the pair. Also toggled by clicking both "
            "thumbsticks together. Switching disengages the arms first."
        ),
    ),
    LiveSettingDef(
        key="reengage",
        label="Re-engage",
        type="select",
        options=("clutch", "ramp"),
        help=(
            "What a grip does when an arm re-engages. clutch: the arm stays "
            "put and your hand's current pose becomes its origin (you match "
            "the arm). ramp: the arm eases out to where your hand is under "
            "the mapping from its previous engage (the arm matches you)."
        ),
    ),
    LiveSettingDef(
        key="hold_to_engage",
        label="Hold to engage",
        type="boolean",
        help=(
            "Grips as dead-man switches: an arm tracks only while its grip is "
            "held. Off: grips toggle each arm between tracking and frozen."
        ),
    ),
    LiveSettingDef(
        key="gripper_torque",
        label="Grip force",
        type="number",
        min=0.1,
        max=3.0,
        step=0.1,
        unit="Nm",
        help=(
            "Peak gripper torque on both hands — how hard a closed gripper "
            "squeezes. Hardware only."
        ),
        source="robot",
    ),
    LiveSettingDef(
        key="position_multiplier",
        label="Reach scale",
        type="number",
        min=0.5,
        max=2.0,
        step=0.1,
        unit="×",
        help=(
            "How far the arm moves per unit of hand motion. Above 1 extends "
            "reach beyond your own; below 1 gives finer control."
        ),
    ),
    LiveSettingDef(
        key="teleop_max_vel",
        label="Arm speed",
        type="number",
        min=0.1,
        max=1.5,
        step=0.1,
        unit="rev/s",
        help=(
            "Joint velocity cap while tracking. Lower it for a careful, "
            "slow-motion session; the arm then lags a fast hand instead of "
            "following it."
        ),
    ),
    LiveSettingDef(
        key="box_jog_speed",
        label="Jog speed",
        type="number",
        min=0.05,
        max=0.4,
        step=0.05,
        unit="m/s",
        help="Box-mode thumbstick jog speed of the arm pair at full deflection.",
    ),
)

_DEFS = {d.key: d for d in LIVE_SETTINGS}

# Joint velocities are configured in rad/s but shown in rev/s (the unit the
# rest of the docs use for teleop_max_vel).
_RAD_PER_REV = 6.283185307179586


class LiveSettings:
    """Applies and publishes the live session settings for one teleop run.

    Owns no state of its own: values are read back from the core / robot on
    every publish, so the announced snapshot can never drift from what the
    control loops actually use.

    Args:
        core: The session's :class:`VRTeleopCore` (mode switches, config).
        robot: The robot object (``left`` / ``right`` ``AxolArm`` attributes
            with an ``_arm_config`` carry the gripper torque limit; anything
            else — the sim — hides the robot-side settings).
        publish: Callback ``(snapshot: dict) -> None`` that ships the
            ``settings`` message to clients and stores it for late joiners.
    """

    def __init__(
        self,
        core: VRTeleopCore,
        robot: object,
        publish: Callable[[dict[str, Any]], None],
    ) -> None:
        self._core = core
        self._robot = robot
        self._publish = publish

    def set_robot(self, robot: object) -> None:
        """Attach (or swap) the robot the robot-side settings act on.

        For adapters that receive the robot after construction; re-announces
        so clients pick up the newly available knobs.
        """
        self._robot = robot
        self.announce()

    # -- Robot-side accessors ----------------------------------------------

    def _arms(self) -> list[Any]:
        arms = []
        for side in ("left", "right"):
            arm = getattr(self._robot, side, None)
            if arm is not None and getattr(arm, "_arm_config", None) is not None:
                arms.append(arm)
        return arms

    def _has_gripper_torque(self) -> bool:
        return any(
            getattr(a._arm_config, "gripper", None) is not None for a in self._arms()
        )

    # -- Public API ----------------------------------------------------------

    def schema(self) -> list[dict[str, Any]]:
        """The settings available in *this* session (robot-side ones only on hardware)."""
        out = []
        for d in LIVE_SETTINGS:
            if d.key == "gripper_torque" and not self._has_gripper_torque():
                continue
            entry = asdict(d)
            entry.pop("source")
            entry["options"] = list(d.options)
            out.append(entry)
        return out

    def values(self) -> dict[str, Any]:
        """Current value of every published setting, in wire units."""
        vals: dict[str, Any] = {}
        for d in LIVE_SETTINGS:
            if d.key == "gripper_torque":
                arms = [a for a in self._arms() if a._arm_config.gripper is not None]
                if not arms:
                    continue
                vals[d.key] = round(float(arms[0]._arm_config.gripper.torque_limit), 2)
            elif d.key == "teleop_max_vel":
                vals[d.key] = round(
                    float(self._core.live_value(d.key)) / _RAD_PER_REV, 2
                )
            else:
                v = self._core.live_value(d.key)
                vals[d.key] = round(float(v), 3) if isinstance(v, float) else v
        return vals

    def snapshot(self) -> dict[str, Any]:
        """``{"schema": [...], "values": {...}}`` as sent to clients."""
        return {"schema": self.schema(), "values": self.values()}

    def announce(self) -> None:
        """Publish the current snapshot (startup, connect, after a change)."""
        self._publish(self.snapshot())

    def apply(self, key: str, value: Any) -> None:
        """Validate and apply one ``set`` request, then publish.

        A boolean setting also accepts the value ``"toggle"``, flipped
        against the *server's* current value — for controller gestures (both
        thumbsticks clicked = box mode) whose client-side mirror may lag the
        server by a round trip, so a repeated gesture can never re-send a
        stale state.

        Raises ``KeyError`` / ``ValueError`` for an unknown key or an
        out-of-range value (the server logs and drops the request).
        """
        d = _DEFS.get(key)
        if d is None:
            raise KeyError(f"unknown live setting {key!r}")
        if d.type == "boolean" and value == "toggle":
            coerced = not bool(self.values().get(d.key))
        else:
            coerced = self._coerce(d, value)
        if d.key == "gripper_torque":
            arms = [a for a in self._arms() if a._arm_config.gripper is not None]
            if not arms:
                raise ValueError("gripper torque is not adjustable on this robot")
            for arm in arms:
                # Read on every gripper command (see AxolArm.motion_control),
                # so the new cap applies from the next control tick.
                arm._arm_config.gripper.torque_limit = coerced
            _logger.info("Gripper torque limit set to %.2f Nm", coerced)
            self.announce()
            return
        if d.key == "teleop_max_vel":
            coerced = coerced * _RAD_PER_REV
        # Applied on the IK thread at the next frame; the core's
        # broadcast_mode callback (wired to on_changed) re-announces then.
        self._core.set_live(d.key, coerced)

    def on_changed(self, key: str, value: Any) -> None:
        """Core change notification (``VRTeleopCore`` ``broadcast_mode``)."""
        del key, value  # the snapshot is re-read in full
        self.announce()

    @staticmethod
    def _coerce(d: LiveSettingDef, value: Any) -> Any:
        if d.type == "boolean":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "on", "yes")
            return bool(value)
        if d.type == "select":
            if value not in d.options:
                raise ValueError(
                    f"{d.key} must be one of {list(d.options)}, not {value!r}"
                )
            return str(value)
        try:
            num = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{d.key} needs a number, not {value!r}") from exc
        if d.min is not None and num < d.min - 1e-9:
            raise ValueError(f"{d.key} below minimum {d.min}: {num}")
        if d.max is not None and num > d.max + 1e-9:
            raise ValueError(f"{d.key} above maximum {d.max}: {num}")
        return round(num, 4)
