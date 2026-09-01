"""FastAPI application for ``axol serve``.

Exposes a tiny JSON API the web control panel uses to list commands, launch
and stop sessions, and stream logs over a WebSocket. When a built web bundle
is available it is served too, with SPA-style fallback to ``index.html``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import secrets
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

from ..constants import (
    ARM_JOINTS,
    CAN_LEFT,
    CAN_MANTIS_LEFT,
    CAN_MANTIS_RIGHT,
    CAN_RIGHT,
    URDF_PATH,
)
from ..utils import adb, ports
from ..utils.can_channels import require_distinct_axol_channels, require_mantis_channels
from ..utils.certs import ACCEPT_PAGE_HTML
from ..utils.state_files import (
    mark_privileged_service,
    privileged_service_active,
    validated_service_dataset_root,
)
from ..utils.sudo import prime_sudo
from .commands import (
    COMMANDS,
    command_specs,
    flag_enabled,
    get_schema,
    normalize_boolean_args,
    operation_ids,
)
from .manager import Session, SessionManager
from .robot_link import STATE_ERROR, RobotLink, scoped_motor_faults
from .runner import OperationRunner
from .settings import SettingsStore, advanced_schema, settings_schema
from .telemetry import DiagnosticsRunStore, TelemetryHub
from .update import SelfUpdater

_logger = logging.getLogger(__name__)


class RunRequest(BaseModel):
    command: str
    args: dict[str, Any] = {}


class DiagnosticsRunRequest(BaseModel):
    """Launch a catalog command as a *diagnostics run*: the session is wrapped
    in a persisted run record with the telemetry observed while it ran."""

    command: str
    args: dict[str, Any] = {}


class OpStartRequest(BaseModel):
    """Start one of the four in-process core operations.

    ``cameras`` (optional) carries the local ZED camera setup for teleop /
    collect-data / run-policy, e.g.::

        {
          "serials": {"overhead": 41234567, "left_arm": ..., "right_arm": ...},
          "mantis_serials": {"left_arm": ..., "right_arm": ...},
          "stream_resolution": "HD1200",   # capture res → headset; "off" disables
          "record_resolution": "SVGA",     # dataset downscale; "off" disables
          "stream": {"overhead": "both", "left_arm": true},   # per-slot headset
          "record": {"overhead": "left", "left_arm": false}   # per-slot dataset
        }

    ``mantis_serials`` is a separate two-camera assignment used whenever the
    operation's Mantis toggle is on; Axol continues to use ``serials``.
    The ``stream`` / ``record`` maps decide per camera whether it takes part in
    each branch: ``false`` opts a camera out, ``true`` opts a mono camera in, and
    an eye name (``"both"`` / ``"left"`` / ``"right"``) opts a stereo camera in
    with that eye selection. The runner folds all of this into the operation's
    config (serials, capture/record resolution, per-camera stream/record enable,
    per-eye selection). Whether a slot is stereo is auto-detected from its
    serial, not passed in. The legacy ``"resolution"`` key is still accepted as
    the streaming resolution.
    """

    op: str
    args: dict[str, Any] = {}
    cameras: dict[str, Any] | None = None


class RobotConnectRequest(BaseModel):
    """Optional CAN interface selection for a robot-link connect.

    ``channelsSet`` distinguishes "connect with the stored/default interfaces"
    (an empty body) from an explicit selection. A ``None`` channel disables
    that arm, so a single non-Axol-hub adapter can drive one arm only. The
    Axol and Mantis selections are persisted independently. Selecting or
    swapping the rig's hub channels therefore cannot overwrite the robot's
    channels, and the next Mantis teleop/collection run uses the same map.
    """

    leftChannel: str | None = None
    rightChannel: str | None = None
    channelsSet: bool = False
    profile: Literal["axol", "mantis"] = "axol"
    # Old clients omit this and therefore remain explicit/manual connects.
    # Browser startup sets it so a successful manual Disconnect can remain
    # authoritative across every tab connected to this serve process.
    automatic: bool = False


class SettingsUpdateRequest(BaseModel):
    """Partial update of the shared operator settings (serve/settings.py).

    ``values`` and ``advanced`` merge per key (``null`` resets a key to its
    default); ``cameras`` replaces the stored camera spec wholesale. Omitted
    sections are left untouched.
    """

    values: dict[str, Any] | None = None
    cameras: dict[str, Any] | None = None
    # Distinguish "clear the cameras" (null) from "don't touch them" (omitted).
    camerasSet: bool = False
    advanced: dict[str, Any] | None = None


class EpisodeRequest(BaseModel):
    """A control command for the running op, as named in its own snapshot.

    ``run-policy`` takes ``start`` | ``s`` | ``r`` | ``q``; ``collect-data``
    takes ``start`` | ``s`` | ``r`` | ``continue``; ``waypoints`` takes
    ``record`` | ``undo`` | ``clear`` | ``grip-left`` | ``grip-right`` |
    ``play`` | ``stop`` | ``quit``. A managed Mantis bridge also accepts
    ``bridge-reset``. The panel sends back whatever the operation published
    in ``controls`` plus that bridge action.
    """

    command: str


class ProximityRequest(BaseModel):
    """Disable (default) or restore the headset's proximity sensor over adb.

    Disabling keeps the Quest awake with nobody wearing it, so headless
    sessions driven from the panel don't die when the headset is set down.
    """

    disabled: bool = True


class SessionInputRequest(BaseModel):
    """A line written to a session's stdin (answers an interactive prompt).

    Empty ``line`` (the default) sends a bare newline — i.e. "press Enter".
    """

    line: str = ""


# Ports the launched commands expose on the serve host.
_VIEWER_PORT = 8002  # viser sim 3D viewer
_VR_PORT = ports.VR_PORT  # VR teleop WebSocket server (shared with the adb tunnel)


def _lan_ip() -> str:
    """Best-effort LAN IP of this machine (the one a headset/peer can reach)."""
    from ..utils.network import local_ip

    return local_ip()


# ARPHRD_CAN in /sys/class/net/<iface>/type — identifies CAN interfaces.
_ARPHRD_CAN = "280"

# The idle link surveys once per second.  A lift-cycle launch consumes its
# arm/capability snapshot as a physical-hardware interlock, so more than three
# missed survey intervals is stale rather than evidence that the requested
# hardware is still attached.
_ROBOT_SURVEY_MAX_AGE_S = 3.0


def _lift_cycle_link_error(
    status: dict[str, Any], args: dict[str, Any], *, now: float | None = None
) -> str | None:
    """Explain why the idle Axol survey cannot authorize ``diag.lift-cycle``."""
    if status.get("profile") != "axol":
        return "Lift cycle requires a connected Axol hardware profile."
    if status.get("state") != "connected" or not status.get("connected"):
        return "Connect the Axol robot link before starting Lift cycle."

    last_ping = status.get("lastPing")
    current = time.time() if now is None else now
    if (
        isinstance(last_ping, bool)
        or not isinstance(last_ping, (int, float))
        or not math.isfinite(float(last_ping))
        or not 0 <= current - float(last_ping) <= _ROBOT_SURVEY_MAX_AGE_S
    ):
        return (
            "The Axol robot survey is stale; reconnect the robot link and wait "
            "for fresh motor status before starting Lift cycle."
        )

    left_selected = not flag_enabled(args.get("no_left"))
    right_selected = not flag_enabled(args.get("no_right"))
    if not left_selected and not right_selected:
        return "Lift cycle must select at least one Axol arm."
    left = args.get("left_channel", CAN_LEFT) if left_selected else None
    right = args.get("right_channel", CAN_RIGHT) if right_selected else None
    try:
        selected = require_distinct_axol_channels((left, right))
    except ValueError as exc:
        return str(exc)
    if (left_selected and selected[0] is None) or (
        right_selected and selected[1] is None
    ):
        return "Lift cycle must assign a CAN channel to every selected Axol arm."
    reported_channels = status.get("channels")
    active = (
        (
            reported_channels.get("left"),
            reported_channels.get("right"),
        )
        if isinstance(reported_channels, dict)
        else (None, None)
    )
    selected_mismatches = (left_selected and active[0] != selected[0]) or (
        right_selected and active[1] != selected[1]
    )
    if selected_mismatches:
        return (
            "Lift cycle's selected arm channels do not match the connected Axol "
            f"survey (selected left={selected[0] or 'disabled'}, "
            f"right={selected[1] or 'disabled'}; connected "
            f"left={active[0] or 'disabled'}, right={active[1] or 'disabled'}). "
            "Reconnect with the requested arm mapping before starting."
        )

    return None


_ROM_COMMANDS = frozenset({"diag.rom-enable", "diag.rom-disable"})


def _survey_timestamp_is_fresh(last_ping: Any, *, now: float | None = None) -> bool:
    """Whether a link timestamp is recent enough to authorize physical motion."""
    current = time.time() if now is None else now
    return not (
        isinstance(last_ping, bool)
        or not isinstance(last_ping, (int, float))
        or not math.isfinite(float(last_ping))
        or not 0 <= current - float(last_ping) <= _ROBOT_SURVEY_MAX_AGE_S
    )


def _motor_survey_error(
    status: dict[str, Any],
    *,
    expected_profile: str | None = None,
    now: float | None = None,
) -> str | None:
    """Require a connected, recent survey for any impending motor motion."""
    profile = status.get("profile")
    profile_label = str(expected_profile or profile).capitalize()
    if expected_profile is not None and profile != expected_profile:
        return (
            f"Connect the {profile_label} robot link before starting; the idle "
            f"link currently represents {str(profile).capitalize() or 'another profile'}."
        )
    if status.get("state") != "connected" or not status.get("connected"):
        return f"Connect the {profile_label} robot link before starting motor motion."
    if not _survey_timestamp_is_fresh(status.get("lastPing"), now=now):
        return (
            f"The {profile_label} robot survey is stale; reconnect the robot link "
            "and wait for fresh motor status before starting."
        )
    return None


def _channel_override(args: dict[str, Any], key: str) -> str | None:
    """Return a meaningful explicit channel value, preserving ``"null"``."""
    if key not in args or args[key] is None:
        return None
    text = str(args[key]).strip()
    return text or None


def _prepare_two_side_motor_args(
    args: dict[str, Any],
    active: tuple[str | None, str | None],
    *,
    command_label: str,
    validate_overrides: bool = True,
) -> str | None:
    """Bind a two-arm command to the surveyed buses, deriving omitted sides."""
    selected = 0
    for index, side in enumerate(("left", "right")):
        channel_key = f"{side}_channel"
        skip_key = f"no_{side}"
        surveyed = active[index]
        override = _channel_override(args, channel_key)
        if validate_overrides and override is not None and override != surveyed:
            return (
                f"{command_label}'s {side} CAN channel override ({override}) does "
                f"not match the connected survey ({surveyed or 'disabled'}). "
                "Reconnect with the requested mapping before starting."
            )

        skip = flag_enabled(args.get(skip_key))
        if surveyed is None:
            if not validate_overrides and override is not None:
                if not skip:
                    selected += 1
                continue
            args.pop(channel_key, None)
            if skip_key in args and not skip:
                return (
                    f"{command_label} selected the {side} side, but the connected "
                    f"survey has no {side} CAN channel."
                )
            args[skip_key] = True
            continue

        # The subprocess must use the exact interface the idle link just surveyed,
        # including custom adapters. An explicit skip still wins and leaves that
        # surveyed side untouched.
        if override is None:
            args[channel_key] = surveyed
        if not skip:
            selected += 1

    if selected == 0:
        return f"{command_label} must select at least one connected side."
    return None


def _prepare_motor_launch_args(
    command_id: str,
    status: dict[str, Any],
    args: dict[str, Any],
    *,
    now: float | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Validate and bind one motor subprocess to the latest idle-link survey.

    Browser-provided profile/channel fields are a convenience, not a safety
    boundary. This pass runs for both launch endpoints while the shared launch
    reservation is held, before the link releases its CAN sockets.
    """
    prepared = dict(args)
    profile = status.get("profile")
    profile_label = str(profile).capitalize() if profile else "Robot"
    survey_error = _motor_survey_error(status, now=now)
    if survey_error is not None:
        return prepared, survey_error

    reported = status.get("channels")
    raw_channels = (
        (reported.get("left"), reported.get("right"))
        if isinstance(reported, dict)
        else (None, None)
    )
    try:
        if profile == "mantis":
            active: tuple[str | None, str | None] = require_mantis_channels(
                raw_channels
            )
        elif profile == "axol":
            active = require_distinct_axol_channels(raw_channels)
        else:
            return prepared, "The connected robot survey has an unknown profile."
    except ValueError as exc:
        return prepared, f"The connected robot survey is unsafe: {exc}"

    if command_id in _ROM_COMMANDS:
        requested_target = prepared.get("target")
        if requested_target is not None and str(requested_target).strip():
            requested = str(requested_target).strip().lower()
            if requested != profile:
                return (
                    prepared,
                    f"ROM target {requested!r} does not match the connected "
                    f"{profile_label} profile.",
                )
        prepared["target"] = profile

        raw_joints = prepared.get("joints")
        joint_names = (
            {
                part.strip().lower()
                for part in str(raw_joints).split(",")
                if part.strip()
            }
            if raw_joints is not None and str(raw_joints).strip()
            else set()
        )
        if profile == "mantis":
            if joint_names and joint_names != {"gripper"}:
                return prepared, "Mantis ROM supports only the gripper joint."
            prepared["joints"] = "gripper"
        elif status.get("hasGripper") is False:
            if "gripper" in joint_names:
                return (
                    prepared,
                    "The connected Axol survey has no grippers; remove gripper "
                    "from the ROM joint selection.",
                )
            if not joint_names:
                prepared["joints"] = ",".join(joint.value for joint in ARM_JOINTS)

        error = _prepare_two_side_motor_args(
            prepared,
            active,
            command_label="ROM",
        )
        return prepared, error

    if command_id == "diag.lift-cycle":
        error = _prepare_two_side_motor_args(
            prepared,
            active,
            command_label="Lift cycle",
            validate_overrides=False,
        )
        if error is None:
            error = _lift_cycle_link_error(status, prepared, now=now)
        return prepared, error

    if command_id == "diag.mantis-trigger":
        for index, side in enumerate(("left", "right")):
            key = f"{side}_channel"
            override = _channel_override(prepared, key)
            if override is not None and override != active[index]:
                return (
                    prepared,
                    f"Mantis trigger's {side} CAN channel override ({override}) "
                    f"does not match the connected survey ({active[index]}). "
                    "Reconnect with the requested mapping before starting.",
                )
            prepared[key] = active[index]
        return prepared, None

    if command_id == "motor.set-zero-pos":
        arm = str(prepared.get("arm") or "").strip().lower()
        if arm not in ("left", "right"):
            return prepared, "Set zero position requires a left or right arm."
        channel = active[0 if arm == "left" else 1]
        override = _channel_override(prepared, "channel")
        if override is not None and override != channel:
            return (
                prepared,
                f"Set zero position's {arm} CAN channel override ({override}) "
                f"does not match the connected survey ({channel or 'disabled'}).",
            )
        if channel is None:
            return (
                prepared,
                f"The connected survey has no {arm} CAN channel for zeroing.",
            )
        prepared["arm"] = arm
        prepared["channel"] = channel

    return prepared, None


def _operation_channel_arg_keys(command_id: str) -> tuple[str, str] | None:
    """The standard left/right channel fields in one operation's schema."""
    emit = get_schema(command_id).emit
    for keys in (
        ("left_channel", "right_channel"),
        ("robot_config.left_channel", "robot_config.right_channel"),
    ):
        if all(key in emit for key in keys):
            return keys
    return None


def _operation_gripper_arg_key(command_id: str) -> str | None:
    """The Axol SKU capability field in one operation's emitted schema."""
    emit = get_schema(command_id).emit
    for key in (
        "axol.has_gripper",
        "robot_config.axol_config.has_gripper",
    ):
        if key in emit:
            return key
    return None


def _prepare_operation_launch_args(
    command_id: str,
    status: dict[str, Any],
    args: dict[str, Any],
    expected_channels: tuple[str | None, str | None],
    *,
    expected_profile: str,
    now: float | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Bind an in-process hardware operation to its fresh idle-link survey."""
    prepared = dict(args)
    survey_error = _motor_survey_error(
        status,
        expected_profile=expected_profile,
        now=now,
    )
    if survey_error is not None:
        return prepared, survey_error

    reported = status.get("channels")
    raw_channels = (
        (reported.get("left"), reported.get("right"))
        if isinstance(reported, dict)
        else (None, None)
    )
    try:
        if expected_profile == "mantis":
            active: tuple[str | None, str | None] = require_mantis_channels(
                raw_channels
            )
            expected = require_mantis_channels(expected_channels)
        else:
            active = require_distinct_axol_channels(raw_channels)
            expected = require_distinct_axol_channels(expected_channels)
    except ValueError as exc:
        return prepared, str(exc)

    if active != expected:
        if expected_profile == "mantis":
            return prepared, _mantis_channel_mismatch_message(active, expected)
        return (
            prepared,
            "The effective Axol CAN mapping does not match the connected survey "
            f"(requested left={expected[0] or 'disabled'}, "
            f"right={expected[1] or 'disabled'}; connected "
            f"left={active[0] or 'disabled'}, right={active[1] or 'disabled'}). "
            "Reconnect with the requested arm mapping before starting.",
        )

    keys = _operation_channel_arg_keys(command_id)
    if keys is not None:
        for key, channel in zip(keys, active, strict=True):
            # None would be omitted by build_argv and restore the config default;
            # the literal null token is the draccus spelling for a disabled arm.
            prepared[key] = channel if channel is not None else "null"

    if expected_profile == "axol":
        gripper_key = _operation_gripper_arg_key(command_id)
        if gripper_key is not None:
            surveyed_has_gripper = status.get("hasGripper")
            if not isinstance(surveyed_has_gripper, bool):
                return (
                    prepared,
                    "The connected Axol survey did not report the gripper SKU; "
                    "reconnect before starting.",
                )
            if (
                surveyed_has_gripper is False
                and gripper_key in prepared
                and flag_enabled(prepared[gripper_key])
            ):
                return (
                    prepared,
                    "The operation enables grippers, but the connected Axol "
                    "survey is gripperless; disable has_gripper before starting.",
                )
            # Bind the parsed operation config to the surveyed SKU just like
            # the channel fields. This explicit CLI layer also overrides a
            # stale has_gripper value hidden inside config_path.
            prepared[gripper_key] = surveyed_has_gripper
    return prepared, None


def _mantis_channel_mismatch_message(
    active: tuple[str | None, str | None],
    expected: tuple[str, str],
) -> str | None:
    """Actionable preflight error when an open link uses an older rig map."""
    if active == expected:
        return None
    formatted = ", ".join(
        f"{side}={channel or 'disabled'}"
        for side, channel in zip(("left", "right"), expected, strict=True)
    )
    return (
        "The saved Mantis CAN mapping changed after this link connected "
        f"({formatted}). Disconnect and reconnect Mantis, then start again. "
        "If the physical sides are reversed, swap the two channels in "
        "Settings → Mantis first."
    )


def _list_can_interfaces() -> list[dict[str, Any]]:
    """Every SocketCAN network interface on this host (name + up state).

    Lets the UI offer real choices when the Axol hub adapter isn't present
    (its named interfaces missing) and the operator must pick the interface(s)
    of whatever CAN adapter is attached instead.
    """
    interfaces: list[dict[str, Any]] = []
    for iface in sorted(Path("/sys/class/net").glob("*")):
        try:
            if iface.joinpath("type").read_text().strip() != _ARPHRD_CAN:
                continue
            flags = int(iface.joinpath("flags").read_text().strip(), 16)
        except (OSError, ValueError):
            continue
        interfaces.append({"name": iface.name, "up": bool(flags & 0x1)})
    return interfaces


def _attached_configured_hub_profiles() -> set[str]:
    """Persisted Axol/Mantis USB identities attached before netdev creation."""
    from ..cli.can.setup import attached_configured_hub_profiles

    return attached_configured_hub_profiles()


def _attached_hub_state() -> Any:
    """Atomic configured/unresolved USB state; serials never leave the server."""
    from ..cli.can.setup import attached_hub_state

    return attached_hub_state()


_CAN_DISCOVERY_STATUSES = {
    "ready",
    "needed",
    "running",
    "configured",
    "partial",
    "unidentified",
    "error",
}
_CAN_DISCOVERY_FORCE_RETRY_SECONDS = 2.0


@dataclass
class _CanDiscoveryCache:
    """Process-local discovery result keyed by physical hardware/config epoch."""

    generation: int = 0
    status: str = "ready"
    message: str | None = None
    candidate_identities: tuple[tuple[str, str], ...] | None = None
    validation_identity: tuple[tuple[str, ...], ...] | None = None
    validated_identity: tuple[tuple[str, ...], ...] | None = None
    last_forced_retry_at: float | None = None

    def observe(self, state: Any, *, running: bool = False) -> None:
        candidates = tuple(state.candidate_identities)
        validation = tuple(state.validation_identity)
        candidates_changed = candidates != self.candidate_identities
        validation_changed = validation != self.validation_identity
        if candidates_changed or validation_changed:
            self.generation += 1
        self.candidate_identities = candidates
        self.validation_identity = validation

        if running:
            self.status = "running"
            self.message = None
            return

        validation_needed = bool(validation) and validation != self.validated_identity
        if candidates_changed or validation_changed or self.status == "running":
            self.status = "needed" if candidates or validation_needed else "ready"
            self.message = None
        elif self.status == "ready" and (candidates or validation_needed):
            self.status = "needed"

    def finish(
        self,
        state: Any,
        *,
        status: str,
        message: str | None,
        validated_identity: tuple[tuple[str, ...], ...] | None = None,
    ) -> None:
        if status not in _CAN_DISCOVERY_STATUSES - {"needed", "running"}:
            raise ValueError(f"invalid terminal CAN discovery status: {status}")
        self.observe(state, running=True)
        classified = (
            tuple(state.validation_identity)
            if validated_identity is None
            else tuple(validated_identity)
        )
        self.validated_identity = classified
        if tuple(state.validation_identity) != classified:
            # setup classified an exact physical/config epoch. Hardware that
            # arrived or changed before this rescan needs its own pass; never
            # bless it with the preceding result merely because the call won.
            self.status = "needed"
            self.message = None
            return
        self.status = status
        self.message = message

    def payload(self) -> dict[str, Any]:
        candidates = self.candidate_identities or ()
        result: dict[str, Any] = {
            "status": self.status,
            "candidateCount": len(candidates),
            "generation": self.generation,
        }
        if self.message:
            result["message"] = self.message
        return result


def _can_profile_presence(
    interfaces: list[dict[str, Any]],
    channels: tuple[str | None, str | None],
    *,
    require_both: bool,
    configured_usb_present: bool = False,
    profile_channels: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Describe whether one configured hardware profile is connectable.

    Interface identity comes from the persisted Axol/Mantis mapping, never from
    guessing what an arbitrary ``can0`` might be. A persisted profile's exact
    USB serial also counts before its driver/netdevs exist, allowing
    :meth:`RobotLink.connect` to restore them. Such a profile and an ordinary
    present-but-down interface both report ``up=False``.
    """
    left, right = channels
    enabled = [channel for channel in channels if channel is not None]
    valid = (len(enabled) == 2 if require_both else bool(enabled)) and len(
        set(enabled)
    ) == len(enabled)
    by_name = {str(interface.get("name")): interface for interface in interfaces}
    all_managed_names = {
        CAN_LEFT,
        CAN_RIGHT,
        CAN_MANTIS_LEFT,
        CAN_MANTIS_RIGHT,
    }
    uses_managed_name = any(channel in all_managed_names for channel in enabled)
    managed_names_match_profile = profile_channels is not None and all(
        channel not in all_managed_names or channel in profile_channels
        for channel in enabled
    )
    named_present = (
        valid
        and all(channel in by_name for channel in enabled)
        and (
            not uses_managed_name
            or (configured_usb_present and managed_names_match_profile)
        )
    )
    usb_bootstrap = (
        valid
        and configured_usb_present
        and profile_channels is not None
        and len(enabled) == 2
        and set(enabled) == set(profile_channels)
    )
    present = named_present or usb_bootstrap
    up = named_present and all(bool(by_name[channel].get("up")) for channel in enabled)
    return {
        "channels": {"left": left, "right": right},
        "present": present,
        "up": up,
    }


def _detect_cameras() -> dict[str, Any]:
    """Enumerate locally connected ZED cameras; never raises.

    Returns ``{"devices": [...], "error": str | None}`` — an empty device
    list with an error message when the ZED SDK / pyzed is unavailable.
    """
    try:
        from ..zed import list_zed_devices

        return {"devices": list_zed_devices(), "error": None}
    except ImportError:
        return {
            "devices": [],
            "error": "pyzed is not installed — run `axol zed.install` first",
        }
    except Exception as exc:  # noqa: BLE001 - SDK errors surface to the UI
        return {"devices": [], "error": f"{type(exc).__name__}: {exc}"}


def _usb_status_dict(status: adb.AdbStatus) -> dict[str, Any]:
    """Serialize the adb device + reverse-tunnel status for the UI."""
    return {
        "installed": status.installed,
        "serial": status.serial,
        "state": status.state,
        "reverseActive": status.reverse_active,
        "ready": status.ready,
    }


def create_app(static_dir: Path | None = None) -> FastAPI:
    # ``create_app`` is the public embedding surface as well as the factory
    # used by ``axol serve``. Mark a root embedding before constructing any
    # API-owned state so it cannot silently bypass the hosted path gates.
    if os.geteuid() == 0:
        mark_privileged_service()
        os.umask(0o027)

    app = FastAPI(title="axol serve")
    # Browser latches must not survive a quick serve restart that happens
    # entirely between two status polls. This value is opaque, process-local,
    # and intentionally regenerated for every app lifetime.
    server_instance_id = secrets.token_hex(16)
    # System setup (Jetson clock pinning, GStreamer install) is owned by the
    # host installer and its boot service (`axol jetson.setup` runs as an
    # ExecStartPre on axol.service; `axol provision` runs at install time). The
    # one exception is the self-updater (below), which delegates upgrades to
    # the hosted transaction and retains a legacy startup-provision self-heal.

    manager = SessionManager()
    hub = TelemetryHub()
    settings = SettingsStore()
    # The link opens the interfaces configured in the shared settings (the
    # Axol hub's persistent names unless the operator picked others).
    left_channel, right_channel = settings.can_channels()
    robot = RobotLink(
        left_channel, right_channel, hub=hub, has_gripper=settings.has_gripper
    )
    runner = OperationRunner(robot, settings=settings)
    runs = DiagnosticsRunStore(hub)
    # ZED devices are exclusive. Hold this across preview capture and operation
    # startup so both paths make their idle check while owning one reservation.
    camera_reservation = asyncio.Lock()
    # Operations and spawned setup/diagnostic commands share hardware (CAN,
    # cameras, and tracker dongles).  Their check-and-start sequences must be
    # atomic in both directions: without this lock, a tracker Identify could
    # pass its idle check while a Mantis operation was still starting (or vice
    # versa), leaving two readers fighting over the same device.
    session_launch_reservation = asyncio.Lock()
    # Resource-owning subprocesses become terminal just before their watcher has
    # completed cleanup. Keep those small windows reserved so a new operation
    # cannot start against a CAN link still being restored or a camera session
    # whose lifetime lease has not been released.
    diagnostic_cleanup_pending: set[str] = set()
    camera_cleanup_pending: set[str] = set()
    # A successful manual Disconnect pauses only the exact profile + channel
    # map that was disconnected.  This lives on the server so opening another
    # browser cannot immediately undo the operator's choice.  It is deliberately
    # process-local: a serve restart starts a fresh hardware-detection epoch.
    manually_disconnected_target: (
        tuple[Literal["axol", "mantis"], str | None, str | None] | None
    ) = None
    can_discovery = _CanDiscoveryCache()
    can_discovery_launch = asyncio.Lock()
    can_discovery_task: asyncio.Task[None] | None = None

    def _diagnostic_session_active() -> bool:
        return bool(diagnostic_cleanup_pending or camera_cleanup_pending) or any(
            session["status"] in ("starting", "running", "stopping")
            for session in manager.list()
        )

    def _camera_session_active() -> bool:
        if camera_cleanup_pending:
            return True
        return any(
            session["status"] in ("starting", "running", "stopping")
            and (command := COMMANDS.get(str(session.get("command")))) is not None
            and command.uses_cameras
            for session in manager.list()
        )

    def _is_idle() -> bool:
        """Safe to hand host ownership to the updater: no operation running.

        A connected robot is fine -- the hosted transaction stops the service
        and the candidate reconnects after verification; only an in-flight
        operation must not be interrupted.
        """
        if runner.is_running():
            return False
        return not _diagnostic_session_active()

    # Surfaces "update available" (a newer release tag, found via read-only
    # `git ls-remote --tags`) to the control panel via /api/update/status and
    # delegates an on-demand exact release to a transient hosted-installer
    # worker via /api/update/start. Nothing upgrades automatically. No-ops for
    # dev checkouts.
    updater = SelfUpdater(_is_idle)

    def _maintenance_launch_response() -> JSONResponse | None:
        reason = updater.launch_block_reason()
        if reason is None:
            return None
        return JSONResponse({"error": reason}, status_code=409)

    def _discovery_running() -> bool:
        return can_discovery_task is not None and not can_discovery_task.done()

    def _observe_can_state(attached: Any, *, running: bool = False) -> None:
        # Development serves commonly run unprivileged after the operator has
        # completed interactive can.setup. A strict, non-conflicting profile
        # with no unresolved hardware is safe to use, but this process cannot
        # run the root-only validation pass. Production root serves still
        # validate every newly attached configured hub once per epoch.
        if (
            os.geteuid() != 0
            and attached.configured_profiles
            and not attached.candidate_identities
        ):
            can_discovery.validated_identity = tuple(attached.validation_identity)
        can_discovery.observe(attached, running=running)

    async def _can_inventory(*, observe_discovery: bool = True) -> dict[str, Any]:
        """Build one authoritative interface/profile/discovery snapshot."""
        interfaces, attached = await asyncio.gather(
            asyncio.to_thread(_list_can_interfaces),
            asyncio.to_thread(_attached_hub_state),
        )
        if observe_discovery:
            _observe_can_state(attached, running=_discovery_running())
        usb_profiles = set(attached.configured_profiles)
        axol_channels = settings.can_channels()
        mantis_channels = settings.mantis_can_channels()
        axol_presence = _can_profile_presence(
            interfaces,
            axol_channels,
            require_both=False,
            configured_usb_present="axol" in usb_profiles,
            profile_channels=(CAN_LEFT, CAN_RIGHT),
        )
        mantis_presence = _can_profile_presence(
            interfaces,
            mantis_channels,
            require_both=True,
            configured_usb_present="mantis" in usb_profiles,
            profile_channels=(CAN_MANTIS_LEFT, CAN_MANTIS_RIGHT),
        )
        axol_presence["automaticConnectSuppressed"] = manually_disconnected_target == (
            "axol",
            axol_channels[0],
            axol_channels[1],
        )
        mantis_presence["automaticConnectSuppressed"] = (
            manually_disconnected_target
            == ("mantis", mantis_channels[0], mantis_channels[1])
        )
        return {
            "serverInstanceId": server_instance_id,
            "interfaces": interfaces,
            "profiles": {
                "axol": axol_presence,
                "mantis": mantis_presence,
            },
            "discovery": can_discovery.payload(),
        }

    async def _run_can_discovery(initial_state: Any) -> None:
        """Own the launch reservation until disconnect, setup, and rescan finish."""
        try:
            disconnected = await asyncio.to_thread(robot.disconnect)
            confirmed = await asyncio.to_thread(robot.status)
            if (
                disconnected.get("state") != "disconnected"
                or disconnected.get("connected") is not False
                or confirmed.get("state") != "disconnected"
                or confirmed.get("connected") is not False
            ):
                raise RuntimeError("robot link did not prove it released CAN")

            from ..cli.can.setup import setup_detected_hubs

            result = await asyncio.to_thread(setup_detected_hubs)
            refreshed = await asyncio.to_thread(_attached_hub_state)
            can_discovery.finish(
                refreshed,
                status=result.status,
                message=result.message,
                validated_identity=result.validation_identity,
            )
        except Exception:  # noqa: BLE001 - terminal state is surfaced safely
            _logger.exception("automatic CAN hardware discovery failed")
            try:
                refreshed = await asyncio.to_thread(_attached_hub_state)
            except Exception:  # noqa: BLE001 - retain the launch snapshot
                refreshed = initial_state
            can_discovery.finish(
                refreshed,
                status="error",
                message=(
                    "Automatic CAN discovery could not safely identify the "
                    "attached hardware. Run `axol can.setup` for details."
                ),
            )
        finally:
            session_launch_reservation.release()

    async def _launch_or_join_can_discovery(
        *, force: bool = False
    ) -> asyncio.Task[None] | JSONResponse:
        """Start one cancellation-safe discovery task or join the current one."""
        nonlocal can_discovery_task
        async with can_discovery_launch:
            if _discovery_running():
                assert can_discovery_task is not None
                return can_discovery_task

            await session_launch_reservation.acquire()
            maintenance = _maintenance_launch_response()
            busy_reason: str | None = None
            if maintenance is not None:
                busy_reason = "host maintenance is active; retry CAN discovery later"
            elif runner.is_running() or _diagnostic_session_active():
                busy_reason = (
                    "an operation or setup/diagnostics session owns hardware; "
                    "retry CAN discovery when it finishes"
                )
            if busy_reason is not None:
                session_launch_reservation.release()
                inventory = await _can_inventory()
                inventory.update({"error": busy_reason, "retryable": True})
                return JSONResponse(inventory, status_code=409)

            try:
                initial_state = await asyncio.to_thread(_attached_hub_state)
            except BaseException:
                session_launch_reservation.release()
                raise
            can_discovery.observe(initial_state)
            if force and can_discovery.status in {
                "partial",
                "unidentified",
                "error",
            }:
                now = time.monotonic()
                last_retry = can_discovery.last_forced_retry_at
                if (
                    last_retry is not None
                    and now - last_retry < _CAN_DISCOVERY_FORCE_RETRY_SECONDS
                ):
                    session_launch_reservation.release()
                    inventory = await _can_inventory()
                    retry_after = max(
                        1,
                        math.ceil(
                            _CAN_DISCOVERY_FORCE_RETRY_SECONDS - (now - last_retry)
                        ),
                    )
                    inventory.update(
                        {
                            "error": "CAN identification was just retried; wait briefly",
                            "retryable": True,
                            "retryAfterSeconds": retry_after,
                        }
                    )
                    return JSONResponse(
                        inventory,
                        status_code=429,
                        headers={"Retry-After": str(retry_after)},
                    )
                can_discovery.last_forced_retry_at = now
                can_discovery.status = "needed"
                can_discovery.message = None
            if can_discovery.status != "needed":
                session_launch_reservation.release()
                inventory = await _can_inventory()
                return JSONResponse(inventory)
            can_discovery.observe(initial_state, running=True)
            can_discovery_task = asyncio.create_task(
                _run_can_discovery(initial_state),
                name="can-hardware-discovery",
            )
            return can_discovery_task

    def _uses_managed_name(channels: tuple[str | None, str | None]) -> bool:
        managed = {
            CAN_LEFT,
            CAN_RIGHT,
            CAN_MANTIS_LEFT,
            CAN_MANTIS_RIGHT,
        }
        return any(channel in managed for channel in channels if channel is not None)

    def _managed_names_match_profile(
        profile: Literal["axol", "mantis"],
        channels: tuple[str | None, str | None],
    ) -> bool:
        expected = (
            (CAN_LEFT, CAN_RIGHT)
            if profile == "axol"
            else (CAN_MANTIS_LEFT, CAN_MANTIS_RIGHT)
        )
        all_managed = {
            CAN_LEFT,
            CAN_RIGHT,
            CAN_MANTIS_LEFT,
            CAN_MANTIS_RIGHT,
        }
        return all(
            channel not in all_managed or channel in expected
            for channel in channels
            if channel is not None
        )

    def _find_session(session_id: str) -> tuple[Session | None, Any]:
        """Resolve a session id to (session, owner) across runner + manager."""
        s = runner.get(session_id)
        if s is not None:
            return s, runner
        return manager.get(session_id), manager

    async def _motor_fault_response(
        scope_args: dict[str, Any] | None = None,
    ) -> JSONResponse | None:
        """Return the shared motor-fault rejection, or ``None`` when clear.

        ``scope_args`` (a diagnostics launch's request args) narrows the check
        to the motors that run will actually touch — an arm/joint-scoped run
        (guided zeroing of a joint subset, a one-arm ROM test, a single-motor
        tool) must not be blocked by faults on motors it never drives, e.g. a
        bench arm with only some motors on the bus.
        """
        faults = await asyncio.to_thread(robot.motor_faults)
        if scope_args:
            faults = scoped_motor_faults(faults, scope_args)
        if not faults:
            return None
        detail = ", ".join(
            f"{f['arm']} {f['joint'].lower()} ({f['problem']})" for f in faults
        )
        return JSONResponse(
            {"error": f"motor fault — fix before starting: {detail}"},
            status_code=409,
        )

    # Allow the Vite dev server (different origin) to call the API directly.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/__accept")
    async def accept_cert() -> HTMLResponse:
        """Self-closing page the web UI opens to approve the self-signed cert.

        Registered before the SPA catch-all (mounted last) so it isn't shadowed.
        """
        return HTMLResponse(ACCEPT_PAGE_HTML)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        """Process readiness used by systemd's post-update verifier.

        This endpoint deliberately does not start lazy provisioning. During a
        candidate boot the durable marker already blocks every hardware launch;
        the verifier needs only to prove that the expected backend stayed up on
        one stable systemd PID before it commits that candidate.
        """
        return {
            "ready": True,
            "version": updater.version,
            "pid": os.getpid(),
        }

    @app.get("/api/info")
    async def get_info() -> dict[str, Any]:
        """Identify the serve host so the UI can build reachable links/hints."""
        # Self-heal a host whose legacy update path skipped this build's new
        # provisioning steps; idempotent, once per process.
        # Startup provisioning mutates the same live tool environment as an
        # update. Enter it through the global launch reservation so it cannot
        # begin between another endpoint's idle check and hardware start.
        async with camera_reservation, session_launch_reservation:
            if _is_idle() and not updater.launches_blocked:
                await updater.ensure_provisioned()
        return {
            "hostname": socket.gethostname(),
            "lanIp": _lan_ip(),
            "viewerPort": _VIEWER_PORT,
            "vrPort": _VR_PORT,
            "version": updater.version,
            # Backend git commit, compared against the commit baked into the
            # web bundle at build time to warn about a UI/backend mismatch.
            # Against a release (tag-pinned) install a hosted UI compares
            # versions instead — its commit tracks main and legitimately
            # differs between releases.
            "commit": updater.commit,
            "releaseInstall": updater.release_install,
        }

    @app.get("/api/update/status")
    async def update_status(refresh: bool = False) -> dict[str, Any]:
        """Installed vs. latest release version so the UI can offer an update.

        ``refresh=1`` forces a synchronous remote check (used on connect / page
        load) so the result is current; the steady-state poll omits it and gets
        the cheap debounced/cached value.
        """
        return await updater.status(force=refresh)

    @app.post("/api/update/start")
    async def update_start() -> JSONResponse:
        """Delegate a user-initiated exact release to the hosted transaction."""
        # Setting the updater's launch barrier and checking global idleness is
        # atomic with every operation/session/camera launch below.
        async with camera_reservation, session_launch_reservation:
            started, reason = updater.start()
        if not started:
            return JSONResponse({"error": reason}, status_code=409)
        return JSONResponse({"started": True})

    # -- host power ----------------------------------------------------------

    async def _host_power(flag: str, verb: str) -> JSONResponse:
        """Run ``shutdown <flag> now`` on the serve host.

        Refused while an operation or session is running — cutting power mid-
        run would drop the arms. The hosted install runs as root; a dev serve
        escalates via ``sudo -n`` so a headless context fails fast instead of
        blocking on a password prompt.
        """
        async with session_launch_reservation:
            maintenance_reason = updater.launch_block_reason()
            if maintenance_reason is not None:
                return JSONResponse(
                    {"error": f"cannot request a host {verb}: {maintenance_reason}"},
                    status_code=409,
                )
            if not _is_idle():
                return JSONResponse(
                    {"error": "an operation or session is running — stop it first"},
                    status_code=409,
                )

            def _run() -> tuple[bool, str]:
                cmd = ["shutdown", flag, "now"]
                if os.geteuid() != 0:
                    if not prime_sudo():
                        return False, "root required (no passwordless sudo)"
                    cmd = ["sudo", "-n", *cmd]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                return proc.returncode == 0, (proc.stderr or proc.stdout).strip()

            ok, detail = await asyncio.to_thread(_run)
        if not ok:
            return JSONResponse(
                {"error": f"{verb} failed: {detail or 'unknown error'}"},
                status_code=500,
            )
        return JSONResponse({"ok": True})

    @app.post("/api/host/shutdown")
    async def host_shutdown() -> JSONResponse:
        """Power off the serve host (``shutdown -h now``)."""
        return await _host_power("-h", "shutdown")

    @app.post("/api/host/restart")
    async def host_restart() -> JSONResponse:
        """Reboot the serve host (``shutdown -r now``)."""
        return await _host_power("-r", "restart")

    # -- robot connection (detached CAN + 1 Hz motor ping) ------------------

    @app.get("/api/robot/status")
    async def robot_status() -> dict[str, Any]:
        return robot.status()

    def _resolve_robot_connect_target(
        req: RobotConnectRequest | None = None,
    ) -> tuple[Literal["axol", "mantis"], tuple[str | None, str | None]] | JSONResponse:
        """Validate and resolve the exact profile + channels a connect targets."""
        profile = req.profile if req is not None else "axol"
        if req is not None and req.channelsSet:
            channels: tuple[str | None, str | None] = (
                req.leftChannel,
                req.rightChannel,
            )
            if profile == "mantis":
                try:
                    channels = require_mantis_channels(channels)
                except ValueError as exc:
                    return JSONResponse({"error": str(exc)}, status_code=400)
            else:
                try:
                    channels = require_distinct_axol_channels(channels)
                except ValueError as exc:
                    return JSONResponse({"error": str(exc)}, status_code=400)
                if channels == (None, None):
                    return JSONResponse(
                        {"error": "select a CAN interface for at least one side"},
                        status_code=400,
                    )
        elif profile == "mantis":
            try:
                channels = require_mantis_channels(settings.mantis_can_channels())
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
        else:
            channels = settings.can_channels()
        return profile, channels

    async def _connect_robot(
        req: RobotConnectRequest | None,
        target: tuple[Literal["axol", "mantis"], tuple[str | None, str | None]],
    ) -> dict[str, Any] | JSONResponse:
        """Connect the link to a validated profile + channel target."""
        profile, channels = target
        if req is not None and req.channelsSet:
            if robot.status()["state"] == "busy":
                return JSONResponse(
                    {"error": "cannot change CAN interfaces while a task owns the bus"},
                    status_code=409,
                )
            if profile == "axol":
                settings.update(
                    values={
                        "robot.left_channel": channels[0] or "null",
                        "robot.right_channel": channels[1] or "null",
                    }
                )
            else:
                settings.update(
                    values={
                        "mantis.left_channel": channels[0],
                        "mantis.right_channel": channels[1],
                    }
                )
        if channels != robot.channels() or profile != robot.profile():
            if robot.status()["state"] == "busy":
                return JSONResponse(
                    {"error": "cannot change CAN interfaces while a task owns the bus"},
                    status_code=409,
                )
            await asyncio.to_thread(robot.disconnect)
            robot.set_channels(*channels, profile=profile)
        return await asyncio.to_thread(robot.connect)

    @app.post("/api/robot/connect", response_model=None)
    async def robot_connect(
        req: RobotConnectRequest | None = None,
    ) -> dict[str, Any] | JSONResponse:
        nonlocal manually_disconnected_target
        async with session_launch_reservation:
            maintenance = _maintenance_launch_response()
            if maintenance is not None:
                return maintenance
            if runner.is_running() or _diagnostic_session_active():
                return JSONResponse(
                    {
                        "error": "cannot connect the robot link while an operation "
                        "or setup/diagnostics session owns hardware"
                    },
                    status_code=409,
                )
            target = _resolve_robot_connect_target(req)
            if isinstance(target, JSONResponse):
                return target
            profile, channels = target
            automatic = req is not None and req.automatic
            uses_managed_name = _uses_managed_name(channels)
            attached = None
            if automatic or uses_managed_name:
                # Do not rely on the browser having won its inventory poll.
                # A direct/stale automatic request must observe raw USB state
                # before it can open any CAN interface.
                attached = await asyncio.to_thread(_attached_hub_state)
                _observe_can_state(attached, running=_discovery_running())
            if automatic and (
                can_discovery.status in {"needed", "running", "unidentified", "error"}
                or (
                    attached is not None
                    and attached.candidate_identities
                    and not uses_managed_name
                )
            ):
                return JSONResponse(
                    {
                        "error": "automatic connection is waiting for CAN "
                        "hardware discovery"
                    },
                    status_code=409,
                )
            if uses_managed_name:
                if not _managed_names_match_profile(profile, channels):
                    return JSONResponse(
                        {
                            "error": "the selected managed CAN interface belongs "
                            "to a different hardware profile"
                        },
                        status_code=409,
                    )
                assert attached is not None
                if (
                    profile not in attached.configured_profiles
                    or can_discovery.status
                    in {"needed", "running", "unidentified", "error"}
                ):
                    return JSONResponse(
                        {
                            "error": "the managed CAN profile has not passed "
                            "hardware discovery for this attachment; retry after "
                            "CAN discovery completes"
                        },
                        status_code=409,
                    )
            target_key = (profile, channels[0], channels[1])
            if automatic and manually_disconnected_target == target_key:
                return JSONResponse(
                    {
                        "error": "automatic connection paused after manual disconnect",
                        "automaticConnectSuppressed": True,
                    },
                    status_code=409,
                )
            if automatic and manually_disconnected_target != target_key:
                # A different selected profile or a changed saved map is new
                # automatic intent and releases the old, narrowly scoped pause.
                manually_disconnected_target = None
            result = await _connect_robot(req, target)
            if not automatic and not isinstance(result, JSONResponse):
                # An actionable manual Connect supersedes a prior Disconnect.
                # A validation/ownership rejection above does not.
                manually_disconnected_target = None
            return result

    @app.post("/api/robot/disconnect", response_model=None)
    async def robot_disconnect() -> dict[str, Any] | JSONResponse:
        nonlocal manually_disconnected_target
        async with session_launch_reservation:
            maintenance = _maintenance_launch_response()
            if maintenance is not None:
                return maintenance
            if runner.is_running() or _diagnostic_session_active():
                return JSONResponse(
                    {
                        "error": "cannot disconnect the robot link while an operation "
                        "or setup/diagnostics session owns hardware"
                    },
                    status_code=409,
                )
            target_key = (robot.profile(), *robot.channels())
            result = await asyncio.to_thread(robot.disconnect)
            if result.get("state") == "disconnected":
                manually_disconnected_target = target_key
            return result

    @app.get("/api/can/interfaces", response_model=None)
    async def can_interfaces() -> dict[str, Any] | JSONResponse:
        """SocketCAN inventory, trusted profiles, and discovery state."""
        # The discovery worker owns the launch reservation across root
        # mutation. Do not wait behind it: expose `running` so every tab
        # suppresses connection while the interfaces are being renamed.
        if _discovery_running():
            maintenance = _maintenance_launch_response()
            if maintenance is not None:
                return maintenance
            return await _can_inventory(observe_discovery=False)
        async with session_launch_reservation:
            maintenance = _maintenance_launch_response()
            if maintenance is not None:
                return maintenance
            return await _can_inventory()

    @app.post("/api/can/discover", response_model=None)
    async def can_discover(force: bool = False) -> dict[str, Any] | JSONResponse:
        """Positively identify and persist fresh Axol/Mantis hub roles."""
        if os.geteuid() != 0:
            return JSONResponse(
                {"error": "automatic CAN discovery requires root axol serve"},
                status_code=403,
            )
        task = await _launch_or_join_can_discovery(force=force)
        if isinstance(task, JSONResponse):
            return task
        # Request cancellation must not cancel a thread performing root setup.
        # The process-owned task remains strongly referenced and shutdown joins it.
        await asyncio.shield(task)
        return await _can_inventory()

    @app.get("/api/robot/motors/{arm}/{joint}")
    async def robot_motor_details(arm: str, joint: str) -> JSONResponse:
        """One-motor full readout (the ``motor.info`` set) over the idle link."""
        async with session_launch_reservation:
            maintenance = _maintenance_launch_response()
            if maintenance is not None:
                return maintenance
            try:
                details = await asyncio.to_thread(robot.motor_details, arm, joint)
            except KeyError:
                return JSONResponse({"error": "unknown motor"}, status_code=404)
            except RuntimeError as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
        return JSONResponse(details)

    # -- motor telemetry (diagnostics dashboard) -----------------------------

    @app.get("/api/telemetry")
    async def telemetry_snapshot() -> dict[str, Any]:
        """Link state + latest fast frame + latest slow sweep for every motor."""
        return hub.snapshot()

    @app.get("/api/telemetry/history")
    async def telemetry_history(
        seconds: float = 120.0, max_frames: int = 2000
    ) -> dict[str, Any]:
        """Buffered telemetry frames for chart backfill on page load."""
        return {"frames": hub.history(seconds, max_frames)}

    @app.websocket("/api/telemetry/ws")
    async def telemetry_ws(ws: WebSocket) -> None:
        """Live telemetry stream: frame / slow / state messages (see telemetry.py)."""
        await ws.accept()
        queue = hub.subscribe()
        try:
            await ws.send_json({"type": "hello", **hub.snapshot()})
            while True:
                await ws.send_json(await queue.get())
        except WebSocketDisconnect:
            pass
        finally:
            hub.unsubscribe(queue)

    # -- diagnostics runs (script launches with telemetry capture) -----------

    async def _watch_diagnostics_run(
        meta: dict[str, Any] | None,
        session: Session,
        uses_can_bus: bool,
        uses_cameras: bool,
    ) -> None:
        """Wait for exit, release lifetime hardware leases, and persist the run."""
        queue = manager.subscribe(session)
        try:
            while session.status in ("starting", "running", "stopping"):
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue  # re-check status: end-of-stream may have raced us
                if line is None:
                    break
        finally:
            manager.unsubscribe(session, queue)
            if uses_can_bus or uses_cameras:
                # The process status is already terminal here, so retain the
                # shared reservation explicitly until the idle link owns CAN
                # again and any camera lease is cleared. The launch lock makes
                # the pending-check and discard atomic with every launch and
                # direct camera endpoint.
                async with session_launch_reservation:
                    try:
                        if uses_can_bus:
                            await asyncio.to_thread(robot.reacquire)
                    finally:
                        if uses_can_bus:
                            diagnostic_cleanup_pending.discard(session.id)
                        if uses_cameras:
                            camera_cleanup_pending.discard(session.id)
        if meta is not None:
            await asyncio.to_thread(
                runs.finalize,
                meta,
                session.status,
                session.exit_code,
                list(session.log),
            )

    async def _launch_subprocess_command(
        command_id: str,
        args: dict[str, Any],
        *,
        stdin_pipe: bool,
    ) -> tuple[Any, Session, bool, bool] | JSONResponse:
        """Atomically reserve shared hardware and spawn one catalog command."""
        command = COMMANDS.get(command_id)
        if command is None:
            return JSONResponse(
                {"error": f"unknown command: {command_id}"}, status_code=400
            )

        async with session_launch_reservation:
            maintenance = _maintenance_launch_response()
            if maintenance is not None:
                return maintenance
            # Subprocess-backed commands do not pass through OperationRunner,
            # so fold their declared shared settings here. This stays inside
            # the maintenance reservation: schema loading may import command
            # code from the tool environment an update replaces. It also must
            # happen before profile/fault scoping and argv construction so the
            # checks, process, session, and history share one effective launch.
            try:
                launch_args = normalize_boolean_args(
                    command_id, settings.merged_args(command_id, args)
                )
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            if runner.is_running():
                return JSONResponse(
                    {"error": "an operation is running — stop it first"},
                    status_code=409,
                )
            if _diagnostic_session_active():
                return JSONResponse(
                    {"error": "another session is running — stop it first"},
                    status_code=409,
                )
            profile = robot.profile()
            if profile not in command.hardware_profiles:
                allowed = " or ".join(command.hardware_profiles)
                return JSONResponse(
                    {
                        "error": f"{command.label} requires the {allowed} "
                        f"hardware profile; the connected profile is {profile}"
                    },
                    status_code=409,
                )
            if command.drives_motors:
                status = robot.status()
                if status.get("profile") != profile:
                    return JSONResponse(
                        {
                            "error": "The robot link profile changed during launch; "
                            "retry after reconnecting."
                        },
                        status_code=409,
                    )
                launch_args, link_error = _prepare_motor_launch_args(
                    command_id,
                    status,
                    launch_args,
                )
                if link_error is not None:
                    return JSONResponse({"error": link_error}, status_code=409)
                # Session metadata retains the full submitted/effective mapping,
                # while build_argv deliberately drops unknown keys. Apply the
                # same schema boundary to safety scoping so an ignored key cannot
                # hide a fault on hardware the spawned process will still drive.
                valid_scope_keys = get_schema(command_id).emit
                fault_scope_args = {
                    key: value
                    for key, value in launch_args.items()
                    if key in valid_scope_keys
                }
                if command_id == "diag.lift-cycle":
                    # Lift cycle never touches grippers. Keep its arm/side
                    # scoping, but do not block it on an unrelated gripper
                    # fault from the idle survey. Only the two side flags from
                    # this command's schema may narrow the scope: arbitrary
                    # request keys are dropped by build_argv and must not be
                    # able to hide a fault on hardware the process will drive.
                    fault_scope_args = {
                        key: launch_args[key]
                        for key in ("no_left", "no_right")
                        if key in launch_args
                    }
                    fault_scope_args["joints"] = ",".join(
                        joint.name for joint in ARM_JOINTS
                    )
                fault_response = await _motor_fault_response(
                    scope_args=fault_scope_args
                )
                if fault_response is not None:
                    return fault_response

            # A camera-only diagnostic (ZED cable check) doesn't touch the CAN
            # bus, so leave the idle motor telemetry streaming while it runs.
            uses_can_bus = command.uses_can_bus
            uses_cameras = command.uses_cameras
            if uses_can_bus:
                try:
                    await asyncio.to_thread(robot.release)
                except Exception as exc:  # noqa: BLE001 - preserve safety lockout
                    return JSONResponse(
                        {
                            "error": "Could not release the robot CAN link; the "
                            "command was not started and hardware remains locked. "
                            f"Reconnect the robot link before retrying: {exc}"
                        },
                        status_code=409,
                    )
            try:
                session = await manager.start(
                    command_id, launch_args, stdin_pipe=stdin_pipe
                )
            except Exception:
                if uses_can_bus:
                    await asyncio.to_thread(robot.reacquire)
                raise

            if uses_can_bus:
                if session.status == "error":
                    await asyncio.to_thread(robot.reacquire)
                else:
                    diagnostic_cleanup_pending.add(session.id)
            if uses_cameras and session.status != "error":
                camera_cleanup_pending.add(session.id)
            return command, session, uses_can_bus, uses_cameras

    @app.post("/api/diagnostics/run")
    async def diagnostics_run(req: DiagnosticsRunRequest) -> JSONResponse:
        # Diagnostics commands open the CAN bus (or reconfigure its interfaces)
        # themselves, so the launch does the same single-owner dance as the
        # in-process operations: refuse while something else owns the bus, and
        # hand the idle link's buses over for the duration of the run.
        # A writable stdin lets the UI answer the diagnostic's hands-on prompts
        # (the "Continue" button) via /input below.
        launched = await _launch_subprocess_command(
            req.command, req.args, stdin_pipe=True
        )
        if isinstance(launched, JSONResponse):
            return launched
        command, session, uses_can_bus, uses_cameras = launched
        # Only the Diagnostics tests are recorded in the run history; the
        # ad-hoc launches (CAN bring-up, motor calibration tools) still get
        # the bus handover + prompt plumbing but leave no record behind.
        record = command.category == "Diagnostics"
        meta = runs.begin(session.id, req.command, session.args) if record else None
        if session.status == "error":
            if meta is not None:
                await asyncio.to_thread(
                    runs.finalize,
                    meta,
                    session.status,
                    session.exit_code,
                    list(session.log),
                )
        else:
            asyncio.create_task(
                _watch_diagnostics_run(meta, session, uses_can_bus, uses_cameras)
            )
        return JSONResponse({"run": meta, "session": session.to_dict()})

    @app.get("/api/diagnostics/runs")
    async def diagnostics_runs() -> dict[str, Any]:
        return {"runs": await asyncio.to_thread(runs.list)}

    @app.delete("/api/diagnostics/runs")
    async def diagnostics_runs_clear() -> dict[str, Any]:
        """Delete the whole run history (the dashboard's Clear button)."""
        return {"removed": await asyncio.to_thread(runs.clear)}

    @app.get("/api/diagnostics/runs/{run_id}")
    async def diagnostics_run_data(run_id: str) -> JSONResponse:
        data = await asyncio.to_thread(runs.load, run_id)
        if data is None:
            return JSONResponse({"error": "unknown run"}, status_code=404)
        return JSONResponse(data)

    # -- local ZED cameras ---------------------------------------------------

    @app.get("/api/cameras/detect", response_model=None)
    async def cameras_detect() -> dict[str, Any] | JSONResponse:
        """List locally connected ZED cameras (serial, model, mono/stereo)."""
        async with camera_reservation, session_launch_reservation:
            maintenance = _maintenance_launch_response()
            if maintenance is not None:
                return maintenance
            if runner.is_running() or _camera_session_active():
                return JSONResponse(
                    {"error": "cannot detect cameras while they are in use"},
                    status_code=409,
                )
            return await asyncio.to_thread(_detect_cameras)

    @app.get("/api/cameras/preview/{serial}", response_model=None)
    async def camera_preview(serial: int) -> Response | JSONResponse:
        """One live JPEG frame from a connected ZED, so operators can tell
        which physical camera a serial belongs to. Cameras are exclusive:
        refused while an operation may be using them."""
        async with camera_reservation, session_launch_reservation:
            maintenance = _maintenance_launch_response()
            if maintenance is not None:
                return maintenance
            if runner.is_running() or _camera_session_active():
                return JSONResponse(
                    {"error": "cannot preview cameras while they are in use"},
                    status_code=409,
                )

            def _capture() -> bytes:
                from ..zed.snapshot import snapshot_jpeg

                return snapshot_jpeg(serial)

            try:
                data = await asyncio.to_thread(_capture)
            except ImportError:
                return JSONResponse(
                    {"error": "pyzed is not installed — run `axol zed.install` first"},
                    status_code=503,
                )
            except KeyError as exc:
                return JSONResponse({"error": str(exc)}, status_code=404)
            except Exception as exc:  # noqa: BLE001 - surface capture errors to the UI
                return JSONResponse(
                    {"error": f"{type(exc).__name__}: {exc}"}, status_code=502
                )
            return Response(
                content=data,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )

    @app.post("/api/cameras/restart-daemon")
    async def cameras_restart_daemon() -> JSONResponse:
        """Restart the ZED X daemon so cameras plugged in after boot enumerate."""
        async with camera_reservation, session_launch_reservation:
            maintenance = _maintenance_launch_response()
            if maintenance is not None:
                return maintenance
            if runner.is_running() or _camera_session_active():
                return JSONResponse(
                    {"error": "cannot restart the ZED daemon while cameras are in use"},
                    status_code=409,
                )

            def _restart() -> dict[str, Any]:
                try:
                    from ..zed import restart_zed_daemon

                    restart_zed_daemon()
                    return {"ok": True, "error": None}
                except Exception as exc:  # noqa: BLE001 - surface to the UI
                    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

            result = await asyncio.to_thread(_restart)
            return JSONResponse(result, status_code=200 if result["ok"] else 500)

    # -- shared operator settings (see serve/settings.py) --------------------

    @app.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        """Stored shared settings + the schemas describing every category."""
        return {
            **settings.snapshot(),
            "schema": settings_schema(),
            "advancedSchema": advanced_schema(),
        }

    @app.put("/api/settings")
    async def put_settings(req: SettingsUpdateRequest) -> JSONResponse:
        try:
            snapshot = settings.update(
                values=req.values,
                cameras=req.cameras
                if (req.camerasSet or req.cameras is not None)
                else ...,
                advanced=req.advanced,
            )
        except (KeyError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except OSError:
            return JSONResponse(
                {"error": "could not securely write the shared settings"},
                status_code=500,
            )
        return JSONResponse(snapshot)

    def quest_calibration_key() -> object:
        snapshot = settings.snapshot()
        values = snapshot.get("values")
        return (
            values.get("mantis.quest_tracker_key") if isinstance(values, dict) else None
        )

    async def redacted_json_body(
        request: Request,
    ) -> tuple[dict[str, Any] | None, JSONResponse | None]:
        """Parse a JSON object without echoing a possibly secret request body."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - parser detail may contain body data
            return None, JSONResponse(
                {"error": "request body must be valid JSON"}, status_code=400
            )
        if not isinstance(body, dict):
            return None, JSONResponse(
                {"error": "request body must be a JSON object"}, status_code=400
            )
        return body, None

    # -- tracker setup files -------------------------------------------------

    @app.get("/api/tracker/ultimate/wifi")
    async def get_ultimate_wifi() -> dict[str, Any]:
        """Non-secret Ultimate shared-map Wi-Fi configuration status."""
        from .tracker_setup import ultimate_wifi_snapshot

        return await asyncio.to_thread(ultimate_wifi_snapshot)

    @app.put("/api/tracker/ultimate/wifi", response_model=None)
    async def put_ultimate_wifi(request: Request) -> JSONResponse:
        """Save shared-map Wi-Fi values without ever returning the password."""
        from .tracker_setup import TrackerSetupError, save_ultimate_wifi

        body, error = await redacted_json_body(request)
        if error is not None:
            return error
        try:
            result = await asyncio.to_thread(save_ultimate_wifi, body)
        except TrackerSetupError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except OSError:
            return JSONResponse(
                {"error": "could not write the Ultimate Wi-Fi configuration"},
                status_code=400,
            )
        return JSONResponse(result)

    @app.get("/api/tracker/calibration/{source}", response_model=None)
    async def get_tracker_calibration(source: str) -> JSONResponse:
        """Measured TCP calibration for the exact active tracker identities."""
        from .tracker_setup import TrackerSetupError, calibration_snapshot

        try:
            result = await asyncio.to_thread(
                calibration_snapshot, source, quest_calibration_key()
            )
        except TrackerSetupError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except OSError:
            return JSONResponse(
                {"error": "could not inspect the tracker calibration"},
                status_code=400,
            )
        return JSONResponse(result)

    @app.put("/api/tracker/calibration/{source}", response_model=None)
    async def put_tracker_calibration(source: str, request: Request) -> JSONResponse:
        """Merge measured TCP calibration for one or both Mantis sides."""
        from .tracker_setup import TrackerSetupError, save_calibration

        body, error = await redacted_json_body(request)
        if error is not None:
            return error
        try:
            result = await asyncio.to_thread(
                save_calibration, source, body, quest_calibration_key()
            )
        except TrackerSetupError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except OSError:
            return JSONResponse(
                {"error": "could not write the tracker calibration"},
                status_code=400,
            )
        return JSONResponse(result)

    @app.delete("/api/tracker/calibration/{source}/{side}", response_model=None)
    async def delete_tracker_calibration(
        source: str, side: str, key: str, active_key: str, revision: str
    ) -> JSONResponse:
        """Remove one selected relevant override under an active-key guard."""
        from .tracker_setup import TrackerSetupError, remove_calibration

        try:
            result = await asyncio.to_thread(
                remove_calibration,
                source,
                side,
                key,
                active_key,
                revision,
                quest_calibration_key(),
            )
        except TrackerSetupError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except OSError:
            return JSONResponse(
                {"error": "could not remove the tracker calibration"},
                status_code=400,
            )
        return JSONResponse(result)

    @app.get("/api/tracker/bindings", response_model=None)
    async def get_tracker_bindings() -> dict[str, Any] | JSONResponse:
        """Saved bindings plus non-invasive setup readiness for each source."""

        def inspect() -> dict[str, Any]:
            from ..cli.tracker_install import lighthouse_readiness
            from ..cli.tracker_ultimate import (
                is_ultimate_tracker_key,
                ultimate_runtime_readiness,
            )
            from ..mantis.calibration import (
                DESIGN_TCP_TRANSFORMS,
                INVALID_TRANSFORM_ENTRY,
                STALE_TRANSFORM_ENTRY,
                candidate_transform_for,
                design_transform_for,
                has_conflicting_transform_override,
                load_tcp_transforms,
                parse_quest_tracker_key,
                select_quest_transform_key,
            )
            from ..tracker import load_tracker_config
            from ..vr.server import get_last_quest_pose_datum
            from .tracker_setup import TrackerSetupError, calibration_snapshot

            config = load_tracker_config()
            transform_entry_statuses: dict[tuple[str, str], str] = {}
            transform_document_errors: list[str] = []
            saved_transforms = load_tcp_transforms(
                entry_statuses=transform_entry_statuses,
                document_errors=transform_document_errors,
            )
            bindings: dict[str, dict[str, Any]] = {}
            source_status: dict[str, dict[str, Any]] = {}

            def transform_status(
                family: str, devices: dict[str, Any]
            ) -> dict[str, str]:
                result: dict[str, str] = {}
                for side in ("left", "right"):
                    device = devices.get(side)
                    key = f"{family}:{device}" if device else family
                    if transform_document_errors:
                        result[side] = "missing"
                    elif key in saved_transforms.get(side, {}):
                        result[side] = "measured"
                    elif (
                        transform_entry_statuses.get((side, key))
                        == STALE_TRANSFORM_ENTRY
                    ):
                        result[side] = "stale"
                    elif (
                        transform_entry_statuses.get((side, key))
                        == INVALID_TRANSFORM_ENTRY
                    ):
                        result[side] = "missing"
                    elif has_conflicting_transform_override(
                        side,
                        key,
                        saved_transforms,
                        transform_entry_statuses,
                    ):
                        result[side] = "missing"
                    elif design_transform_for(side, key) is not None:
                        result[side] = "factory"
                    elif candidate_transform_for(side, key) is not None:
                        result[side] = "candidate"
                    else:
                        result[side] = "missing"
                return result

            resolved: dict[str, dict[str, Any]] = {}
            for backend in ("survive", "ultimate"):
                saved = config.bindings.get(backend, {})
                left = saved.get("left")
                right = saved.get("right")
                if backend == config.backend:
                    left = config.left or left
                    right = config.right or right
                resolved[backend] = {"left": left, "right": right}
                bindings[backend] = {
                    "complete": bool(
                        left
                        and right
                        and left != right
                        and (
                            backend != "ultimate"
                            or (
                                is_ultimate_tracker_key(left)
                                and is_ultimate_tracker_key(right)
                            )
                        )
                    ),
                    "left": left,
                    "right": right,
                }

            common_quest_keys = {
                key
                for key in set(saved_transforms.get("left", {}))
                & set(saved_transforms.get("right", {}))
                if parse_quest_tracker_key(key) is not None
            }
            common_quest_keys.update(
                key
                for key, sides in DESIGN_TCP_TRANSFORMS.items()
                if "left" in sides
                and "right" in sides
                and parse_quest_tracker_key(key) is not None
            )
            configured_quest_key = settings.snapshot()["values"].get(
                "mantis.quest_tracker_key"
            )
            if configured_quest_key is not None:
                configured_quest_key = str(configured_quest_key).strip() or None
            quest_key = configured_quest_key or select_quest_transform_key(
                saved_transforms
            )
            quest_datum = (
                parse_quest_tracker_key(quest_key) if quest_key is not None else None
            )
            quest_transforms = {
                side: (
                    "measured"
                    if quest_key is not None
                    and quest_key in saved_transforms.get(side, {})
                    else "factory"
                    if quest_key is not None
                    and design_transform_for(side, quest_key) is not None
                    else "candidate"
                    if quest_key is not None
                    and candidate_transform_for(side, quest_key) is not None
                    else "missing"
                )
                for side in ("left", "right")
            }
            source_status["quest"] = {
                "binding": "automatic-handedness",
                "installed": True,
                "transforms": quest_transforms,
                "calibrationKey": quest_key,
                "controllerProfile": quest_datum[0] if quest_datum else None,
                "poseSpace": quest_datum[1] if quest_datum else None,
                "availableCalibrationKeys": sorted(common_quest_keys),
                "datumStatus": (
                    "configured"
                    if quest_datum is not None
                    else "invalid"
                    if configured_quest_key is not None
                    else "ambiguous"
                    if len(common_quest_keys) > 1
                    else "missing"
                ),
                "liveDatum": get_last_quest_pose_datum(),
            }
            lighthouse = lighthouse_readiness()
            source_status["lighthouse"] = {
                **lighthouse,
                "binding": bindings["survive"],
                "transforms": transform_status("survive", resolved["survive"]),
            }
            ultimate = ultimate_runtime_readiness()
            try:
                ultimate_calibration = calibration_snapshot("ultimate")
            except (OSError, TrackerSetupError):
                # Readiness remains a diagnostic endpoint when an operator-
                # editable calibration file is malformed. The production CLI
                # preflight still fails closed because no transform loads.
                ultimate_transforms = {"left": "missing", "right": "missing"}
            else:
                ultimate_transforms = {
                    side: (
                        "missing"
                        if ultimate_calibration[side]["status"] == "unbound"
                        else ultimate_calibration[side]["status"]
                    )
                    for side in ("left", "right")
                }
            source_status["ultimate"] = {
                **ultimate,
                "binding": bindings["ultimate"],
                "transforms": ultimate_transforms,
                "quatOrder": config.ultimate_quat_order,
                "upAxis": config.ultimate_up_axis,
            }
            left_channel, right_channel = settings.mantis_can_channels()
            return {
                "bindings": bindings,
                "sources": source_status,
                "channels": {"left": left_channel, "right": right_channel},
            }

        async with session_launch_reservation:
            maintenance = _maintenance_launch_response()
            if maintenance is not None:
                return maintenance
            return await asyncio.to_thread(inspect)

    # -- datasets on disk (the operation panels' shared repo-id picker) --------

    @app.get("/api/datasets")
    async def get_datasets() -> dict[str, Any]:
        """LeRobot datasets on this host, newest first.

        A hosted root service always scans its validated immutable dataset
        store. A plain non-root embedding retains the shared ``recording.root``
        setting/default used by direct CLI commands.
        """
        from pathlib import Path

        from ..recording.datasets import list_datasets

        if privileged_service_active():
            base = validated_service_dataset_root()
        else:
            stored_root = settings.snapshot()["values"].get("recording.root")
            base = Path(str(stored_root)).expanduser() if stored_root else None
        found = await asyncio.to_thread(list_datasets, base)
        return {
            "datasets": [
                {
                    "repoId": d.repo_id,
                    "root": d.root,
                    "episodes": d.episodes,
                    "fps": d.fps,
                }
                for d in found
            ]
        }

    # -- robot model (URDF + meshes for the pose editor) ---------------------

    @app.get("/api/urdf/{asset_path:path}", response_model=None)
    async def urdf_asset(asset_path: str) -> FileResponse | JSONResponse:
        """Serve the robot URDF and its STL meshes to the web pose editor."""
        base = URDF_PATH.parent.resolve()
        target = (base / asset_path).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        media = "model/stl" if target.suffix == ".stl" else "application/xml"
        return FileResponse(target, media_type=media)

    # -- Quest-over-USB (adb reverse pose tunnel) ---------------------------

    @app.get("/api/usb/status", response_model=None)
    async def usb_status() -> dict[str, Any] | JSONResponse:
        """adb device + reverse-tunnel status for the Quest-over-USB pose link."""
        async with session_launch_reservation:
            maintenance = _maintenance_launch_response()
            if maintenance is not None:
                return maintenance
            return _usb_status_dict(await asyncio.to_thread(adb.status))

    @app.post("/api/usb/connect", response_model=None)
    async def usb_connect() -> dict[str, Any] | JSONResponse:
        """Forward the headset's localhost:VR_PORT to this host via `adb reverse`.

        The first adb command against a freshly plugged-in headset also triggers
        the USB-debugging authorization popup on the device.
        """
        async with session_launch_reservation:
            maintenance = _maintenance_launch_response()
            if maintenance is not None:
                return maintenance
            return _usb_status_dict(await asyncio.to_thread(adb.connect))

    @app.post("/api/usb/proximity")
    async def usb_proximity(req: ProximityRequest) -> JSONResponse:
        """Disable/restore the headset's proximity sensor (`adb shell am broadcast`).

        Disabled, the headset stays awake with nobody wearing it — headless
        sessions driven from the panel keep their pose stream and camera relay.
        The override holds until restored or the headset reboots. Needs an
        attached, authorized headset (same requirement as the pose tunnel).
        """
        async with session_launch_reservation:
            maintenance = _maintenance_launch_response()
            if maintenance is not None:
                return maintenance
            ok, error = await asyncio.to_thread(
                adb.set_proximity_disabled, req.disabled
            )
        if not ok:
            return JSONResponse(
                {"error": error or "adb broadcast failed"}, status_code=502
            )
        return JSONResponse({"ok": True})

    # -- in-process operations (teleop / gravity / collect / policy) --------

    @app.get("/api/op/status")
    async def op_status() -> dict[str, Any]:
        session = runner.current()
        return {
            "running": runner.is_running(),
            "session": session.to_dict() if session else None,
            "policy": runner.policy_state(),
        }

    @app.post("/api/op/start")
    async def op_start(req: OpStartRequest) -> JSONResponse:
        if req.op not in operation_ids():
            return JSONResponse(
                {"error": f"unknown operation: {req.op}"}, status_code=400
            )
        async with camera_reservation, session_launch_reservation:
            maintenance = _maintenance_launch_response()
            if maintenance is not None:
                return maintenance
            if _diagnostic_session_active():
                return JSONResponse(
                    {
                        "error": "a setup or diagnostics session is running — "
                        "stop it first"
                    },
                    status_code=409,
                )
            # A faulted motor (over-temp, stall, encoder error, unreachable, …)
            # must block every hardware operation — driving through a fault risks
            # the arm. A sim run never touches the motors, and a robot-free run
            # (teleop's cart_only) never touches the *arms*, so both stay allowed.
            cmd = COMMANDS[req.op]
            try:
                launch_args = normalize_boolean_args(
                    req.op, settings.merged_args(req.op, req.args)
                )
                requested_mantis = flag_enabled(launch_args.get("mantis"))
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)

            if requested_mantis and not cmd.supports_mantis:
                return JSONResponse(
                    {
                        "error": f"{req.op} does not support Mantis; "
                        "use teleop or collect-data"
                    },
                    status_code=400,
                )
            mantis_mode = cmd.supports_mantis and requested_mantis
            is_sim = cmd.sim_flag is not None and flag_enabled(
                launch_args.get(cmd.sim_flag)
            )
            robot_free = is_sim or any(
                flag_enabled(launch_args.get(flag)) for flag in cmd.robot_free_flags
            )
            hardware_profile = "mantis" if mantis_mode else "axol"
            needs_motor_survey = cmd.uses_can_bus and (
                not robot_free or hardware_profile == "mantis"
            )
            if needs_motor_survey:
                try:
                    if mantis_mode:
                        expected_channels = require_mantis_channels(
                            settings.effective_mantis_can_channels(req.op, launch_args)
                        )
                    else:
                        expected_channels = settings.effective_axol_can_channels(
                            req.op, launch_args
                        )
                except (KeyError, ValueError) as exc:
                    return JSONResponse({"error": str(exc)}, status_code=400)

                launch_args, link_error = _prepare_operation_launch_args(
                    req.op,
                    robot.status(),
                    launch_args,
                    expected_channels,
                    expected_profile=hardware_profile,
                )
                if link_error is not None:
                    return JSONResponse({"error": link_error}, status_code=409)

            if mantis_mode and needs_motor_survey:
                try:
                    expected_channels = require_mantis_channels(expected_channels)
                except ValueError as exc:
                    return JSONResponse({"error": str(exc)}, status_code=400)
                interfaces = {
                    item["name"]: bool(item["up"]) for item in _list_can_interfaces()
                }
                missing_interfaces = [
                    channel
                    for channel in expected_channels
                    if channel not in interfaces
                ]
                if missing_interfaces:
                    return JSONResponse(
                        {
                            "error": "Configured Mantis CAN interface not present: "
                            + ", ".join(missing_interfaces)
                            + ". Plug in the hub (or run `axol can.setup`), then "
                            "reconnect Mantis before starting."
                        },
                        status_code=409,
                    )
                down_interfaces = [
                    channel for channel in expected_channels if not interfaces[channel]
                ]
                if down_interfaces:
                    return JSONResponse(
                        {
                            "error": "Configured Mantis CAN interface is down: "
                            + ", ".join(down_interfaces)
                            + ". Reconnect Mantis to bring both channels up before "
                            "starting."
                        },
                        status_code=409,
                    )
            if needs_motor_survey:
                fault_response = await _motor_fault_response()
                if fault_response is not None:
                    return fault_response
            try:
                session = runner.start(
                    req.op,
                    launch_args,
                    cameras=req.cameras,
                    loop=asyncio.get_running_loop(),
                )
            except RuntimeError as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
            # OperationRunner records synchronous config failures in the
            # session, but a RobotLink release failure is also a bus-ownership
            # lockout. Surface that one as an actionable HTTP failure while
            # retaining the terminal session for status/log inspection.
            if (
                session.status == "error"
                and needs_motor_survey
                and robot.status().get("state") == STATE_ERROR
            ):
                return JSONResponse(
                    {
                        "error": "Could not release the robot CAN link; the "
                        "operation was not started and hardware remains locked. "
                        "Reconnect the robot link before retrying: "
                        f"{session.error or 'unknown release failure'}",
                        "session": session.to_dict(),
                    },
                    status_code=409,
                )
            return JSONResponse(session.to_dict())

    @app.post("/api/op/stop")
    async def op_stop() -> JSONResponse:
        session = await asyncio.to_thread(runner.stop)
        if session is None:
            return JSONResponse({"error": "no operation running"}, status_code=404)
        return JSONResponse(session.to_dict())

    @app.post("/api/op/episode")
    async def op_episode(req: EpisodeRequest) -> JSONResponse:
        ok = runner.episode_command(req.command)
        if not ok:
            return JSONResponse({"error": "no episode control active"}, status_code=409)
        return JSONResponse({"ok": True})

    @app.get("/api/commands")
    async def get_commands() -> list[dict[str, Any]]:
        return command_specs()

    @app.get("/api/sessions")
    async def get_sessions() -> list[dict[str, Any]]:
        sessions = manager.list()
        current = runner.current()
        if current is not None:
            sessions.append(current.to_dict())
        return sessions

    @app.post("/api/run")
    async def run(req: RunRequest) -> JSONResponse:
        # Legacy/plain command launch shares exactly the same reservation as
        # the diagnostics dashboard; otherwise it would be an API-level bypass
        # around the operation/diagnostic single-owner guarantee.
        launched = await _launch_subprocess_command(
            req.command, req.args, stdin_pipe=False
        )
        if isinstance(launched, JSONResponse):
            return launched
        _command, session, uses_can_bus, uses_cameras = launched
        if session.status != "error":
            asyncio.create_task(
                _watch_diagnostics_run(None, session, uses_can_bus, uses_cameras)
            )
        return JSONResponse(session.to_dict())

    @app.post("/api/sessions/{session_id}/stop")
    async def stop(session_id: str) -> JSONResponse:
        # In-process operation sessions are stopped through the runner.
        if runner.get(session_id) is not None:
            session = await asyncio.to_thread(runner.stop)
            return JSONResponse(session.to_dict() if session else {"ok": True})
        ok = await manager.stop(session_id)
        if not ok:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        session = manager.get(session_id)
        return JSONResponse(session.to_dict() if session else {"ok": True})

    @app.post("/api/sessions/{session_id}/input")
    async def session_input(session_id: str, req: SessionInputRequest) -> JSONResponse:
        """Answer a session's interactive prompt (the diagnostics Continue button)."""
        session, _owner = _find_session(session_id)
        if session is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        ok = await session.send_input(req.line)
        if not ok:
            return JSONResponse(
                {"error": "session is not accepting input"}, status_code=409
            )
        return JSONResponse({"ok": True})

    @app.get("/api/sessions/{session_id}/log")
    async def get_log(session_id: str, offset: int = 0) -> JSONResponse:
        """Offset-based log poll (HTTP alternative to the WebSocket below)."""
        session, _owner = _find_session(session_id)
        if session is None:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        lines, next_offset = session.read_log(offset)
        return JSONResponse(
            {
                "lines": lines,
                "nextOffset": next_offset,
                "status": session.status,
                "exitCode": session.exit_code,
            }
        )

    @app.websocket("/api/sessions/{session_id}/logs")
    async def logs(ws: WebSocket, session_id: str) -> None:
        await ws.accept()
        session, owner = _find_session(session_id)
        if session is None:
            await ws.send_json({"type": "error", "message": "unknown session"})
            await ws.close()
            return

        queue = owner.subscribe(session)
        try:
            # Replay the buffered backlog first.
            for line in list(session.log):
                await ws.send_json({"type": "log", "line": line})
            await ws.send_json({"type": "status", "session": session.to_dict()})

            while True:
                line = await queue.get()
                if line is None:
                    await ws.send_json({"type": "status", "session": session.to_dict()})
                    break
                await ws.send_json({"type": "log", "line": line})
        except WebSocketDisconnect:
            pass
        finally:
            owner.unsubscribe(session, queue)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        # Freeze maintenance first and reap any locally owned root child before
        # tearing down the sessions/hardware it is gated against. A confirmed
        # transient update worker is independently owned by systemd and is not
        # stopped by this drain.
        await updater.shutdown()
        if can_discovery_task is not None and not can_discovery_task.done():
            await asyncio.shield(can_discovery_task)
        await runner.shutdown()
        await manager.shutdown()
        await asyncio.to_thread(robot.shutdown)

    if static_dir is not None:
        _mount_spa(app, static_dir)

    return app


def _mount_spa(app: FastAPI, static_dir: Path) -> None:
    """Serve the built web bundle with client-side-routing fallback.

    Vite emits content-hashed files under ``assets/`` (safe to cache forever);
    everything else — crucially ``index.html`` — is served ``no-cache`` so a
    rebuild is picked up immediately instead of the browser serving a stale
    ``index.html`` that points at deleted asset hashes.
    """
    # Resolve the boundary once, then delegate file lookup to StaticFiles. Its
    # lookup resolves every candidate and checks ``commonpath`` before returning
    # a FileResponse, so an asset symlink cannot escape the bundle directory.
    # Keep the explicit validation below as a fail-closed boundary for raw ASGI
    # paths as well: clients such as ``curl --path-as-is`` can send ``..``
    # components that ordinary browsers normalize before making a request.
    static_root = static_dir.resolve(strict=True)
    static_files = StaticFiles(directory=static_root, follow_symlink=False)
    immutable = {"Cache-Control": "public, max-age=31536000, immutable"}
    no_cache = {"Cache-Control": "no-cache"}

    def safe_relative_path(full_path: str) -> bool:
        if os.path.isabs(full_path) or "\\" in full_path:
            return False
        if any(component in {".", ".."} for component in full_path.split("/")):
            return False
        try:
            candidate = (static_root / full_path).resolve(strict=False)
            candidate.relative_to(static_root)
        except (OSError, RuntimeError, ValueError):
            return False
        return True

    @app.get("/{full_path:path}", response_model=None)
    async def spa(full_path: str, request: Request) -> Response:
        if full_path.startswith("api/"):
            return JSONResponse({"error": "not found"}, status_code=404)
        if not safe_relative_path(full_path):
            return JSONResponse({"error": "not found"}, status_code=404)

        if full_path:
            try:
                response = await static_files.get_response(full_path, request.scope)
            except StarletteHTTPException as exc:
                if exc.status_code != 404:
                    raise
            else:
                response.headers.update(
                    immutable if full_path.startswith("assets/") else no_cache
                )
                return response

        # Only safe, in-bound application routes receive the SPA fallback.
        try:
            response = await static_files.get_response("index.html", request.scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise
            return JSONResponse({"error": "web bundle not built"}, status_code=404)
        response.headers.update(no_cache)
        return response
