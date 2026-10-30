"""The robot's shared settings: one file, one vocabulary, every surface.

Almost nothing an operator tunes is per-run — camera assignment, arm
stiffness, per-joint gains, teleop rate, recording fps, the inference server
address — they are properties of *this robot*. This module gives them one
home on the host, ``~/.almond/settings.json``, read by the control panel,
the ``axol`` CLI and the SDK alike (see :mod:`almond_axol.settings`).

The file is a nested tree keyed by **canonical sections**
(:data:`SECTIONS`)::

    {
      "version": 2,
      "axol":       {"left_stiffness": 0.8, "left": {"elbow": {"kp": 60}}},
      "teleop":     {"frequency": 240, "rest_pose_left": [...]},
      "kinematics": {"pos_weight": 100},
      "robot":      {"left_channel": "can0", "right_channel": "null"},
      "gravity":    {"kd": 0.5},
      "recording":  {"fps": 30, "root": "/data"},
      "cameras":    {...camera spec...}
    }

Each subsystem section (``axol``, ``teleop``, ``kinematics``, ``jelly``,
``vr_server``) is the *same shape as its config dataclass*, so for ``axol
teleop`` the file is literally a partial ``--config_path`` file. The same ops
embed those subsystems at different dotted paths (stiffness is
``axol.left_stiffness`` on teleop but
``robot_config.axol_config.left_stiffness`` on collect-data); a section
declares the prefix it is grafted onto per op, and every value under it is
translated at start. A handful of op-level knobs whose leaf names differ
between ops (``recording.fps`` is ``fps`` on collect-data but
``recording.replay_fps`` is ``fps`` on replay) carry explicit per-op targets
on their :class:`SettingDef` instead.

:data:`SETTINGS` is the curated subset the panel renders as first-class
controls, grouped into UI categories; every other leaf of a section is
reachable through the panel's Advanced tree. Curated vs advanced is purely a
presentation split — the file and the stored keys do not distinguish them.

:class:`SettingsStore` persists the file (plus the camera spec) and folds it
into an op start's args. The request's own args always win, so a per-run
value can still override a shared one.

Precedence at op start (later wins): dataclass defaults → shared settings →
the request's args → the camera spec fold-in.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..tracker.trigger import validate_trigger_full_scale
from ..utils.can_channels import require_distinct_axol_channels, require_mantis_channels
from ..utils.paths import almond_path
from ..utils.state_files import secure_atomic_write_json, secure_read_text
from .commands import flag_enabled, normalize_boolean_args, parse_boolean

_logger = logging.getLogger(__name__)

SETTINGS_PATH = almond_path("settings.json")
SETTINGS_VERSION = 2

# The built-in operations, whose dotted config paths the tables below spell
# out. A registered operation joins the same tables through its
# ``settings_like`` alias rather than re-declaring them (see _settings_op).
_OPS = ("teleop", "gravity-comp", "collect-data", "run-policy", "replay-dataset")
_LEROBOT_OPS = ("collect-data", "run-policy", "replay-dataset")

# Dotted paths into the lerobot-based ops' shared robot config.
_ROBOT = "robot_config"
_AXOL = f"{_ROBOT}.axol_config"
_TELEOP_CFG = "teleop_config"
_VRT = f"{_TELEOP_CFG}.vr_teleop_config"
_KIN = f"{_TELEOP_CFG}.kinematics_config"

# Top-level keys of the settings file that are not sections.
_RESERVED_KEYS = frozenset({"version", "cameras"})


@dataclass(frozen=True)
class Section:
    """One canonical top-level key of the settings file.

    ``targets`` maps an operation id to the dotted prefix the section's
    subtree is grafted onto for that op (``""`` = the op's top level). A
    section without targets holds only curated leaves whose per-op paths are
    spelled out on their :class:`SettingDef` (their leaf names differ between
    ops, so no single prefix works).

    ``ref_op`` / ``ref_prefix`` name the op config subtree the panel's
    Advanced form for this section is built from; ``None`` keeps the section
    out of the Advanced tree (its leaves are all curated, or its op-level
    neighbours are per-run fields that must not become shared).
    ``drop_children`` removes subtrees owned elsewhere (cameras by the
    Cameras tab; axol_config by the ``axol`` section).
    """

    key: str
    label: str
    targets: dict[str, str] = field(default_factory=dict)
    ref_op: str | None = None
    ref_prefix: str | None = None
    drop_children: tuple[str, ...] = ()

    def graft(self, op: str, subpath: str) -> tuple[str, ...]:
        """The op config key(s) a ``<section>.<subpath>`` value drives."""
        prefix = self.targets.get(op)
        if prefix is None:
            return ()
        return (f"{prefix}.{subpath}" if prefix else subpath,)


SECTIONS: tuple[Section, ...] = (
    # -- shared subsystem configs: the subtree *is* the dataclass -----------
    Section(
        key="axol",
        label="Axol",
        targets={
            "teleop": "axol",
            "gravity-comp": "axol",
            **{op: _AXOL for op in _LEROBOT_OPS},
        },
        ref_op="teleop",
        ref_prefix="axol",
    ),
    Section(
        key="teleop",
        label="Teleop",
        targets={"teleop": "teleop", "collect-data": _VRT},
        ref_op="teleop",
        ref_prefix="teleop",
        # Follows axol.gripper (adopt_robot_gripper); not a setting of its own.
        drop_children=("gripper",),
    ),
    Section(
        key="kinematics",
        label="Kinematics",
        targets={"teleop": "kinematics", "collect-data": _KIN},
        ref_op="teleop",
        ref_prefix="kinematics",
    ),
    Section(
        key="jelly",
        label="Jelly",
        targets={"teleop": "jelly", "collect-data": f"{_TELEOP_CFG}.jelly"},
        ref_op="teleop",
        ref_prefix="jelly",
    ),
    Section(
        key="vr_server",
        label="VR server",
        targets={
            "teleop": "vr_server",
            "collect-data": f"{_TELEOP_CFG}.vr_server_config",
        },
        ref_op="teleop",
        ref_prefix="vr_server",
    ),
    # -- LeRobot wrappers around the shared configs --------------------------
    Section(
        key="lerobot",
        label="LeRobot robot",
        targets={op: _ROBOT for op in _LEROBOT_OPS},
        # Replay exposes the same robot subtree without importing the optional
        # ZED stack, so these controls remain available on camera-less hosts.
        ref_op="replay-dataset",
        ref_prefix=_ROBOT,
        drop_children=("cameras", "axol config"),
    ),
    Section(
        key="lerobot_teleop",
        label="LeRobot teleoperator",
        targets={"collect-data": _TELEOP_CFG},
    ),
    # -- op-level knobs grafted onto each op's top level ----------------------
    Section(
        key="robot",
        label="Robot",
        targets={
            "teleop": "",
            "gravity-comp": "",
            "diag.lift-cycle": "",
            **{op: _ROBOT for op in _LEROBOT_OPS},
        },
    ),
    Section(key="gravity", label="Gravity comp", targets={"gravity-comp": ""}),
    Section(key="inference", label="Inference", targets={"run-policy": ""}),
    # -- op-level knobs whose leaf names differ per op: explicit targets ----
    Section(key="mantis", label="Mantis"),
    Section(key="recording", label="Recording"),
    Section(key="system", label="System"),
)

_SECTIONS_BY_KEY: dict[str, Section] = {s.key: s for s in SECTIONS}


@dataclass(frozen=True)
class SettingDef:
    """One curated setting: how it renders and (if irregular) what it drives.

    ``key`` is the canonical ``<section>.<subpath>`` the value is stored
    under. Its op config keys normally follow from the section's prefix map
    (:meth:`Section.graft`); ``targets`` overrides that for the irregular
    ones — a leaf whose name differs per op, or one that fans out to several
    paths (``teleop.frequency`` also sets collect-data's ``teleop_hz``).
    ``ui`` carries optional widget hints for the front-end (slider ranges,
    the pose editor). ``options`` makes it a dropdown. ``effective_default``
    describes what actually happens when the config default is ``None``
    (e.g. "the LeRobot cache dir", "inference runs locally") — shown as the
    placeholder instead of an unhelpful "unset"; a callable is resolved on
    the serve host.
    """

    key: str
    label: str
    type: str  # "number" | "boolean" | "text" | "select"
    help: str
    targets: dict[str, tuple[str, ...]] | None = None
    options: tuple[str, ...] | None = None
    ui: dict[str, Any] = field(default_factory=dict)
    effective_default: "str | Callable[[], str] | None" = None


@dataclass(frozen=True)
class SettingCategory:
    key: str
    label: str
    description: str
    settings: tuple[SettingDef, ...]


def _all_ops(*keys: str) -> dict[str, tuple[str, ...]]:
    return {op: keys for op in _OPS}


def _settings_op(op_id: str) -> str:
    """Resolve an operation to the id whose targets it uses.

    An operation registered by a downstream package usually embeds the same
    config dataclasses as a built-in one, so its dotted paths are identical;
    declaring ``settings_like="collect-data"`` on its
    :class:`~almond_axol.serve.commands.CommandDef` inherits the whole table
    instead of restating ~50 mappings. Unknown or unaliased ids resolve to
    themselves and simply match nothing.
    """
    from .commands import COMMANDS

    cmd = COMMANDS.get(op_id)
    return cmd.settings_like if cmd is not None and cmd.settings_like else op_id


def _confine_hosted_runtime_paths(target_op: str, merged: dict[str, Any]) -> None:
    """Remove privileged path overrides before an operation config is parsed.

    The root control-panel service shares operator settings, but it must not
    let either those saved values or one request redirect libraries that open
    paths by name. VR certificate setup can create/replace both configured
    files when one half of the pair is missing, while LeRobot constructors
    create their calibration directory and read ``<id>.json``. Pin TLS to the
    one managed pair and let LeRobot use its ordinary service-local defaults.

    ``target_op`` deliberately follows :func:`_settings_op`, so operations such
    as ``collect-dagger`` that alias the collect-data config shape inherit the
    same boundary automatically. Direct CLI use and non-root embeddings never
    call this helper.
    """
    tls_prefix = {
        "teleop": "vr_server",
        "collect-data": "teleop_config.vr_server_config",
    }.get(target_op)
    if tls_prefix is not None:
        from ..utils.certs import CERTFILE, KEYFILE

        merged[f"{tls_prefix}.certfile"] = CERTFILE
        merged[f"{tls_prefix}.keyfile"] = KEYFILE

    calibration_fields = {
        "collect-data": (
            "robot_config.calibration_dir",
            "robot_config.id",
            "teleop_config.calibration_dir",
            "teleop_config.id",
        ),
        "run-policy": (
            "robot_config.calibration_dir",
            "robot_config.id",
        ),
        "replay-dataset": (
            "robot_config.calibration_dir",
            "robot_config.id",
        ),
    }.get(target_op, ())
    for field_name in calibration_fields:
        merged.pop(field_name, None)


def _mantis_channels_from_values(
    values: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Resolve stored Mantis channel values, including reset/default semantics."""
    from ..constants import CAN_MANTIS_LEFT, CAN_MANTIS_RIGHT

    def norm(key: str, default: str) -> str | None:
        value = values.get(key)
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return None if text.lower() in ("null", "none") else text

    return (
        norm("mantis.left_channel", CAN_MANTIS_LEFT),
        norm("mantis.right_channel", CAN_MANTIS_RIGHT),
    )


def _axol_channels_from_values(
    values: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Resolve defaults/nulls and validate the persisted Axol arm map."""
    from ..constants import CAN_LEFT, CAN_RIGHT

    def norm(key: str, default: str) -> str | None:
        value = values.get(key)
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return None if text.lower() in ("null", "none") else text

    return require_distinct_axol_channels(
        (
            norm("robot.left_channel", CAN_LEFT),
            norm("robot.right_channel", CAN_RIGHT),
        )
    )


def _lerobot_dataset_root() -> str:
    """The directory datasets actually land in when ``root`` is unset."""
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME

        return str(HF_LEROBOT_HOME)
    except Exception:  # noqa: BLE001 - lerobot extra may be absent (dev host)
        return "~/.cache/huggingface/lerobot"


def _parcel_only() -> dict[str, Any]:
    """``ui`` hint for box mode's settings: shown only while the parcel
    gripper is selected (the only gripper with box mode)."""
    return {"showWhen": {"key": "axol.gripper", "equals": "parcel"}}


SETTINGS: tuple[SettingCategory, ...] = (
    SettingCategory(
        key="robot",
        label="Robot",
        description=(
            "Axol's mount, arms and grippers. Every operation on this robot uses "
            "these settings."
        ),
        settings=(
            SettingDef(
                key="kinematics.mount",
                label="Mount",
                type="select",
                options=("classic", "standalone", "ox_cart", "owl"),
                help=(
                    "What Axol is mounted on: classic (Axol Legacy), standalone, "
                    "ox_cart (Ox Cart) or owl (Owl Mount). Axol uses it to keep the "
                    "arms from hitting the mount, and the 3D views show it. When Jelly "
                    "is attached, Axol on Jelly is used instead."
                ),
                ui={"widget": "mount"},
                effective_default="classic (legacy)",
            ),
            SettingDef(
                key="axol.left_stiffness",
                label="Left arm stiffness",
                type="number",
                help=(
                    "How stiff the left arm is, from 0 to 1. 1 (the default) is full "
                    "stiffness, and lower values make the arm softer. When you run a "
                    "policy, use the value you collected its data with."
                ),
                ui={"widget": "slider", "min": 0, "max": 1, "step": 0.05},
            ),
            SettingDef(
                key="axol.right_stiffness",
                label="Right arm stiffness",
                type="number",
                help=(
                    "How stiff the right arm is, from 0 to 1. 1 (the default) is full "
                    "stiffness, and lower values make the arm softer. When you run a "
                    "policy, use the value you collected its data with."
                ),
                ui={"widget": "slider", "min": 0, "max": 1, "step": 0.05},
            ),
            SettingDef(
                key="robot.left_channel",
                label="Left arm CAN channel",
                type="text",
                help=(
                    "SocketCAN interface for the left arm, such as can_alm_axol_l, or "
                    "can0 for an adapter other than the Axol hub. Enter null to turn "
                    "the arm off. The diagnostics dashboard uses this interface too."
                ),
            ),
            SettingDef(
                key="robot.right_channel",
                label="Right arm CAN channel",
                type="text",
                help=(
                    "SocketCAN interface for the right arm, such as can_alm_axol_r, or "
                    "can1 for an adapter other than the Axol hub. Enter null to turn "
                    "the arm off. The diagnostics dashboard uses this interface too."
                ),
            ),
            SettingDef(
                key="axol.has_gripper",
                label="Grippers fitted",
                type="boolean",
                help=(
                    "Turn off if this Axol has no grippers. The gripper motors then "
                    "stay off, gripper commands are ignored, and recorded datasets "
                    "leave out the gripper channels. The gripper torque and speed "
                    "settings have no effect while this is off."
                ),
            ),
            SettingDef(
                key="axol.gripper",
                label="Gripper",
                type="select",
                options=("parallel", "parcel"),
                help=(
                    "The gripper fitted to both arms. parallel is the stock two-finger "
                    "gripper. parcel is the hinged-plate parcel gripper, which adds "
                    "box mode settings to Teleop & VR. At startup, Axol measures the "
                    "gripper's travel and stops if it looks like the other type."
                ),
            ),
            SettingDef(
                key="axol.left.gripper.torque_limit",
                label="Left gripper torque limit (Nm)",
                type="number",
                help="Peak output torque of the left gripper.",
            ),
            SettingDef(
                key="axol.right.gripper.torque_limit",
                label="Right gripper torque limit (Nm)",
                type="number",
                help="Peak output torque of the right gripper.",
            ),
            SettingDef(
                key="axol.left.gripper.max_speed",
                label="Left gripper max speed (rad/s)",
                type="number",
                help="Maximum joint speed of the left gripper.",
            ),
            SettingDef(
                key="axol.right.gripper.max_speed",
                label="Right gripper max speed (rad/s)",
                type="number",
                help="Maximum joint speed of the right gripper.",
            ),
            SettingDef(
                key="robot.arms",
                label="Axol arms",
                type="boolean",
                help=(
                    "Drive Axol's arms during teleoperation. Turn off to drive only Jelly, "
                    "with or without the arms connected. Data collection and policies "
                    "always use the arms."
                ),
                targets={"teleop": ("arms",)},
            ),
            SettingDef(
                key="teleop.reset_torque_threshold",
                label="Return-to-rest contact stop (Nm)",
                type="number",
                help=(
                    "Stops a move back to rest when a joint meets this much unexpected "
                    "torque (Nm), and lets the arms go limp. Free the arms by hand, "
                    "then press X in VR or Return to rest in the panel to try again "
                    "from where they are. Raise it if normal returns stop by mistake, "
                    "or set 0 to turn the check off."
                ),
                # Also a top-level field of the ops without a teleop subtree.
                targets={
                    "teleop": ("teleop.reset_torque_threshold",),
                    "collect-data": (f"{_VRT}.reset_torque_threshold",),
                    "run-policy": ("reset_torque_threshold",),
                    "replay-dataset": ("reset_torque_threshold",),
                },
            ),
            SettingDef(
                key="teleop.teleop_torque_threshold",
                label="Teleop contact stop (Nm)",
                type="number",
                help=(
                    "Stops tracking or replay when a joint meets this much unexpected "
                    "torque (Nm), and lets the arms go limp. Free the arms by hand, "
                    "then reset to return to rest. Off by default, because "
                    "teleoperation often pushes on things on purpose. Raise it if "
                    "normal contact trips it."
                ),
                # One combined control: a switch that arms the watchdog
                # (filling in the suggested threshold) with the value
                # editable next to it; off stores 0 = disabled.
                ui={"widget": "toggle-number", "onValue": 16},
                targets={
                    "teleop": ("teleop.teleop_torque_threshold",),
                    "collect-data": (f"{_VRT}.teleop_torque_threshold",),
                    "replay-dataset": ("teleop_torque_threshold",),
                },
            ),
            SettingDef(
                key="inference.policy_torque_threshold",
                label="Policy contact stop (Nm)",
                type="number",
                help=(
                    "Ends the policy episode without saving it when a joint meets this "
                    "much unexpected torque (Nm), and lets the arms go limp. Off by "
                    "default, because a policy often pushes on things on purpose. "
                    "Raise it if normal contact trips it."
                ),
                ui={"widget": "toggle-number", "onValue": 16},
            ),
            SettingDef(
                key="gravity.kd",
                label="Gravity compensation damping (kd)",
                type="number",
                help="Damping on the joints you move by hand in Gravity compensation.",
            ),
            SettingDef(
                key="gravity.rate_hz",
                label="Gravity compensation rate (Hz)",
                type="number",
                help="Control-loop rate of Gravity compensation.",
            ),
        ),
    ),
    # Jelly (the powered omni base and telescoping lift the Axol rides on)
    # has its own settings scope in the panel, opened from its connection
    # tile: the two hardware switches here, and the drive parameters from
    # the Jelly config tree as its Parameters tab.
    SettingCategory(
        key="jelly",
        label="Jelly",
        description=(
            "Which parts of Jelly teleoperation drives. Axol detects Jelly when it is "
            "attached. Use these switches to turn parts off."
        ),
        settings=(
            SettingDef(
                key="jelly.wheels",
                label="Jelly wheels",
                type="boolean",
                help=(
                    "Drive Jelly's wheels when they are attached. The left stick moves "
                    "the base and the right stick turns it. Jelly's "
                    "motion is never recorded or driven by a policy. Set speeds and "
                    "heading hold in Jelly's Parameters tab."
                ),
            ),
            SettingDef(
                key="jelly.lift",
                label="Jelly lift",
                type="boolean",
                help=(
                    "Run Jelly's lift from the stick clicks when its controller is "
                    "attached: left lowers, right raises. Turn off to leave the lift "
                    "still even when it is attached."
                ),
            ),
        ),
    ),
    SettingCategory(
        key="teleop",
        label="Teleop & VR",
        description="How VR controller motion drives the arms (teleop and data collection).",
        settings=(
            SettingDef(
                key="mantis.source",
                label="Mantis tracking",
                type="select",
                options=("lighthouse", "ultimate", "quest"),
                help=(
                    "How Mantis data collection tracks the rigs. Quest uses the "
                    "headset's controllers, and Lighthouse and Ultimate use VIVE "
                    "trackers. Mantis teleop doesn't track the rigs. The triggers "
                    "drive the grippers."
                ),
                targets={
                    "collect-data": ("mantis_source",),
                },
            ),
            SettingDef(
                key="mantis.quest_tracker_key",
                label="Quest calibration key",
                type="text",
                help=(
                    "Identifies the Quest controllers Mantis is calibrated for, in the "
                    "form quest:<controller-profile>:grip. Follow the Headset step "
                    "below to fill it in. Used only when Mantis tracking is Quest."
                ),
                targets={
                    "collect-data": (f"{_VRT}.tracker_key",),
                },
            ),
            SettingDef(
                key="mantis.left_channel",
                label="Left Mantis CAN channel",
                type="text",
                help=("CAN interface of the left Mantis."),
                # Applied conditionally by SettingsStore.merged_args only when
                # that run has Mantis enabled; Axol runs retain robot.*.
                targets={},
                effective_default="can_mantis_l",
            ),
            SettingDef(
                key="mantis.right_channel",
                label="Right Mantis CAN channel",
                type="text",
                help=("CAN interface of the right Mantis."),
                targets={},
                effective_default="can_mantis_r",
            ),
            SettingDef(
                key="mantis.left_trigger_full_scale",
                label="Left trigger full squeeze",
                type="number",
                help=("The left trigger's reading at a full squeeze."),
                targets={
                    "teleop": ("mantis_left_trigger_full_scale",),
                    "collect-data": ("mantis_left_trigger_full_scale",),
                },
                ui={"widget": "slider", "min": 0.5, "max": 1.0, "step": 0.01},
            ),
            SettingDef(
                key="mantis.right_trigger_full_scale",
                label="Right trigger full squeeze",
                type="number",
                help=("The right trigger's reading at a full squeeze."),
                targets={
                    "teleop": ("mantis_right_trigger_full_scale",),
                    "collect-data": ("mantis_right_trigger_full_scale",),
                },
                ui={"widget": "slider", "min": 0.5, "max": 1.0, "step": 0.01},
            ),
            SettingDef(
                key="teleop.frequency",
                label="Teleop rate (Hz)",
                type="number",
                help="Control-loop rate for VR teleoperation (whole Hz).",
                # Fans out: collect-data also paces its loop from ``teleop_hz``.
                targets={
                    "teleop": ("teleop.frequency",),
                    "collect-data": ("teleop_hz", f"{_VRT}.frequency"),
                },
            ),
            SettingDef(
                key="teleop.position_multiplier",
                label="Position multiplier",
                type="number",
                help=(
                    "How far the gripper moves for each unit of controller motion. 1 "
                    "matches your hand, and higher values cover more workspace with "
                    "less hand travel."
                ),
                ui={"widget": "slider", "min": 0.5, "max": 3, "step": 0.1},
            ),
            SettingDef(
                key="teleop.rotation_multiplier",
                label="Rotation multiplier",
                type="number",
                help=(
                    "How far the gripper turns for each unit of wrist rotation. 1 "
                    "matches your wrist, and higher values turn further with less "
                    "twist."
                ),
                ui={"widget": "slider", "min": 0.5, "max": 3, "step": 0.1},
            ),
            SettingDef(
                key="teleop.hold_to_engage",
                label="Hold grips to engage",
                type="boolean",
                help=(
                    "Each arm moves only while you hold its grip, even from rest, and "
                    "freezes where it is when you let go. When off (the default), "
                    "click both grips to engage from rest, then click a grip to freeze "
                    "or resume that arm."
                ),
            ),
            SettingDef(
                key="teleop.park_to_zero",
                label="Park to zero on stop",
                type="boolean",
                help=(
                    "When you press Stop (or Ctrl+C), the arms go to rest, then hang "
                    "straight down at the all-zero pose before the motors turn off. "
                    "They keep their grasp, so nothing drops. When off, the arms stop "
                    "at the rest pose."
                ),
                # Also a top-level field of the ops without a teleop subtree.
                targets={
                    "teleop": ("teleop.park_to_zero",),
                    "collect-data": (f"{_VRT}.park_to_zero",),
                    "run-policy": ("park_to_zero",),
                    "replay-dataset": ("park_to_zero",),
                },
            ),
            SettingDef(
                key="teleop.box_mode",
                ui=_parcel_only(),
                label="Box mode at startup",
                type="boolean",
                help=(
                    "Start sessions in box mode, where the grippers clamp a box like two "
                    "flat hands and one controller moves and turns both arms. "
                    "While you lead, push the right thumbstick forward or back to set the "
                    "clamp width. The rest of the thumbsticks drive the base as usual. "
                    "Toggle it from the headset menu or by clicking both thumbsticks, which "
                    "engages the arms right away."
                ),
                targets={
                    "teleop": ("teleop.box_mode",),
                    "collect-data": (f"{_VRT}.box_mode",),
                },
            ),
            SettingDef(
                key="teleop.box_lead_hand",
                ui=_parcel_only(),
                label="Box-mode lead hand",
                type="select",
                options=("right", "left"),
                help=(
                    "The controller that leads when box mode turns on. The arms engage "
                    "right away, led by this hand. The other grip can take over the "
                    "lead, and you can change this from the headset menu."
                ),
                targets={
                    "teleop": ("teleop.box_lead_hand",),
                    "collect-data": (f"{_VRT}.box_lead_hand",),
                },
            ),
            SettingDef(
                key="teleop.box_grasp",
                ui=_parcel_only(),
                label="Box-mode grasp",
                type="select",
                options=("straight", "flush"),
                help=(
                    "The grasp box mode starts in. straight (the default) points the "
                    "fingers forward and measures the width between the mounts. flush "
                    "turns each gripper 39° inward about its contact face and measures "
                    "the width between the faces. In the headset, click either "
                    "thumbstick while leading to switch."
                ),
                targets={
                    "teleop": ("teleop.box_grasp",),
                    "collect-data": (f"{_VRT}.box_grasp",),
                },
            ),
            SettingDef(
                key="teleop.box_elbow_out",
                ui=_parcel_only(),
                label="Box-mode elbows out (°)",
                type="number",
                help=(
                    "How far out box mode holds the elbows, from 0 (hanging straight "
                    "down) to 90 (out level). This is the starting value. In the "
                    "headset, push either thumbstick forward to move the elbows apart "
                    "or back to bring them in, and the value you end with is saved "
                    "here."
                ),
                targets={
                    "teleop": ("teleop.box_elbow_out",),
                    "collect-data": (f"{_VRT}.box_elbow_out",),
                },
            ),
            SettingDef(
                key="teleop.box_elbow_weight",
                ui=_parcel_only(),
                label="Box-mode elbow hint weight",
                type="number",
                help=(
                    "How strongly IK steers the elbows toward the Box-mode elbows out "
                    "angle. The default is 10, against 100 for gripper position and 40 "
                    "for orientation. Set 0 to turn it off: the elbows go where IK "
                    "puts them, and the thumbstick elbow control does nothing."
                ),
                targets={
                    "teleop": ("teleop.box_elbow_weight",),
                    "collect-data": (f"{_VRT}.box_elbow_weight",),
                },
            ),
            SettingDef(
                key="teleop.box_squeeze_torque",
                ui=_parcel_only(),
                label="Box squeeze cap (Nm)",
                type="number",
                help=(
                    "Caps the shoulder torque that squeezes the box. Jogging the width "
                    "past contact then "
                    "presses with at most this torque. It also slows an arm moving "
                    "toward the other, so leave it at 0 (off, the default) unless you "
                    "need it. Also in the headset menu as Squeeze cap. No effect in sim."
                ),
                targets={
                    "teleop": ("teleop.box_squeeze_torque",),
                    "collect-data": (f"{_VRT}.box_squeeze_torque",),
                },
            ),
            SettingDef(
                key="teleop.box_squeeze_lean",
                ui=_parcel_only(),
                label="Box squeeze lean (×)",
                type="number",
                help=(
                    "Leans each gripper into the box as the squeeze builds, so the "
                    "clamp presses evenly on the gripper's facet and tip. At 1×, that "
                    "is about 1.3° of inward yaw per cm past contact. Raise it if the "
                    "tip lifts as you squeeze, lower it if the facet lifts, or set 0 "
                    "to turn it off. Also in the headset menu as Squeeze lean. "
                    "No effect in sim."
                ),
                targets={
                    "teleop": ("teleop.box_squeeze_lean",),
                    "collect-data": (f"{_VRT}.box_squeeze_lean",),
                },
            ),
            SettingDef(
                key="teleop.box_squeeze_trim",
                ui=_parcel_only(),
                label="Box squeeze trim (°)",
                type="number",
                help=(
                    "Turns the grippers further into the box while they press, until "
                    "their faces sit parallel to it. This helps the blade tips clamp "
                    "as hard as the roots on heavy boxes. Sets the most inward yaw it "
                    "can add, and 0 turns it off. Also in the headset menu as Squeeze "
                    "trim. No effect in sim."
                ),
                targets={
                    "teleop": ("teleop.box_squeeze_trim",),
                    "collect-data": (f"{_VRT}.box_squeeze_trim",),
                },
            ),
            SettingDef(
                key="teleop.box_squeeze_force",
                ui=_parcel_only(),
                label="Box squeeze force (N)",
                type="number",
                help=(
                    "Caps how hard each arm clamps the box. Without a cap, the force "
                    "grows by about 6 N for each cm you jog past contact. 0 (off, the "
                    "default) leaves the squeeze to the width you set, and 8 N is a "
                    "good working value. Also in the headset menu as Squeeze force. "
                    "No effect in sim."
                ),
                targets={
                    "teleop": ("teleop.box_squeeze_force",),
                    "collect-data": (f"{_VRT}.box_squeeze_force",),
                },
            ),
            SettingDef(
                key="teleop.reengage",
                label="Re-engage",
                type="select",
                options=("clutch", "ramp"),
                help=(
                    "What happens when an arm engages again after a pause, such as a "
                    "freeze or a move by hand. clutch: the arm stays put and your "
                    "controller's current pose becomes the new origin. ramp: the arm "
                    "eases back to match where your controller is. You can also switch "
                    "this from the headset menu."
                ),
                targets={
                    "teleop": ("teleop.reengage",),
                    "collect-data": (f"{_VRT}.reengage",),
                },
            ),
            SettingDef(
                key="teleop.rest_pose_left",
                label="Left arm rest pose",
                type="text",
                help=(
                    "The left arm's rest and start pose: seven joint angles in "
                    "radians. Edit it with the pose editor."
                ),
                ui={"widget": "pose"},
            ),
            SettingDef(
                key="teleop.rest_pose_right",
                label="Right arm rest pose",
                type="text",
                help=(
                    "The right arm's rest and start pose: seven joint angles in "
                    "radians. Edit it with the pose editor."
                ),
                ui={"widget": "pose"},
            ),
            SettingDef(
                key="lerobot_teleop.id",
                label="Teleoperator ID",
                type="text",
                help="LeRobot identifier used for this teleoperator instance.",
            ),
            SettingDef(
                key="lerobot_teleop.calibration_dir",
                label="Teleoperator calibration directory",
                type="text",
                help="Directory containing LeRobot teleoperator calibration files.",
            ),
        ),
    ),
    SettingCategory(
        key="kinematics",
        label="Kinematics",
        description=(
            "The IK solver every operation uses, and the jax solver's weights for "
            "teleop and data collection. The weights trade off following your hands "
            "against posture and joint limits."
        ),
        settings=(
            SettingDef(
                key="kinematics.backend",
                label="IK solver",
                type="select",
                options=("jax", "mink"),
                help=(
                    "jax (the default) uses the Pyroki solver. mink uses the MuJoCo "
                    "solver for teleop tracking, Cartesian policy actions, and "
                    "Cartesian return-to-rest and park paths. The weights in this "
                    "section tune jax only."
                ),
                # One switch for both config fields that name the backend.
                # collect-dagger inherits collect-data's (settings_like).
                targets={
                    "teleop": ("kinematics.backend",),
                    "collect-data": (
                        f"{_KIN}.backend",
                        f"{_ROBOT}.cartesian_controller",
                    ),
                    "run-policy": (f"{_ROBOT}.cartesian_controller",),
                    "replay-dataset": (f"{_ROBOT}.cartesian_controller",),
                },
            ),
            SettingDef(
                key="kinematics.pos_weight",
                label="Position weight",
                type="number",
                help="Weight on end-effector position error.",
            ),
            SettingDef(
                key="kinematics.ori_weight",
                label="Orientation weight",
                type="number",
                help="Weight on end-effector orientation error.",
            ),
            SettingDef(
                key="kinematics.elbow_weight",
                label="Elbow weight",
                type="number",
                help=(
                    "How strongly the arm follows your elbow position. 0 (the default) "
                    "turns elbow tracking off, and the solver's posture terms place "
                    "the elbow instead."
                ),
            ),
            SettingDef(
                key="kinematics.rest_weight",
                label="Rest weight",
                type="number",
                help=(
                    "How strongly each joint is held near its current angle. Higher "
                    "values reduce drift, and lower values track your hands more "
                    "closely."
                ),
            ),
            SettingDef(
                key="kinematics.max_joint_delta",
                label="Max joint delta (rad)",
                type="number",
                help="Maximum change of any joint between consecutive IK solutions.",
            ),
        ),
    ),
    SettingCategory(
        key="recording",
        label="Recording",
        description="Dataset recording for Collect data and Run policy.",
        settings=(
            SettingDef(
                key="recording.fps",
                label="Recording fps",
                type="number",
                help=(
                    "Frame rate of the recorded dataset and of policy control. The "
                    "default is 30, and 60 doubles the recording load. A policy must "
                    "run at the fps it was trained on. Above 60 the cameras capture "
                    "faster to match, which the ZED X supports only at SVGA "
                    "resolution."
                ),
                targets={
                    "collect-data": ("fps",),
                    "run-policy": ("fps",),
                },
            ),
            SettingDef(
                key="recording.replay_fps",
                label="Replay fps",
                type="number",
                help="Replay frame rate. 0 uses the dataset's recorded frame rate.",
                targets={"replay-dataset": ("fps",)},
            ),
            SettingDef(
                key="lerobot.observe_torques",
                label="Observe torques",
                type="boolean",
                help=(
                    "Include joint torques in the recorded observations. A "
                    "policy must run with the same observation shape it was "
                    "trained on."
                ),
            ),
            SettingDef(
                key="lerobot.observe_cartesian",
                label="Observe Cartesian",
                type="boolean",
                help=(
                    "Use Cartesian space for observations and actions: each "
                    "arm's end-effector as a 6-axis pose plus gripper, instead "
                    "of the 7 joint angles. Must match between data collection "
                    "and running the policy."
                ),
            ),
            SettingDef(
                key="recording.vcodec",
                label="Video codec",
                type="text",
                help=(
                    "Codec for the recorded dataset video (auto, h264, "
                    "libsvtav1, …). Leave unset for the platform default."
                ),
                targets={
                    "collect-data": ("vcodec",),
                    "run-policy": ("vcodec",),
                },
            ),
            SettingDef(
                key="recording.root",
                label="Dataset root",
                type="text",
                help=(
                    "Local folder that datasets are written to and read from. Defaults "
                    "to the LeRobot cache ($HF_LEROBOT_HOME, usually "
                    "~/.cache/huggingface/lerobot)."
                ),
                effective_default=_lerobot_dataset_root,
                targets={
                    "collect-data": ("root",),
                    "run-policy": ("root",),
                    "replay-dataset": ("root",),
                },
            ),
            SettingDef(
                key="recording.push_to_hub",
                label="Push to HuggingFace Hub",
                type="boolean",
                help="Upload the dataset to the HuggingFace Hub when recording ends.",
                targets={
                    "collect-data": ("push_to_hub",),
                    "run-policy": ("push_to_hub",),
                },
            ),
            SettingDef(
                key="recording.rerun_ip",
                label="Rerun viewer IP",
                type="text",
                help="Stream live visualization to a Rerun viewer at this address.",
                effective_default="off (not streaming to a Rerun viewer)",
                targets={
                    "collect-data": ("rerun_ip",),
                    "run-policy": ("rerun_ip",),
                },
            ),
            SettingDef(
                key="recording.rerun_port",
                label="Rerun viewer port",
                type="number",
                help="Port of the Rerun viewer.",
                targets={
                    "collect-data": ("rerun_port",),
                    "run-policy": ("rerun_port",),
                },
            ),
        ),
    ),
    SettingCategory(
        key="inference",
        label="Inference",
        description="Where and how Run policy runs the policy.",
        settings=(
            SettingDef(
                key="inference.device",
                label="Device",
                type="select",
                options=("cuda", "cpu", "mps"),
                help="Device the policy runs on (LeRobot policies only).",
            ),
            SettingDef(
                key="inference.server_host",
                label="Inference server host",
                type="text",
                help=(
                    "Address of a remote `axol inference-server`, or of your custom "
                    "policy endpoint. Leave unset to run LeRobot inference on this "
                    "machine, or to use a custom endpoint here. Start custom endpoints "
                    "yourself."
                ),
                effective_default="local (inference runs on this machine)",
            ),
            SettingDef(
                key="inference.server_port",
                label="Inference server port",
                type="number",
                help=(
                    "Port of the inference server, local or remote, or of your custom "
                    "policy endpoint."
                ),
            ),
            SettingDef(
                key="inference.episode_time_s",
                label="Episode length (s)",
                type="number",
                help="Maximum length of one policy episode.",
            ),
            SettingDef(
                key="inference.actions_per_chunk",
                label="Actions per chunk",
                type="number",
                help="Actions requested from the policy per inference call.",
            ),
            SettingDef(
                key="inference.chunk_size_threshold",
                label="Chunk size threshold",
                type="number",
                help=(
                    "Queue fraction below which the next chunk is requested "
                    "(LeRobot policies only)."
                ),
            ),
            SettingDef(
                key="inference.aggregate_fn",
                label="Chunk aggregation",
                type="select",
                options=(
                    "temporal_ensemble",
                    "weighted_average",
                    "latest_only",
                    "average",
                    "conservative",
                ),
                help="How overlapping action chunks are combined (LeRobot policies only).",
            ),
            SettingDef(
                key="inference.temporal_ensemble_coeff",
                label="Temporal ensemble coefficient",
                type="number",
                help=(
                    "Exponential weight for the temporal_ensemble aggregation "
                    "(LeRobot policies only)."
                ),
            ),
        ),
    ),
    SettingCategory(
        key="system",
        label="System",
        description="Logging and diagnostics.",
        settings=(
            SettingDef(
                key="system.hardware_profile",
                label="Active device",
                type="select",
                options=("axol", "mantis"),
                help=(
                    "Whether operations run on Axol's arms or the handheld Mantis "
                    "rigs. Change it with Select on the Axol or Mantis tile of the "
                    "control panel. Operations that need Axol wait until Axol is "
                    "selected again."
                ),
                # Not a config key: the panel translates it into each run's
                # ``mantis`` flag, so it stays visible to every operator device.
                targets={},
                effective_default="axol",
            ),
            SettingDef(
                key="system.log_level",
                label="Log level",
                type="select",
                options=("DEBUG", "INFO", "WARNING", "ERROR"),
                help="Verbosity of the operation log shown in the console.",
                targets=_all_ops("log_level"),
            ),
        ),
    ),
)

_SETTINGS_BY_KEY: dict[str, SettingDef] = {
    s.key: s for cat in SETTINGS for s in cat.settings
}


def targets_for(key: str, op: str) -> tuple[str, ...]:
    """The op config key(s) a stored canonical ``key`` drives on ``op``.

    A curated setting with explicit ``targets`` wins; every other key is
    grafted through its section's per-op prefix. Unknown sections (or a bare
    section name) drive nothing.
    """
    setting = _SETTINGS_BY_KEY.get(key)
    if setting is not None and setting.targets is not None:
        return setting.targets.get(op, ())
    section_key, _, subpath = key.partition(".")
    section = _SECTIONS_BY_KEY.get(section_key)
    if section is None or not subpath:
        return ()
    return section.graft(op, subpath)


def is_known_key(key: str) -> bool:
    """Whether a canonical key may be stored.

    Any leaf under a grafted section is fine (the op schemas decide what it
    means; ``build_argv`` drops what they don't know). Sections that only
    carry explicit-target leaves accept just their curated keys.
    """
    if key in _SETTINGS_BY_KEY:
        return True
    section_key, _, subpath = key.partition(".")
    section = _SECTIONS_BY_KEY.get(section_key)
    return section is not None and bool(subpath) and bool(section.targets)


# Curated keys of the previous flat file layout (``values``) and their
# canonical homes. One old key may fan out to several leaves (the gripper
# limits used to set both sides at once).
_LEGACY_KEYS: dict[str, tuple[str, ...]] = {
    "robot.left_stiffness": ("axol.left_stiffness",),
    "robot.right_stiffness": ("axol.right_stiffness",),
    "robot.has_gripper": ("axol.has_gripper",),
    "robot.gripper_torque_limit": (
        "axol.left.gripper.torque_limit",
        "axol.right.gripper.torque_limit",
    ),
    "robot.gripper_max_speed": (
        "axol.left.gripper.max_speed",
        "axol.right.gripper.max_speed",
    ),
    # Jelly used to be opted into by hand; it is now inferred from the CAN
    # interfaces attached (with jelly.wheels / jelly.lift as the opt-out), so
    # the old switch has no home and is dropped on load.
    "robot.jelly_enabled": (),
    "jelly.enabled": (),
    "robot.reset_torque_threshold": ("teleop.reset_torque_threshold",),
    "robot.teleop_torque_threshold": ("teleop.teleop_torque_threshold",),
    "robot.policy_torque_threshold": ("inference.policy_torque_threshold",),
    "robot.gravity_kd": ("gravity.kd",),
    "robot.gravity_rate_hz": ("gravity.rate_hz",),
    "teleop.mantis_source": ("mantis.source",),
    "teleop.id": ("lerobot_teleop.id",),
    "teleop.calibration_dir": ("lerobot_teleop.calibration_dir",),
    "recording.observe_torques": ("lerobot.observe_torques",),
    "recording.observe_cartesian": ("lerobot.observe_cartesian",),
    # The Advanced tree's old name for the teleop subsystem, and the Quest
    # datum that used to be buried in it.
    "vr_teleop.tracker_key": ("mantis.quest_tracker_key",),
    # Box mode's tool select, superseded by the robot-wide axol.gripper.
    "teleop.box_tool": (),
}


def _curated_section_aliases() -> dict[str, str]:
    """Section paths that a curated key with explicit targets already drives.

    ``mantis.quest_tracker_key`` sets ``teleop_config.vr_teleop_config
    .tracker_key`` on collect-data, which is the ``teleop`` section's
    ``teleop.tracker_key``. That path must not be a second, unscoped home
    for the same knob (the curated one is applied only for Quest Mantis
    runs), so it is folded back onto the curated key and hidden from the
    Advanced tree. Computed once from the tables, never spelled twice.
    """
    aliases: dict[str, str] = {}
    for setting in _SETTINGS_BY_KEY.values():
        if setting.targets is None:
            continue
        for op, keys in setting.targets.items():
            for key in keys:
                for section in SECTIONS:
                    prefix = section.targets.get(op)
                    if prefix and key.startswith(prefix + "."):
                        path = section.key + key[len(prefix) :]
                        if path != setting.key:
                            aliases[path] = setting.key
    return aliases


_SECTION_ALIASES: dict[str, str] = _curated_section_aliases()


def canonical_keys(key: str) -> tuple[str, ...]:
    """Translate a possibly pre-v2 (or aliased) key into its canonical key(s)."""
    mapped = _LEGACY_KEYS.get(key)
    if mapped is not None:
        return mapped
    if key.startswith("vr_teleop."):
        key = "teleop." + key.removeprefix("vr_teleop.")
    return (_SECTION_ALIASES.get(key, key),)


def flatten_tree(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """``{"axol": {"left": {"kp": 1}}}`` → ``{"axol.left.kp": 1}``.

    Only dicts nest; lists (poses, per-joint stiffness) are leaves.
    """
    flat: dict[str, Any] = {}
    for name, value in tree.items():
        key = f"{prefix}{name}"
        if isinstance(value, dict):
            flat.update(flatten_tree(value, key + "."))
        else:
            flat[key] = value
    return flat


def nest_tree(flat: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`flatten_tree`; sections in registry order, leaves sorted."""
    order = {section.key: i for i, section in enumerate(SECTIONS)}
    tree: dict[str, Any] = {}
    for key in sorted(
        flat, key=lambda k: (order.get(k.partition(".")[0], len(order)), k)
    ):
        node = tree
        *parents, leaf = key.split(".")
        for part in parents:
            child = node.get(part)
            if not isinstance(child, dict):
                child = node[part] = {}
            node = child
        node[leaf] = flat[key]
    return tree


# ---------------------------------------------------------------------------
# The Advanced tree: every non-curated leaf of the grafted sections, built
# from a reference op's schema and re-rooted under the canonical section key.
# ---------------------------------------------------------------------------


def _rekey_nodes(
    nodes: list[dict[str, Any]],
    old_prefix: str,
    new_prefix: str,
    managed: set[str],
    drop: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Re-root a schema subtree under a canonical prefix, pruning owned keys."""
    out: list[dict[str, Any]] = []
    for node in nodes:
        if node["label"] in drop:
            continue
        leaf = node["key"].rsplit(".", 1)[-1]
        # ChoiceRegistry discriminators ("type") are implementation detail.
        if node["kind"] == "field" and leaf == "type":
            continue
        new_key = new_prefix + node["key"][len(old_prefix) :]
        if node["kind"] == "field":
            if new_key in managed:
                continue
            out.append({**node, "key": new_key})
            if new_key.startswith("teleop.box_"):
                out[-1].update(_parcel_only())
        else:
            children = _rekey_nodes(
                node["children"], old_prefix, new_prefix, managed, ()
            )
            if children:
                out.append({**node, "key": new_key, "children": children})
    return out


def advanced_schema() -> list[dict[str, Any]]:
    """The unified Advanced tree: one canonical section per shared subsystem.

    Curated keys — and the section paths they drive under another name —
    are pruned so every knob has exactly one home. Best-effort per section:
    a subsystem whose reference op can't build its schema (missing extras)
    is simply omitted.
    """
    from .commands import get_schema

    managed = set(_SETTINGS_BY_KEY) | set(_SECTION_ALIASES)
    sections: list[dict[str, Any]] = []
    for section in SECTIONS:
        if section.ref_op is None or section.ref_prefix is None:
            continue
        try:
            schema = get_schema(section.ref_op)
        except Exception:  # noqa: BLE001 - optional extras may be missing
            continue
        subtree = next(
            (
                n
                for n in schema.nodes
                if n["kind"] == "group" and n["key"] == section.ref_prefix
            ),
            None,
        )
        if subtree is None:
            continue
        nodes = _rekey_nodes(
            subtree["children"],
            section.ref_prefix,
            section.key,
            managed,
            section.drop_children,
        )
        if nodes:
            sections.append(
                {"key": section.key, "label": section.label, "nodes": nodes}
            )
    return sections


def _schema_defaults() -> dict[str, dict[str, Any]]:
    """``op_id -> {leaf key -> default}`` from the introspected op schemas.

    Best-effort: an op whose config can't build (missing lerobot / ZED extras)
    simply contributes no defaults, and the affected settings render without a
    placeholder.
    """
    from .commands import get_schema

    out: dict[str, dict[str, Any]] = {}
    for op in _OPS:
        try:
            schema = get_schema(op)
        except Exception:  # noqa: BLE001 - optional extras may be missing
            continue
        leaves: dict[str, Any] = {}

        def _walk(nodes: list[dict[str, Any]]) -> None:
            for node in nodes:
                if node["kind"] == "group":
                    _walk(node["children"])
                else:
                    leaves[node["key"]] = node["default"]

        _walk(schema.nodes)
        out[op] = leaves
    return out


def settings_schema() -> list[dict[str, Any]]:
    """Serializable settings categories for the UI, with resolved defaults.

    Each setting's default comes from the first op config that declares it, so
    the UI's placeholders and reset targets always match the code's defaults.
    """
    defaults = _schema_defaults()
    categories: list[dict[str, Any]] = []
    for cat in SETTINGS:
        fields: list[dict[str, Any]] = []
        for s in cat.settings:
            targets = {
                op: keys
                for op in (*_OPS, "diag.lift-cycle")
                if (keys := targets_for(s.key, op))
            }
            default: Any = None
            for op, keys in targets.items():
                leaf = defaults.get(op, {}).get(keys[0])
                if leaf is not None:
                    default = leaf
                    break
            # A None default usually still *does* something (the LeRobot cache
            # dir, local inference, …) — spell that out instead of "unset".
            default_text: str | None = None
            if default is None and s.effective_default is not None:
                try:
                    default_text = (
                        s.effective_default()
                        if callable(s.effective_default)
                        else s.effective_default
                    )
                except Exception:  # noqa: BLE001 - placeholder is cosmetic
                    default_text = None
            fields.append(
                {
                    "key": s.key,
                    "label": s.label,
                    "type": s.type,
                    "help": s.help,
                    "options": list(s.options) if s.options else None,
                    "default": default,
                    "defaultText": default_text,
                    "ui": s.ui,
                    "targets": {op: list(keys) for op, keys in targets.items()},
                }
            )
        categories.append(
            {
                "key": cat.key,
                "label": cat.label,
                "description": cat.description,
                "settings": fields,
            }
        )
    return categories


class SettingsStore:
    """Thread-safe JSON persistence for the robot's shared settings.

    On disk the file is the nested tree described in the module docstring
    (``version`` 2): one object per canonical section plus the panel's
    ``cameras`` spec. In memory the same values are kept flat under their
    dotted canonical keys (``"axol.left.elbow.kp"``), which is also how the
    HTTP API exchanges them. Version-1 files (flat ``values`` + ``advanced``
    maps under the old curated names) are migrated on load and rewritten in
    the new layout on the next save.

    A missing file is an empty store. An *unreadable* one (corrupt JSON, a
    permission error, a symlinked path component) is logged and treated as
    empty by default — serve must come up so the operator can repair it —
    but ``strict=True`` re-raises instead: the CLI and SDK use that, since
    silently running the calibrated defaults in place of the robot's saved
    gravity model is exactly the failure the shared file exists to prevent.
    """

    def __init__(self, path: Path = SETTINGS_PATH, *, strict: bool = False) -> None:
        self._path = path
        self._strict = strict
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            raw = json.loads(secure_read_text(self._path))
            if isinstance(raw, dict):
                if "values" in raw or "advanced" in raw:
                    values = self._migrate_v1(raw)
                else:
                    flat = flatten_tree(
                        {
                            key: value
                            for key, value in raw.items()
                            if key not in _RESERVED_KEYS and isinstance(value, dict)
                        }
                    )
                    # A hand-edited file may spell a knob by its section path
                    # (``teleop.tracker_key``); fold it onto its one home.
                    values = {}
                    for key, value in flat.items():
                        for canonical in canonical_keys(key):
                            values[canonical] = value
                if not self._strict:
                    self._repair_axol_channels(values)
                return {"values": values, "cameras": raw.get("cameras")}
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001 - a corrupt file must not kill serve
            if self._strict:
                raise
            _logger.exception("failed to load %s; starting empty", self._path)
        return {"values": {}, "cameras": None}

    def _repair_axol_channels(self, values: dict[str, Any]) -> None:
        """Drop a persisted Axol arm map that the store would refuse to save.

        Files written before ``update`` validated the pair (or hand-edited
        since) can map both arms onto one interface. ``can_channels`` rejects
        that, and serve resolves it at ``create_app`` — so an invalid pair
        would crash every restart while the only UI able to fix it is the
        panel serve hosts. Fall back to the hub's default names (logged, not
        rewritten until the next save) so the operator can repair it in
        Settings. The strict CLI/SDK store keeps the values and fails on use.
        """
        keys = ("robot.left_channel", "robot.right_channel")
        try:
            _axol_channels_from_values(values)
        except ValueError as exc:
            _logger.error(
                "ignoring the Axol CAN channels saved in %s (%s); using the "
                "defaults until Settings → Robot is corrected",
                self._path,
                exc,
            )
            for key in keys:
                values.pop(key, None)

    @staticmethod
    def _migrate_v1(raw: dict[str, Any]) -> dict[str, Any]:
        """Flat curated ``values`` + canonical ``advanced`` → canonical keys.

        Curated values win over an advanced entry for the same leaf: they
        were the visible control, the advanced one was hidden as "managed".
        """
        values: dict[str, Any] = {}
        for key, value in dict(raw.get("advanced") or {}).items():
            for canonical in canonical_keys(key):
                values[canonical] = value
        for key, value in dict(raw.get("values") or {}).items():
            for canonical in canonical_keys(key):
                values[canonical] = value
        return values

    def _save_locked(self) -> None:
        # nest_tree already orders sections by registry and leaves by name;
        # keep that so the file reads top-down like the config it mirrors.
        secure_atomic_write_json(self._path, self._document_locked(), sort_keys=False)

    def _document_locked(self) -> dict[str, Any]:
        return {
            "version": SETTINGS_VERSION,
            **nest_tree(self._data["values"]),
            "cameras": self._data["cameras"],
        }

    # -- API surface ---------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Stored values (flat canonical keys) and the camera spec."""
        with self._lock:
            return {
                "values": dict(self._data["values"]),
                "cameras": self._data["cameras"],
            }

    def document(self) -> dict[str, Any]:
        """The settings file as written to disk (nested, versioned)."""
        with self._lock:
            return self._document_locked()

    def update(
        self,
        values: dict[str, Any] | None = None,
        cameras: Any = ...,  # sentinel: ``...`` means "not provided"
        advanced: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply a partial update and persist. Returns the new snapshot.

        ``values`` merges per canonical key; a ``None`` value removes the key
        (reset to default). Pre-v2 key names (and the old ``advanced`` map)
        are still accepted from cached panels and imported files and land
        on their canonical keys. ``cameras`` replaces the whole camera spec
        (``None`` clears it).
        """
        if values is not None or advanced is not None:
            incoming: dict[str, Any] = {}
            for key, value in {**(advanced or {}), **(values or {})}.items():
                for canonical in canonical_keys(key):
                    incoming[canonical] = value
            unknown = [k for k in incoming if not is_known_key(k)]
            if unknown:
                raise KeyError(f"unknown settings: {', '.join(sorted(unknown))}")
            for key, value in incoming.items():
                setting = _SETTINGS_BY_KEY.get(key)
                if (
                    setting is not None
                    and setting.type == "boolean"
                    and value is not None
                ):
                    incoming[key] = parse_boolean(value, key=key)
                if (
                    key
                    in {
                        "mantis.left_trigger_full_scale",
                        "mantis.right_trigger_full_scale",
                    }
                    and value is not None
                ):
                    try:
                        incoming[key] = validate_trigger_full_scale(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{key}: {exc}") from exc
            values = incoming
        with self._lock:
            if (
                values is not None
                and {
                    "robot.left_channel",
                    "robot.right_channel",
                }
                & values.keys()
            ):
                proposed = dict(self._data["values"])
                for key, value in values.items():
                    if value is None:
                        proposed.pop(key, None)
                    else:
                        proposed[key] = value
                _axol_channels_from_values(proposed)
            if (
                values is not None
                and {
                    "mantis.left_channel",
                    "mantis.right_channel",
                }
                & values.keys()
            ):
                for key in (
                    "mantis.left_channel",
                    "mantis.right_channel",
                ):
                    if (
                        key in values
                        and values[key] is not None
                        and not str(values[key]).strip()
                    ):
                        side = key.removeprefix("mantis.").removesuffix("_channel")
                        raise ValueError(
                            f"Mantis {side} CAN channel is empty; use null to "
                            "reset it to the default"
                        )
                proposed = dict(self._data["values"])
                for key, value in values.items():
                    if value is None:
                        proposed.pop(key, None)
                    else:
                        proposed[key] = value
                left, right = _mantis_channels_from_values(proposed)
                require_mantis_channels((left, right))
            if values is not None:
                for k, v in values.items():
                    if v is None:
                        self._data["values"].pop(k, None)
                    else:
                        self._data["values"][k] = v
            if cameras is not ...:
                self._data["cameras"] = cameras
            self._save_locked()
        return self.snapshot()

    def cameras(self) -> dict[str, Any] | None:
        with self._lock:
            cams = self._data["cameras"]
            return dict(cams) if isinstance(cams, dict) else None

    def can_channels(self) -> tuple[str | None, str | None]:
        """The robot's (left, right) CAN interfaces from the shared settings.

        Unset values fall back to the Axol hub's persistent interface names;
        the ops' ``null`` convention (or an empty value) disables that arm —
        e.g. a single non-hub adapter driving one arm only. The robot link and
        the operations both resolve their channels from here, so a custom
        adapter chosen once (Settings, or the dashboard's CAN adapter picker)
        applies everywhere.
        """
        with self._lock:
            values = dict(self._data["values"])
        return _axol_channels_from_values(values)

    def mantis_can_channels(self) -> tuple[str | None, str | None]:
        """The handheld rig's persisted (left, right) SocketCAN mapping."""
        with self._lock:
            values = dict(self._data["values"])
        return _mantis_channels_from_values(values)

    def effective_axol_can_channels(
        self, op_id: str, args: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        """Resolve the exact Axol arm channels an operation request will open."""
        from ..constants import CAN_LEFT, CAN_RIGHT

        merged = self.merged_args(op_id, args)
        target_op = _settings_op(op_id)
        if target_op in {"teleop", "gravity-comp"}:
            keys = ("left_channel", "right_channel")
        elif target_op in {
            "collect-data",
            "run-policy",
            "replay-dataset",
        }:
            keys = (f"{_ROBOT}.left_channel", f"{_ROBOT}.right_channel")
        else:
            return self.can_channels()

        def resolve(key: str, default: str) -> str | None:
            # build_argv omits JSON null/blank values, so draccus then uses the
            # config default. The literal string "null" is what decodes to None.
            value = merged.get(key)
            if value is None or not str(value).strip():
                return default
            text = str(value).strip()
            return None if text.lower() in ("null", "none") else text

        return require_distinct_axol_channels(
            (resolve(keys[0], CAN_LEFT), resolve(keys[1], CAN_RIGHT))
        )

    def effective_mantis_can_channels(
        self, op_id: str, args: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        """Resolve the exact Mantis channels an operation request will open.

        This mirrors :meth:`merged_args` plus the command-level Mantis profile:
        stored rig channels are folded below request overrides, and untouched
        Axol defaults are translated to the Mantis hub names.  The serve API
        uses this result to ensure its already-open diagnostics link represents
        the same buses the impending operation will borrow.
        """
        from ..constants import CAN_LEFT, CAN_MANTIS_LEFT, CAN_MANTIS_RIGHT, CAN_RIGHT

        merged = self.merged_args(op_id, args)
        target_op = _settings_op(op_id)
        if target_op == "teleop":
            left = merged.get("left_channel", CAN_LEFT)
            right = merged.get("right_channel", CAN_RIGHT)
        elif target_op == "collect-data":
            left = merged.get(f"{_ROBOT}.left_channel", CAN_LEFT)
            right = merged.get(f"{_ROBOT}.right_channel", CAN_RIGHT)
        else:
            return self.mantis_can_channels()

        def resolve(value: Any, arm_default: str, rig_default: str) -> str | None:
            if value is None:
                return None
            text = str(value).strip()
            if not text or text == arm_default:
                return rig_default
            return None if text.lower() in ("null", "none") else text

        return (
            resolve(left, CAN_LEFT, CAN_MANTIS_LEFT),
            resolve(right, CAN_RIGHT, CAN_MANTIS_RIGHT),
        )

    def has_gripper(self) -> bool:
        """Whether this robot is the gripper-equipped SKU (default ``True``).

        Read from ``axol.has_gripper``; the robot link uses it to skip the
        gripper motors' pings, and the UI reads the mirrored ``hasGripper``
        field of ``/api/robot/status``.
        """
        with self._lock:
            value = self._data["values"].get("axol.has_gripper")
        if value is None:
            return True
        return parse_boolean(value, key="axol.has_gripper")

    def merged_args(self, op_id: str, args: dict[str, Any]) -> dict[str, Any]:
        """Fold the shared settings into one op start's args.

        Every stored canonical key is translated to this op's dotted paths
        (:func:`targets_for`); the request's own args then win. Keys the
        op's schema doesn't know are dropped later by ``build_argv``, so a
        stale entry can never inject anything — which is also what makes an
        inherited target table safe when an aliased op only shares part of
        the original's config.
        """
        target_op = _settings_op(op_id)
        merged: dict[str, Any] = {}
        with self._lock:
            values = dict(self._data["values"])
        for key, value in values.items():
            if value is None:
                continue
            for target in targets_for(key, target_op):
                merged[target] = value

        # The Mantis and Axol channel maps are deliberately independent. The
        # curated target table cannot express a conditional target, so fold
        # the rig map in only for a Mantis run and still leave request args as
        # the final override below.
        raw_mantis = args.get("mantis")
        mantis = flag_enabled(raw_mantis)
        if mantis:
            left, right = self.mantis_can_channels()
            if target_op == "teleop":
                merged["left_channel"] = left
                merged["right_channel"] = right
            elif target_op == "collect-data":
                merged[f"{_ROBOT}.left_channel"] = left
                merged[f"{_ROBOT}.right_channel"] = right
        # The curated Quest datum is intentionally source-scoped. Without
        # this removal, switching the UI to Lighthouse/Ultimate would keep a
        # Quest key as an explicit override and prevent those tracker-specific
        # transforms from resolving. Request args below may still make an
        # intentional one-run override. (Teleop never receives tracking keys:
        # Mantis teleop is grippers-only.)
        source = args.get("mantis_source", merged.get("mantis_source"))
        if target_op == "collect-data" and (not mantis or source != "quest"):
            merged.pop(f"{_VRT}.tracker_key", None)
        merged.update(args)
        # Teleop has no tracking config: Mantis teleop is grippers-only.
        # Older cached panels still snapshot the saved source into every
        # Mantis start; drop it instead of failing the run's config parse.
        if target_op == "teleop":
            merged.pop("mantis_source", None)

        # A hosted root process must never honor saved or per-request paths for
        # TLS generation or LeRobot calibration state. Keep this after request
        # precedence so the security boundary is the final authority.
        from ..utils.state_files import privileged_service_active

        if privileged_service_active():
            _confine_hosted_runtime_paths(target_op, merged)

        # ``diag.lift-cycle`` is argparse-backed: unlike draccus, the literal
        # string "null" would be treated as an interface name. Translate a
        # disabled saved/request channel into the diagnostic's explicit skip
        # flag so its argv and the serve-side hardware preflight agree.
        if target_op == "diag.lift-cycle":
            for side in ("left", "right"):
                channel_key = f"{side}_channel"
                channel = merged.get(channel_key)
                if channel is not None and str(channel).strip().lower() in (
                    "",
                    "null",
                    "none",
                ):
                    merged.pop(channel_key, None)
                    merged[f"no_{side}"] = True

        return normalize_boolean_args(op_id, merged)
