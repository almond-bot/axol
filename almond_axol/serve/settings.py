"""Shared operator settings for the control panel.

The control panel used to bury every tunable inside each operation's own
config form, persisted per-browser in localStorage. Almost none of those
values are actually per-run — camera assignment, arm stiffness, teleop rate,
recording fps, the inference server address — they are properties of *this
robot*, shared by every operation and every operator device. This module
gives them one home on the serve host:

- :data:`SETTINGS` — the curated registry of shared settings, grouped into
  UI categories. Each setting maps to the dotted config key(s) it drives on
  each operation (the same ops have the same knob at different paths, e.g.
  stiffness is ``axol.left_stiffness`` on teleop but
  ``robot_config.axol_config.left_stiffness`` on collect-data).
- :class:`SettingsStore` — JSON persistence at ``~/.almond/settings.json``
  (the camera spec, the shared setting values, and per-op advanced
  overrides), plus the merge that folds them into an op start's args. The
  request's own args always win, so a per-run value can still override a
  shared one.

Precedence at op start (later wins): dataclass defaults → shared settings →
per-op advanced overrides → the request's args → the camera spec fold-in.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

SETTINGS_PATH = Path.home() / ".almond" / "settings.json"

# Operation ids (mirrors serve.commands / serve.runner).
_OPS = ("teleop", "gravity-comp", "collect-data", "run-policy", "replay-dataset")

# Dotted paths into the lerobot-based ops' shared robot config.
_ROBOT = "robot_config"
_AXOL = f"{_ROBOT}.axol_config"
_VRT = "teleop_config.vr_teleop_config"
_KIN = "teleop_config.kinematics_config"


@dataclass(frozen=True)
class SettingDef:
    """One shared setting: how it renders and which op config keys it drives.

    ``targets`` maps an operation id to the dotted config key(s) this setting
    sets on that op (build_argv turns them into CLI-style overrides). ``ui``
    carries optional widget hints for the front-end (slider ranges, the pose
    editor). ``options`` makes it a dropdown.
    """

    key: str
    label: str
    type: str  # "number" | "boolean" | "text" | "select"
    help: str
    targets: dict[str, tuple[str, ...]]
    options: tuple[str, ...] | None = None
    ui: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SettingCategory:
    key: str
    label: str
    description: str
    settings: tuple[SettingDef, ...]


def _all_ops(*keys: str) -> dict[str, tuple[str, ...]]:
    return {op: keys for op in _OPS}


SETTINGS: tuple[SettingCategory, ...] = (
    SettingCategory(
        key="robot",
        label="Robot",
        description="Arm behaviour shared by every operation on this robot.",
        settings=(
            SettingDef(
                key="robot.left_stiffness",
                label="Left arm stiffness",
                type="number",
                help=(
                    "Compliance ↔ stiffness blend in [0, 1] for the left arm. "
                    "Match the value used at data-collection time when running "
                    "a policy."
                ),
                ui={"widget": "slider", "min": 0, "max": 1, "step": 0.05},
                targets={
                    "teleop": ("axol.left_stiffness",),
                    "gravity-comp": ("axol.left_stiffness",),
                    "collect-data": (f"{_AXOL}.left_stiffness",),
                    "run-policy": (f"{_AXOL}.left_stiffness",),
                    "replay-dataset": (f"{_AXOL}.left_stiffness",),
                },
            ),
            SettingDef(
                key="robot.right_stiffness",
                label="Right arm stiffness",
                type="number",
                help=("Compliance ↔ stiffness blend in [0, 1] for the right arm."),
                ui={"widget": "slider", "min": 0, "max": 1, "step": 0.05},
                targets={
                    "teleop": ("axol.right_stiffness",),
                    "gravity-comp": ("axol.right_stiffness",),
                    "collect-data": (f"{_AXOL}.right_stiffness",),
                    "run-policy": (f"{_AXOL}.right_stiffness",),
                    "replay-dataset": (f"{_AXOL}.right_stiffness",),
                },
            ),
            SettingDef(
                key="robot.left_channel",
                label="Left arm CAN channel",
                type="text",
                help="SocketCAN interface for the left arm (e.g. can_left).",
                targets={
                    "teleop": ("left_channel",),
                    "gravity-comp": ("left_channel",),
                    "collect-data": (f"{_ROBOT}.left_channel",),
                    "run-policy": (f"{_ROBOT}.left_channel",),
                    "replay-dataset": (f"{_ROBOT}.left_channel",),
                },
            ),
            SettingDef(
                key="robot.right_channel",
                label="Right arm CAN channel",
                type="text",
                help="SocketCAN interface for the right arm (e.g. can_right).",
                targets={
                    "teleop": ("right_channel",),
                    "gravity-comp": ("right_channel",),
                    "collect-data": (f"{_ROBOT}.right_channel",),
                    "run-policy": (f"{_ROBOT}.right_channel",),
                    "replay-dataset": (f"{_ROBOT}.right_channel",),
                },
            ),
            SettingDef(
                key="robot.gripper_torque_limit",
                label="Gripper torque limit (Nm)",
                type="number",
                help="Peak gripper output torque, applied to both grippers.",
                targets={
                    "teleop": (
                        "axol.left.gripper.torque_limit",
                        "axol.right.gripper.torque_limit",
                    ),
                    "gravity-comp": (
                        "axol.left.gripper.torque_limit",
                        "axol.right.gripper.torque_limit",
                    ),
                    "collect-data": (
                        f"{_AXOL}.left.gripper.torque_limit",
                        f"{_AXOL}.right.gripper.torque_limit",
                    ),
                    "run-policy": (
                        f"{_AXOL}.left.gripper.torque_limit",
                        f"{_AXOL}.right.gripper.torque_limit",
                    ),
                    "replay-dataset": (
                        f"{_AXOL}.left.gripper.torque_limit",
                        f"{_AXOL}.right.gripper.torque_limit",
                    ),
                },
            ),
            SettingDef(
                key="robot.gripper_max_speed",
                label="Gripper max speed (rad/s)",
                type="number",
                help="Maximum gripper joint speed, applied to both grippers.",
                targets={
                    "teleop": (
                        "axol.left.gripper.max_speed",
                        "axol.right.gripper.max_speed",
                    ),
                    "gravity-comp": (
                        "axol.left.gripper.max_speed",
                        "axol.right.gripper.max_speed",
                    ),
                    "collect-data": (
                        f"{_AXOL}.left.gripper.max_speed",
                        f"{_AXOL}.right.gripper.max_speed",
                    ),
                    "run-policy": (
                        f"{_AXOL}.left.gripper.max_speed",
                        f"{_AXOL}.right.gripper.max_speed",
                    ),
                    "replay-dataset": (
                        f"{_AXOL}.left.gripper.max_speed",
                        f"{_AXOL}.right.gripper.max_speed",
                    ),
                },
            ),
            SettingDef(
                key="robot.gravity_kd",
                label="Gravity comp damping (kd)",
                type="number",
                help="Velocity damping applied to freed joints in gravity comp.",
                targets={"gravity-comp": ("kd",)},
            ),
            SettingDef(
                key="robot.gravity_rate_hz",
                label="Gravity comp rate (Hz)",
                type="number",
                help="Gravity compensation control-loop rate.",
                targets={"gravity-comp": ("rate_hz",)},
            ),
        ),
    ),
    SettingCategory(
        key="teleop",
        label="Teleop & VR",
        description="How VR controller motion drives the arms (teleop and data collection).",
        settings=(
            SettingDef(
                key="teleop.frequency",
                label="Teleop rate (Hz)",
                type="number",
                help="Control-loop rate for VR teleoperation (whole Hz).",
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
                    "Scale factor from controller motion to end-effector motion "
                    "(1 = 1:1; larger covers more workspace with less hand travel)."
                ),
                ui={"widget": "slider", "min": 0.5, "max": 3, "step": 0.1},
                targets={
                    "teleop": ("teleop.position_multiplier",),
                    "collect-data": (f"{_VRT}.position_multiplier",),
                },
            ),
            SettingDef(
                key="teleop.rotation_multiplier",
                label="Rotation multiplier",
                type="number",
                help=(
                    "Scale factor from wrist rotation to end-effector rotation "
                    "(1 = 1:1; larger rotates further with less wrist twist)."
                ),
                ui={"widget": "slider", "min": 0.5, "max": 3, "step": 0.1},
                targets={
                    "teleop": ("teleop.rotation_multiplier",),
                    "collect-data": (f"{_VRT}.rotation_multiplier",),
                },
            ),
            SettingDef(
                key="teleop.reset_speed",
                label="Return-to-rest speed (rad/s)",
                type="number",
                help="Peak joint speed while the arms return to the rest pose.",
                targets={
                    "teleop": ("teleop.reset_speed",),
                    "collect-data": (f"{_VRT}.reset_speed",),
                },
            ),
            SettingDef(
                key="teleop.rest_pose_left",
                label="Left arm rest pose",
                type="text",
                help=(
                    "Left arm rest/start configuration in radians, 7 joints in "
                    "ARM_JOINTS order. Edited with the pose editor."
                ),
                ui={"widget": "pose"},
                targets={
                    "teleop": ("teleop.rest_pose_left",),
                    "collect-data": (f"{_VRT}.rest_pose_left",),
                },
            ),
            SettingDef(
                key="teleop.rest_pose_right",
                label="Right arm rest pose",
                type="text",
                help=(
                    "Right arm rest/start configuration in radians, 7 joints in "
                    "ARM_JOINTS order. Edited with the pose editor."
                ),
                ui={"widget": "pose"},
                targets={
                    "teleop": ("teleop.rest_pose_right",),
                    "collect-data": (f"{_VRT}.rest_pose_right",),
                },
            ),
        ),
    ),
    SettingCategory(
        key="kinematics",
        label="Kinematics",
        description=(
            "IK solver cost weights for teleop and data collection — how the "
            "arms trade off tracking the hands against posture and limits."
        ),
        settings=(
            SettingDef(
                key="kinematics.pos_weight",
                label="Position weight",
                type="number",
                help="Weight on end-effector position error.",
                targets={
                    "teleop": ("kinematics.pos_weight",),
                    "collect-data": (f"{_KIN}.pos_weight",),
                },
            ),
            SettingDef(
                key="kinematics.ori_weight",
                label="Orientation weight",
                type="number",
                help="Weight on end-effector orientation error.",
                targets={
                    "teleop": ("kinematics.ori_weight",),
                    "collect-data": (f"{_KIN}.ori_weight",),
                },
            ),
            SettingDef(
                key="kinematics.elbow_weight",
                label="Elbow weight",
                type="number",
                help=(
                    "Weight on the elbow position hint — how strongly the arm "
                    "follows the operator's elbow (position only)."
                ),
                targets={
                    "teleop": ("kinematics.elbow_weight",),
                    "collect-data": (f"{_KIN}.elbow_weight",),
                },
            ),
            SettingDef(
                key="kinematics.rest_weight",
                label="Rest weight",
                type="number",
                help=(
                    "Weight pulling joints toward the current configuration — "
                    "higher damps drift, lower tracks more aggressively."
                ),
                targets={
                    "teleop": ("kinematics.rest_weight",),
                    "collect-data": (f"{_KIN}.rest_weight",),
                },
            ),
            SettingDef(
                key="kinematics.max_joint_delta",
                label="Max joint delta (rad)",
                type="number",
                help="Maximum change of any joint between consecutive IK solutions.",
                targets={
                    "teleop": ("kinematics.max_joint_delta",),
                    "collect-data": (f"{_KIN}.max_joint_delta",),
                },
            ),
        ),
    ),
    SettingCategory(
        key="recording",
        label="Recording",
        description="Dataset recording shared by collect-data and run-policy.",
        settings=(
            SettingDef(
                key="recording.fps",
                label="Recording fps",
                type="number",
                help="Frame rate of the recorded dataset (and policy control rate).",
                targets={
                    "collect-data": ("fps",),
                    "run-policy": ("fps",),
                },
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
                help="Local directory datasets are written to / read from.",
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
        description="Where and how run-policy executes the policy.",
        settings=(
            SettingDef(
                key="inference.device",
                label="Device",
                type="select",
                options=("cuda", "cpu", "mps"),
                help="Device the policy runs on.",
                targets={"run-policy": ("device",)},
            ),
            SettingDef(
                key="inference.server_host",
                label="Inference server host",
                type="text",
                help=(
                    "Address of a remote `axol inference-server`. Leave unset "
                    "to run inference locally."
                ),
                targets={"run-policy": ("server_host",)},
            ),
            SettingDef(
                key="inference.server_port",
                label="Inference server port",
                type="number",
                help="Port of the inference server (local or remote).",
                targets={"run-policy": ("server_port",)},
            ),
            SettingDef(
                key="inference.episode_time_s",
                label="Episode length (s)",
                type="number",
                help="Maximum length of one policy episode.",
                targets={"run-policy": ("episode_time_s",)},
            ),
            SettingDef(
                key="inference.actions_per_chunk",
                label="Actions per chunk",
                type="number",
                help="Actions requested from the policy per inference call.",
                targets={"run-policy": ("actions_per_chunk",)},
            ),
            SettingDef(
                key="inference.chunk_size_threshold",
                label="Chunk size threshold",
                type="number",
                help="Queue fraction below which the next chunk is requested.",
                targets={"run-policy": ("chunk_size_threshold",)},
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
                help="How overlapping action chunks are combined.",
                targets={"run-policy": ("aggregate_fn",)},
            ),
            SettingDef(
                key="inference.temporal_ensemble_coeff",
                label="Temporal ensemble coeff",
                type="number",
                help="Exponential weight for the temporal_ensemble aggregation.",
                targets={"run-policy": ("temporal_ensemble_coeff",)},
            ),
        ),
    ),
    SettingCategory(
        key="system",
        label="System",
        description="Logging and diagnostics.",
        settings=(
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
            default: Any = None
            for op, keys in s.targets.items():
                leaf = defaults.get(op, {}).get(keys[0])
                if leaf is not None:
                    default = leaf
                    break
            fields.append(
                {
                    "key": s.key,
                    "label": s.label,
                    "type": s.type,
                    "help": s.help,
                    "options": list(s.options) if s.options else None,
                    "default": default,
                    "ui": s.ui,
                    "targets": {op: list(keys) for op, keys in s.targets.items()},
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
    """Thread-safe JSON persistence for the shared operator settings.

    File shape (all sections optional)::

        {
          "version": 1,
          "values": {"robot.left_stiffness": 0.8, ...},
          "cameras": {...camera spec, see app.OpStartRequest...},
          "op_overrides": {"collect-data": {"teleop_hz": 90, ...}, ...}
        }
    """

    def __init__(self, path: Path = SETTINGS_PATH) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self._path.read_text())
            if isinstance(raw, dict):
                return {
                    "values": dict(raw.get("values") or {}),
                    "cameras": raw.get("cameras"),
                    "op_overrides": {
                        op: dict(v)
                        for op, v in (raw.get("op_overrides") or {}).items()
                        if isinstance(v, dict)
                    },
                }
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001 - a corrupt file must not kill serve
            _logger.exception("failed to load %s; starting empty", self._path)
        return {"values": {}, "cameras": None, "op_overrides": {}}

    def _save_locked(self) -> None:
        payload = {"version": 1, **self._data}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, self._path)

    # -- API surface ---------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "values": dict(self._data["values"]),
                "cameras": self._data["cameras"],
                "opOverrides": {
                    op: dict(v) for op, v in self._data["op_overrides"].items()
                },
            }

    def update(
        self,
        values: dict[str, Any] | None = None,
        cameras: Any = ...,  # sentinel: ``...`` means "not provided"
        op_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply a partial update and persist. Returns the new snapshot.

        ``values`` merges per key; a ``None`` value removes the key (reset to
        default). ``cameras`` replaces the whole camera spec (``None`` clears
        it). ``op_overrides`` replaces per op: each op's map is swapped
        wholesale (an empty/None map removes the op's overrides).
        """
        if values is not None:
            unknown = [k for k in values if k not in _SETTINGS_BY_KEY]
            if unknown:
                raise KeyError(f"unknown settings: {', '.join(sorted(unknown))}")
        if op_overrides is not None:
            bad = [op for op in op_overrides if op not in _OPS]
            if bad:
                raise KeyError(f"unknown operations: {', '.join(sorted(bad))}")
        with self._lock:
            if values is not None:
                for k, v in values.items():
                    if v is None:
                        self._data["values"].pop(k, None)
                    else:
                        self._data["values"][k] = v
            if cameras is not ...:
                self._data["cameras"] = cameras
            if op_overrides is not None:
                for op, overrides in op_overrides.items():
                    if not overrides:
                        self._data["op_overrides"].pop(op, None)
                    else:
                        self._data["op_overrides"][op] = dict(overrides)
            self._save_locked()
        return self.snapshot()

    def cameras(self) -> dict[str, Any] | None:
        with self._lock:
            cams = self._data["cameras"]
            return dict(cams) if isinstance(cams, dict) else None

    def merged_args(self, op_id: str, args: dict[str, Any]) -> dict[str, Any]:
        """Fold the shared settings into one op start's args.

        Later wins: shared setting values → the op's advanced overrides → the
        request's own args. Keys the op's schema doesn't know are dropped later
        by ``build_argv``, so a stale entry can never inject anything.
        """
        merged: dict[str, Any] = {}
        with self._lock:
            values = dict(self._data["values"])
            overrides = dict(self._data["op_overrides"].get(op_id) or {})
        for key, value in values.items():
            setting = _SETTINGS_BY_KEY.get(key)
            if setting is None or value is None:
                continue
            for target in setting.targets.get(op_id, ()):
                merged[target] = value
        merged.update(overrides)
        merged.update(args)
        return merged
