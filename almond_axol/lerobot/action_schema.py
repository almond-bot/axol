"""Fail-closed policy action-schema discovery and async negotiation.

LeRobot policy configs retain an action *width*, but most policy families do
not retain the ordered dataset action names.  Width is not a safe deployment
contract for Axol: a gripperless joint policy and a Cartesian policy are both
14-dimensional while assigning entirely different physical meaning to every
entry.  This module recovers the authoritative names from checkpoint metadata
and provides the versioned payload used by Axol's async-inference handshake.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from lerobot.async_inference.helpers import RemotePolicyConfig

ACTION_SCHEMA_PROTOCOL_VERSION = 1
ACTION_SCHEMA_METADATA_KEY = "axol-action-schema-bin"

# ``PolicySetup.data`` is an opaque bytes field in LeRobot 0.6.1.  Upstream
# fills it with a pickle received directly from the network.  Axol instead
# uses this small, versioned JSON envelope and reconstructs the upstream
# dataclass locally on the server.  These limits are deliberately much larger
# than an Axol deployment needs (one state feature and at most a handful of
# cameras), while still bounding parser/resource work on an unauthenticated
# endpoint.
POLICY_SETUP_PROTOCOL = "almond-axol-policy-setup"
POLICY_SETUP_PROTOCOL_VERSION = 1
MAX_POLICY_SETUP_BYTES = 64 * 1024
MAX_ACTION_SCHEMA_CONFIRMATION_BYTES = 32 * 1024
MAX_ACTION_DIMENSIONS = 256
MAX_ACTION_NAME_BYTES = 128
MAX_POLICY_PATH_BYTES = 1024
MAX_FEATURES = 32
MAX_CAMERA_DIMENSION = 8192
MAX_ACTIONS_PER_CHUNK = 1024
MAX_TRAINING_FPS = 1000

SUPPORTED_AXOL_POLICY_TYPES = frozenset(
    {"act", "smolvla", "diffusion", "tdmpc", "vqbet", "pi0", "pi05", "groot"}
)
_DEVICE_PATTERN = re.compile(r"(?:cpu|mps|cuda(?::[0-9]{1,2})?|xpu(?::[0-9]{1,2})?)\Z")
_POLICY_SETUP_KEYS = frozenset(
    {
        "protocol",
        "version",
        "policy_type",
        "pretrained_name_or_path",
        "lerobot_features",
        "actions_per_chunk",
        "device",
        "action_schema",
    }
)


class ActionSchemaError(ValueError):
    """A policy's ordered action semantics cannot be proven compatible."""


@dataclass
class AxolRemotePolicyConfig(RemotePolicyConfig):
    """LeRobot's policy request plus Axol's expected ordered action schema."""

    action_schema: tuple[str, ...] = ()
    action_schema_protocol: int = ACTION_SCHEMA_PROTOCOL_VERSION


def _bounded_text(value: Any, source: str, *, max_bytes: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ActionSchemaError(
            f"Malformed {source}: expected a non-empty string without surrounding whitespace."
        )
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ActionSchemaError(f"Malformed {source}: invalid Unicode.") from exc
    if len(encoded) > max_bytes:
        raise ActionSchemaError(
            f"Malformed {source}: exceeds the {max_bytes}-byte limit."
        )
    if any(not character.isprintable() for character in value):
        raise ActionSchemaError(f"Malformed {source}: contains control characters.")
    return value


def _schema(value: Any, source: str) -> tuple[str, ...] | None:
    """Validate a serialized ordered-name list, returning ``None`` if absent."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ActionSchemaError(
            f"Malformed action schema in {source}: expected an ordered list of names."
        )
    names = tuple(value)
    if not names:
        raise ActionSchemaError(f"Malformed action schema in {source}: it is empty.")
    if len(names) > MAX_ACTION_DIMENSIONS:
        raise ActionSchemaError(
            f"Malformed action schema in {source}: exceeds the "
            f"{MAX_ACTION_DIMENSIONS}-dimension limit."
        )
    names = tuple(
        _bounded_text(
            name,
            f"action name in {source}",
            max_bytes=MAX_ACTION_NAME_BYTES,
        )
        for name in names
    )
    if len(names) != len(set(names)):
        raise ActionSchemaError(
            f"Malformed action schema in {source}: action names must be unique."
        )
    return names


def _exact_object_keys(value: Any, expected: frozenset[str], source: str) -> dict:
    if not isinstance(value, dict):
        raise ActionSchemaError(f"Malformed {source}: expected a JSON object.")
    keys = frozenset(value)
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise ActionSchemaError(f"Malformed {source}: {'; '.join(details)}.")
    return value


def _wire_int(value: Any, source: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ActionSchemaError(
            f"Malformed {source}: expected an integer in [{minimum}, {maximum}]."
        )
    return value


def _wire_names(value: Any, source: str) -> list[str]:
    if not isinstance(value, list):
        raise ActionSchemaError(f"Malformed {source}: expected a JSON array.")
    names = _schema(value, source)
    assert names is not None
    return list(names)


def _decode_lerobot_features(value: Any) -> dict[str, dict[str, Any]]:
    """Validate the exact feature shape emitted by Axol's RobotClient.

    There is no reason for a network peer to send arbitrary processor config:
    Axol always emits one float32 ``observation.state`` vector and one or more
    image features.  Rebuilding these dictionaries also ensures no attacker-
    supplied mapping subclass or nested object can reach upstream code.
    """
    if not isinstance(value, dict) or not 2 <= len(value) <= MAX_FEATURES:
        raise ActionSchemaError(
            "Malformed lerobot_features: expected state plus 1-31 camera features."
        )

    result: dict[str, dict[str, Any]] = {}
    for raw_key, raw_spec in value.items():
        key = _bounded_text(raw_key, "feature key", max_bytes=128)
        if "/" in key:
            raise ActionSchemaError("Malformed feature key: '/' is not allowed.")

        if key == "observation.state":
            spec = _exact_object_keys(
                raw_spec,
                frozenset({"dtype", "shape", "names"}),
                "observation.state feature",
            )
            if spec["dtype"] != "float32":
                raise ActionSchemaError(
                    "Malformed observation.state feature: dtype must be 'float32'."
                )
            shape = spec["shape"]
            if not isinstance(shape, list) or len(shape) != 1:
                raise ActionSchemaError(
                    "Malformed observation.state feature: shape must be [D]."
                )
            width = _wire_int(
                shape[0],
                "observation.state width",
                minimum=1,
                maximum=MAX_ACTION_DIMENSIONS,
            )
            names = _wire_names(spec["names"], "observation.state names")
            if len(names) != width:
                raise ActionSchemaError(
                    "Malformed observation.state feature: shape and names disagree."
                )
            if "task" in names:
                raise ActionSchemaError(
                    "Malformed observation.state feature: 'task' is reserved."
                )
            result[key] = {"dtype": "float32", "shape": (width,), "names": names}
            continue

        prefix = "observation.images."
        if not key.startswith(prefix) or not key.removeprefix(prefix):
            raise ActionSchemaError(
                f"Malformed feature key {key!r}: only Axol state/images are allowed."
            )
        spec = _exact_object_keys(
            raw_spec,
            frozenset({"dtype", "shape", "names", "info"}),
            f"{key} feature",
        )
        if spec["dtype"] != "image":
            raise ActionSchemaError(f"Malformed {key} feature: dtype must be 'image'.")
        shape = spec["shape"]
        if not isinstance(shape, list) or len(shape) != 3:
            raise ActionSchemaError(f"Malformed {key} feature: shape must be [H,W,C].")
        height = _wire_int(
            shape[0], f"{key} height", minimum=1, maximum=MAX_CAMERA_DIMENSION
        )
        width = _wire_int(
            shape[1], f"{key} width", minimum=1, maximum=MAX_CAMERA_DIMENSION
        )
        channels = _wire_int(shape[2], f"{key} channels", minimum=1, maximum=3)
        if channels not in (1, 3):
            raise ActionSchemaError(
                f"Malformed {key} feature: channels must be 1 or 3."
            )
        if spec["names"] != ["height", "width", "channels"]:
            raise ActionSchemaError(
                f"Malformed {key} feature: names must be height/width/channels."
            )
        info = _exact_object_keys(
            spec["info"], frozenset({"is_depth_map"}), f"{key} info"
        )
        if type(info["is_depth_map"]) is not bool:
            raise ActionSchemaError(
                f"Malformed {key} feature: is_depth_map must be a boolean."
            )
        if info["is_depth_map"] != (channels == 1):
            raise ActionSchemaError(
                f"Malformed {key} feature: depth-map flag and channel count disagree."
            )
        result[key] = {
            "dtype": "image",
            "shape": (height, width, channels),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": channels == 1},
        }

    if "observation.state" not in result or len(result) == 1:
        raise ActionSchemaError(
            "Malformed lerobot_features: both state and at least one camera are required."
        )
    state_names = set(result["observation.state"]["names"])
    camera_names = {
        key.removeprefix("observation.images.")
        for key in result
        if key != "observation.state"
    }
    collisions = camera_names & (state_names | {"task"})
    if collisions:
        raise ActionSchemaError(
            "Malformed lerobot_features: camera names collide with state/task keys: "
            f"{sorted(collisions)}."
        )
    return result


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ActionSchemaError(f"Malformed policy setup: duplicate key {key!r}.")
        result[key] = value
    return result


def decode_axol_policy_setup(
    data: Any,
    *,
    allowed_policy_types: Iterable[str] = SUPPORTED_AXOL_POLICY_TYPES,
) -> AxolRemotePolicyConfig:
    """Parse an untrusted ``PolicySetup.data`` payload without using pickle."""
    if not isinstance(data, bytes) or not 1 <= len(data) <= MAX_POLICY_SETUP_BYTES:
        raise ActionSchemaError(
            "Malformed policy setup: payload must be 1-65536 bytes."
        )
    try:
        text = data.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ActionSchemaError(
                    f"Malformed policy setup: JSON constant {value!r} is not allowed."
                )
            ),
        )
    except ActionSchemaError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ActionSchemaError(
            "Malformed policy setup: expected bounded UTF-8 JSON."
        ) from exc

    payload = _exact_object_keys(payload, _POLICY_SETUP_KEYS, "policy setup")
    if payload["protocol"] != POLICY_SETUP_PROTOCOL:
        raise ActionSchemaError("Client did not send the Axol policy-setup protocol.")
    if (
        type(payload["version"]) is not int
        or payload["version"] != POLICY_SETUP_PROTOCOL_VERSION
    ):
        raise ActionSchemaError(
            "Client policy-setup protocol version does not match this server."
        )

    policy_type = _bounded_text(payload["policy_type"], "policy_type", max_bytes=32)
    allowed = frozenset(allowed_policy_types)
    if policy_type not in allowed:
        raise ActionSchemaError(
            f"Unsupported policy_type {policy_type!r}; allowed: {sorted(allowed)}."
        )
    policy_path = _bounded_text(
        payload["pretrained_name_or_path"],
        "pretrained_name_or_path",
        max_bytes=MAX_POLICY_PATH_BYTES,
    )
    device = _bounded_text(payload["device"], "device", max_bytes=32)
    if _DEVICE_PATTERN.fullmatch(device) is None:
        raise ActionSchemaError("Malformed device: use cpu, mps, cuda[:N], or xpu[:N].")
    actions_per_chunk = _wire_int(
        payload["actions_per_chunk"],
        "actions_per_chunk",
        minimum=1,
        maximum=MAX_ACTIONS_PER_CHUNK,
    )
    features = _decode_lerobot_features(payload["lerobot_features"])
    action_schema = _wire_names(payload["action_schema"], "client action schema")

    return AxolRemotePolicyConfig(
        policy_type=policy_type,
        pretrained_name_or_path=policy_path,
        lerobot_features=features,
        actions_per_chunk=actions_per_chunk,
        device=device,
        # Axol never needs a client-controlled rename map.  Keeping this local
        # prevents a peer from changing observation semantics on the server.
        rename_map={},
        action_schema=tuple(action_schema),
        action_schema_protocol=ACTION_SCHEMA_PROTOCOL_VERSION,
    )


def encode_axol_policy_setup(config: AxolRemotePolicyConfig) -> bytes:
    """Encode and self-validate the only setup envelope Axol sends."""
    if config.action_schema_protocol != ACTION_SCHEMA_PROTOCOL_VERSION:
        raise ActionSchemaError(
            "Client action-schema protocol version does not match this release."
        )
    if config.rename_map:
        raise ActionSchemaError(
            "Axol's network policy setup does not permit observation rename maps."
        )
    payload = {
        "protocol": POLICY_SETUP_PROTOCOL,
        "version": POLICY_SETUP_PROTOCOL_VERSION,
        "policy_type": config.policy_type,
        "pretrained_name_or_path": config.pretrained_name_or_path,
        "lerobot_features": config.lerobot_features,
        "actions_per_chunk": config.actions_per_chunk,
        "device": config.device,
        "action_schema": list(config.action_schema),
    }
    try:
        data = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ActionSchemaError("Policy setup contains non-JSON data.") from exc
    # Use the exact untrusted-input decoder as the encoder's final contract
    # check, so client and server cannot silently drift on types or limits.
    decode_axol_policy_setup(data)
    return data


def require_exact_action_schema(
    policy_schema: Iterable[str],
    robot_schema: Iterable[str],
    *,
    policy_label: str = "Policy",
) -> tuple[str, ...]:
    """Require exact ordered equality, including for same-width schemas."""
    policy_names = _schema(tuple(policy_schema), policy_label)
    robot_names = _schema(tuple(robot_schema), "configured robot")
    assert policy_names is not None and robot_names is not None
    if policy_names == robot_names:
        return policy_names

    mismatch = next(
        (
            index
            for index, (policy_name, robot_name) in enumerate(
                zip(policy_names, robot_names, strict=False)
            )
            if policy_name != robot_name
        ),
        min(len(policy_names), len(robot_names)),
    )
    detail = (
        f"first mismatch at index {mismatch}"
        if mismatch < min(len(policy_names), len(robot_names))
        else "different lengths"
    )
    raise ActionSchemaError(
        f"{policy_label} action schema does not exactly match the configured "
        f"robot ({detail}).\n"
        f"Policy ({len(policy_names)}): {list(policy_names)}\n"
        f"Robot  ({len(robot_names)}): {list(robot_names)}\n"
        "Refusing to map policy outputs positionally; select a checkpoint "
        "trained with this exact robot action layout."
    )


def _read_json(path: Path, source: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ActionSchemaError(f"Could not read {source} at {path}.") from exc
    if not isinstance(value, dict):
        raise ActionSchemaError(f"Malformed {source} at {path}: expected an object.")
    return value


def _hub_json(
    repo_id: str,
    filename: str,
    *,
    repo_type: str = "model",
    revision: str | None = None,
) -> dict[str, Any] | None:
    """Best-effort fetch of an optional JSON file; malformed files still fail."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
        )
    except (EntryNotFoundError, HfHubHTTPError, OSError):
        return None
    return _read_json(Path(path), f"{repo_type} metadata {repo_id}/{filename}")


def _metadata_fps(value: Any, source: str) -> int | None:
    """Validate an optional serialized dataset rate without lossy coercion."""
    if value is None:
        return None
    fps: int | None = None
    if isinstance(value, int) and not isinstance(value, bool):
        if 0 < value <= MAX_TRAINING_FPS:
            fps = value
    elif (
        isinstance(value, float)
        and math.isfinite(value)
        and value.is_integer()
        and 0 < value <= MAX_TRAINING_FPS
    ):
        fps = int(value)
    if fps is None:
        rendered = (
            f"<integer with {value.bit_length()} bits>"
            if isinstance(value, int)
            and not isinstance(value, bool)
            and value.bit_length() > 256
            else repr(value)
        )
        raise ActionSchemaError(
            f"Malformed training fps in {source}: expected an integer from 1 "
            f"through {MAX_TRAINING_FPS}, got {rendered}."
        )
    return fps


def resolve_policy_training_fps(policy_path: str) -> int | None:
    """Resolve and cross-check a checkpoint's authoritative training rate.

    Only bounded JSON metadata is read: local ``train_config.json`` and
    training-dataset ``meta/info.json`` files, or the same files downloaded
    through Hugging Face Hub.  No checkpoint code or pickle is imported.

    Returns ``None`` when the metadata contains no conclusive fps.  Malformed
    or conflicting values fail closed with :class:`ActionSchemaError`.
    """
    model_path = Path(policy_path)
    local = model_path.is_dir()
    if local:
        train_config = _read_json(model_path / "train_config.json", "train_config.json")
        if train_config is None:
            # Accommodate hand-copied ``pretrained_model`` directories from
            # older runs that left train_config.json at the checkpoint level.
            train_config = _read_json(
                model_path.parent / "train_config.json", "train_config.json"
            )
    else:
        train_config = _hub_json(policy_path, "train_config.json")

    if train_config is None:
        return None

    candidates: list[tuple[int, str]] = []

    def add(value: Any, source: str) -> None:
        fps = _metadata_fps(value, source)
        if fps is not None:
            candidates.append((fps, source))

    if "fps" in train_config:
        add(train_config["fps"], "checkpoint train_config.json")

    dataset = train_config.get("dataset")
    if isinstance(dataset, dict):
        repo_id = dataset.get("repo_id")
        root = dataset.get("root")
        revision = dataset.get("revision")
        info_candidates: list[tuple[Path, str]] = []
        # Local checkpoints may deliberately refer to a colocated/custom
        # dataset root or the local LeRobot cache.  A Hub checkpoint's JSON is
        # untrusted remote input and must never select arbitrary local paths;
        # for Hub policies, only fetch the declared dataset repo below.
        if local and isinstance(root, str) and root:
            root_path = Path(root).expanduser()
            info_candidates.append(
                (root_path / "meta" / "info.json", "training dataset")
            )
            if not root_path.is_absolute():
                info_candidates.extend(
                    [
                        (
                            model_path / root_path / "meta" / "info.json",
                            "training dataset",
                        ),
                        (
                            model_path.parent / root_path / "meta" / "info.json",
                            "training dataset",
                        ),
                    ]
                )
        if local and isinstance(repo_id, str) and repo_id:
            from lerobot.utils.constants import HF_LEROBOT_HOME

            info_candidates.append(
                (
                    HF_LEROBOT_HOME / repo_id / "meta" / "info.json",
                    "cached training dataset",
                )
            )

        seen_paths: set[Path] = set()
        for info_path, label in info_candidates:
            if info_path in seen_paths:
                continue
            seen_paths.add(info_path)
            info = _read_json(info_path, f"{label} metadata")
            if info is not None and "fps" in info:
                add(info["fps"], f"{label} meta/info.json")

        # A Hub policy must not rely on the training host's stale local cache.
        # Fetch the declared dataset metadata as the portable source of truth.
        if not local and isinstance(repo_id, str) and repo_id:
            info = _hub_json(
                repo_id,
                "meta/info.json",
                repo_type=str(dataset.get("repo_type") or "dataset"),
                revision=revision if isinstance(revision, str) else None,
            )
            if info is not None and "fps" in info:
                add(info["fps"], f"Hub dataset {repo_id} meta/info.json")

    if not candidates:
        return None

    expected, expected_source = candidates[0]
    for fps, source in candidates[1:]:
        if fps != expected:
            raise ActionSchemaError(
                "Conflicting training fps values in policy metadata:\n"
                f"{expected_source}: {expected}\n"
                f"{source}: {fps}"
            )
    return expected


def _action_names_from_feature_metadata(
    metadata: Any, source: str
) -> tuple[str, ...] | None:
    if not isinstance(metadata, dict):
        return None
    action = metadata.get("action")
    if not isinstance(action, dict):
        return None
    return _schema(action.get("names"), source)


def _processor_action_names(value: Any) -> list[Any]:
    """Find serialized ``action_names`` values in a processor config."""
    found: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "action_names" and child is not None:
                found.append(child)
            else:
                found.extend(_processor_action_names(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_processor_action_names(child))
    return found


def _add_candidate(
    candidates: list[tuple[tuple[str, ...], str]], value: Any, source: str
) -> None:
    names = _schema(value, source)
    if names is not None:
        candidates.append((names, source))


def _loaded_processor_candidates(
    processors: Iterable[Any], candidates: list[tuple[tuple[str, ...], str]]
) -> None:
    for processor in processors:
        for step in getattr(processor, "steps", ()):
            if hasattr(step, "action_names"):
                _add_candidate(
                    candidates,
                    getattr(step, "action_names"),
                    f"loaded processor {type(step).__name__}.action_names",
                )


def _policy_config_candidates(
    config: Any, candidates: list[tuple[tuple[str, ...], str]]
) -> None:
    if config is None:
        return
    if getattr(config, "action_feature_names", None) is not None:
        _add_candidate(
            candidates,
            config.action_feature_names,
            f"{type(config).__name__}.action_feature_names",
        )
    dataset_names = getattr(config, "dataset_feature_names", None)
    if isinstance(dataset_names, dict) and dataset_names.get("action") is not None:
        _add_candidate(
            candidates,
            dataset_names["action"],
            f"{type(config).__name__}.dataset_feature_names[action]",
        )


def _policy_action_width(config: Any) -> int | None:
    output = getattr(config, "output_features", None)
    if not isinstance(output, dict) or "action" not in output:
        return None
    feature = output["action"]
    shape = (
        feature.get("shape")
        if isinstance(feature, dict)
        else getattr(feature, "shape", None)
    )
    if not isinstance(shape, (list, tuple)) or len(shape) != 1:
        raise ActionSchemaError(
            f"Policy output feature 'action' has malformed shape {shape!r}; expected (D,)."
        )
    raw_width = shape[0]
    try:
        width = int(raw_width)
    except (TypeError, ValueError) as exc:
        raise ActionSchemaError(
            f"Policy output feature 'action' has malformed shape {shape!r}."
        ) from exc
    if isinstance(raw_width, bool) or raw_width != width or width <= 0:
        raise ActionSchemaError(
            f"Policy output feature 'action' has malformed shape {shape!r}."
        )
    return width


def resolve_policy_action_schema(
    policy_path: str,
    *,
    policy_config: Any = None,
    processors: Iterable[Any] = (),
) -> tuple[str, ...]:
    """Resolve authoritative ordered action names for a loaded checkpoint.

    Sources are cross-checked when more than one is present.  Current LeRobot
    checkpoints always save ``train_config.json`` beside the model; it points
    to the training dataset whose ``meta/info.json`` owns the ordered names.
    Policy families that embed names directly, and Mantis/relative processor
    configs that serialize ``action_names``, are also supported.  A width-only
    checkpoint is intentionally rejected.
    """
    candidates: list[tuple[tuple[str, ...], str]] = []
    _policy_config_candidates(policy_config, candidates)
    _loaded_processor_candidates(processors, candidates)

    model_path = Path(policy_path)
    local = model_path.is_dir()
    # A loaded policy/processor name list was deserialized from the checkpoint
    # itself, independently of the client's claimed robot schema, and is
    # length-checked against the policy output below. It is authoritative and
    # avoids making a working deployment depend on the training dataset still
    # being reachable. For local checkpoints, cheap on-disk sources are still
    # cross-checked so conflicting copied metadata fails closed.
    fetch_remote_metadata = not candidates

    def model_json(filename: str) -> dict[str, Any] | None:
        if local:
            return _read_json(model_path / filename, filename)
        return _hub_json(policy_path, filename) if fetch_remote_metadata else None

    config_json = model_json("config.json")
    if config_json is not None:
        if config_json.get("action_feature_names") is not None:
            _add_candidate(
                candidates,
                config_json["action_feature_names"],
                "checkpoint config.json action_feature_names",
            )
        dataset_names = config_json.get("dataset_feature_names")
        if isinstance(dataset_names, dict) and dataset_names.get("action") is not None:
            _add_candidate(
                candidates,
                dataset_names["action"],
                "checkpoint config.json dataset_feature_names[action]",
            )

    for filename in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processor_json = model_json(filename)
        if processor_json is None:
            continue
        for value in _processor_action_names(processor_json):
            _add_candidate(candidates, value, f"checkpoint {filename} action_names")

    train_config = model_json("train_config.json")
    if train_config is None and local:
        # Accommodate hand-copied ``pretrained_model`` directories from older
        # runs that left train_config.json at the checkpoint level.
        train_config = _read_json(
            model_path.parent / "train_config.json", "train_config.json"
        )

    if train_config is not None:
        dataset = train_config.get("dataset")
        if isinstance(dataset, dict):
            embedded = _action_names_from_feature_metadata(
                dataset.get("features"), "train_config.json dataset features"
            )
            if embedded is not None:
                candidates.append((embedded, "train_config.json dataset features"))

            repo_id = dataset.get("repo_id")
            root = dataset.get("root")
            revision = dataset.get("revision")
            info_candidates: list[tuple[Path, str]] = []
            # A Hub checkpoint's train_config.json is remote input and must
            # never select arbitrary files on the deployment host.  Local
            # checkpoints may intentionally point at local/cached datasets;
            # Hub checkpoints use embedded metadata or the declared Hub
            # dataset repo/revision below.
            if local and isinstance(root, str) and root:
                root_path = Path(root).expanduser()
                info_candidates.append(
                    (root_path / "meta" / "info.json", "training dataset")
                )
                if not root_path.is_absolute():
                    info_candidates.extend(
                        [
                            (
                                model_path / root_path / "meta" / "info.json",
                                "training dataset",
                            ),
                            (
                                model_path.parent / root_path / "meta" / "info.json",
                                "training dataset",
                            ),
                        ]
                    )
            if local and isinstance(repo_id, str) and repo_id:
                from lerobot.utils.constants import HF_LEROBOT_HOME

                info_candidates.append(
                    (
                        HF_LEROBOT_HOME / repo_id / "meta" / "info.json",
                        "cached training dataset",
                    )
                )

            seen_paths: set[Path] = set()
            for info_path, label in info_candidates:
                if info_path in seen_paths:
                    continue
                seen_paths.add(info_path)
                info = _read_json(info_path, f"{label} metadata")
                if info is None:
                    continue
                names = _action_names_from_feature_metadata(
                    info.get("features"), f"{label} meta/info.json"
                )
                if names is not None:
                    candidates.append((names, f"{label} meta/info.json"))

            if isinstance(repo_id, str) and repo_id and not candidates:
                info = _hub_json(
                    repo_id,
                    "meta/info.json",
                    repo_type=str(dataset.get("repo_type") or "dataset"),
                    revision=revision if isinstance(revision, str) else None,
                )
                if info is not None:
                    names = _action_names_from_feature_metadata(
                        info.get("features"),
                        f"Hub dataset {repo_id} meta/info.json",
                    )
                    if names is not None:
                        candidates.append(
                            (names, f"Hub dataset {repo_id} meta/info.json")
                        )

    if not candidates:
        raise ActionSchemaError(
            f"Could not prove the ordered action schema for policy {policy_path!r}. "
            "The checkpoint only exposes an action width. Keep its "
            "train_config.json and training dataset meta/info.json with the "
            "checkpoint (or use a policy/processor config that records "
            "action_feature_names); width-only positional mapping is unsafe."
        )

    expected, expected_source = candidates[0]
    for names, source in candidates[1:]:
        if names != expected:
            raise ActionSchemaError(
                "Conflicting action schemas in policy metadata:\n"
                f"{expected_source}: {list(expected)}\n"
                f"{source}: {list(names)}"
            )

    width = _policy_action_width(policy_config)
    if width is not None and width != len(expected):
        raise ActionSchemaError(
            f"Policy output width is {width}, but {expected_source} declares "
            f"{len(expected)} ordered action names."
        )
    return expected


def encode_action_schema_confirmation(schema: Iterable[str]) -> bytes:
    names = _schema(tuple(schema), "policy-server confirmation")
    assert names is not None
    data = json.dumps(
        {
            "protocol": ACTION_SCHEMA_PROTOCOL_VERSION,
            "action_schema": list(names),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    if len(data) > MAX_ACTION_SCHEMA_CONFIRMATION_BYTES:
        raise ActionSchemaError(
            "Policy-server action-schema confirmation exceeds its size limit."
        )
    return data


def decode_action_schema_confirmation(value: Any) -> tuple[str, ...]:
    if (
        not isinstance(value, bytes)
        or not 1 <= len(value) <= MAX_ACTION_SCHEMA_CONFIRMATION_BYTES
    ):
        raise ActionSchemaError(
            "Policy server returned a malformed action-schema confirmation."
        )
    try:
        payload = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ActionSchemaError(
                    f"Malformed action-schema confirmation constant {constant!r}."
                )
            ),
        )
    except ActionSchemaError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ActionSchemaError(
            "Policy server returned an unreadable action-schema confirmation."
        ) from exc
    payload = _exact_object_keys(
        payload,
        frozenset({"protocol", "action_schema"}),
        "policy-server confirmation",
    )
    if (
        type(payload["protocol"]) is not int
        or payload["protocol"] != ACTION_SCHEMA_PROTOCOL_VERSION
    ):
        raise ActionSchemaError(
            "Policy server does not support the required Axol action-schema "
            f"protocol v{ACTION_SCHEMA_PROTOCOL_VERSION} "
            f"(reported {payload['protocol']!r})."
        )
    names = _schema(payload.get("action_schema"), "policy-server confirmation")
    assert names is not None
    return names


def confirmed_schema_from_metadata(metadata: Any) -> tuple[str, ...]:
    values = [
        value for key, value in (metadata or ()) if key == ACTION_SCHEMA_METADATA_KEY
    ]
    if len(values) != 1:
        raise ActionSchemaError(
            "Policy server did not provide exactly one versioned action-schema "
            "confirmation. Upgrade/restart it with the same almond-axol release."
        )
    return decode_action_schema_confirmation(values[0])
