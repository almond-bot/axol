"""Managed tracker lifecycle shared by direct Mantis CLI commands.

The control-panel runner owns this lifecycle itself. Direct ``teleop`` and
``collect-data`` use :func:`managed_mantis_bridge` so ``--mantis_source`` has
the same meaning everywhere: Quest waits for WebXR, while Lighthouse and
Ultimate start the selected local tracker backend automatically.
"""

from __future__ import annotations

import _thread
import contextlib
import sys
import threading
import uuid
from collections.abc import Iterator

from ..constants import CAN_LEFT, CAN_MANTIS_LEFT, CAN_MANTIS_RIGHT, CAN_RIGHT
from ..utils.can_channels import require_mantis_channels

_READY_TIMEOUT_S = 45.0
_STOP_TIMEOUT_S = 5.0


def _rig_channel(channel: str | None, arm_default: str, rig_default: str) -> str | None:
    return rig_default if channel == arm_default else channel


def set_managed_pose_source_id(cfg: object) -> str:
    """Give an external-tracker run one exact server/bridge producer token."""
    server = getattr(cfg, "vr_server", None)
    if server is None:
        teleop = getattr(cfg, "teleop_config", None)
        server = getattr(teleop, "vr_server_config", None)
    if server is None:
        raise ValueError("Mantis config has no VR server configuration")
    if getattr(server, "pose_source_kind", None) != "tracker":
        raise ValueError(
            "managed pose-source tokens are only valid for an external tracker run"
        )
    pose_source_id = f"mantis-{uuid.uuid4()}"
    server.expected_pose_source_id = pose_source_id
    return pose_source_id


def load_direct_mantis_fallback(
    *, collection: bool
) -> tuple[dict[str, object], str | None]:
    """Load UI-saved Mantis defaults below direct config-file/CLI overrides.

    The source and logical CAN map describe the physical handheld rig, not a
    browser session. Direct commands therefore inherit the same saved values
    as the control panel. The Quest calibration selector is returned
    separately because callers must add it only after the effective source
    (including a config-file or CLI override) resolves to Quest.
    """
    from ..serve.settings import SettingsStore

    settings = SettingsStore()
    left, right = require_mantis_channels(settings.mantis_can_channels())
    values = settings.snapshot()["values"]
    source_value = values.get("teleop.mantis_source")
    source = str(source_value).strip() if source_value is not None else ""
    quest_value = values.get("mantis.quest_tracker_key")
    quest_key = str(quest_value).strip() if quest_value is not None else ""

    fallback: dict[str, object] = {}
    if source:
        fallback["mantis_source"] = source
    if collection:
        fallback["robot_config"] = {
            "left_channel": left,
            "right_channel": right,
        }
    else:
        fallback["left_channel"] = left
        fallback["right_channel"] = right
    return fallback, quest_key or None


def add_quest_key_to_direct_fallback(
    fallback: dict[str, object], quest_key: str, *, collection: bool
) -> None:
    """Add a source-scoped saved Quest datum to a direct-command overlay."""
    if collection:
        fallback["teleop_config"] = {"vr_teleop_config": {"tracker_key": quest_key}}
    else:
        fallback["teleop"] = {"tracker_key": quest_key}


def require_mantis_tracker_readiness(source: str) -> None:
    """Fail closed unless ``source`` has Axol's supported runtime and access.

    The control panel displays these checks, but operations can also start via
    the direct CLI or REST API.  Keeping the authoritative gate here prevents
    those paths from collecting with a stale libsurvive/pyvut build or, for
    Ultimate, without the dongle, protected shared-map Wi-Fi config, and
    operator HID access that the bridge needs.
    """
    if source == "quest":
        return
    if source == "lighthouse":
        from .tracker_install import lighthouse_readiness

        readiness = lighthouse_readiness()
        if readiness["installed"]:
            return
        issues = readiness.get("issues")
        details = "; ".join(str(issue) for issue in issues or [])
        raise RuntimeError(
            "Lighthouse tracking support is incomplete"
            + (f": {details}" if details else "")
            + ". Run `axol tracker.install`, then retry."
        )
    if source == "ultimate":
        from .tracker_ultimate import ultimate_runtime_readiness

        readiness = ultimate_runtime_readiness(cached=False)
        problems: list[str] = []
        if not readiness["installed"]:
            problems.extend(str(issue) for issue in readiness.get("issues") or [])
        if not readiness["dongleConnected"]:
            problems.append("the Ultimate wireless dongle is not connected")
        elif readiness["endpointStatus"] != "accessible":
            problems.append(
                f"Ultimate HID interface 0 is {readiness['endpointStatus']}"
            )
        if not readiness["operatorAccess"]:
            problems.append("this operator cannot access the Ultimate dongle")
        if readiness["wifiConfig"] != "valid":
            problems.append(
                "the shared-map Wi-Fi config is "
                f"{readiness['wifiConfig']} (create a valid mode-0600 config)"
            )
        if not problems:
            return
        # Keep ordering useful but avoid repeating endpoint/access issues that
        # can be reported by both the runtime's generic list and live checks.
        details = "; ".join(dict.fromkeys(problems))
        raise RuntimeError(
            f"Ultimate tracking setup is incomplete: {details}. Run `axol "
            "tracker.ultimate.install` if runtime support is missing, then "
            "`axol tracker.ultimate.check`."
        )
    raise ValueError(
        f"Mantis source must be quest, lighthouse, or ultimate; got {source!r}"
    )


@contextlib.contextmanager
def managed_mantis_bridge(
    source: str,
    *,
    left_channel: str | None,
    right_channel: str | None,
    port: int,
    auto_engage: bool = True,
    pose_source_id: str | None = None,
) -> Iterator[None]:
    """Run the selected tracker bridge beside a direct CLI operation."""
    left, right = require_mantis_channels(
        (
            _rig_channel(left_channel, CAN_LEFT, CAN_MANTIS_LEFT),
            _rig_channel(right_channel, CAN_RIGHT, CAN_MANTIS_RIGHT),
        )
    )
    backend = {"lighthouse": "survive", "ultimate": "ultimate"}.get(source)
    if source == "quest":
        yield
        return
    if backend is None:
        raise ValueError(
            f"Mantis source must be quest, lighthouse, or ultimate; got {source!r}"
        )
    if (
        not isinstance(pose_source_id, str)
        or not pose_source_id.strip()
        or len(pose_source_id) > 128
    ):
        raise RuntimeError(
            "managed Mantis bridge requires the operation's exact pose-source token "
            "(a non-empty string of at most 128 characters)"
        )
    require_mantis_tracker_readiness(source)

    from ..tracker import load_tracker_config
    from ..tracker.bridge import ManagedStdinControls
    from ..tracker.config import select_tracker_backend
    from .tracker_bridge import run_configured_bridge

    config = load_tracker_config()
    select_tracker_backend(config, backend)
    if (config.left is None or config.right is None) and not config.allow_single_side:
        raise RuntimeError(
            f"No complete {source} tracker binding is saved. Run "
            f"`axol tracker.identify --backend {backend}` first."
        )

    # A managed operation has one authoritative left/right map: the channels
    # its Mantis grippers open.  Never retain tracker.bridge's standalone
    # trigger overrides here, otherwise swapping a hub in the operation config
    # can silently leave the trigger controls attached to the old logical side.
    config.trigger_can_left = left
    config.trigger_can_right = right

    stop = threading.Event()
    ready = threading.Event()
    owner_active = threading.Event()
    requested_quit = threading.Event()
    failure: list[BaseException] = []
    failure_lock = threading.Lock()

    def record_failure(exc: BaseException) -> None:
        with failure_lock:
            if not failure:
                failure.append(exc)

    def current_failure() -> BaseException | None:
        with failure_lock:
            return failure[0] if failure else None

    def interrupt_owner() -> None:
        """Wake the main CLI loop so its normal robot cleanup can run."""
        if owner_active.is_set():
            _thread.interrupt_main()

    def request_quit() -> None:
        requested_quit.set()
        interrupt_owner()

    controls = ManagedStdinControls(
        stop,
        request_quit,
        activation_event=owner_active,
    )

    def target() -> None:
        try:
            run_configured_bridge(
                config,
                port=port,
                controls=controls,
                on_ready=ready.set,
                auto_engage=auto_engage,
                require_live_inputs=True,
                pose_source_id=pose_source_id,
            )
        except BaseException as exc:  # noqa: BLE001 - relay to the main thread
            record_failure(exc)
            ready.set()
            if not stop.is_set():
                interrupt_owner()
        else:
            # A managed bridge is expected to live exactly as long as its
            # owner. Returning first without an explicit q/stop is fatal too.
            if not stop.is_set():
                record_failure(RuntimeError("tracker bridge exited unexpectedly"))
                ready.set()
                interrupt_owner()

    thread = threading.Thread(target=target, name="mantis-tracker-bridge", daemon=True)
    thread.start()
    try:
        if not ready.wait(_READY_TIMEOUT_S):
            raise RuntimeError(
                f"{source} tracker bridge did not initialize within "
                f"{_READY_TIMEOUT_S:.0f}s"
            )
        bridge_failure = current_failure()
        if bridge_failure is not None:
            raise RuntimeError(
                f"{source} tracker bridge failed: "
                f"{type(bridge_failure).__name__}: {bridge_failure}"
            ) from bridge_failure
        print(
            f"Mantis {source} inputs are live; r = reset, q = stop; "
            "starting the operation and waiting for the alignment gesture..."
        )
        try:
            owner_active.set()
            # Close the small on_ready→owner_active race: if the reader died
            # in between, fail before starting the robot operation.
            bridge_failure = current_failure()
            if bridge_failure is not None:
                raise RuntimeError(
                    f"{source} tracker bridge failed: "
                    f"{type(bridge_failure).__name__}: {bridge_failure}"
                ) from bridge_failure
            yield
        except KeyboardInterrupt:
            bridge_failure = current_failure()
            if bridge_failure is not None:
                raise RuntimeError(
                    f"{source} tracker bridge failed after startup: "
                    f"{type(bridge_failure).__name__}: {bridge_failure}"
                ) from bridge_failure
            if not requested_quit.is_set():
                raise
        else:
            # collect-data performs its own Ctrl+C cleanup and intentionally
            # swallows KeyboardInterrupt, so inspect the background result
            # again after the operation returns.
            bridge_failure = current_failure()
            if bridge_failure is not None:
                raise RuntimeError(
                    f"{source} tracker bridge failed after startup: "
                    f"{type(bridge_failure).__name__}: {bridge_failure}"
                ) from bridge_failure
    finally:
        active_error = sys.exception()
        owner_active.clear()
        stop.set()
        thread.join(_STOP_TIMEOUT_S)
        teardown_failure = current_failure()
        if thread.is_alive():
            teardown_failure = RuntimeError(
                f"{source} tracker bridge did not stop cleanly; tracker ownership "
                "is uncertain"
            )
        if teardown_failure is not None:
            message = (
                f"{source} tracker bridge teardown failed: "
                f"{type(teardown_failure).__name__}: {teardown_failure}"
            )
            if active_error is not None:
                active_error.add_note(message)
            else:
                raise RuntimeError(message) from teardown_failure
