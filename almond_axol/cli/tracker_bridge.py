"""
axol tracker.bridge

Stream Vive tracker poses into a running teleop session as VRFrame JSON.

Normal ``teleop --mantis`` and ``collect-data --mantis`` sessions start and
own this bridge automatically. This standalone command is for diagnostics or
for a generic server such as ``axol teleop --sim``: it opens the configured
tracker backend, composes VRFrames at 120 Hz, and connects like a headset.

With ``--backend static`` it needs no tracker hardware and holds fixed poses,
which is useful for a standalone bridge/protocol dry run. It is not a Mantis
source option and must not be launched alongside a managed Mantis session.

Backend + left/right binding come from ``~/.almond/tracker/config.json``
(written by ``axol tracker.identify``); every field can be overridden on
the command line. The rig's CAN trigger nodes default to the Mantis
gripper buses (override with ``--trigger-can-left`` /
``--trigger-can-right``), and their analog trigger position drives the
grip command proportionally (squeeze = close, release = open); engage/
reset stay on stdin (Enter toggles engage, ``r`` resets, ``q`` quits)
until the button PCB exists.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from ..tracker.base import (
    TRACKER_PAIR_MAX_SKEW_S,
    TRACKER_POSE_MAX_AGE_S,
    TrackerSourceError,
    valid_tracker_pose,
)
from ..utils.ports import VR_PORT

_logger = logging.getLogger(__name__)
_INPUT_READY_TIMEOUT_S = 25.0


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``tracker.bridge`` subcommand."""
    parser = subparsers.add_parser(
        "tracker.bridge",
        help="Stream Vive tracker poses to the VR server (headset-free teleop).",
    )
    parser.add_argument(
        "--backend",
        choices=("survive", "ultimate", "synthetic", "static"),
        default=None,
        help="Tracker backend (default: the saved config, else survive). "
        "Use static only for a standalone fixed-pose protocol dry run.",
    )
    parser.add_argument(
        "--left",
        default=None,
        help="Left-side device key (libsurvive codename / Ultimate MAC); "
        "overrides the saved binding.",
    )
    parser.add_argument(
        "--right",
        default=None,
        help="Right-side device key; overrides the saved binding.",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="VR server host (the teleop machine). Default: localhost.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=VR_PORT,
        help=f"VR server port. Default: {VR_PORT}.",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=120.0,
        help="Frame streaming rate. Default: 120.",
    )
    parser.add_argument(
        "--trigger-can-left",
        default=None,
        help="SocketCAN interface of the left rig's trigger node; overrides "
        "the saved config, which defaults to the Mantis gripper bus.",
    )
    parser.add_argument(
        "--trigger-can-right",
        default=None,
        help="SocketCAN interface of the right rig's trigger node; overrides "
        "the saved config, which defaults to the Mantis gripper bus.",
    )
    parser.add_argument(
        "--allow-single-side",
        action="store_true",
        help="Run with only one side's tracker bound. WARNING: absolute-mode "
        "(Mantis) engagement fits the base transform from BOTH controller "
        "positions, so the placeholder pose streamed for the unbound side "
        "corrupts it.",
    )
    parser.set_defaults(func=run)


def run(args) -> None:  # type: ignore[no-untyped-def]
    """Open the tracker backend (and trigger PCBs) and stream frames until quit."""
    logging.basicConfig(level=logging.INFO, force=True)

    from ..tracker import load_tracker_config
    from ..tracker.config import select_tracker_backend

    config = load_tracker_config()
    if args.backend is not None:
        select_tracker_backend(config, args.backend)
    if args.left is not None:
        config.left = args.left
    if args.right is not None:
        config.right = args.right

    if args.trigger_can_left is not None:
        config.trigger_can_left = args.trigger_can_left
    if args.trigger_can_right is not None:
        config.trigger_can_right = args.trigger_can_right
    config.allow_single_side = args.allow_single_side or config.allow_single_side

    run_configured_bridge(
        config,
        host=args.host,
        port=args.port,
        hz=args.hz,
    )


def run_configured_bridge(
    config: Any,
    *,
    host: str = "localhost",
    port: int = VR_PORT,
    hz: float = 120.0,
    controls: Any = None,
    on_ready: Callable[[], None] | None = None,
    auto_engage: bool = False,
    require_live_inputs: bool = False,
    pose_source_id: str | None = None,
) -> None:
    """Run one configured bridge, optionally under headless lifecycle controls.

    ``on_ready`` fires after the tracker backend, trigger readers, and bridge
    object are ready. With ``require_live_inputs``, it additionally waits for
    fresh, synchronized tracked poses on both bound sides and a fresh frame
    from every configured trigger. The WebSocket connection may not exist yet:
    the bridge deliberately starts before the operation's VR server and
    reconnects until that server begins listening.
    """
    from ..tracker import HARDWARE_FREE_BINDINGS, create_source
    from ..tracker.bridge import TrackerBridge
    from ..tracker.trigger import TriggerReader

    left, right = config.left, config.right
    binding = HARDWARE_FREE_BINDINGS.get(config.backend)
    if binding is not None and left is None and right is None:
        left, right = binding

    if require_live_inputs:
        if (
            not isinstance(pose_source_id, str)
            or not pose_source_id.strip()
            or len(pose_source_id) > 128
        ):
            raise RuntimeError(
                "managed Mantis bridge requires the operation's exact pose-source "
                "token (a non-empty string of at most 128 characters)"
            )
        if left is None or right is None:
            raise RuntimeError(
                "managed Mantis operation requires a tracker bound to each side; "
                "run `axol tracker.identify`"
            )
        if left == right:
            raise RuntimeError(
                f"left and right are both bound to {left!r}; run "
                "`axol tracker.identify` and bind two distinct trackers"
            )
        missing_trigger_sides = [
            side
            for side, channel in (
                ("left", config.trigger_can_left),
                ("right", config.trigger_can_right),
            )
            if not channel
        ]
        if missing_trigger_sides:
            raise RuntimeError(
                "managed Mantis operation requires both trigger CAN channels; "
                "configure: " + ", ".join(missing_trigger_sides)
            )

    triggers: dict[str, TriggerReader] = {}
    source = create_source(config)
    source.start()
    try:
        for side, channel in (
            ("left", config.trigger_can_left),
            ("right", config.trigger_can_right),
        ):
            if not channel:
                continue
            try:
                triggers[side] = TriggerReader(channel)
            except Exception as exc:
                if require_live_inputs:
                    raise RuntimeError(
                        f"{side} Mantis trigger could not open {channel}: {exc}"
                    ) from exc
                # No trigger node reachable on this host (sim dry run, a rig
                # without the PCB, or the CAN bus not brought up). That side
                # streams fully open rather than failing the whole session.
                _logger.warning(
                    "%s trigger node on %s could not be opened — that side's "
                    "gripper will stream fully open",
                    side,
                    channel,
                    exc_info=True,
                )
        if require_live_inputs:
            _wait_for_live_inputs(source, left, right, triggers)
        bridge = TrackerBridge(
            source,
            left=left,
            right=right,
            host=host,
            port=port,
            hz=hz,
            controls=controls,
            left_trigger=triggers.get("left"),
            right_trigger=triggers.get("right"),
            allow_single_side=config.allow_single_side,
            auto_engage=auto_engage,
            confirm_auto_engage=require_live_inputs,
            pose_source_id=pose_source_id,
        )
        if on_ready is not None:
            on_ready()
        asyncio.run(bridge.run())
    except KeyboardInterrupt:
        pass
    finally:
        cleanup_failures: list[tuple[str, BaseException]] = []
        for reader in triggers.values():
            try:
                reader.close()
            except BaseException as exc:
                cleanup_failures.append(("trigger", exc))
        try:
            source.stop()
        except BaseException as exc:
            cleanup_failures.append(("tracker source", exc))
        if cleanup_failures:
            label, failure = cleanup_failures[0]
            for extra_label, extra in cleanup_failures[1:]:
                failure.add_note(
                    f"additional {extra_label} cleanup failure: "
                    f"{type(extra).__name__}: {extra}"
                )
            raise TrackerSourceError(
                f"{label} teardown failed; tracker ownership is uncertain"
            ) from failure


def _wait_for_live_inputs(
    source: Any,
    left: str | None,
    right: str | None,
    triggers: dict[str, Any],
    timeout_s: float = _INPUT_READY_TIMEOUT_S,
) -> None:
    """Block until both tracker poses and every trigger are currently live."""
    bindings = {"left": left, "right": right}
    deadline = time.perf_counter() + timeout_s
    missing: list[str] = []
    while True:
        poses = source.poses()
        now = time.perf_counter()
        missing = []
        ready_poses: dict[str, Any] = {}
        for side, key in bindings.items():
            if key is None:
                missing.append(f"{side} tracker is not bound")
                continue
            pose = poses.get(key)
            if pose is None:
                missing.append(f"{side} tracker {key!r} is not reporting")
            elif not valid_tracker_pose(pose):
                missing.append(f"{side} tracker {key!r} reported an invalid pose")
            elif not pose.tracking:
                missing.append(f"{side} tracker {key!r} has not converged")
            elif not now - pose.t <= TRACKER_POSE_MAX_AGE_S:
                missing.append(f"{side} tracker {key!r} is stale")
            else:
                ready_poses[side] = pose
        if len(ready_poses) == 2:
            skew_s = abs(ready_poses["left"].t - ready_poses["right"].t)
            if skew_s > TRACKER_PAIR_MAX_SKEW_S:
                missing.append(
                    "left/right tracker samples are not synchronized "
                    f"({skew_s * 1000.0:.0f} ms apart; maximum "
                    f"{TRACKER_PAIR_MAX_SKEW_S * 1000.0:.0f} ms)"
                )
        for side, trigger in triggers.items():
            if trigger.grip() is None or trigger.is_stale():
                missing.append(f"{side} trigger has no fresh CAN frames")
        if not missing:
            _logger.info("both Mantis trackers and triggers are live")
            return
        if now >= deadline:
            raise RuntimeError(
                "Mantis inputs were not ready within "
                f"{timeout_s:.0f}s: "
                + "; ".join(missing)
                + ". Power and move both trackers, verify their map/base-station "
                "visibility, and check the Mantis CAN channel mapping."
            )
        time.sleep(0.05)
