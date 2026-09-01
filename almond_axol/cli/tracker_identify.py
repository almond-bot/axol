"""
axol tracker.identify

Bind discovered Vive trackers to the left/right Mantis rig sides.

Trackers report under backend-native keys (a libsurvive codename like
``T20``, an Ultimate tracker MAC) that say nothing about which rig they
are bolted to. This command discovers the powered-on trackers, then asks
the operator to shake each rig in turn; the device with the most motion
during the capture window is bound to that side. The binding (plus the
backend choice) is saved to ``~/.almond/tracker/config.json`` and picked
up by ``axol tracker.bridge``.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from ..tracker.base import TRACKER_POSE_MAX_AGE_S, valid_tracker_pose

_DISCOVER_TIMEOUT_S = 30.0
_CAPTURE_S = 3.0
_FRESH_S = TRACKER_POSE_MAX_AGE_S
_MAX_CAPTURE_ATTEMPTS = 5
# A tracker must move at least this much path length (m) during the shake
# window to count — anything less is sensor noise on a resting device.
_MIN_MOTION_M = 0.10


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``tracker.identify`` subcommand."""
    parser = subparsers.add_parser(
        "tracker.identify",
        help="Bind Vive trackers to the left/right Mantis rig sides.",
    )
    parser.add_argument(
        "--backend",
        choices=("survive", "ultimate", "synthetic"),
        default=None,
        help="Tracker backend to use and save (default: the saved config, "
        "else survive).",
    )
    parser.add_argument(
        "--web-prompts",
        action="store_true",
        help="Emit guided prompt markers for the web control panel.",
    )
    parser.set_defaults(func=run)


def _confirm(instruction: str, web_prompts: bool) -> None:
    """Wait for a motion-capture step from a terminal or the web UI."""
    if web_prompts:
        print(f"[prompt] {instruction}", flush=True)
        sys.stdin.readline()
    else:
        input(f"{instruction} Press Enter to begin ... ")


def _motion(source, window_s: float) -> tuple[dict[str, float], set[str]]:
    """Fresh, fully-tracked position path length per device over a window.

    Also returns the keys that reported at all, so a device whose poses went
    stale mid-capture can be named rather than silently dropping out.
    """
    last: dict[str, np.ndarray] = {}
    travelled: dict[str, float] = {}
    seen: set[str] = set()
    deadline = time.perf_counter() + window_s
    while time.perf_counter() < deadline:
        for key, sample in source.poses().items():
            seen.add(key)
            if (
                not valid_tracker_pose(sample)
                or not sample.tracking
                or time.perf_counter() - sample.t > _FRESH_S
            ):
                last.pop(key, None)
                continue
            prev = last.get(key)
            if prev is not None:
                travelled[key] = travelled.get(key, 0.0) + float(
                    np.linalg.norm(sample.pos - prev)
                )
            last[key] = sample.pos
        time.sleep(0.01)
    return travelled, seen


def _motion_summary(
    travelled: dict[str, float], seen: set[str], assigned: dict[str, str]
) -> str:
    """One line per device: how far it moved, or that it was stale/bound."""
    bound = {key: side for side, key in assigned.items()}
    parts = []
    for key in sorted(seen):
        if key in bound:
            parts.append(
                f"{key} {travelled.get(key, 0.0):.2f} m (already {bound[key]})"
            )
        elif key in travelled:
            parts.append(f"{key} {travelled[key]:.2f} m")
        else:
            parts.append(f"{key} no fresh poses")
    return ", ".join(parts) if parts else "no devices reported"


def _report_backend_warnings(source) -> None:  # type: ignore[no-untyped-def]
    """Surface backend setup warnings (libsurvive logs them, poses still flow)."""
    warnings = getattr(source, "warnings", None)
    if warnings is None:
        return
    for message in warnings():
        print(f"  backend warning: {message}")


def _check_lighthouse_channels(source, live: list[str]) -> None:  # type: ignore[no-untyped-def]
    """Refuse to bind while base stations share a channel.

    A clash makes tracker poses jump or stall as soon as a rig moves, so the
    motion capture would either fail repeatedly or bind the wrong device. The
    survey is persisted either way so the control panel can show it.
    """
    lighthouse_survey = getattr(source, "lighthouse_survey", None)
    if lighthouse_survey is None:
        return
    from ..tracker.lighthouse_survey import save_lighthouse_survey

    survey = lighthouse_survey()
    survey.trackers = set(live)
    save_lighthouse_survey(survey)
    clashes = survey.clash_problems()
    if clashes:
        raise SystemExit(
            "Base-station channel clash: " + "; ".join(clashes) + ". Identify "
            "cannot bind trackers reliably until every base station is on its "
            "own channel."
        )


def run(args) -> None:  # type: ignore[no-untyped-def]
    """Discover trackers, capture per-side motion, save the binding."""
    from ..tracker import create_source, load_tracker_config
    from ..tracker.config import save_tracker_config, select_tracker_backend

    config = load_tracker_config()
    if args.backend is not None:
        select_tracker_backend(config, args.backend)

    runtime_source = {
        "survive": "lighthouse",
        "ultimate": "ultimate",
    }.get(config.backend)
    if runtime_source is not None:
        # Identify persists device ownership, so it must use the same pinned
        # runtime and host-access policy as the managed operation that will
        # consume that binding.  Synthetic remains available for tests.
        from .mantis_bridge import require_mantis_tracker_readiness

        try:
            require_mantis_tracker_readiness(runtime_source)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from None

    source = create_source(config)
    print(f"Starting the {config.backend} backend...")
    try:
        # Enter cleanup ownership before start(): a backend can acquire its USB
        # device or launch a helper before an interrupt/failure reaches us.
        source.start()
        print(
            "Waiting for trackers to report (power them on and move them a little)..."
        )
        deadline = time.perf_counter() + _DISCOVER_TIMEOUT_S
        live: list[str] = []
        while len(live) < 2:
            now = time.perf_counter()
            live = sorted(
                key
                for key, sample in source.poses().items()
                if valid_tracker_pose(sample)
                and sample.tracking
                and now - sample.t <= _FRESH_S
            )
            if time.perf_counter() >= deadline:
                source_hint = (
                    "check power, pairing, and base-station visibility"
                    if config.backend == "survive"
                    else "check power, dongle permissions, and that both trackers "
                    "have localized in the Windows-created SLAM map"
                    if config.backend == "ultimate"
                    else "check the tracker source"
                )
                raise SystemExit(
                    "Two fresh, fully tracked devices did not report within "
                    f"{_DISCOVER_TIMEOUT_S:.0f}s (live: "
                    f"{', '.join(live) or 'none'}) — {source_hint}."
                )
            time.sleep(0.2)
        # Give stragglers a moment to appear too.
        time.sleep(2.0)
        print(f"Discovered and tracking: {', '.join(live)}")
        _report_backend_warnings(source)
        _check_lighthouse_channels(source, live)
        print()

        assigned: dict[str, str] = {}
        for side in ("left", "right"):
            for attempt in range(1, _MAX_CAPTURE_ATTEMPTS + 1):
                _confirm(
                    f"Hold every rig still. When ready, move ONLY the {side.upper()} "
                    f"Mantis for {_CAPTURE_S:.0f} seconds.",
                    args.web_prompts,
                )
                travelled, seen = _motion(source, _CAPTURE_S)
                summary = _motion_summary(travelled, seen, assigned)
                candidates = {
                    k: v
                    for k, v in travelled.items()
                    if v >= _MIN_MOTION_M and k not in assigned.values()
                }
                if not candidates:
                    print(
                        "  no unassigned, fully tracked device moved at least "
                        f"{_MIN_MOTION_M:.2f} m ({summary}) — restore tracking "
                        "and try again."
                    )
                    _report_backend_warnings(source)
                    continue
                key = max(candidates, key=candidates.get)  # type: ignore[arg-type]
                others = sorted(
                    (v for k, v in candidates.items() if k != key), reverse=True
                )
                if others and others[0] > 0.5 * candidates[key]:
                    print(
                        f"  two trackers moved a similar amount ({summary}) — "
                        "hold the other rig still and try again."
                    )
                    continue
                print(f"  {side} = {key} ({candidates[key]:.2f} m of motion)")
                assigned[side] = key
                break
            else:
                raise SystemExit(
                    f"Could not identify the {side} tracker after "
                    f"{_MAX_CAPTURE_ATTEMPTS} attempts. No binding was changed."
                )

        config.left = assigned["left"]
        config.right = assigned["right"]
        save_tracker_config(config)
        from ..tracker.config import TRACKER_CONFIG_FILE

        print(f"\nSaved to {TRACKER_CONFIG_FILE}")
        print("axol tracker.bridge will use this binding automatically.")
    finally:
        source.stop()
