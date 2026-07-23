"""
axol tracker.identify

Bind discovered Vive trackers to the left/right UMI rig sides.

Trackers report under backend-native keys (a libsurvive codename like
``T20``, an Ultimate tracker MAC) that say nothing about which rig they
are bolted to. This command discovers the powered-on trackers, then asks
the operator to shake each rig in turn; the device with the most motion
during the capture window is bound to that side. The binding (plus the
backend choice) is saved to ``~/.almond/tracker/config.json`` and picked
up by ``axol tracker.bridge``.
"""

from __future__ import annotations

import time

import numpy as np

_DISCOVER_TIMEOUT_S = 30.0
_CAPTURE_S = 3.0
# A tracker must move at least this much path length (m) during the shake
# window to count — anything less is sensor noise on a resting device.
_MIN_MOTION_M = 0.10


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``tracker.identify`` subcommand."""
    parser = subparsers.add_parser(
        "tracker.identify",
        help="Bind Vive trackers to the left/right UMI rig sides.",
    )
    parser.add_argument(
        "--backend",
        choices=("survive", "ultimate", "synthetic"),
        default=None,
        help="Tracker backend to use and save (default: the saved config, "
        "else survive).",
    )
    parser.set_defaults(func=run)


def _motion(source, window_s: float) -> dict[str, float]:
    """Per-device position path length (m) accumulated over ``window_s``."""
    last: dict[str, np.ndarray] = {}
    travelled: dict[str, float] = {}
    deadline = time.perf_counter() + window_s
    while time.perf_counter() < deadline:
        for key, sample in source.poses().items():
            prev = last.get(key)
            if prev is not None:
                travelled[key] = travelled.get(key, 0.0) + float(
                    np.linalg.norm(sample.pos - prev)
                )
            last[key] = sample.pos
        time.sleep(0.01)
    return travelled


def run(args) -> None:  # type: ignore[no-untyped-def]
    """Discover trackers, capture per-side motion, save the binding."""
    from ..tracker import create_source, load_tracker_config
    from ..tracker.config import save_tracker_config

    config = load_tracker_config()
    if args.backend is not None:
        config.backend = args.backend

    source = create_source(config)
    print(f"Starting the {config.backend} backend...")
    source.start()
    try:
        print(
            "Waiting for trackers to report (power them on and move them a little)..."
        )
        deadline = time.perf_counter() + _DISCOVER_TIMEOUT_S
        while not source.poses():
            if time.perf_counter() >= deadline:
                raise SystemExit(
                    "No trackers reported within "
                    f"{_DISCOVER_TIMEOUT_S:.0f}s — check power, pairing, and "
                    "(for Tracker 3.0) base-station visibility."
                )
            time.sleep(0.2)
        # Give stragglers a moment to appear too.
        time.sleep(2.0)
        keys = sorted(source.poses())
        print(f"Discovered: {', '.join(keys)}\n")

        assigned: dict[str, str] = {}
        for side in ("left", "right"):
            while True:
                input(
                    f"[{side.upper()}] Hold every rig still, press Enter, then "
                    f"shake ONLY the {side} rig for {_CAPTURE_S:.0f}s... "
                )
                travelled = _motion(source, _CAPTURE_S)
                candidates = {
                    k: v
                    for k, v in travelled.items()
                    if v >= _MIN_MOTION_M and k not in assigned.values()
                }
                if not candidates:
                    print("  no tracker moved enough — try again, shake harder.")
                    continue
                key = max(candidates, key=candidates.get)  # type: ignore[arg-type]
                others = sorted(
                    (v for k, v in candidates.items() if k != key), reverse=True
                )
                if others and others[0] > 0.5 * candidates[key]:
                    print(
                        "  two trackers moved a similar amount — hold the other "
                        "rig still and try again."
                    )
                    continue
                print(f"  {side} = {key} ({candidates[key]:.2f} m of motion)")
                assigned[side] = key
                break

        config.left = assigned["left"]
        config.right = assigned["right"]
        save_tracker_config(config)
        from ..tracker.config import TRACKER_CONFIG_FILE

        print(f"\nSaved to {TRACKER_CONFIG_FILE}")
        print("axol tracker.bridge will use this binding automatically.")
    finally:
        source.stop()
