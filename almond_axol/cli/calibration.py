"""
axol calibration.pull

Fetch this robot's factory calibration (friction + gravity, all joints —
written by ``axol tune.factory``) from the cloud and cache it locally.

The robot is identified by its Axol hub adapter's USB serial — the hub
travels with the arms, so the calibration follows the robot across compute
hosts and reflashes. The fetched document is written to
``~/.almond/factory_calibration.json``, which every ``AxolConfig`` overlays
between the coded defaults and the local calibration file:

    coded config  <-  factory calibration (this cache)  <-  calibration.json

so anything you later tune locally (``tune.friction --save``, ``tune.pid
--save``, ...) still wins over the factory values.

No credentials needed — the calibration objects live in a public bucket
(``axol can.setup`` also runs this pull automatically at the end of setup).

Examples:
    axol calibration.pull
    axol calibration.pull --hub-serial 004800345542501420373234
"""

import argparse
from typing import Any

from ..constants import ARM_JOINTS
from ..robot.calibration import save_factory_calibration
from ..robot.calibration_cloud import fetch_calibration
from .can.setup import hub_serial


def add_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``calibration.pull`` subcommand."""
    p = subparsers.add_parser(
        "calibration.pull",
        help="Fetch this robot's factory calibration (by hub adapter serial) "
        "into the local cache.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--hub-serial",
        default=None,
        metavar="SERIAL",
        help="Robot identity (default: the attached Axol hub adapter's USB serial)",
    )
    p.set_defaults(func=run)


def _summarize(document: dict[str, Any]) -> None:
    for side in ("left", "right"):
        joints = document.get(side)
        if not isinstance(joints, dict) or not joints:
            print(f"  {side}: (no data)")
            continue
        parts = []
        for j in ARM_JOINTS:
            entry = joints.get(j.value)
            if not isinstance(entry, dict):
                continue
            tags = [
                t for t, k in (("friction", "friction"), ("com", "com")) if k in entry
            ]
            parts.append(f"{j.value} ({'+'.join(tags)})" if tags else j.value)
        print(f"  {side}: {', '.join(parts) if parts else '(no data)'}")


def run(args: argparse.Namespace) -> None:
    """Fetch and cache the factory calibration for this robot."""
    serial = args.hub_serial or hub_serial()
    if serial is None:
        raise SystemExit(
            "No Axol hub adapter detected — plug the robot in (or pass "
            "--hub-serial) so the fetch knows which robot's calibration "
            "to pull."
        )

    print(f"Fetching factory calibration for hub {serial} ...")
    try:
        document = fetch_calibration(serial)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}")
    if document is None:
        raise SystemExit(
            f"No factory calibration stored for hub {serial} — run "
            "axol tune.factory on the robot first."
        )
    path = save_factory_calibration(document)
    print(f"Saved to {path}:")
    _summarize(document)
    print(
        "\nEvery AxolConfig now overlays these values between the coded "
        "defaults and the local calibration file (local tuning still wins)."
    )
