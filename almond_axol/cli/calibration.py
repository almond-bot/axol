"""
axol calibration.pull / calibration.push

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

``calibration.push`` goes the other way: it uploads this machine's local
calibration file (``~/.almond/calibration.json`` — everything ``tune.friction``,
``tune.gravity``, ``tune.pid`` and ``motion``-side tools saved on this robot)
to the cloud under the hub serial, merged per joint over what is already
stored there, so a robot calibrated joint by joint needs no re-run of
``tune.factory`` to share its values. It needs the Supabase write key
(``AXOL_SUPABASE_KEY``, as ``tune.factory`` does).

Examples:
    axol calibration.pull
    axol calibration.pull --hub-serial 004800345542501420373234
    axol calibration.push --dry-run
    axol calibration.push
"""

import argparse
from typing import Any

from ..constants import ARM_JOINTS
from ..robot.calibration import (
    CALIBRATION_PATH,
    load_calibration,
    save_factory_calibration,
)
from ..robot.calibration_cloud import (
    fetch_calibration,
    push_calibration,
    supabase_credentials,
)
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

    u = subparsers.add_parser(
        "calibration.push",
        help="Upload this robot's local calibration file to the cloud (by hub "
        "adapter serial), merged per joint over what is stored there.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    u.add_argument(
        "--hub-serial",
        default=None,
        metavar="SERIAL",
        help="Robot identity (default: the attached Axol hub adapter's USB serial)",
    )
    u.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )
    u.set_defaults(func=run_push)


def merge_documents(
    cloud: dict[str, Any] | None, local: dict[str, dict[str, Any]], serial: str
) -> dict[str, Any]:
    """The cloud document with every joint the local file has replaced by the
    local entry (a joint calibrated here wins as a whole; joints only the
    cloud has are kept)."""
    merged: dict[str, Any] = {"version": 1, "hub_serial": serial}
    for side in ("left", "right"):
        old = (cloud or {}).get(side)
        side_doc = dict(old) if isinstance(old, dict) else {}
        for joint, entry in (local.get(side) or {}).items():
            side_doc[joint] = entry
        if side_doc:
            merged[side] = side_doc
    return merged


def run_push(args: argparse.Namespace) -> None:
    """Upload the local calibration file for this robot."""
    serial = args.hub_serial or hub_serial()
    if serial is None:
        raise SystemExit(
            "No Axol hub adapter detected — plug the robot in (or pass "
            "--hub-serial) so the upload knows which robot it is."
        )
    local = load_calibration(CALIBRATION_PATH, expected_hub_serial=serial)
    if not any(local.get(s) for s in ("left", "right")):
        raise SystemExit(
            f"No calibration for hub {serial} in {CALIBRATION_PATH} — nothing to push."
        )
    try:
        cloud = fetch_calibration(serial)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: could not read the stored calibration: {exc}")
    merged = merge_documents(cloud, local, serial)
    print(f"Calibration for hub {serial} ({CALIBRATION_PATH}):")
    _summarize(merged)
    if args.dry_run:
        print("\n(dry run — nothing uploaded)")
        return
    creds = supabase_credentials()
    if creds is None:
        raise SystemExit(
            "No Supabase write key (AXOL_SUPABASE_KEY, plus AXOL_SUPABASE_URL "
            "if not baked in) — cannot upload."
        )
    try:
        push_calibration(creds, serial, merged)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}")
    print(
        f"\nUploaded to the cloud as {serial}; any machine fetches it with "
        "axol calibration.pull."
    )


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
                t
                for t, k in (
                    ("friction", "friction"),
                    ("stribeck", "stribeck_dfs"),
                    ("com", "com"),
                    ("gains", "kp"),
                )
                if k in entry
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
        # Scope the empty cache to this robot so values cached for a previous
        # Axol can never survive a confirmed cloud miss.
        save_factory_calibration({"version": 1}, hub_serial=serial)
        raise SystemExit(
            f"No factory calibration stored for hub {serial} — run "
            "axol tune.factory on the robot first."
        )
    path = save_factory_calibration(document, hub_serial=serial)
    print(f"Saved to {path}:")
    _summarize(document)
    print(
        "\nEvery AxolConfig now overlays these values between the coded "
        "defaults and the local calibration file (local tuning still wins)."
    )
