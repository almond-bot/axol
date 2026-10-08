"""Read-only snapshot of a robot for whoever is tuning it (person or agent).

Everything that had to be checked by hand while tuning customer robots, in
one JSON document: code version, whether the realtime core is built for it
and allowed real-time scheduling, the CAN links, what holds the buses,
the panel settings that override gains or link masses, the calibration
(and whether it belongs to this robot's hub), the effective per-joint
gains of both arms side by side, and the latest tuning runs.

It never talks to a motor and never writes anything except ``--bundle``.

    uv run python scripts/robot_report.py                  # JSON to stdout
    uv run python scripts/robot_report.py --bundle ~/report.tar.gz
        # + settings.json, calibration files and recent run metadata
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Callable

import almond_axol  # noqa: E402 - the checkout being reported on
from almond_axol.utils.paths import almond_home  # noqa: E402

REPO = Path(almond_axol.__file__).resolve().parents[1]
ALMOND = almond_home()
_FIELDS = (
    "kp",
    "kd",
    "kd_host",
    "kd_host_hz",
    "stribeck_gain",
    "mass",
    "com",
    "wire_mode",
)


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()
    except Exception as exc:  # noqa: BLE001 - reported, not fatal
        return f"<{exc!r}>"


def _section(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - one broken section must not hide the rest
        return {"error": repr(exc)}


def code() -> dict:
    from importlib.metadata import version

    dirty = _run(["git", "-C", str(REPO), "status", "--porcelain"])
    return {
        "repo": str(REPO),
        "branch": _run(["git", "-C", str(REPO), "branch", "--show-current"]),
        "commit": _run(["git", "-C", str(REPO), "log", "--oneline", "-1"]),
        "dirty_files": len([x for x in dirty.splitlines() if x.strip()]),
        "package_version": version("almond-axol"),
        "python": sys.version.split()[0],
    }


def core() -> dict:
    from almond_axol.rt.link import find_binary

    path = Path(find_binary())
    src = REPO / "rust" / "axol-rt" / "src"
    newest_src = max((p.stat().st_mtime for p in src.glob("*.rs")), default=0.0)
    built = path.stat().st_mtime
    caps = _run(["getcap", str(path)]) if shutil.which("getcap") else "<getcap missing>"
    return {
        "binary": str(path),
        "built": time.strftime("%F %T", time.localtime(built)),
        "stale": bool(src.is_dir() and newest_src > built),
        "realtime_caps": "cap_sys_nice" in caps,
        "getcap": caps,
    }


def system() -> dict:
    links = _run(["ip", "-br", "link"])
    procs = _run(["pgrep", "-af", "axol|axol-rt"])
    busy = [
        p
        for p in procs.splitlines()
        if any(
            k in p
            for k in ("serve", "tune.", "teleop", "axol-rt", "collect", "gravity-comp")
        )
        and "robot_report" not in p
    ]
    service = (
        _run(["systemctl", "is-active", "axol.service"])
        if shutil.which("systemctl")
        else "n/a"
    )
    return {
        "host": socket.gethostname(),
        "machine": platform.machine(),
        "jetson": Path("/etc/nv_tegra_release").exists(),
        "can_links": [x for x in links.splitlines() if x.startswith("can")],
        "axol_service": service,
        "processes_on_the_robot": busy,
    }


def identity() -> dict:
    from almond_axol.robot.identity import hub_serial

    return {"hub_serial": hub_serial()}


def settings() -> dict:
    from almond_axol.settings import load_store

    values = load_store().snapshot()["values"]
    axol = {k: v for k, v in values.items() if k.startswith("axol.")}
    from almond_axol.cli.tune.factory import settings_link_overrides

    return {
        "path": str(ALMOND / "settings.json"),
        "axol_overrides": axol,
        "link_mass_com_overrides": settings_link_overrides(["left", "right"]),
        "has_gripper": values.get("robot.has_gripper", values.get("axol.has_gripper")),
        "note": "panel settings override the calibration file for every field they set",
    }


def calibration() -> dict:
    from almond_axol.robot.identity import hub_serial

    out: dict[str, Any] = {}
    serial = hub_serial()
    for name in ("calibration.json", "factory_calibration.json"):
        path = ALMOND / name
        if not path.is_file():
            out[name] = None
            continue
        raw = json.loads(path.read_text())
        doc: dict[str, Any] = {
            "hub_serial": raw.get("hub_serial"),
            "matches_this_hub": raw.get("hub_serial") == serial if serial else None,
        }
        for side in ("left", "right"):
            joints = raw.get(side) or {}
            doc[side] = {
                j: {
                    "fields": sorted(k for k in e if k != "updated_at"),
                    "fc": (e.get("friction") or {}).get("fc"),
                    "fo": (e.get("friction") or {}).get("fo"),
                    "stribeck_gain": e.get("stribeck_gain"),
                    "updated_at": e.get("updated_at"),
                }
                for j, e in joints.items()
                if isinstance(e, dict)
            }
        out[name] = doc
    return out


def effective() -> dict:
    """The per-joint values the robot runs (settings over calibration over
    defaults), both arms side by side — asymmetries stand out."""
    from almond_axol.constants import ARM_JOINTS
    from almond_axol.settings import shared_axol_config

    cfg = shared_axol_config()
    out = {}
    for j in ARM_JOINTS:
        row = {}
        for side in ("left", "right"):
            jc = getattr(getattr(cfg, side), j.value)
            row[side] = {
                f: (list(getattr(jc, f)) if f == "com" else getattr(jc, f))
                for f in _FIELDS
                if hasattr(jc, f)
            }
            fr = jc.friction
            row[side]["friction"] = {"fc": fr.fc, "fo": fr.fo, "fv": fr.fv}
        out[j.value] = row
    return out


def recent_runs(n: int = 10) -> list[dict]:
    from almond_axol.tuning.runs import TUNING_RUNS_DIR

    metas = sorted(
        TUNING_RUNS_DIR.glob("*/meta.json"), key=lambda p: p.stat().st_mtime
    )[-n:]
    out = []
    for p in reversed(metas):
        m = json.loads(p.read_text())
        met = m.get("metrics") or {}
        out.append(
            {
                "id": m.get("id", p.parent.name),
                "kind": m.get("kind"),
                "label": m.get("label"),
                "side": m.get("side"),
                "completed": met.get("completed"),
                "guard_trips": len(met.get("guard_trips") or []),
            }
        )
    return out


def report() -> dict:
    return {
        "generated": time.strftime("%F %T"),
        "code": _section(code),
        "core": _section(core),
        "system": _section(system),
        "identity": _section(identity),
        "settings": _section(settings),
        "calibration": _section(calibration),
        "effective_config": _section(effective),
        "motions": sorted(p.name for p in (ALMOND / "motions").glob("*.npz")),
        "recent_runs": _section(recent_runs),
    }


def bundle(path: Path, rep: dict) -> None:
    from almond_axol.tuning.runs import TUNING_RUNS_DIR

    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        data = json.dumps(rep, indent=1, default=str).encode()
        info = tarfile.TarInfo("report.json")
        info.size = len(data)
        info.mtime = int(time.time())
        import io

        tar.addfile(info, io.BytesIO(data))
        for name in ("settings.json", "calibration.json", "factory_calibration.json"):
            if (ALMOND / name).is_file():
                tar.add(ALMOND / name, arcname=name)
        for run in rep.get("recent_runs") or []:
            meta = TUNING_RUNS_DIR / str(run.get("id")) / "meta.json"
            if meta.is_file():
                tar.add(meta, arcname=f"runs/{run['id']}/meta.json")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bundle", type=Path, help="also write a .tar.gz with the files")
    args = p.parse_args()
    rep = report()
    print(json.dumps(rep, indent=1, default=str))
    if args.bundle:
        bundle(args.bundle.expanduser(), rep)
        print(f"bundle: {args.bundle}", file=sys.stderr)


if __name__ == "__main__":
    main()
