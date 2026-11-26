"""The calibration procedure, from two recordings to a package and a file.

1. Fit Recording A: each camera's pose on the robot (the head camera's left
   eye in ``root``, each wrist camera in its arm's last link).
2. Lock the fitted values (``locked_A.json``, with A's SHA-256) before
   Recording B is examined.
3. Check Recording B with A's values held fixed (:mod:`.checks`), and fit B
   on its own for the repeatability check.
4. Write the evidence package; on PASS, the calibration file.

The same code runs on recordings from a robot and from the sim.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .board import Board
from .checks import (
    Outcome,
    ProcedureLimits,
    Rig,
    camera_errors,
    cross_camera,
    judge,
    placements,
    view_counts,
)
from .file import document, due_after, transform_entry, utc_text, write
from .package import PACKAGE_FORMAT, next_id, store_recording, write_json
from .recording import Recording, read
from .solve import CameraSpec, Limits, View, solve

_logger = logging.getLogger(__name__)

# Recording camera -> solver camera (the head is fitted through its left eye).
_SPEC_OF = {
    "head_left": "head",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
}
_FRAMES = {
    "head": "head_camera_left_optical",
    "left_wrist": "left_wrist_camera_optical",
    "right_wrist": "right_wrist_camera_optical",
}
_MOUNTS = {"head": "root", "left_wrist": "left_w2", "right_wrist": "right_w2"}
_LABELS = {
    "head": "Head camera",
    "left_wrist": "Left wrist camera",
    "right_wrist": "Right wrist camera",
}
_CAMERA_OF = {
    "head": "head_left",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
}


@dataclass
class SessionResult:
    passed: bool
    calibration_id: str
    package: Path
    outcome: Outcome
    mounts: dict[str, np.ndarray]
    file: Path | None
    notes: list[str] = field(default_factory=list)


def specs_for(recording: Recording, tree: Any) -> dict[str, CameraSpec]:
    """The cameras to fit, with the recording's lenses and CAD mounts."""
    specs = {}
    for camera, key in _SPEC_OF.items():
        lens = recording.lenses.get(camera)
        if lens is None or not recording.frames.get(camera):
            continue
        frame, mount = _FRAMES[key], _MOUNTS[key]
        specs[key] = CameraSpec(
            key=key,
            label=_LABELS[key],
            frame=frame,
            mount=mount,
            cad=tree.fk(mount, frame, {}),
            lens=lens,
            moves=mount != "root",
        )
    return specs


def fit(
    recording: Recording, board: Board, tree: Any
) -> tuple[dict[str, np.ndarray], Any]:
    """Each camera's pose in its mount from one recording's table views."""
    specs = specs_for(recording, tree)
    views: list[View] = []
    for camera, key in _SPEC_OF.items():
        if key not in specs:
            continue
        mount = _MOUNTS[key]
        seen: set[int] = set()
        for frame in recording.frames.get(camera, []):
            if frame.phase != "table":
                continue
            if mount == "root":
                # The board lies still: one head view per placement is enough.
                if frame.placement in seen:
                    continue
                seen.add(frame.placement)
                pose = np.eye(4)
            else:
                pose = tree.fk("root", mount, frame.joints)
            views.append(View(key, frame.placement, pose, frame.detection))
    loose = Limits(
        min_wrist_views=1,
        min_head_views=1,
        min_head_placements=1,
        max_rms_px=1e9,
        max_change_mm=1e9,
        max_change_deg=1e9,
        min_turn_deg=0.0,
    )
    solution = solve(specs, views, board, loose)
    mounts = {
        key: result.transform
        for key, result in solution.cameras.items()
        if result.transform is not None
    }
    return mounts, solution


def run(
    path_a: Path,
    path_b: Path,
    *,
    board: Board,
    tree: Any,
    arm_joints: Mapping[str, list[str]],
    robot_serial: str,
    packages: Path,
    calibration_path: Path | None,
    limits: ProcedureLimits | None = None,
    valid_days: float = 30.0,
    identity: Mapping[str, Any] | None = None,
    truth: Mapping[str, np.ndarray] | None = None,
    progress: Callable[[str], None] = print,
) -> SessionResult:
    """Fit A, lock it, check B, write the package (and the file on PASS)."""
    limits = limits or ProcedureLimits()
    now = datetime.now(UTC).replace(microsecond=0)
    calibration_id = next_id(robot_serial, now, packages)
    folder = packages / calibration_id
    folder.mkdir(parents=True, exist_ok=False)
    calibrated_at = utc_text(now)
    due_at = due_after(calibrated_at, valid_days)

    progress("Reading Recording A")
    rec_a = read(path_a, board, arm_joints=dict(arm_joints), progress=progress)
    progress("Fitting Recording A")
    mounts_a, solution_a = fit(rec_a, board, tree)
    locked = {
        "calibration_id": calibration_id,
        "locked_at": utc_text(datetime.now(UTC)),
        "recording_a_sha256": rec_a.sha256,
        "board": _board_identity(board),
        "mounts": {
            key: transform_entry(_MOUNTS[key], _FRAMES[key], value)
            for key, value in mounts_a.items()
        },
        "intrinsics": rec_a.intrinsics,
        "stereo": None if rec_a.stereo is None else rec_a.stereo.tolist(),
    }
    locked_sha = write_json(folder / "locked_A.json", locked)
    progress(f"Locked Recording A's values (SHA-256 {locked_sha[:16]})")

    progress("Reading Recording B")
    rec_b = read(path_b, board, arm_joints=dict(arm_joints), progress=progress)
    stereo = rec_a.stereo if rec_a.stereo is not None else np.eye(4)
    rig = Rig(tree, mounts_a, stereo)
    places = placements(rig, rec_b, board)
    errors_b = camera_errors(rig, rec_b, board, places)
    counts_a = view_counts(rec_a, board)
    counts_b = view_counts(rec_b, board)
    cross_a = cross_camera(rec_a, board, stereo) if "head_right" in rec_a.lenses else {}
    cross_b = cross_camera(rec_b, board, stereo) if "head_right" in rec_b.lenses else {}
    progress("Fitting Recording B on its own, for the repeatability check")
    mounts_b, _solution_b = fit(rec_b, board, tree)
    cad = {key: tree.fk(_MOUNTS[key], _FRAMES[key], {}) for key in _FRAMES}
    cameras = [
        c
        for c in ("head_left", "head_right", "left_wrist", "right_wrist")
        if c in rec_b.lenses
    ]
    outcome = judge(
        limits,
        errors_b=errors_b,
        counts_a=counts_a,
        counts_b=counts_b,
        cross_a=cross_a,
        cross_b=cross_b,
        mounts_a=mounts_a,
        mounts_b=mounts_b,
        cad=cad,
        cameras=cameras,
    )
    same_mode = rec_a.intrinsics == rec_b.intrinsics
    outcome.add(
        "image mode",
        "all",
        "A and B match" if same_mode else "A and B differ",
        "same as production, same in A and B",
        same_mode,
    )
    missing = [k for k in _FRAMES if k not in mounts_a]
    outcome.add(
        "cameras linked",
        "all",
        "every camera fitted" if not missing else f"not fitted: {', '.join(missing)}",
        "all",
        not missing,
    )
    passed = outcome.passed
    status = "pass" if passed else "fail"

    # The package.
    stored_a = store_recording(Path(path_a), folder, "recording_A.mcap")
    stored_b = store_recording(Path(path_b), folder, "recording_B.mcap")
    write_json(folder / "corners_A.json", {"recording": "A", "frames": rec_a.corners})
    write_json(folder / "corners_B.json", {"recording": "B", "frames": rec_b.corners})
    entries = [
        transform_entry(
            _MOUNTS[key], _FRAMES[key], mounts_a[key], camera_serial=_serial(rec_a, key)
        )
        for key in _FRAMES
        if key in mounts_a
    ]
    doc = document(
        robot_serial=robot_serial,
        calibration_id=calibration_id,
        calibrated_at=calibrated_at,
        due_at=due_at,
        status=status,
        transforms=entries,
        board=_board_identity(board),
        package=str(folder),
    )
    file_in_package = write(doc, folder / "camera_calibration.json")
    notes: list[str] = []
    if truth:
        for key, value in mounts_a.items():
            if key in truth:
                from .checks import change

                mm, deg = change(truth[key], value)
                notes.append(
                    f"{_LABELS[key]}: {mm:.2f} mm and {deg:.3f} degrees from the sim's true mount"
                )
    package = {
        "format": PACKAGE_FORMAT,
        "version": 1,
        "result": "PASS" if passed else "FAIL",
        "identity": {
            "robot_serial": robot_serial,
            "calibration_id": calibration_id,
            "calibrated_at": calibrated_at,
            "due_at": due_at,
            "board": _board_identity(board),
            "cameras": _cameras(rec_a),
            "imu": {
                "camera_serial": rec_a.metadata.get("recording", {}).get(
                    "head_imu.serial", ""
                ),
                "frame_id": rec_a.metadata.get("recording", {}).get(
                    "head_imu.frame_id", ""
                ),
            },
            "software_version": rec_a.metadata.get("robot", {}).get(
                "software_version", ""
            ),
            "image_mode": _image_mode(rec_a),
            **dict(identity or {}),
        },
        "intrinsics": {
            camera: {
                **data,
                "source": "factory calibration",
                "crop": "none",
                "resize": "none",
                "rotation": "none",
                "binning": "none",
            }
            for camera, data in rec_a.intrinsics.items()
        },
        "transforms": {
            "convention": (
                "child pose in parent: translation in metres, unit quaternion "
                "x, y, z, w; right-handed; optical frames z forward, x right, y down"
            ),
            "calibrated": entries,
            "verified_factory": [
                transform_entry(
                    "head_camera_left_optical", "head_camera_right_optical", stereo
                )
            ],
            "head_stereo_baseline_mm": float(1000.0 * np.linalg.norm(stereo[:3, 3])),
        },
        "recordings": {
            "A": {
                "file": stored_a.name,
                "recording_id": rec_a.recording_id,
                "sha256": rec_a.sha256,
                "used_for": "fitting",
            },
            "B": {
                "file": stored_b.name,
                "recording_id": rec_b.recording_id,
                "sha256": rec_b.sha256,
                "used_for": "checking (and the repeatability fit)",
            },
        },
        "detected_corners": {"A": "corners_A.json", "B": "corners_B.json"},
        "imu_samples": "the /head/imu topic of each recording",
        "metrics": {
            "per_camera_B": errors_b,
            "views_A": counts_a,
            "views_B": counts_b,
            "head_pair_A": cross_a,
            "head_pair_B": cross_b,
            "repeatability": {
                key: dict(
                    zip(
                        ("translation_mm", "rotation_deg"),
                        _change(mounts_a[key], mounts_b[key]),
                        strict=True,
                    )
                )
                for key in mounts_a
                if key in mounts_b
            },
        },
        "checks": [c.as_dict() for c in outcome.checks],
        "statement": (
            f"The calibrated values were fitted on Recording A only and locked at "
            f"{locked['locked_at']} (locked_A.json, SHA-256 {locked_sha}) before "
            "Recording B was examined. Recording B was used only to check them and "
            "for the separate repeatability fit; its values replace nothing."
        ),
        "calibration_file": file_in_package.name,
        "notes": notes,
    }
    write_json(folder / "package.json", package)
    written = None
    if passed and calibration_path is not None:
        written = write(doc, calibration_path)
    return SessionResult(
        passed, calibration_id, folder, outcome, mounts_a, written, notes
    )


def _change(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    from .checks import change

    return change(a, b)


def _serial(recording: Recording, key: str) -> int | None:
    meta = recording.metadata.get("recording", {})
    prefix = {
        "head": "head_left",
        "left_wrist": "wrist_left",
        "right_wrist": "wrist_right",
    }[key]
    value = meta.get(f"{prefix}.serial")
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None


def _cameras(recording: Recording) -> dict[str, Any]:
    meta = recording.metadata.get("recording", {})
    out = {}
    for camera, prefix in (
        ("head_left", "head_left"),
        ("head_right", "head_right"),
        ("left_wrist", "wrist_left"),
        ("right_wrist", "wrist_right"),
    ):
        if f"{prefix}.serial" in meta:
            out[camera] = {
                "serial": meta.get(f"{prefix}.serial"),
                "frame_id": meta.get(f"{prefix}.frame_id"),
            }
    return out


def _image_mode(recording: Recording) -> dict[str, Any]:
    meta = recording.metadata.get("recording", {})
    out = {}
    for prefix in ("head_left", "head_right", "wrist_left", "wrist_right"):
        if f"{prefix}.width" in meta:
            out[prefix] = {
                "width": meta.get(f"{prefix}.width"),
                "height": meta.get(f"{prefix}.height"),
                "fps": meta.get(f"{prefix}.fps"),
                "codec": meta.get("video_codec"),
                "images": meta.get("images"),
                "timestamps": meta.get("clock"),
            }
    return out


def _board_identity(board: Board) -> dict[str, Any]:
    c = board.config
    return {
        "id": c.id,
        "squares": [c.squares_x, c.squares_y],
        "square_mm": c.square_mm,
        "marker_mm": c.marker_mm,
        "dictionary": c.dictionary,
    }


def report(result: SessionResult) -> list[str]:
    """Lines for the operator."""
    lines = [
        f"Calibration {result.calibration_id}: {'PASS' if result.passed else 'FAIL'}.",
    ]
    for check in result.outcome.checks:
        mark = {True: "PASS", False: "FAIL", None: "    "}[check.passed]
        lines.append(
            f"  {mark}  {check.camera}: {check.name}: {check.value} ({check.limit})"
        )
    lines += [f"  {note}" for note in result.notes]
    lines.append(f"Evidence package: {result.package}")
    if result.file is not None:
        lines.append(f"Saved the calibration to {result.file}.")
    elif not result.passed:
        lines.append(
            "Nothing was saved; the robot keeps its previous calibration until it expires."
        )
    return lines
