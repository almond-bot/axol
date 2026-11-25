"""Every ``/tf_static`` transform of one episode, and where each came from.

The URDF places every fixed frame from CAD. Three sources replace or add to
it, per episode:

- **The camera calibration file** (``axol calibration.cameras``), when it
  exists, passed, and names this robot. Each of its transforms gives a
  frame's pose in another frame rigidly attached to it (for example the head
  camera's left optical frame in ``root``). The tree keeps its shape: the
  measured pose becomes the frame's own fixed joint, solved through the CAD
  chain between the two frames, so a TF lookup between them returns the
  file's value.
- **The ZED factory stereo calibration** places the right head eye from the
  left one (after the left eye's own measured or CAD placement).
- **Frames the URDF lacks**: each gripper's tool centre (``<side>_tcp``) and
  each camera IMU (``head_camera_imu``, from the camera's factory
  camera-to-IMU transform).

Moving joints nothing measures are written once at zero, as before.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...camera_calibration.file import (
    CalibrationFileError,
    CameraCalibration,
    FrameTransform,
)
from ...camera_calibration.file import load as load_calibration_file
from .scene import Scene, camera_frame_id

_logger = logging.getLogger(__name__)

CALIBRATION = "calibration"
ZED_FACTORY = "zed_factory"
URDF = "urdf"
GRIPPER_MODEL = "gripper_model"
ASSUMED_ZERO = "urdf_at_zero"

RIGHT_EYE = "head_camera_right_optical"
LEFT_EYE = "head_camera_left_optical"


@dataclass
class StaticFrame:
    parent: str
    child: str
    transform: np.ndarray
    source: str


@dataclass
class EpisodeFrames:
    """The static frames of one episode."""

    frames: list[StaticFrame]
    # The calibration file used, or None.
    calibration: CameraCalibration | None = None
    # Why no file was used ("" when one was).
    problem: str = ""
    # File transforms left out, each with the reason.
    skipped: list[str] = field(default_factory=list)
    # calibrated, partial or uncalibrated (see :func:`resolve`).
    status: str = "uncalibrated"
    # Fixed joints whose transform differs from the URDF: child -> transform.
    edges: dict[str, np.ndarray] = field(default_factory=dict)
    # Frames that are not in the URDF: (parent, child, transform).
    added: list[tuple[str, str, np.ndarray]] = field(default_factory=list)

    def sources(self) -> dict[str, str]:
        return {f.child: f.source for f in self.frames}


def read_calibration(
    path: Path, robot_serial: str | None
) -> tuple[CameraCalibration | None, str]:
    """The usable calibration file at ``path`` and ``""``, or ``None`` and why not."""
    try:
        calibration = load_calibration_file(path)
    except CalibrationFileError as exc:
        return None, str(exc)
    if calibration is None:
        return None, f"there is no camera calibration file ({path})"
    problem = calibration.problem_for(robot_serial)
    if problem is not None:
        return None, problem
    return calibration, ""


def _rigid_chain(scene: Scene, frame: str) -> list[str]:
    """``frame`` and the links above it reached through fixed joints only."""
    tree = scene.tree
    out = [frame]
    link = frame
    while True:
        joint = tree.parent_joint.get(link)
        if joint is None or joint.moving:
            return out
        link = joint.parent
        out.append(link)


def _pose_in(
    scene: Scene, edges: dict[str, np.ndarray], frame: str, ancestor: str
) -> np.ndarray:
    """``frame``'s pose in ``ancestor`` through fixed joints (``edges`` first)."""
    tree = scene.tree
    chain = []
    link = frame
    while link != ancestor:
        joint = tree.parent_joint[link]
        chain.append(joint)
        link = joint.parent
    out = np.eye(4)
    for joint in reversed(chain):
        out = out @ edges.get(joint.child, joint.origin)
    return out


def _measured_edge(
    scene: Scene, edges: dict[str, np.ndarray], entry: FrameTransform
) -> tuple[np.ndarray | None, str, set[str]]:
    """The child's own joint transform that puts it where ``entry`` says.

    Returns ``(transform, "", frames it depends on)``, or ``(None, reason, _)``.
    """
    tree = scene.tree
    parent, child = entry.parent, entry.child
    for name in (parent, child):
        if not tree.has_link(name):
            return None, f"{child}: there is no {name} frame on this robot", set()
    joint = tree.parent_joint.get(child)
    if joint is None or joint.moving:
        return None, f"{child}: not a fixed frame", set()
    child_chain = _rigid_chain(scene, joint.parent)
    parent_chain = _rigid_chain(scene, parent)
    if child in parent_chain:
        return (
            None,
            f"{child}: placed relative to {parent}, which hangs below it",
            set(),
        )
    common = next((link for link in parent_chain if link in child_chain), None)
    if common is None:
        return (
            None,
            f"{child}: {parent} moves relative to it (a moving joint is between them)",
            set(),
        )
    depends = set(parent_chain[: parent_chain.index(common)]) | set(
        child_chain[: child_chain.index(common)]
    )
    to_parent = _pose_in(scene, edges, parent, common)
    to_mount = _pose_in(scene, edges, joint.parent, common)
    return np.linalg.inv(to_mount) @ to_parent @ entry.matrix(), "", depends


def _camera_serials(scene: Scene) -> dict[str, int]:
    """The serial of the camera behind each recorded optical frame."""
    out: dict[str, int] = {}
    for camera in scene.cameras.values():
        if camera.serial is None:
            continue
        out[camera.frame_id] = camera.serial
        if camera.camera == "overhead" and camera.stereo:
            for eye in ("left", "right"):
                out[camera_frame_id(camera.camera, eye)] = camera.serial
    return out


def _apply_file(
    scene: Scene, calibration: CameraCalibration, edges: dict[str, np.ndarray]
) -> tuple[set[str], list[str]]:
    """Put the file's frames into ``edges``. Returns (frames placed, skipped)."""
    serials = _camera_serials(scene)
    pending: list[FrameTransform] = []
    skipped: list[str] = []
    for entry in calibration.transforms:
        fitted = serials.get(entry.child)
        if (
            entry.camera_serial is not None
            and fitted is not None
            and fitted != entry.camera_serial
        ):
            skipped.append(
                f"{entry.child}: measured on camera SN{entry.camera_serial}, "
                f"but SN{fitted} is fitted now"
            )
            continue
        pending.append(entry)
    placed: set[str] = set()
    children = {e.child for e in pending}
    # A frame placed relative to another measured frame waits for that one.
    while pending:
        progress = False
        for entry in list(pending):
            transform, reason, depends = _measured_edge(scene, edges, entry)
            if transform is None:
                skipped.append(reason)
                pending.remove(entry)
                children.discard(entry.child)
                progress = True
                continue
            if (depends & children) - placed - {entry.child}:
                continue
            edges[entry.child] = transform
            placed.add(entry.child)
            pending.remove(entry)
            progress = True
        if not progress:
            skipped += [
                f"{e.child}: placed in a loop of measured frames" for e in pending
            ]
            break
    return placed, skipped


def resolve(
    scene: Scene,
    calibration: CameraCalibration | None = None,
    problem: str = "",
    imu_poses: dict[str, np.ndarray] | None = None,
) -> EpisodeFrames:
    """Every static frame of an episode.

    ``imu_poses`` maps a camera IMU's frame to its pose in the camera's left
    optical frame (the camera's factory value).
    """
    tree = scene.tree
    edges: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    skipped: list[str] = []
    if calibration is not None:
        placed, skipped = _apply_file(scene, calibration, edges)
        for child in placed:
            sources[child] = CALIBRATION
        if not placed:
            problem = (
                f"none of the transforms in {calibration.path.name} apply to this robot"
            )
            calibration = None
    left_eye = tree.parent_joint.get(LEFT_EYE)
    right_eye = tree.parent_joint.get(RIGHT_EYE)
    if (
        scene.stereo is not None
        and RIGHT_EYE not in edges
        and left_eye is not None
        and right_eye is not None
        and left_eye.parent == right_eye.parent
        and not left_eye.moving
        and not right_eye.moving
    ):
        edges[RIGHT_EYE] = edges.get(LEFT_EYE, left_eye.origin) @ scene.stereo
        sources[RIGHT_EYE] = ZED_FACTORY

    frames: list[StaticFrame] = []
    for joint in tree.joints.values():
        if joint.moving:
            continue
        frames.append(
            StaticFrame(
                joint.parent,
                joint.child,
                edges.get(joint.child, joint.origin),
                sources.get(joint.child, URDF),
            )
        )
    for name in scene.assumed_joints():
        joint = tree.joints[name]
        frames.append(
            StaticFrame(
                joint.parent,
                joint.child,
                joint.transform(tree.value(joint, {})),
                ASSUMED_ZERO,
            )
        )
    added: list[tuple[str, str, np.ndarray]] = []
    for parent, child, transform in scene.tool_frames():
        added.append((parent, child, transform))
        frames.append(StaticFrame(parent, child, transform, GRIPPER_MODEL))
    for child, transform in (imu_poses or {}).items():
        parent = scene.imu_parent(child)
        if parent is None or tree.has_link(child):
            continue
        added.append((parent, child, transform))
        frames.append(StaticFrame(parent, child, transform, ZED_FACTORY))

    out = EpisodeFrames(
        frames=frames,
        calibration=calibration,
        problem=problem,
        skipped=skipped,
        edges=edges,
        added=added,
    )
    out.status = _status(scene, out.sources())
    return out


def _status(scene: Scene, sources: dict[str, str]) -> str:
    """``calibrated`` when every recorded camera is placed from measurements,
    ``uncalibrated`` when none is, else ``partial``."""
    measured = []
    for camera in scene.cameras.values():
        source = sources.get(camera.frame_id)
        if camera.frame_id == RIGHT_EYE and source == ZED_FACTORY:
            source = sources.get(LEFT_EYE)
        measured.append(source == CALIBRATION)
    if measured and all(measured):
        return "calibrated"
    return "partial" if any(measured) else "uncalibrated"


def episode_frames(
    scene: Scene,
    calibration_path: Path,
    robot_serial: str | None,
    imu_poses: dict[str, np.ndarray] | None = None,
) -> EpisodeFrames:
    """Read the calibration file and resolve the episode's static frames."""
    calibration, problem = read_calibration(calibration_path, robot_serial)
    frames = resolve(scene, calibration, problem, imu_poses)
    for reason in frames.skipped:
        _logger.warning("Camera calibration: %s", reason)
    return frames
