"""Where each camera sits, from views of a board that stays still.

The board lies somewhere on the table (a **placement**) while the arms move.
Each view is one camera's detection of the board at one moment, with the pose
of the link the camera is fixed to (its **mount**) at that moment, in the
robot's ``root`` frame:

- a wrist camera's mount is the arm's last link (``<side>_w2``), whose pose
  comes from the joint readings and the URDF;
- the head camera's mount is ``root`` itself.

The unknowns are each camera's pose in its mount and each placement's pose in
``root``. One least-squares solve over every view finds them all by
minimising the reprojection error of the board corners, with a robust loss
and one round of rejecting views that disagree with the rest. A placement is
tied to the robot only through wrist views, so the head camera is solved from
placements a wrist camera also saw.

A camera passes when it has enough views, small reprojection error, enough
variety in how its mount turned (wrist cameras), and a result close to CAD
(a result far from CAD means a bad view or a loose mount, not a real offset).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .board import Board, Detection
from .optics import Lens

# A view whose error is this many times the median (or more than the floor)
# is dropped before the final solve.
_OUTLIER_FACTOR = 3.0
_OUTLIER_FLOOR_PX = 2.0
_HUBER_PX = 1.0


@dataclass(frozen=True)
class CameraSpec:
    """One camera to calibrate."""

    key: str  # "head", "left_wrist", "right_wrist"
    label: str  # "Head camera", for the operator
    frame: str  # its optical frame (the file's child)
    mount: str  # the link it is fixed to (the file's parent)
    cad: np.ndarray  # CAD pose of ``frame`` in ``mount`` (4x4)
    lens: Lens
    serial: int | None = None
    moves: bool = True  # False for a camera fixed to root (the head)


@dataclass
class View:
    """One camera's detection of the board at one moment."""

    camera: str  # CameraSpec.key
    placement: int
    mount_pose: np.ndarray  # the camera's mount in root (4x4)
    detection: Detection
    capture: int = -1


@dataclass
class Limits:
    """What a camera must reach to pass."""

    min_wrist_views: int = 10
    min_head_views: int = 3
    # Board positions the head camera saw (each also seen by a wrist camera).
    # A camera far from the board needs several to pin down sideways shift
    # against tilt.
    min_head_placements: int = 3
    max_rms_px: float = 3.0
    max_change_mm: float = 25.0
    max_change_deg: float = 6.0
    # How far the wrist must turn about its second most varied axis (degrees,
    # standard deviation). Turning about one axis only leaves the mount
    # undetermined along it.
    min_turn_deg: float = 7.0


@dataclass
class CameraResult:
    spec: CameraSpec
    transform: np.ndarray | None
    views: int = 0
    rejected: int = 0
    rms_px: float = float("nan")
    change_mm: float = float("nan")
    change_deg: float = float("nan")
    turn_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.transform is not None and not self.problems


@dataclass
class Solution:
    cameras: dict[str, CameraResult]
    placements: dict[int, np.ndarray]
    # Per input view: its RMS error (px), NaN if it wasn't used.
    view_rms: list[float]

    @property
    def passed(self) -> bool:
        return bool(self.cameras) and all(c.passed for c in self.cameras.values())


def _exp(rotvec: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(rotvec).as_matrix()


def rotation_spread(rotations: list[np.ndarray]) -> tuple[float, float, float]:
    """Standard deviation (degrees) of ``rotations`` about their three
    principal axes, largest first."""
    from scipy.spatial.transform import Rotation

    if len(rotations) < 2:
        return (0.0, 0.0, 0.0)
    rot = Rotation.from_matrix(np.asarray(rotations))
    mean = rot.mean()
    vecs = (mean.inv() * rot).as_rotvec()
    cov = vecs.T @ vecs / len(vecs)
    values = np.sqrt(np.clip(np.linalg.eigvalsh(cov), 0.0, None))
    out = np.degrees(np.sort(values)[::-1])
    return (float(out[0]), float(out[1]), float(out[2]))


def _average_pose(poses: list[np.ndarray]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    out = np.eye(4)
    out[:3, :3] = (
        Rotation.from_matrix(np.asarray([p[:3, :3] for p in poses])).mean().as_matrix()
    )
    out[:3, 3] = np.mean([p[:3, 3] for p in poses], axis=0)
    return out


def _difference(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Translation (mm) and rotation (degrees) between two poses."""
    dt = float(np.linalg.norm(a[:3, 3] - b[:3, 3])) * 1000.0
    r = a[:3, :3].T @ b[:3, :3]
    angle = np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    return dt, float(np.degrees(angle))


class _Problem:
    """The least-squares problem over the cameras and placements in use."""

    def __init__(
        self,
        specs: dict[str, CameraSpec],
        views: list[View],
        board: Board,
        cameras: dict[str, np.ndarray],
        placements: dict[int, np.ndarray],
    ) -> None:
        self.specs = specs
        self.views = views
        self.cam_keys = sorted(cameras)
        self.place_keys = sorted(placements)
        self.cam0 = cameras
        self.place0 = placements
        self.points = [board.points[v.detection.ids] for v in views]

    def n_params(self) -> int:
        return 6 * (len(self.cam_keys) + len(self.place_keys))

    def unpack(
        self, x: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray]]:
        def pose(base: np.ndarray, block: np.ndarray) -> np.ndarray:
            out = base.copy()
            out[:3, :3] = base[:3, :3] @ _exp(block[:3])
            out[:3, 3] = base[:3, 3] + block[3:]
            return out

        cams = {
            key: pose(self.cam0[key], x[6 * i : 6 * i + 6])
            for i, key in enumerate(self.cam_keys)
        }
        n = len(self.cam_keys)
        places = {
            key: pose(self.place0[key], x[6 * (n + j) : 6 * (n + j) + 6])
            for j, key in enumerate(self.place_keys)
        }
        return cams, places

    def view_residuals(
        self, cams: dict[str, np.ndarray], places: dict[int, np.ndarray], i: int
    ) -> np.ndarray:
        view = self.views[i]
        root_cam = view.mount_pose @ cams[view.camera]
        cam_board = np.linalg.inv(root_cam) @ places[view.placement]
        pts = self.points[i] @ cam_board[:3, :3].T + cam_board[:3, 3]
        projected = self.specs[view.camera].lens.project(pts)
        return (projected - view.detection.pixels).reshape(-1)

    def residuals(self, x: np.ndarray) -> np.ndarray:
        cams, places = self.unpack(x)
        return np.concatenate(
            [self.view_residuals(cams, places, i) for i in range(len(self.views))]
        )

    def solve(self) -> np.ndarray:
        from scipy.optimize import least_squares

        result = least_squares(
            self.residuals,
            np.zeros(self.n_params()),
            loss="huber",
            f_scale=_HUBER_PX,
            x_scale="jac",
            method="trf",
            tr_solver="exact",
            max_nfev=100,
        )
        return result.x


def _initial(
    specs: dict[str, CameraSpec],
    views: list[View],
    board: Board,
    cameras: dict[str, np.ndarray],
) -> tuple[list[View], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Views with a board pose, each placement's first estimate, and each
    view's board pose in the camera (by ``id``).

    A placement is estimated from its wrist views (CAD mounts); views of a
    placement no wrist camera saw are dropped.
    """
    estimates: dict[int, list[np.ndarray]] = {}
    posed: list[View] = []
    board_in_camera: dict[int, np.ndarray] = {}
    for view in views:
        spec = specs.get(view.camera)
        if spec is None:
            continue
        pose = spec.lens.board_pose(
            board.points[view.detection.ids], view.detection.pixels
        )
        if pose is None:
            continue
        posed.append(view)
        board_in_camera[id(view)] = pose
        if spec.moves:
            estimates.setdefault(view.placement, []).append(
                view.mount_pose @ cameras[view.camera] @ pose
            )
    placements = {k: _average_pose(v) for k, v in estimates.items()}
    kept = [view for view in posed if view.placement in placements]
    return kept, placements, board_in_camera


def _refine(
    specs: dict[str, CameraSpec],
    views: list[View],
    board: Board,
    cameras: dict[str, np.ndarray],
    placements: dict[int, np.ndarray],
) -> tuple[_Problem, dict[str, np.ndarray], dict[int, np.ndarray]]:
    """One least-squares solve over ``views``; returns the solved poses."""
    problem = _Problem(
        specs,
        views,
        board,
        {k: cameras[k] for k in {v.camera for v in views}},
        {k: placements[k] for k in {v.placement for v in views}},
    )
    cams, places = problem.unpack(problem.solve())
    return problem, cams, places


def solve(
    specs: dict[str, CameraSpec],
    views: list[View],
    board: Board,
    limits: Limits | None = None,
) -> Solution:
    """Each camera's pose in its mount, with the pass/fail checks."""
    limits = limits or Limits()
    cameras = {key: spec.cad.copy() for key, spec in specs.items()}
    used, placements, board_in_camera = _initial(specs, views, board, cameras)
    rejected: list[View] = []
    view_rms = {id(v): float("nan") for v in views}

    # The wrist cameras and placements first; then each fixed camera starts
    # from its own view of those placements. A fixed camera far from the
    # board trades sideways shift against tilt almost freely, so it needs
    # this close start.
    wrist = [v for v in used if specs[v.camera].moves]
    if wrist:
        _problem, cams, places = _refine(specs, wrist, board, cameras, placements)
        cameras.update(cams)
        placements.update(places)
    for key, spec in specs.items():
        seen = [
            placements[v.placement] @ np.linalg.inv(board_in_camera[id(v)])
            for v in used
            if v.camera == key and not spec.moves
        ]
        if seen:
            cameras[key] = _average_pose(seen)

    for _round in range(2):
        if not used or not placements:
            break
        problem, cams, places = _refine(specs, used, board, cameras, placements)
        cameras.update(cams)
        placements.update(places)
        rms = []
        for i, _view in enumerate(used):
            r = problem.view_residuals(cams, places, i).reshape(-1, 2)
            rms.append(float(np.sqrt(np.mean(np.sum(r * r, axis=1)))))
        for view, value in zip(used, rms, strict=True):
            view_rms[id(view)] = value
        cutoff = max(_OUTLIER_FACTOR * float(np.median(rms)), _OUTLIER_FLOOR_PX)
        bad = [v for v, value in zip(used, rms, strict=True) if value > cutoff]
        if not bad or _round == 1:
            break
        rejected += bad
        for view in bad:
            view_rms[id(view)] = float("nan")
        dropped = {id(v) for v in bad}
        used = [v for v in used if id(v) not in dropped]

    results: dict[str, CameraResult] = {}
    for key, spec in specs.items():
        mine = [v for v in used if v.camera == key]
        result = CameraResult(
            spec=spec,
            transform=cameras[key] if mine else None,
            views=len(mine),
            rejected=sum(1 for v in rejected if v.camera == key),
        )
        need = limits.min_wrist_views if spec.moves else limits.min_head_views
        if not mine:
            result.problems.append(
                "no usable views of the board"
                if spec.moves
                else "no usable views of the board where a wrist camera also saw it"
            )
            results[key] = result
            continue
        values = [view_rms[id(v)] for v in mine]
        result.rms_px = float(np.sqrt(np.mean(np.square(values))))
        result.change_mm, result.change_deg = _difference(spec.cad, cameras[key])
        if len(mine) < need:
            result.problems.append(f"{len(mine)} views; capture at least {need}")
        places = len({v.placement for v in mine})
        if not spec.moves and places < limits.min_head_placements:
            result.problems.append(
                f"saw the board in {places} place(s); move it to at least "
                f"{limits.min_head_placements} places"
            )
        if result.rms_px > limits.max_rms_px:
            result.problems.append(
                f"reprojection error {result.rms_px:.2f} px is above "
                f"{limits.max_rms_px:g} px"
            )
        if (
            result.change_mm > limits.max_change_mm
            or result.change_deg > limits.max_change_deg
        ):
            result.problems.append(
                f"{result.change_mm:.0f} mm and {result.change_deg:.1f} degrees from "
                f"CAD, more than {limits.max_change_mm:g} mm or "
                f"{limits.max_change_deg:g} degrees. Check the camera mount is tight"
            )
        if spec.moves:
            result.turn_deg = rotation_spread([v.mount_pose[:3, :3] for v in mine])
            if result.turn_deg[1] < limits.min_turn_deg:
                result.problems.append(
                    "the wrist turned too little between views; tilt the camera "
                    "toward the board from more directions"
                )
        results[key] = result
    return Solution(
        cameras=results,
        placements=placements,
        view_rms=[view_rms[id(v)] for v in views],
    )


def reprojection(
    solution: Solution, spec: CameraSpec, view: View, board: Board
) -> np.ndarray | None:
    """Where the solved calibration puts the board corners in ``view`` (px)."""
    result = solution.cameras.get(spec.key)
    place = solution.placements.get(view.placement)
    if result is None or result.transform is None or place is None:
        return None
    root_cam = view.mount_pose @ result.transform
    cam_board = np.linalg.inv(root_cam) @ place
    pts = board.points[view.detection.ids] @ cam_board[:3, :3].T + cam_board[:3, 3]
    return spec.lens.project(pts)
