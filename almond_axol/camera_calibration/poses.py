"""Calibration views: generated camera poses, and how varied a set of views is.

Wrist cameras. The board lies still on the table; each arm moves its wrist
camera through generated poses around it: near and far, straight down and
tilted from every side, turned about the lens, and aimed so the board lands
in every part of the image. Each pose is checked with the arm's inverse
kinematics (reachable, away from joint limits, every link above the table)
before it is used.

Head camera. The board moves in front of the head (by hand, or the sim's
stand-in for a hand) through poses that put it in every part of the image,
near and far, tilted.

The view measures follow the calibration procedure: a pose is **distinct**
when it differs from every kept pose by at least 50 mm or 5 degrees (the
camera's pose relative to the board); a view is **tilted** when the board is
turned 15 degrees or more from facing the camera; **coverage** counts the
cells of a 4 x 4 grid over the image that board corners fell in.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

# The procedure's thresholds.
DISTINCT_MM = 50.0
DISTINCT_DEG = 5.0
TILT_DEG = 15.0
TILT_APART_DEG = 10.0
GRID = 4
# Outer band of the image that counts as its edge (fraction of each side).
EDGE_FRACTION = 0.2


def _rotation_angle(r: np.ndarray) -> float:
    return math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(r) - 1.0) / 2.0))))


def look_at(
    position: np.ndarray, target: np.ndarray, x_hint: np.ndarray, roll_deg: float = 0.0
) -> np.ndarray:
    """An optical frame (+z toward ``target``) at ``position``; its +x is
    ``x_hint`` made square to +z, then turned ``roll_deg`` about +z."""
    z = np.asarray(target, float) - np.asarray(position, float)
    z /= np.linalg.norm(z)
    x = np.asarray(x_hint, float) - np.dot(x_hint, z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    a = math.radians(roll_deg)
    x, y = math.cos(a) * x + math.sin(a) * y, -math.sin(a) * x + math.cos(a) * y
    out = np.eye(4)
    out[:3, 0], out[:3, 1], out[:3, 2] = x, y, z
    out[:3, 3] = position
    return out


def tilt(camera_in_board: np.ndarray) -> tuple[float, float, float]:
    """``(distance m, tilt deg, tilt direction deg)`` of a camera seeing the
    board: how far the board turns from facing the camera, and which way."""
    p = camera_in_board[:3, 3]
    # The board faces the camera when the camera looks along the board's +z.
    view = camera_in_board[:3, 2]
    angle = math.degrees(math.acos(min(1.0, max(-1.0, float(view[2])))))
    direction = math.degrees(math.atan2(float(view[1]), float(view[0])))
    return float(np.linalg.norm(p)), angle, direction


def distinct(poses: Sequence[np.ndarray]) -> list[int]:
    """Indices of ``poses`` (camera in board) kept by the distinct-pose rule,
    in order: each differs from every kept one by 50 mm or 5 degrees."""
    kept: list[int] = []
    for i, pose in enumerate(poses):
        ok = True
        for j in kept:
            other = poses[j]
            mm = 1000.0 * float(np.linalg.norm(pose[:3, 3] - other[:3, 3]))
            deg = _rotation_angle(other[:3, :3].T @ pose[:3, :3])
            if mm < DISTINCT_MM and deg < DISTINCT_DEG:
                ok = False
                break
        if ok:
            kept.append(i)
    return kept


def tilted_views(poses: Sequence[np.ndarray]) -> int:
    """How many views are tilted 15 degrees or more, with tilt directions
    at least 10 degrees apart (the most such views, chosen greedily)."""
    directions = []
    for pose in poses:
        _d, angle, direction = tilt(pose)
        if angle >= TILT_DEG:
            directions.append(direction)
    kept: list[float] = []
    for d in sorted(directions):
        if all(abs((d - k + 180.0) % 360.0 - 180.0) >= TILT_APART_DEG for k in kept):
            kept.append(d)
    return len(kept)


@dataclass
class Coverage:
    """Where in the image board corners were seen."""

    width: int
    height: int
    cells: set[tuple[int, int]] = field(default_factory=set)
    centre: bool = False
    edge: bool = False

    def add(self, pixels: np.ndarray) -> None:
        if len(pixels) == 0:
            return
        u = np.clip(pixels[:, 0] / self.width, 0.0, 0.999999)
        v = np.clip(pixels[:, 1] / self.height, 0.0, 0.999999)
        for cu, cv in zip((u * GRID).astype(int), (v * GRID).astype(int), strict=True):
            self.cells.add((int(cu), int(cv)))
        if np.any((np.abs(u - 0.5) < 0.1) & (np.abs(v - 0.5) < 0.1)):
            self.centre = True
        if np.any(
            (u < EDGE_FRACTION / 2)
            | (u > 1 - EDGE_FRACTION / 2)
            | (v < EDGE_FRACTION / 2)
            | (v > 1 - EDGE_FRACTION / 2)
        ):
            self.edge = True

    @property
    def quadrants(self) -> int:
        half = GRID // 2
        return len({(cu >= half, cv >= half) for cu, cv in self.cells})


def in_edge(pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    """Which pixels are in the outer 20% of the image."""
    u = pixels[:, 0] / width
    v = pixels[:, 1] / height
    m = EDGE_FRACTION
    return (u < m) | (u > 1 - m) | (v < m) | (v > 1 - m)


# ---------------------------------------------------------------------------
# Generating poses


@dataclass
class WristPose:
    """One generated wrist camera pose and the arm's joints for it."""

    joints: dict[str, float]
    camera: np.ndarray  # camera in root (from CAD)
    camera_in_board: np.ndarray
    pixels: np.ndarray  # predicted board corners in the image


def _project(
    lens: object, camera_in_board: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Board corners in the image, for those in front and inside it."""
    board_in_camera = np.linalg.inv(camera_in_board)
    pts = points @ board_in_camera[:3, :3].T + board_in_camera[:3, 3]
    px = lens.project(pts)  # type: ignore[attr-defined]
    w, h = lens.width, lens.height  # type: ignore[attr-defined]
    keep = (
        (pts[:, 2] > 0.05)
        & (px[:, 0] >= 0)
        & (px[:, 0] < w)
        & (px[:, 1] >= 0)
        & (px[:, 1] < h)
    )
    return px[keep]


def wrist_poses(
    kin: object,
    side: str,
    *,
    board_in_root: np.ndarray,
    board_size: tuple[float, float],
    points: np.ndarray,
    lens: object,
    camera_in_gripper: np.ndarray,
    table_z: float,
    seed: Mapping[str, float],
    count: int,
    rng: np.random.Generator,
    tries: int = 260,
    min_corners: int = 24,
) -> list[WristPose]:
    """``count`` reachable, varied wrist camera poses around the board.

    ``kin`` is an :class:`~almond_axol.synthetic.motion.ArmKinematics`;
    ``table_z`` the table top in ``root`` (every arm link stays above it).
    """
    width, height = board_size
    board_x = board_in_root[:3, 0]
    gripper_in_camera = np.linalg.inv(camera_in_gripper)
    tool = kin.tool  # type: ignore[attr-defined]
    names = kin.joints[side]  # type: ignore[attr-defined]
    lo, hi = kin.limits[side].T  # type: ignore[attr-defined]
    candidates: list[WristPose] = []
    last = dict(seed)
    for _ in range(tries):
        distance = rng.uniform(0.2, 0.42)
        angle = rng.choice([0.0, 0.0, 12.0, 20.0, 28.0, 36.0])
        azimuth = rng.uniform(0.0, 360.0)
        roll = rng.uniform(-45.0, 45.0)
        aim = (
            board_in_root
            @ np.array(
                [
                    width / 2 + rng.uniform(-0.3, 0.3) * width,
                    height / 2 + rng.uniform(-0.3, 0.3) * height,
                    0.0,
                    1.0,
                ]
            )
        )[:3]
        # Up from the board (its -z), turned by the tilt toward the azimuth.
        up = -board_in_root[:3, 2]
        a = math.radians(azimuth)
        side_dir = (
            math.cos(a) * board_in_root[:3, 0] + math.sin(a) * board_in_root[:3, 1]
        )
        t = math.radians(angle)
        direction = math.cos(t) * up + math.sin(t) * side_dir
        camera = look_at(aim + distance * direction, aim, board_x, roll)
        gripper = camera @ gripper_in_camera
        q, err = kin.solve(side, gripper @ tool, last, iterations=60)  # type: ignore[attr-defined]
        if err > 1e-3:
            q, err = kin.solve(side, gripper @ tool, seed, iterations=80)  # type: ignore[attr-defined]
            if err > 1e-3:
                continue
        values = np.array([q[n] for n in names])
        if np.any(values - lo < 0.08) or np.any(hi - values < 0.08):
            continue
        reached = kin.tcp(side, q) @ np.linalg.inv(tool) @ camera_in_gripper  # type: ignore[attr-defined]
        if _rotation_angle(reached[:3, :3].T @ camera[:3, :3]) > 0.5:
            continue
        if not _above_table(kin, side, q, table_z):
            continue
        camera_in_board = np.linalg.inv(board_in_root) @ camera
        pixels = _project(lens, camera_in_board, points)
        if len(pixels) < min_corners:
            continue
        last = q
        candidates.append(WristPose(q, camera, camera_in_board, pixels))
    return _select(candidates, count, lens)


def _above_table(
    kin: object, side: str, q: Mapping[str, float], table_z: float
) -> bool:
    tree = kin.tree  # type: ignore[attr-defined]
    base = kin.base  # type: ignore[attr-defined]
    for link in (f"{side}_w0", f"{side}_w1", f"{side}_w2", f"{side}_gripper"):
        if tree.has_link(link) and tree.fk(base, link, q)[2, 3] < table_z + 0.06:
            return False
    tcp = kin.tcp(side, q)  # type: ignore[attr-defined]
    return bool(tcp[2, 3] > table_z + 0.03)


def route(
    kin: object,
    side: str,
    q_from: np.ndarray,
    q_to: np.ndarray,
    *,
    table_z: float,
    via: np.ndarray,
    samples: int = 16,
) -> list[np.ndarray]:
    """Joint targets from ``q_from`` to ``q_to`` that keep the arm above the
    table: straight in joint space, or through ``via`` (a raised pose) when
    the straight move would bring a link too close to it."""
    names = kin.joints[side]  # type: ignore[attr-defined]
    for s in np.linspace(0.0, 1.0, samples):
        q = dict(zip(names, q_from + s * (q_to - q_from), strict=True))
        if not _above_table(kin, side, q, table_z - 0.02):
            return [np.asarray(via, float), np.asarray(q_to, float)]
    return [np.asarray(q_to, float)]


def _select(candidates: list[WristPose], count: int, lens: object) -> list[WristPose]:
    """Tilted views from many sides first, then views that cover new parts
    of the image, then the most different of the rest; in a short path."""
    chosen: list[WristPose] = []
    cov = Coverage(lens.width, lens.height)  # type: ignore[attr-defined]

    def is_distinct(pose: WristPose) -> bool:
        return (
            len(distinct([*(c.camera_in_board for c in chosen), pose.camera_in_board]))
            == len(chosen) + 1
        )

    tilted = sorted(
        (c for c in candidates if tilt(c.camera_in_board)[1] >= TILT_DEG + 3),
        key=lambda c: tilt(c.camera_in_board)[2],
    )
    directions: list[float] = []
    for c in tilted:
        d = tilt(c.camera_in_board)[2]
        if all(abs((d - k + 180.0) % 360.0 - 180.0) >= 40.0 for k in directions):
            if is_distinct(c):
                chosen.append(c)
                cov.add(c.pixels)
                directions.append(d)
        if len(chosen) >= min(6, count):
            break
    rest = [c for c in candidates if c not in chosen]
    while len(chosen) < count and rest:

        def gain(c: WristPose) -> tuple[int, float]:
            probe = Coverage(cov.width, cov.height, set(cov.cells))
            probe.add(c.pixels)
            spread = min(
                (
                    float(
                        np.linalg.norm(
                            c.camera_in_board[:3, 3] - o.camera_in_board[:3, 3]
                        )
                    )
                    for o in chosen
                ),
                default=1.0,
            )
            return len(probe.cells) - len(cov.cells), spread

        rest.sort(key=gain, reverse=True)
        pick = next((c for c in rest if is_distinct(c)), None)
        if pick is None:
            break
        chosen.append(pick)
        cov.add(pick.pixels)
        rest.remove(pick)
    return _short_path(chosen)


def _short_path(poses: list[WristPose]) -> list[WristPose]:
    if not poses:
        return poses
    names = sorted(poses[0].joints)
    left = list(poses)
    out = [left.pop(0)]
    while left:
        here = np.array([out[-1].joints[n] for n in names])
        i = min(
            range(len(left)),
            key=lambda k: float(
                np.abs(np.array([left[k].joints[n] for n in names]) - here).max()
            ),
        )
        out.append(left.pop(i))
    return out


def head_board_poses(
    lens: object,
    head_in_root: np.ndarray,
    *,
    board_size: tuple[float, float],
    points: np.ndarray,
    count: int,
    rng: np.random.Generator,
    min_z: float,
    min_x: float = 0.2,
) -> list[np.ndarray]:
    """Board poses (in root) in front of the head camera that put the board
    in every part of the image, near and far, some tilted."""
    from .optics import Lens

    assert isinstance(lens, Lens)
    width, height = board_size
    out: list[np.ndarray] = []
    seen: list[np.ndarray] = []
    cells = [(i, j) for j in range(GRID) for i in range(GRID)]
    tries = 0
    while len(out) < count and tries < count * 40:
        tries += 1
        i, j = cells[len(out) % len(cells)]
        u = (i + rng.uniform(0.2, 0.8)) / GRID * lens.width
        v = (j + rng.uniform(0.2, 0.8)) / GRID * lens.height
        ray = np.append(lens.normalized(np.array([[u, v]]))[0], 1.0)
        ray /= np.linalg.norm(ray)
        distance = rng.uniform(0.42, 0.8)
        centre_cam = ray * distance
        angle = rng.choice([0.0, 10.0, 20.0, 30.0])
        azimuth = rng.uniform(0.0, 360.0)
        roll = rng.uniform(-35.0, 35.0)
        # The board's +z points away from the camera (into the board).
        a, t = math.radians(azimuth), math.radians(angle)
        normal = math.cos(t) * ray + math.sin(t) * (
            math.cos(a) * np.cross(ray, [0.0, 1.0, 0.0])
            + math.sin(a) * np.cross(ray, [1.0, 0.0, 0.0])
        )
        normal /= np.linalg.norm(normal)
        frame = look_at(np.zeros(3), normal, np.array([1.0, 0.0, 0.0]), roll)
        board_in_camera = frame.copy()
        board_in_camera[:3, 3] = centre_cam - frame[:3, :3] @ np.array(
            [width / 2, height / 2, 0.0]
        )
        board = head_in_root @ board_in_camera
        corners = (
            board
            @ np.array(
                [
                    [0, 0, 0, 1],
                    [width, 0, 0, 1],
                    [0, height, 0, 1],
                    [width, height, 0, 1],
                ],
                float,
            ).T
        )
        if corners[2].min() < min_z or corners[0].min() < min_x:
            continue
        camera_in_board = np.linalg.inv(board_in_camera)
        pixels = _project(lens, camera_in_board, points)
        if len(pixels) < 24:
            continue
        if len(distinct([*seen, camera_in_board])) <= len(seen):
            continue
        seen.append(camera_in_board)
        out.append(board)
    return out


def views_summary(
    camera_in_board: Iterable[np.ndarray],
    pixels: Iterable[np.ndarray],
    width: int,
    height: int,
) -> dict[str, float]:
    """The procedure's counts for one camera's views."""
    poses = list(camera_in_board)
    cov = Coverage(width, height)
    for p in pixels:
        cov.add(p)
    distances = [tilt(p)[0] for p in poses]
    return {
        "views": len(poses),
        "distinct_poses": len(distinct(poses)),
        "tilted_views": tilted_views(poses),
        "coverage_cells": len(cov.cells),
        "quadrants": cov.quadrants,
        "centre": cov.centre,
        "edge": cov.edge,
        "nearest_m": min(distances) if distances else float("nan"),
        "farthest_m": max(distances) if distances else float("nan"),
    }
