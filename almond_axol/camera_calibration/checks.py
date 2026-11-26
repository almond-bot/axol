"""The calibration procedure's pass/fail checks.

Recording A is fitted; its values are locked; Recording B is checked with
them held fixed. The lens intrinsics and the head stereo transform are the
cameras' factory values (each camera is calibrated individually at the
factory); the fit finds where each camera sits on the robot.

On Recording B, every corner's error is the distance (native image pixels)
between where it was found and where the locked values put it:

- a view of the board lying on the table: the board's place on the table is
  found from B's wrist views with A's camera mounts held fixed, then every
  camera's view of it is predicted through the robot's joints and A's
  mounts (the head's right eye through the factory stereo transform);
- a view of the board held in front of the head: the board's pose is found
  from that one image with the factory lens, so the error tests the lens.

The head pair's cross-camera error predicts each eye's corners from the
other eye's view through the factory stereo transform. The repeatability
check fits Recording B on its own and compares it with A; A's values are
never replaced.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .board import Board
from .optics import Lens
from .poses import Coverage, distinct, in_edge, tilted_views
from .recording import Frame, Recording


@dataclass
class ProcedureLimits:
    """The procedure's limits (all on Recording B unless named A)."""

    median_px: float = 0.5
    p95_px: float = 2.0
    within_3px: float = 0.95
    edge_p95_px: float = 5.0
    cross_median_px: float = 1.5
    cross_p95_px: float = 5.0
    distinct_a: int = 20
    distinct_b: int = 10
    simultaneous_a: int = 10
    simultaneous_b: int = 10
    tilted: int = 4
    coverage_cells: int = 12
    repeat_deg: float = 0.5
    repeat_mm: float = 2.5
    repeat_baseline_pct: float = 2.0
    # A sanity bound on the fit (not in the procedure): far from CAD means a
    # loose mount or a bad view, not a real offset.
    max_cad_mm: float = 25.0
    max_cad_deg: float = 6.0


@dataclass
class Check:
    name: str
    camera: str
    value: str
    limit: str
    passed: bool | None  # None: reported, not judged

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.name,
            "camera": self.camera,
            "value": self.value,
            "limit": self.limit,
            "result": "report"
            if self.passed is None
            else ("pass" if self.passed else "fail"),
        }


@dataclass
class Outcome:
    checks: list[Check] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        judged = [c for c in self.checks if c.passed is not None]
        return bool(judged) and all(c.passed for c in judged)

    def add(
        self, name: str, camera: str, value: str, limit: str, passed: bool | None
    ) -> None:
        self.checks.append(Check(name, camera, value, limit, passed))


def _stats(errors: np.ndarray) -> dict[str, float]:
    if len(errors) == 0:
        return {
            "corners": 0,
            "median_px": math.nan,
            "p95_px": math.nan,
            "rms_px": math.nan,
            "within_3px": math.nan,
        }
    return {
        "corners": int(len(errors)),
        "median_px": float(np.median(errors)),
        "p95_px": float(np.percentile(errors, 95)),
        "rms_px": float(np.sqrt(np.mean(errors**2))),
        "within_3px": float(np.mean(errors <= 3.0)),
    }


def board_pose(lens: Lens, board: Board, frame: Frame) -> np.ndarray | None:
    return lens.board_pose(board.points[frame.detection.ids], frame.detection.pixels)


def _project(
    lens: Lens, camera_in_root: np.ndarray, board_in_root: np.ndarray, pts: np.ndarray
) -> np.ndarray:
    board_in_camera = np.linalg.inv(camera_in_root) @ board_in_root
    return lens.project(pts @ board_in_camera[:3, :3].T + board_in_camera[:3, 3])


def _average(poses: list[np.ndarray]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    out = np.eye(4)
    out[:3, :3] = (
        Rotation.from_matrix(np.array([p[:3, :3] for p in poses])).mean().as_matrix()
    )
    out[:3, 3] = np.median(np.array([p[:3, 3] for p in poses]), axis=0)
    return out


class Rig:
    """Camera poses in ``root`` from the joints and the locked values."""

    def __init__(
        self,
        tree: Any,
        mounts: Mapping[str, np.ndarray],
        stereo: np.ndarray,
    ) -> None:
        self.tree = tree
        self.mounts = dict(mounts)  # head, left_wrist, right_wrist (camera in mount)
        self.stereo = stereo

    def camera(self, key: str, joints: Mapping[str, float]) -> np.ndarray:
        if key == "head_left":
            return self.mounts["head"]
        if key == "head_right":
            return self.mounts["head"] @ self.stereo
        side = key.split("_", 1)[0]
        return self.tree.fk("root", f"{side}_w2", joints) @ self.mounts[key]


def placements(rig: Rig, recording: Recording, board: Board) -> dict[int, np.ndarray]:
    """Each table placement's board pose in root, from the wrist views with
    the locked mounts (refined by least squares over all their corners)."""
    from scipy.optimize import least_squares

    from .solve import _exp

    out: dict[int, np.ndarray] = {}
    for key in ("left_wrist", "right_wrist"):
        pass
    groups: dict[int, list[Frame]] = {}
    for key in ("left_wrist", "right_wrist"):
        for frame in recording.frames.get(key, []):
            if frame.phase == "table":
                groups.setdefault(frame.placement, []).append(frame)
    for placement, frames in groups.items():
        guesses = []
        for frame in frames:
            pose = board_pose(recording.lenses[frame.camera], board, frame)
            if pose is not None:
                guesses.append(rig.camera(frame.camera, frame.joints) @ pose)
        if not guesses:
            continue
        start = _average(guesses)

        def residuals(
            x: np.ndarray, frames: list[Frame] = frames, start: np.ndarray = start
        ) -> np.ndarray:
            pose = start.copy()
            pose[:3, :3] = start[:3, :3] @ _exp(x[:3])
            pose[:3, 3] = start[:3, 3] + x[3:]
            res = []
            for frame in frames:
                pts = board.points[frame.detection.ids]
                px = _project(
                    recording.lenses[frame.camera],
                    rig.camera(frame.camera, frame.joints),
                    pose,
                    pts,
                )
                res.append((px - frame.detection.pixels).reshape(-1))
            return np.concatenate(res)

        fit = least_squares(residuals, np.zeros(6), loss="huber", f_scale=1.0)
        pose = start.copy()
        pose[:3, :3] = start[:3, :3] @ _exp(fit.x[:3])
        pose[:3, 3] = start[:3, 3] + fit.x[3:]
        out[placement] = pose
    return out


def camera_errors(
    rig: Rig, recording: Recording, board: Board, places: Mapping[int, np.ndarray]
) -> dict[str, dict[str, Any]]:
    """Per camera: corner errors on this recording with the locked values."""
    out: dict[str, dict[str, Any]] = {}
    for key, frames in recording.frames.items():
        lens = recording.lenses.get(key)
        if lens is None:
            continue
        errors: list[np.ndarray] = []
        edge: list[np.ndarray] = []
        kinds = {"table": 0, "head": 0}
        for frame in frames:
            pts = board.points[frame.detection.ids]
            if frame.phase == "table":
                place = places.get(frame.placement)
                if place is None:
                    continue
                px = _project(lens, rig.camera(key, frame.joints), place, pts)
            else:
                pose = board_pose(lens, board, frame)
                if pose is None:
                    continue
                px = lens.project(pts @ pose[:3, :3].T + pose[:3, 3])
            e = np.linalg.norm(px - frame.detection.pixels, axis=1)
            errors.append(e)
            edge.append(e[in_edge(frame.detection.pixels, lens.width, lens.height)])
            kinds[frame.phase] += 1
        all_e = np.concatenate(errors) if errors else np.zeros(0)
        edge_e = np.concatenate(edge) if edge else np.zeros(0)
        stats = _stats(all_e)
        stats["edge_corners"] = int(len(edge_e))
        stats["edge_p95_px"] = (
            float(np.percentile(edge_e, 95)) if len(edge_e) else math.nan
        )
        stats["table_views"] = kinds["table"]
        stats["head_views"] = kinds["head"]
        out[key] = stats
    return out


def view_counts(recording: Recording, board: Board) -> dict[str, dict[str, Any]]:
    """Per camera: distinct poses, tilted views, image coverage, distances."""
    out: dict[str, dict[str, Any]] = {}
    for key, frames in recording.frames.items():
        lens = recording.lenses.get(key)
        if lens is None:
            continue
        poses = []
        cov = Coverage(lens.width, lens.height)
        for frame in frames:
            pose = board_pose(lens, board, frame)
            if pose is None:
                continue
            poses.append(np.linalg.inv(pose))
            cov.add(frame.detection.pixels)
        distances = [float(np.linalg.norm(p[:3, 3])) for p in poses]
        out[key] = {
            "usable_views": len(frames),
            "distinct_poses": len(distinct(poses)),
            "tilted_views": tilted_views(poses),
            "coverage_cells": len(cov.cells),
            "quadrants": cov.quadrants,
            "centre": cov.centre,
            "edge": cov.edge,
            "nearest_m": min(distances) if distances else math.nan,
            "farthest_m": max(distances) if distances else math.nan,
        }
    return out


def _pairs(recording: Recording) -> list[tuple[Frame, Frame]]:
    right = {f.stamp_ns: f for f in recording.frames.get("head_right", [])}
    return [
        (f, right[f.stamp_ns])
        for f in recording.frames.get("head_left", [])
        if f.stamp_ns in right
    ]


def cross_camera(
    recording: Recording, board: Board, stereo: np.ndarray
) -> dict[str, Any]:
    """Each head eye's corners predicted from the other eye's view, and the
    epipolar residual of the matched corners."""
    left_lens = recording.lenses["head_left"]
    right_lens = recording.lenses["head_right"]
    errors: list[np.ndarray] = []
    epipolar: list[np.ndarray] = []
    pairs = _pairs(recording)
    r, t = stereo[:3, :3], stereo[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    essential = r.T @ tx
    for left, right in pairs:
        for src, dst, src_lens, dst_lens, transform in (
            (left, right, left_lens, right_lens, np.linalg.inv(stereo)),
            (right, left, right_lens, left_lens, stereo),
        ):
            pose = board_pose(src_lens, board, src)
            if pose is None:
                continue
            common, _i, j = np.intersect1d(
                src.detection.ids, dst.detection.ids, return_indices=True
            )
            if len(common) == 0:
                continue
            in_dst = transform @ pose
            pts = board.points[common]
            px = dst_lens.project(pts @ in_dst[:3, :3].T + in_dst[:3, 3])
            errors.append(np.linalg.norm(px - dst.detection.pixels[j], axis=1))
        common, i, j = np.intersect1d(
            left.detection.ids, right.detection.ids, return_indices=True
        )
        if len(common):
            xl = np.c_[left_lens.normalized(left.detection.pixels[i]), np.ones(len(i))]
            xr = np.c_[
                right_lens.normalized(right.detection.pixels[j]), np.ones(len(j))
            ]
            lines = xl @ essential.T
            d = np.abs(np.sum(lines * xr, axis=1)) / np.linalg.norm(
                lines[:, :2], axis=1
            )
            epipolar.append(d * float(right_lens.k[0, 0]))
    e = np.concatenate(errors) if errors else np.zeros(0)
    ep = np.concatenate(epipolar) if epipolar else np.zeros(0)
    return {
        "simultaneous_views": len(pairs),
        "cross_median_px": float(np.median(e)) if len(e) else math.nan,
        "cross_p95_px": float(np.percentile(e, 95)) if len(e) else math.nan,
        "epipolar_median_px": float(np.median(ep)) if len(ep) else math.nan,
        "epipolar_p95_px": float(np.percentile(ep, 95)) if len(ep) else math.nan,
    }


def change(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """``(mm, degrees)`` between two poses."""
    r = a[:3, :3].T @ b[:3, :3]
    deg = math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(r) - 1) / 2))))
    return 1000.0 * float(np.linalg.norm(a[:3, 3] - b[:3, 3])), deg


def judge(
    limits: ProcedureLimits,
    *,
    errors_b: Mapping[str, Mapping[str, Any]],
    counts_a: Mapping[str, Mapping[str, Any]],
    counts_b: Mapping[str, Mapping[str, Any]],
    cross_a: Mapping[str, Any],
    cross_b: Mapping[str, Any],
    mounts_a: Mapping[str, np.ndarray],
    mounts_b: Mapping[str, np.ndarray],
    cad: Mapping[str, np.ndarray],
    cameras: Iterable[str],
) -> Outcome:
    out = Outcome()
    L = limits
    for key in cameras:
        e = errors_b.get(key, {})
        a = counts_a.get(key, {})
        b = counts_b.get(key, {})

        def num(x: Any) -> float:
            return float(x) if x is not None else math.nan

        med, p95, w3, edge, rms = (
            num(e.get(k))
            for k in ("median_px", "p95_px", "within_3px", "edge_p95_px", "rms_px")
        )
        out.add(
            "reprojection median",
            key,
            f"{med:.3f} px",
            f"<= {L.median_px} px",
            med <= L.median_px,
        )
        out.add(
            "reprojection p95",
            key,
            f"{p95:.3f} px",
            f"<= {L.p95_px} px",
            p95 <= L.p95_px,
        )
        out.add(
            "corners within 3 px",
            key,
            f"{100 * w3:.1f}%",
            f">= {100 * L.within_3px:g}%",
            w3 >= L.within_3px,
        )
        out.add(
            "edge-region p95",
            key,
            f"{edge:.3f} px ({e.get('edge_corners', 0)} corners)",
            f"<= {L.edge_p95_px} px",
            bool(e.get("edge_corners")) and edge <= L.edge_p95_px,
        )
        out.add("reprojection RMS", key, f"{rms:.3f} px", "report", None)
        da, db = int(a.get("distinct_poses", 0)), int(b.get("distinct_poses", 0))
        out.add(
            "distinct poses A / B",
            key,
            f"{da} / {db}",
            f">= {L.distinct_a} / >= {L.distinct_b}",
            da >= L.distinct_a and db >= L.distinct_b,
        )
        tilted = min(int(a.get("tilted_views", 0)), int(b.get("tilted_views", 0)))
        out.add(
            "tilted views",
            key,
            f"{a.get('tilted_views', 0)} / {b.get('tilted_views', 0)}",
            f">= {L.tilted} in each",
            tilted >= L.tilted,
        )
        cells = int(a.get("coverage_cells", 0))
        quads = int(a.get("quadrants", 0))
        out.add(
            "image coverage (A)",
            key,
            f"{cells} of 16 cells, {quads} quadrants",
            f">= {L.coverage_cells} cells, all 4 quadrants",
            cells >= L.coverage_cells and quads == 4,
        )
    for name, data in (("A", cross_a), ("B", cross_b)):
        n = int(data.get("simultaneous_views", 0))
        need = L.simultaneous_a if name == "A" else L.simultaneous_b
        out.add(
            f"simultaneous views ({name})", "head pair", str(n), f">= {need}", n >= need
        )
    cm, cp = (
        cross_b.get("cross_median_px", math.nan),
        cross_b.get("cross_p95_px", math.nan),
    )
    out.add(
        "cross-camera median",
        "head pair",
        f"{cm:.3f} px",
        f"<= {L.cross_median_px} px",
        cm <= L.cross_median_px,
    )
    out.add(
        "cross-camera p95",
        "head pair",
        f"{cp:.3f} px",
        f"<= {L.cross_p95_px} px",
        cp <= L.cross_p95_px,
    )
    out.add(
        "epipolar residual",
        "head pair",
        f"median {cross_b.get('epipolar_median_px', math.nan):.3f} px, p95 {cross_b.get('epipolar_p95_px', math.nan):.3f} px",
        "report",
        None,
    )
    for key in ("head", "left_wrist", "right_wrist"):
        if key not in mounts_a:
            continue
        label = "head pair" if key == "head" else key
        if key in mounts_b:
            mm, deg = change(mounts_a[key], mounts_b[key])
            out.add(
                "repeatability: rotation",
                label,
                f"{deg:.3f} deg",
                f"<= {L.repeat_deg} deg",
                deg <= L.repeat_deg,
            )
            out.add(
                "repeatability: translation",
                label,
                f"{mm:.2f} mm",
                f"<= {L.repeat_mm} mm",
                mm <= L.repeat_mm,
            )
        else:
            out.add(
                "repeatability",
                label,
                "B could not be fitted on its own",
                "fit B",
                False,
            )
        mm, deg = change(cad[key], mounts_a[key])
        out.add(
            "distance from CAD",
            label,
            f"{mm:.1f} mm, {deg:.2f} deg",
            f"<= {L.max_cad_mm} mm, {L.max_cad_deg} deg",
            mm <= L.max_cad_mm and deg <= L.max_cad_deg,
        )
    out.add(
        "repeatability: baseline",
        "head pair",
        "0.00% (factory value, not fitted)",
        f"<= {L.repeat_baseline_pct}%",
        True,
    )
    return out
