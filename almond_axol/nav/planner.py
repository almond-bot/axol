"""Plan collision-free paths on a map's occupancy grid.

The grid is resampled to :attr:`PlannerConfig.resolution`, obstacles are
grown by the robot's radius, and the cheapest 8-connected path is found with
SciPy's Dijkstra on a sparse graph (compiled code, so large maps plan in
well under a second). Cells near obstacles cost more, so paths keep to the
middle of corridors and doorways. The raw cell path is then shortened with
line-of-sight checks into a few straight segments.

Unknown cells are not driven through: a robot should only go where the map
has seen floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .mapping import FREE, OCCUPIED, OccupancyGrid


class NoPath(RuntimeError):
    """No collision-free path between the two points."""


@dataclass
class PlannerConfig:
    """Tuning for :class:`Planner`.

    Attributes:
        robot_radius: Radius of a circle around the base that must stay clear
                      of obstacles (m). Jelly is 0.6 x 0.6 m, so 0.42 m
                      reaches its corners.
        clearance:    Extra distance (m) beyond ``robot_radius`` the path
                      prefers to keep from obstacles.
        clearance_cost: How strongly the path avoids that extra distance.
        resolution:   Planning cell size (m).
        start_escape: Distance (m) around the start the robot may cross even
                      if it is closer to an obstacle than ``robot_radius``
                      (it is already there).
    """

    robot_radius: float = 0.45
    clearance: float = 0.30
    clearance_cost: float = 4.0
    resolution: float = 0.1
    start_escape: float = 0.4


_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class Planner:
    def __init__(
        self, grid: OccupancyGrid, config: PlannerConfig | None = None
    ) -> None:
        self.config = config or PlannerConfig()
        self.source = grid
        res = self.config.resolution
        k = max(1, int(round(res / grid.resolution)))
        h, w = grid.shape
        H, W = -(-h // k), -(-w // k)
        pad = np.zeros((H * k, W * k), np.uint8)
        pad[:h, :w] = grid.data
        blocks = pad.reshape(H, k, W, k)
        self.occupied = (blocks == OCCUPIED).any(axis=(1, 3))
        self.known_free = (blocks == FREE).sum(axis=(1, 3)) * 2 >= k * k
        self.res = grid.resolution * k
        self.origin = (grid.origin_x, grid.origin_y)
        self.shape = (H, W)
        self._extra = np.zeros_like(self.occupied)
        self._update()

    # -- layers ---------------------------------------------------------

    def _update(self) -> None:
        blocked = self.occupied | self._extra
        self.distance = ndimage.distance_transform_edt(~blocked) * self.res
        self.traversable = (
            self.known_free & ~blocked & (self.distance > self.config.robot_radius)
        )

    def set_obstacles(self, xy: np.ndarray | None) -> None:
        """Replace the temporary obstacles (map-frame points seen live)."""
        self._extra[:] = False
        if xy is not None and len(xy):
            r, c = self.cell(xy[:, 0], xy[:, 1])
            ok = (r >= 0) & (r < self.shape[0]) & (c >= 0) & (c < self.shape[1])
            self._extra[r[ok], c[ok]] = True
        self._update()

    # -- coordinates ------------------------------------------------------

    def cell(self, x, y):
        c = np.floor((np.asarray(x) - self.origin[0]) / self.res).astype(int)
        r = np.floor((np.asarray(y) - self.origin[1]) / self.res).astype(int)
        return r, c

    def world(self, r, c):
        return (
            self.origin[0] + (np.asarray(c) + 0.5) * self.res,
            self.origin[1] + (np.asarray(r) + 0.5) * self.res,
        )

    def _in(self, r: int, c: int) -> bool:
        return 0 <= r < self.shape[0] and 0 <= c < self.shape[1]

    def is_clear(self, x: float, y: float) -> bool:
        r, c = self.cell(x, y)
        return bool(self._in(int(r), int(c)) and self.traversable[r, c])

    def clearance_at(self, x: float, y: float) -> float:
        r, c = self.cell(x, y)
        return float(self.distance[r, c]) if self._in(int(r), int(c)) else 0.0

    # -- planning ---------------------------------------------------------

    def plan(
        self, start: tuple[float, float], goal: tuple[float, float]
    ) -> list[tuple[float, float]]:
        cfg = self.config
        sr, sc = (int(v) for v in self.cell(*start))
        gr, gc = (int(v) for v in self.cell(*goal))
        if not self._in(sr, sc):
            raise NoPath("The robot is outside the map.")
        if not self._in(gr, gc) or not self.traversable[gr, gc]:
            raise NoPath(
                "The goal is too close to an obstacle or outside the mapped area."
            )
        ok = self.traversable.copy()
        # Let the robot leave a tight spot it is already in.
        esc = int(math.ceil(cfg.start_escape / self.res))
        r0, r1 = max(0, sr - esc), min(self.shape[0], sr + esc + 1)
        c0, c1 = max(0, sc - esc), min(self.shape[1], sc + esc + 1)
        blocked = self.occupied | self._extra
        yy, xx = np.mgrid[r0:r1, c0:c1]
        near = (yy - sr) ** 2 + (xx - sc) ** 2 <= esc**2
        ok[r0:r1, c0:c1] |= (
            near & ~blocked[r0:r1, c0:c1] & (self.distance[r0:r1, c0:c1] > self.res)
        )

        H, W = self.shape
        idx = np.arange(H * W).reshape(H, W)
        extra = np.clip(
            (cfg.robot_radius + cfg.clearance - self.distance)
            / max(cfg.clearance, 1e-6),
            0,
            1,
        )
        node_cost = 1.0 + cfg.clearance_cost * extra
        rows, cols, weights = [], [], []
        for dr, dc in _NEIGHBOURS:
            a = ok[max(0, -dr) : H - max(0, dr), max(0, -dc) : W - max(0, dc)]
            b = ok[
                max(0, dr) : H + min(0, dr) or None, max(0, dc) : W + min(0, dc) or None
            ]
            both = a & b
            ia = idx[max(0, -dr) : H - max(0, dr), max(0, -dc) : W - max(0, dc)][both]
            ib = idx[
                max(0, dr) : H + min(0, dr) or None, max(0, dc) : W + min(0, dc) or None
            ][both]
            step = math.hypot(dr, dc) * self.res
            ca = node_cost.ravel()[ia]
            cb = node_cost.ravel()[ib]
            rows.append(ia)
            cols.append(ib)
            weights.append(step * 0.5 * (ca + cb))
        graph = csr_matrix(
            (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
            shape=(H * W, H * W),
        )
        dist, pred = dijkstra(graph, indices=idx[sr, sc], return_predecessors=True)
        target = idx[gr, gc]
        if not np.isfinite(dist[target]):
            raise NoPath("No clear path to the goal. Something blocks every way there.")
        cells = []
        node = target
        while node >= 0 and node != idx[sr, sc]:
            cells.append(divmod(int(node), W))
            node = pred[node]
        cells.append((sr, sc))
        cells.reverse()
        # Shortcuts must keep half the preferred clearance, so a shortened path
        # still keeps to the middle of a corridor.
        strict = ok & (self.distance > cfg.robot_radius + 0.5 * cfg.clearance)
        path = self._shortcut(cells, strict)
        pts = [tuple(map(float, self.world(r, c))) for r, c in path]
        pts[0] = (float(start[0]), float(start[1]))
        pts[-1] = (float(goal[0]), float(goal[1]))
        return pts

    def _line_ok(self, a: tuple[int, int], b: tuple[int, int], ok: np.ndarray) -> bool:
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
        rr = np.round(np.linspace(a[0], b[0], n)).astype(int)
        cc = np.round(np.linspace(a[1], b[1], n)).astype(int)
        return bool(ok[rr, cc].all())

    def _shortcut(
        self, cells: list[tuple[int, int]], ok: np.ndarray
    ) -> list[tuple[int, int]]:
        if len(cells) <= 2:
            return cells
        out = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not self._line_ok(cells[i], cells[j], ok):
                j -= 1
            out.append(cells[j])
            i = j
        return out


def path_length(path: list[tuple[float, float]]) -> float:
    return float(sum(math.dist(a, b) for a, b in zip(path, path[1:], strict=False)))
