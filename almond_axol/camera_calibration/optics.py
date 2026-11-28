"""Projection and board pose for the ZED camera models.

A :class:`Lens` wraps the factory intrinsics of one raw (unrectified) camera
image, as :func:`almond_axol.recording.mcap.zed_conf.camera_intrinsics` gives
them: ``plumb_bob`` or ``rational_polynomial`` for the head camera, the
``equidistant`` fisheye model for the wrist cameras. These are the same
intrinsics the MCAP recorder writes to ``camera_info``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Projected in front of the lens closer than this (m) counts as behind it.
_MIN_DEPTH = 1e-3
# Residual (px) for a point the camera could not see.
BEHIND_PX = 1000.0


@dataclass(frozen=True)
class Lens:
    width: int
    height: int
    k: np.ndarray  # 3x3
    d: np.ndarray  # distortion coefficients
    fisheye: bool

    @classmethod
    def from_intrinsics(cls, intrinsics: object) -> Lens:
        fx = float(intrinsics.fx)  # type: ignore[attr-defined]
        fy = float(intrinsics.fy)  # type: ignore[attr-defined]
        cx = float(intrinsics.cx)  # type: ignore[attr-defined]
        cy = float(intrinsics.cy)  # type: ignore[attr-defined]
        model = str(intrinsics.distortion_model)  # type: ignore[attr-defined]
        return cls(
            width=int(intrinsics.width),  # type: ignore[attr-defined]
            height=int(intrinsics.height),  # type: ignore[attr-defined]
            k=np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
            d=np.asarray(intrinsics.d, dtype=np.float64),  # type: ignore[attr-defined]
            fisheye=model == "equidistant",
        )

    def project(self, points: np.ndarray) -> np.ndarray:
        """Pixels of camera-frame points (N, 3); far off-image if behind."""
        import cv2

        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        out = np.full((len(points), 2), BEHIND_PX * 10, dtype=np.float64)
        ahead = points[:, 2] > _MIN_DEPTH
        if not ahead.any():
            return out
        zero = np.zeros(3)
        if self.fisheye:
            px, _ = cv2.fisheye.projectPoints(
                points[ahead].reshape(-1, 1, 3), zero, zero, self.k, self.d
            )
        else:
            px, _ = cv2.projectPoints(points[ahead], zero, zero, self.k, self.d)
        out[ahead] = px.reshape(-1, 2)
        return out

    def normalized(self, pixels: np.ndarray) -> np.ndarray:
        """Undistorted normalized coordinates (N, 2) of pixels."""
        import cv2

        pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        if self.fisheye:
            out = cv2.fisheye.undistortPoints(pts, self.k, self.d)
        else:
            out = cv2.undistortPoints(pts, self.k, self.d)
        return out.reshape(-1, 2)

    def board_pose(
        self, object_points: np.ndarray, pixels: np.ndarray
    ) -> np.ndarray | None:
        """The board's pose in the camera (4x4) from its corners, or ``None``."""
        import cv2

        if len(pixels) < 4:
            return None
        norm = self.normalized(pixels)
        obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        ok, rvec, tvec = cv2.solvePnP(
            obj, norm, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE
        )
        if not ok:
            return None
        ok, rvec, tvec = cv2.solvePnP(
            obj,
            norm,
            np.eye(3),
            None,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or float(tvec.reshape(-1)[2]) <= 0:
            return None
        out = np.eye(4)
        out[:3, :3] = cv2.Rodrigues(rvec)[0]
        out[:3, 3] = tvec.reshape(3)
        return out
