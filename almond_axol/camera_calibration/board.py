"""The ChArUco calibration board: its geometry, detection, and images.

A ChArUco board is a chessboard with an ArUco marker in every white square.
The markers name each chessboard corner, so a partly visible board still
gives corners with known positions on the board.

The default is a bought board: 12 x 9 squares of 30 mm with 22 mm markers
from the 5 x 5 ArUco dictionary, made on a rigid flat plate. Calibrate with
a bought board, not a printed one: paper bends and printers scale. Give the
board's ID (printed on it, or one you assign) so each calibration records
which board it used.

Board coordinates are metres, with the origin at the chessboard's top-left
outer corner (as the board's own image shows it), x along the long side, y
down the image and z into the board.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MM_PER_INCH = 25.4
# Printable pages (mm, landscape).
PAPER_MM = {"letter": (279.4, 215.9), "a4": (297.0, 210.0), "a3": (420.0, 297.0)}
# Fewer corners than this on a board is too little to estimate a pose from.
MIN_CORNERS = 6
# Space above a printed board (mm).
_TOP_MM = 6.0


@dataclass(frozen=True)
class Detection:
    """The board's chessboard corners found in one image."""

    ids: np.ndarray  # (N,) corner ids
    pixels: np.ndarray  # (N, 2) image positions (px)
    markers: int = 0  # ArUco markers identified

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class BoardConfig:
    """The ChArUco board."""

    # Squares along the long side and the short side.
    squares_x: int = 12
    squares_y: int = 9
    square_mm: float = 30.0
    marker_mm: float = 22.0
    dictionary: str = "DICT_5X5_100"
    # The board's ID, recorded in every calibration made with it.
    id: str = ""
    # Paper for --print_board (letter, a4 or a3).
    paper: str = "a3"


class Board:
    """A :class:`BoardConfig` made into OpenCV's board and detector."""

    def __init__(self, config: BoardConfig | None = None) -> None:
        import cv2

        self.config = config or BoardConfig()
        c = self.config
        if c.squares_x < 3 or c.squares_y < 3:
            raise ValueError("the board needs at least 3 x 3 squares")
        if not 0.3 * c.square_mm <= c.marker_mm <= 0.9 * c.square_mm:
            raise ValueError("marker_mm must be between 30% and 90% of square_mm")
        name = c.dictionary.upper()
        if not hasattr(cv2.aruco, name):
            raise ValueError(f"unknown ArUco dictionary {c.dictionary!r}")
        self.square_m = c.square_mm / 1000.0
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        markers = (c.squares_x * c.squares_y) // 2
        if markers > dictionary.bytesList.shape[0]:
            raise ValueError(
                f"{c.dictionary} has too few markers for a {c.squares_x} x "
                f"{c.squares_y} board"
            )
        self.cv_board = cv2.aruco.CharucoBoard(
            (c.squares_x, c.squares_y),
            self.square_m,
            c.marker_mm / 1000.0,
            dictionary,
        )
        self.points = np.asarray(
            self.cv_board.getChessboardCorners(), dtype=np.float64
        ).reshape(-1, 3)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.CharucoDetector(
            self.cv_board, cv2.aruco.CharucoParameters(), params
        )

    @property
    def size_m(self) -> tuple[float, float]:
        c = self.config
        return c.squares_x * self.square_m, c.squares_y * self.square_m

    @property
    def corners_x(self) -> int:
        """Inner chessboard corners along x (a corner id's column count)."""
        return self.config.squares_x - 1

    def describe(self) -> str:
        c = self.config
        name = f"board {c.id}: " if c.id else ""
        return (
            f"{name}{c.squares_x} x {c.squares_y} squares of {c.square_mm:g} mm, "
            f"{c.marker_mm:g} mm {c.dictionary} markers"
        )

    def detect(self, image: np.ndarray) -> Detection | None:
        """The board's corners in a grey or colour image, or ``None``."""
        import cv2

        gray = image
        if image.ndim == 3:
            code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(image, code)
        corners, ids, _markers, marker_ids = self._detector.detectBoard(gray)
        if ids is None or corners is None or len(ids) < MIN_CORNERS:
            return None
        return Detection(
            ids=np.asarray(ids, dtype=np.int64).reshape(-1),
            pixels=np.asarray(corners, dtype=np.float64).reshape(-1, 2),
            markers=0 if marker_ids is None else len(marker_ids),
        )

    def usable(
        self, detection: Detection | None, min_markers: int = 4, min_corners: int = 12
    ) -> bool:
        """At least ``min_markers`` markers and ``min_corners`` corners that
        don't all lie on one line of the board."""
        if detection is None:
            return False
        if detection.markers < min_markers or len(detection) < min_corners:
            return False
        cols = detection.ids % self.corners_x
        rows = detection.ids // self.corners_x
        # Not collinear: some corner off the line through any two others.
        pts = np.stack([cols, rows], axis=1).astype(float)
        centred = pts - pts.mean(axis=0)
        return bool(np.linalg.svd(centred, compute_uv=False)[-1] > 0.5)

    # -- images --------------------------------------------------------------

    def image(
        self, px_per_square: int = 120, margin_squares: float = 0.5
    ) -> np.ndarray:
        """The board as a grey image with a white margin, for rendering."""
        c = self.config
        bw, bh = c.squares_x * px_per_square, c.squares_y * px_per_square
        board = self.cv_board.generateImage((bw, bh), marginSize=0, borderBits=1)
        m = round(margin_squares * px_per_square)
        out = np.full((bh + 2 * m, bw + 2 * m), 255, dtype=np.uint8)
        out[m : m + bh, m : m + bw] = board
        return out

    def render(self, dpi: int = 600) -> np.ndarray:
        """The whole printable page (grey, white margins) at ``dpi``."""
        c = self.config
        if c.paper not in PAPER_MM:
            raise ValueError(f"paper must be one of {', '.join(PAPER_MM)}")
        page_w_mm, page_h_mm = PAPER_MM[c.paper]
        board_w_mm, board_h_mm = (
            c.squares_x * c.square_mm,
            c.squares_y * c.square_mm,
        )
        # Room for the margins and the instruction line under the board.
        if board_w_mm > page_w_mm - 20 or board_h_mm > page_h_mm - 20:
            raise ValueError(
                f"a {board_w_mm:g} x {board_h_mm:g} mm board does not fit on "
                f"{c.paper} paper"
            )
        px = dpi / MM_PER_INCH
        page = np.full(
            (round(page_h_mm * px), round(page_w_mm * px)), 255, dtype=np.uint8
        )
        bw, bh = round(board_w_mm * px), round(board_h_mm * px)
        board = self.cv_board.generateImage((bw, bh), marginSize=0, borderBits=1)
        x0 = (page.shape[1] - bw) // 2
        y0 = round(_TOP_MM * px)
        page[y0 : y0 + bh, x0 : x0 + bw] = board
        return page

    def save_page(self, path: Path, dpi: int = 600) -> Path:
        """Write the printable page (PDF, or PNG by suffix) at its real size."""
        from PIL import Image, ImageDraw, ImageFont

        c = self.config
        page = self.render(dpi)
        image = Image.fromarray(page)
        px = dpi / MM_PER_INCH
        text = (
            f"Axol camera calibration board: {self.describe()}. Print at 100% "
            f"(actual size), then check that one square measures {c.square_mm:g} mm."
        )
        font = ImageFont.load_default(size=round(3.2 * px))
        draw = ImageDraw.Draw(image)
        y = round((_TOP_MM + c.squares_y * c.square_mm + 4) * px)
        draw.text((round(15 * px), y), text, fill=0, font=font)
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".pdf":
            image.save(path, "PDF", resolution=float(dpi))
        else:
            image.save(path, dpi=(dpi, dpi))
        return path
