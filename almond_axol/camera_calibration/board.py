"""The printed ChArUco board: its geometry, detection, and printable page.

A ChArUco board is a chessboard with an ArUco marker in every white square.
The markers name each chessboard corner, so a partly visible board still
gives corners with known positions on the board. The default board (6 x 4
squares of 40 mm) fits on a Letter or A4 page in landscape.

Board coordinates are metres, with the origin at one outer corner of the
chessboard and z = 0 on the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MM_PER_INCH = 25.4
# Printable pages (mm, landscape).
PAPER_MM = {"letter": (279.4, 215.9), "a4": (297.0, 210.0)}
# Fewer corners than this on a board is too little to estimate a pose from.
MIN_CORNERS = 6


@dataclass(frozen=True)
class Detection:
    """The board's chessboard corners found in one image."""

    ids: np.ndarray  # (N,) corner ids
    pixels: np.ndarray  # (N, 2) image positions (px)

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class BoardConfig:
    """The ChArUco board. Measure a printed square and set ``square_mm`` to it
    if the printer scaled the page."""

    squares_x: int = 6
    squares_y: int = 4
    square_mm: float = 40.0
    # The marker inside each white square, as a fraction of the square.
    marker_ratio: float = 0.75
    dictionary: str = "DICT_4X4_50"
    paper: str = "letter"


class Board:
    """A :class:`BoardConfig` made into OpenCV's board and detector."""

    def __init__(self, config: BoardConfig | None = None) -> None:
        import cv2

        self.config = config or BoardConfig()
        c = self.config
        if c.squares_x < 3 or c.squares_y < 3:
            raise ValueError("the board needs at least 3 x 3 squares")
        if not 0.3 <= c.marker_ratio <= 0.9:
            raise ValueError("marker_ratio must be between 0.3 and 0.9")
        name = c.dictionary.upper()
        if not hasattr(cv2.aruco, name):
            raise ValueError(f"unknown ArUco dictionary {c.dictionary!r}")
        self.square_m = c.square_mm / 1000.0
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        self.cv_board = cv2.aruco.CharucoBoard(
            (c.squares_x, c.squares_y),
            self.square_m,
            self.square_m * c.marker_ratio,
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

    def describe(self) -> str:
        c = self.config
        return (
            f"{c.squares_x} x {c.squares_y} squares of {c.square_mm:g} mm, "
            f"{c.dictionary} markers"
        )

    def detect(self, image: np.ndarray) -> Detection | None:
        """The board's corners in a grey or colour image, or ``None``."""
        import cv2

        gray = image
        if image.ndim == 3:
            code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(image, code)
        corners, ids, _markers, _marker_ids = self._detector.detectBoard(gray)
        if ids is None or corners is None or len(ids) < MIN_CORNERS:
            return None
        return Detection(
            ids=np.asarray(ids, dtype=np.int64).reshape(-1),
            pixels=np.asarray(corners, dtype=np.float64).reshape(-1, 2),
        )

    # -- printing ------------------------------------------------------------

    def render(self, dpi: int = 600) -> np.ndarray:
        """The whole page (grey, white margins) at ``dpi``."""
        c = self.config
        if c.paper not in PAPER_MM:
            raise ValueError(f"paper must be one of {', '.join(PAPER_MM)}")
        page_w_mm, page_h_mm = PAPER_MM[c.paper]
        board_w_mm, board_h_mm = (
            c.squares_x * c.square_mm,
            c.squares_y * c.square_mm,
        )
        # Room for the instruction line under the board.
        if board_w_mm > page_w_mm - 20 or board_h_mm > page_h_mm - 30:
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
        y0 = round(10 * px)
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
        y = round((10 + c.squares_y * c.square_mm + 6) * px)
        draw.text((round(15 * px), y), text, fill=0, font=font)
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".pdf":
            image.save(path, "PDF", resolution=float(dpi))
        else:
            image.save(path, dpi=(dpi, dpi))
        return path
