"""Camera intrinsics and the head stereo transform from ZED factory calibration.

The ZED SDK caches each camera's factory calibration in
``/usr/local/zed/settings/SN<serial>.conf`` (see :mod:`almond_axol.zed.calibration`).
It is an INI file with one section per eye and sensor resolution, for example::

    [LEFT_CAM_FHD1200]
    fx=733.4
    fy=733.1
    cx=969.2
    cy=592.8
    k1=...

    [STEREO]
    Baseline=50.02
    TY=0.012
    TZ=-0.09
    CV_FHD1200=0.0012
    RX_FHD1200=0.0009
    RZ_FHD1200=-0.0002

Reading the file needs no camera and no SDK, so the recorder can describe
every camera's calibration while the relay owns the cameras.

Distortion model. A pinhole calibration stores tangential terms (``p1``,
``p2``) next to the radial ones; a fisheye calibration stores the four
equidistant terms as ``k1`` to ``k4`` with no tangential terms. ROS calls those
``plumb_bob`` / ``rational_polynomial`` and ``equidistant``.

Stereo transform. ``Baseline``, ``TY`` and ``TZ`` (mm) place the right eye in
the left eye's optical frame; ``RX``, ``CV`` and ``RZ`` (rad) are the rotation
as a Rodrigues vector, the layout the SDK's ``stereo_transform`` reports.
"""

from __future__ import annotations

import configparser
import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

# Recorder resolution name -> (conf section suffix, width, height).
_CONF_RESOLUTIONS: dict[str, tuple[str, int, int]] = {
    "HD1200": ("FHD1200", 1920, 1200),
    "HD1080": ("FHD", 1920, 1080),
    "SVGA": ("SVGA", 960, 600),
}


@dataclass(frozen=True)
class Intrinsics:
    """One camera's calibration at one image size, in ROS terms."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    d: tuple[float, ...]

    @property
    def k(self) -> tuple[float, ...]:
        return (self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0)

    @property
    def p(self) -> tuple[float, ...]:
        return (
            self.fx,
            0.0,
            self.cx,
            0.0,
            0.0,
            self.fy,
            self.cy,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        )

    def scaled(self, width: int, height: int) -> Intrinsics:
        """The same lens at another image size (a pure resize of the frame).

        Distortion coefficients act on normalized image coordinates, so only
        the focal lengths and principal point change. The principal point
        scales about pixel centres, which keeps a resize by the VIC exact.
        """
        if (width, height) == (self.width, self.height):
            return self
        sx = width / self.width
        sy = height / self.height
        return replace(
            self,
            width=width,
            height=height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=(self.cx + 0.5) * sx - 0.5,
            cy=(self.cy + 0.5) * sy - 0.5,
        )


@dataclass(frozen=True)
class CalibrationFile:
    """A parsed ``SN<serial>.conf`` and where it came from."""

    serial: int
    path: Path
    sections: dict[str, dict[str, float]]
    sha256: str
    mtime: float

    def section(self, name: str) -> dict[str, float] | None:
        return self.sections.get(name.upper())


def parse_conf_text(text: str) -> dict[str, dict[str, float]]:
    """Sections of a ZED ``.conf`` file, upper-cased names, lower-cased keys.

    Values that are not numbers are skipped; nothing the recorder needs is
    text.
    """
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str.lower  # type: ignore[method-assign,assignment]
    parser.read_string(text)
    out: dict[str, dict[str, float]] = {}
    for name in parser.sections():
        values: dict[str, float] = {}
        for key, raw in parser.items(name):
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value):
                values[key] = value
        out[name.upper()] = values
    return out


def load_calibration(serial: int, directory: Path) -> CalibrationFile | None:
    """The cached factory calibration for ``serial``, or ``None`` if absent."""
    path = Path(directory) / f"SN{int(serial)}.conf"
    try:
        raw = path.read_bytes()
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return CalibrationFile(
        serial=int(serial),
        path=path,
        sections=parse_conf_text(raw.decode("utf-8", errors="replace")),
        sha256=hashlib.sha256(raw).hexdigest(),
        mtime=mtime,
    )


def _eye_section(
    calib: CalibrationFile, eye: str, resolution: str
) -> dict[str, float] | None:
    suffix = _CONF_RESOLUTIONS[resolution][0]
    names = [f"{eye.upper()}_CAM_{suffix}"]
    if eye == "left":
        # A mono camera may name its only section without an eye.
        names.append(f"CAM_{suffix}")
    for name in names:
        section = calib.section(name)
        if section is not None and {"fx", "fy", "cx", "cy"} <= section.keys():
            return section
    return None


def distortion(
    values: dict[str, float], *, fisheye: bool
) -> tuple[str, tuple[float, ...]]:
    """ROS distortion model and coefficients for one calibration section.

    ``fisheye`` says the lens is a fisheye (the Camera Kit's wrist cameras).
    It only decides the cases the file leaves open: a section with non-zero
    tangential terms is always a pinhole model.
    """

    def k(name: str) -> float:
        return float(values.get(name, 0.0))

    tangential = any(k(name) != 0.0 for name in ("p1", "p2"))
    high_order = any(k(name) != 0.0 for name in ("k5", "k6"))
    if not tangential and not high_order and (fisheye or "k4" in values):
        return "equidistant", (k("k1"), k("k2"), k("k3"), k("k4"))
    if high_order or k("k4") != 0.0:
        return "rational_polynomial", (
            k("k1"),
            k("k2"),
            k("p1"),
            k("p2"),
            k("k3"),
            k("k4"),
            k("k5"),
            k("k6"),
        )
    return "plumb_bob", (k("k1"), k("k2"), k("p1"), k("p2"), k("k3"))


def camera_intrinsics(
    calib: CalibrationFile,
    *,
    eye: str,
    capture_resolution: str,
    width: int,
    height: int,
    fisheye: bool,
) -> Intrinsics | None:
    """Calibration of the recorded image, or ``None`` if the file lacks it.

    ``capture_resolution`` is the sensor mode the camera ran in (the file has
    one section per mode). ``width`` x ``height`` is the recorded image, which
    can be a downscale of that mode.
    """
    if capture_resolution not in _CONF_RESOLUTIONS:
        return None
    section = _eye_section(calib, eye, capture_resolution)
    if section is None:
        return None
    _, cap_w, cap_h = _CONF_RESOLUTIONS[capture_resolution]
    model, coeffs = distortion(section, fisheye=fisheye)
    base = Intrinsics(
        width=cap_w,
        height=cap_h,
        fx=section["fx"],
        fy=section["fy"],
        cx=section["cx"],
        cy=section["cy"],
        distortion_model=model,
        d=coeffs,
    )
    return base.scaled(width, height)


def rodrigues(vector: np.ndarray) -> np.ndarray:
    """Rotation matrix for a rotation vector (axis times angle, rad)."""
    v = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    axis = v / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew


def stereo_transform(
    calib: CalibrationFile, capture_resolution: str
) -> np.ndarray | None:
    """The right eye's optical frame in the left eye's, as a 4x4 (metres).

    ``None`` when the file has no ``[STEREO]`` section or no baseline.
    """
    stereo = calib.section("STEREO")
    if not stereo or "baseline" not in stereo:
        return None
    suffix = _CONF_RESOLUTIONS.get(capture_resolution, ("",))[0].lower()

    def value(name: str) -> float:
        for key in (f"{name}_{suffix}", name):
            if key in stereo:
                return float(stereo[key])
        return 0.0

    transform = np.eye(4)
    transform[:3, :3] = rodrigues(np.array([value("rx"), value("cv"), value("rz")]))
    transform[:3, 3] = (
        np.array([stereo["baseline"], value("ty"), value("tz")], dtype=np.float64)
        / 1000.0
    )
    return transform
