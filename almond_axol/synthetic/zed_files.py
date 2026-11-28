"""Synthetic ZED factory calibration files, laid out like the real ones.

The values are typical of the Camera Kit's cameras (a ZED X Mini head
camera, ZED X One S fisheye wrist cameras), changed a little per serial so no
two files are the same. The serials are made up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HEAD_SERIAL = 59990001
LEFT_WRIST_SERIAL = 309990001
RIGHT_WRIST_SERIAL = 309990002


def _jitter(serial: int, scale: float, n: int) -> np.ndarray:
    return np.random.default_rng(serial).normal(0.0, scale, n)


def head_conf(serial: int = HEAD_SERIAL) -> str:
    """A ZED X Mini file: per-resolution intrinsics, per-eye distortion, stereo."""
    j = _jitter(serial, 1.0, 8)
    lines: list[str] = []
    for eye, dx in (("LEFT", 0.0), ("RIGHT", 1.0)):
        for name, scale in (("FHD1200", 1.0), ("FHD", 1.0), ("SVGA", 0.5)):
            cy_shift = -60.0 if name == "FHD" else 0.0
            lines += [
                f"[{eye}_CAM_{name}]",
                f"fx={(734.95 + j[0] + dx * 0.4) * scale:.6f}",
                f"fy={(734.77 + j[1] + dx * 0.4) * scale:.6f}",
                f"cx={(947.04 + j[2] + dx * 15.5) * scale:.6f}",
                f"cy={(607.01 + j[3] - dx * 8.7 + cy_shift) * scale:.6f}",
                "",
            ]
    for eye, d in (
        ("LEFT", (0.10937, 0.06317, -0.03925, 0.11709, 0.11350, -0.05260)),
        ("RIGHT", (0.11012, 0.06108, -0.03881, 0.11623, 0.11291, -0.05187)),
    ):
        lines += [f"[{eye}_DISTO]"]
        lines += [f"k{i + 1}={v * (1 + 0.01 * j[4]):.8f}" for i, v in enumerate(d)]
        lines += [f"p1={-0.00021 + 0.00001 * j[5]:.8f}", f"p2={-0.00016:.8f}", ""]
    lines += [
        "[STEREO]",
        f"Baseline={49.8 + 0.05 * j[6]:.6f}",
        "TY=-0.088",
        "TZ=0.142",
    ]
    for name in ("FHD", "SVGA", "FHD1200"):
        lines += [
            f"CV_{name}={0.0086 + 0.0002 * j[7]:.8f}",
            f"RX_{name}=0.0022",
            f"RZ_{name}=0.0010",
        ]
    lines += ["", "[MISC]", "Sensor_ID=2", ""]
    return "\n".join(lines)


def wrist_conf(serial: int) -> str:
    """A ZED X One S file: per-resolution intrinsics, one fisheye distortion."""
    j = _jitter(serial, 1.0, 6)
    lines: list[str] = []
    for name, scale in (("FHD1200", 1.0), ("FHD", 1.0), ("SVGA", 0.5)):
        cy_shift = -60.0 if name == "FHD" else 0.0
        lines += [
            f"[CAM_{name}]",
            f"fx={(456.2 + j[0]) * scale:.6f}",
            f"fy={(456.0 + j[0] * 0.98) * scale:.6f}",
            f"cx={(965.0 + 6 * j[1]) * scale:.6f}",
            f"cy={(597.0 + 6 * j[2] + cy_shift) * scale:.6f}",
            "",
        ]
    lines += [
        "[DISTO]",
        f"k1={0.0684528 * (1 + 0.02 * j[3]):.8f}",
        f"k2={0.00473935 * (1 + 0.02 * j[4]):.8f}",
        f"k3={-0.00222603 * (1 + 0.02 * j[5]):.8f}",
        "k4=-4.37426e-05",
        "k5=0",
        "k6=0",
        "p1=0",
        "p2=0",
        "",
        "[MISC]",
        "Sensor_ID=3",
        "",
    ]
    return "\n".join(lines)


def write_files(directory: Path) -> dict[str, int]:
    """Write the three cameras' files; returns ``{slot: serial}``."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"SN{HEAD_SERIAL}.conf").write_text(head_conf(HEAD_SERIAL))
    for serial in (LEFT_WRIST_SERIAL, RIGHT_WRIST_SERIAL):
        (directory / f"SN{serial}.conf").write_text(wrist_conf(serial))
    return {
        "overhead": HEAD_SERIAL,
        "left_arm": LEFT_WRIST_SERIAL,
        "right_arm": RIGHT_WRIST_SERIAL,
    }
