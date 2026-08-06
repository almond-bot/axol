"""IK backend registry + factory.

``pink-qp`` won the 2026-07 offline solver bake-off (see ``bench.py``) and is
the default; the other candidates are kept for on-hardware A/B comparison
(``axol teleop --kinematics.backend=...``). Backends are imported lazily so
selecting one never pays the import cost (or requires the dependencies) of
the others — e.g. ``mink-qp`` works without JAX warm-up, and ``pyroki-lm``
works without mink/pink installed.
"""

from __future__ import annotations

from ..base import IKBackend
from ..config import KinematicsConfig

BACKEND_NAMES: tuple[str, ...] = (
    "pink-qp",
    "pyroki-lm",
    "pyroki-diff",
    "mink-qp",
    "dls",
)


def create_backend(config: KinematicsConfig, dt: float) -> IKBackend:
    """Instantiate the IK backend selected by ``config.backend``.

    Args:
        config: Solver parameters (including the ``backend`` selector).
        dt: Integration timestep in seconds — the interval between
            consecutive :meth:`IKBackend.ik` calls (the teleop IK rate).

    Returns:
        A ready (warmed-up) backend instance.
    """
    name = config.backend
    if name == "pink-qp":
        from .pink_qp import PinkBackend

        return PinkBackend(config, dt)
    if name == "pyroki-lm":
        from .pyroki_lm import PyrokiLMBackend

        return PyrokiLMBackend(config)
    if name == "pyroki-diff":
        from .pyroki_diff import PyrokiDiffBackend

        return PyrokiDiffBackend(config, dt)
    if name == "mink-qp":
        from .mink_qp import MinkBackend

        return MinkBackend(config, dt)
    if name == "dls":
        from .dls import DLSBackend

        return DLSBackend(config, dt)
    raise ValueError(
        f"Unknown IK backend {name!r}; expected one of {list(BACKEND_NAMES)}"
    )


__all__ = ["BACKEND_NAMES", "create_backend", "IKBackend"]
