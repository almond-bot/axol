"""Public re-exports for almond_axol.kinematics.

``KinematicsSolver`` (the original pyroki Levenberg-Marquardt solver) is
imported lazily: the teleop IK path now goes through the pluggable backend
factory (:func:`create_backend`), and eagerly importing the solver would pull
JAX in even when a non-JAX backend is selected.
"""

from typing import TYPE_CHECKING, Any

from .backends import BACKEND_NAMES, create_backend
from .base import IKBackend
from .config import KinematicsConfig

if TYPE_CHECKING:
    from .solver import KinematicsSolver

__all__ = [
    "BACKEND_NAMES",
    "IKBackend",
    "KinematicsConfig",
    "KinematicsSolver",
    "create_backend",
]


def __getattr__(name: str) -> Any:
    if name == "KinematicsSolver":
        from .solver import KinematicsSolver

        return KinematicsSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
