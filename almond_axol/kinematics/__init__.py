"""Kinematics API with optional backends loaded only when requested."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import KinematicsConfig
from .jax_cache import enable_persistent_compilation_cache

Pose = tuple[np.ndarray, np.ndarray]

if TYPE_CHECKING:
    from .path import PathPlanningError, plan_linear_segment, tip_poses
    from .solver import KinematicsSolver


def __getattr__(name: str) -> Any:
    modules = {
        "KinematicsSolver": ".solver",
        "PathPlanningError": ".path",
        "plan_linear_segment": ".path",
        "tip_poses": ".path",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "KinematicsConfig",
    "KinematicsSolver",
    "PathPlanningError",
    "Pose",
    "enable_persistent_compilation_cache",
    "plan_linear_segment",
    "tip_poses",
]
