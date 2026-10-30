"""Precompile the IK so a session starts without waiting for it.

The first IK start for a robot model (or after an update changes the solver)
compiles the solve with XLA: ~50 s on a Jetson, while the operator waits for
teleop to become ready. Later starts reuse it — the exported program
(:mod:`almond_axol.kinematics.aot`) and the persistent compilation cache.

:func:`warm_ik` does that first start ahead of time: it builds the IK worker
exactly as ``axol teleop`` / ``axol collect-data`` would on this robot — the
shared settings, the robot model (Jelly if attached, else the saved mount)
and gripper, the worker's XLA flags — and runs its startup (both solve graph
shapes, the startup trajectory), so every program a session will ask for is
already on disk. ``axol provision`` runs it last, so an update ends ready to
go; ``axol ik.warm`` runs it by hand (e.g. after pulling a dev checkout).
Nothing moves: no hardware is opened.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace

_logger = logging.getLogger(__name__)


def warm_ik(store: object | None = None) -> str:
    """Compile this robot's IK as a session would; returns what was built.

    Runs in the calling process, whose XLA backend must not have started yet
    (:func:`~almond_axol.teleop.worker.configure_ik_process` sets the worker's
    flags, which the compilation cache keys on).
    """
    from .worker import configure_ik_process

    configure_ik_process()

    import numpy as np

    from ..constants import AxolModel
    from ..kinematics.config import KinematicsConfig
    from ..robot.jelly import JellyConfig, detect_jelly
    from ..settings import (
        load_store,
        resolve_gripper,
        resolve_robot_model,
        shared_axol_config,
        shared_config,
    )
    from .config import VRTeleopConfig, adopt_robot_gripper
    from .worker import IKWorker

    if store is None:
        store = load_store()
    config = shared_config(VRTeleopConfig, "teleop", "teleop", store=store)
    kinematics = shared_config(KinematicsConfig, "teleop", "kinematics", store=store)
    if kinematics.backend != "jax":
        # Mink solves a QP per call: there is nothing to compile ahead.
        return f"IK backend {kinematics.backend}: nothing to precompile"
    adopt_robot_gripper(config, shared_axol_config(store).gripper)
    # The model a session on this robot resolves (VRTeleop): Jelly when it is
    # attached and on, else the saved mount, unless one is forced.
    jelly = detect_jelly(shared_config(JellyConfig, "teleop", "jelly", store=store))
    model = resolve_robot_model(
        kinematics.robot_model,
        jelly_enabled=True if jelly is not None else None,
        mount=kinematics.mount,
    )
    kinematics = replace(kinematics, robot_model=model.value)
    if model is AxolModel.MOBILE:
        kinematics = replace(
            kinematics, gripper=resolve_gripper(kinematics.gripper or config.gripper)
        )

    started = time.perf_counter()
    worker = IKWorker(config, kinematics)
    q_rest = worker.get_rest_q()
    worker.compute_path_trajectory(np.zeros_like(q_rest), ("zero", "rest"))
    gripper = f", {kinematics.gripper} gripper" if kinematics.gripper else ""
    return (
        f"IK ready for {model.value}{gripper} ({time.perf_counter() - started:.0f} s)"
    )


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``ik.warm`` subcommand."""
    parser = subparsers.add_parser(
        "ik.warm",
        help=(
            "precompile this robot's IK (its model and gripper, as teleop "
            "would run it) so the next session starts without the ~1 min "
            "first-start compile; moves nothing"
        ),
    )
    parser.set_defaults(func=run)


def run(args: object = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(warm_ik(), flush=True)
