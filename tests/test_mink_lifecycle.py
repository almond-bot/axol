"""Real spawned Mink reset worker with an ideal joint plant, without CAN."""

from __future__ import annotations

import sys
from unittest import mock

import numpy as np
import pytest

from almond_axol.kinematics.config import KinematicsConfig
from almond_axol.lerobot import rollout
from almond_axol.teleop.config import VRTeleopConfig

from .test_rollout_park import Clock, Plant


class LifecyclePlant(Plant):
    @property
    def positions(self):
        return self.parking_positions()


def test_selected_backend_is_copied_into_reset_worker_config():
    config = KinematicsConfig(backend="mink", max_joint_delta=0.03)
    controller = rollout.IKResetController(kinematics_config=config)
    config.backend = "jax"
    assert controller._kin_cfg.backend == "mink"
    assert controller._kin_cfg.max_joint_delta == 0.03


def test_reset_preserves_selected_mink_clearance_and_playback_rate():
    motion = VRTeleopConfig(mink_reset_collision_margin=0.015, frequency=60)
    controller = rollout.IKResetController(
        kinematics_config=KinematicsConfig(backend="mink"), vr_teleop_config=motion
    )
    motion.mink_reset_collision_margin = 0.02
    assert controller._vr_cfg.mink_reset_collision_margin == 0.015
    assert controller._vr_cfg.frequency == 60


def test_real_worker_returns_repeats_and_parks_without_hardware():
    """Exercise the real process protocol, planner, playback and settle checks."""
    controller = rollout.IKResetController(
        kinematics_config=KinematicsConfig(backend="mink")
    )
    plant = LifecyclePlant()
    stock = VRTeleopConfig()
    rest = np.r_[stock.rest_pose_left, stock.rest_pose_right]
    before = set(sys.modules)
    try:
        controller.start()
        assert controller.wait_ready(timeout=30)
        with mock.patch.object(rollout, "time", Clock()):
            plant.q[3] = 10.0
            with pytest.raises(RuntimeError, match="Reset trajectory refused"):
                controller.return_to_rest(plant)
            assert not plant.actions
            assert controller._proc.is_alive()
            plant.q[:] = 0.0
            assert controller.return_to_rest(plant)
            np.testing.assert_allclose(plant.q, rest, atol=0.05)
            # A subsequent scene-reset cycle must not retain the prior plan.
            plant.q[3] += 0.03
            assert controller.return_to_rest(plant)
            np.testing.assert_allclose(plant.q, rest, atol=0.05)
            assert controller.park(plant)
            np.testing.assert_allclose(plant.q, np.zeros(14), atol=0.05)
        assert plant.actions
        # No real-time sleeps are needed for the ideal plant, but process
        # cleanup uses the real clock and must leave no worker behind.
        process = controller._proc
    finally:
        controller.stop()
    assert not process.is_alive()
    assert not {name.split(".")[0] for name in set(sys.modules) - before} & {
        "jax",
        "jaxlib",
        "jaxlie",
        "jaxls",
        "pyroki",
    }


def test_invalid_backend_is_rejected_before_starting_worker():
    # Backend validation must occur before a child or hardware owner starts.
    config = KinematicsConfig(backend="missing")
    with mock.patch("multiprocessing.get_context") as spawn:
        try:
            rollout.IKResetController(kinematics_config=config)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid reset backend accepted")
        spawn.assert_not_called()
