from __future__ import annotations

import struct

import numpy as np
import pytest

from almond_axol.robot.cart import deadzone, mix
from almond_axol.robot.config import AxolConfig
from almond_axol.robot.lift import _decode_status
from almond_axol.teleop.worker import (
    _quat_xyzw_to_matrix,
    _scale_rotation_np,
    _vr_to_flu_np,
)


def test_cart_deadzone_rescales_and_clamps() -> None:
    assert deadzone(0.1, 0.15) == 0.0
    assert deadzone(0.15, 0.15) == 0.0
    assert deadzone(1.0, 0.15) == 1.0
    assert deadzone(-1.0, 0.15) == -1.0
    assert deadzone(0.575, 0.15) == pytest.approx(0.5)


def test_cart_mix_preserves_limits_and_symmetry() -> None:
    wheels = mix(vx=1.0, vy=1.0, wz=1.0, max_speed=20.0, turn_scale=1.0)
    assert len(wheels) == 4
    assert max(abs(v) for v in wheels) == pytest.approx(20.0)
    assert mix(0.0, 0.0, 0.0, 20.0, 1.0) == [0.0] * 4


def test_lift_status_decodes_wire_flags_and_unhomed_position() -> None:
    data = struct.pack("<HhBb", 0xFFFF, -120, 0b10101011, -3)
    status = _decode_status(data)

    assert status.position_permille is None
    assert status.height_percent is None
    assert status.velocity == -120
    assert status.homed and status.moving and status.stall_fault and status.at_upper
    assert status.jog


def test_robot_config_stiffness_is_applied_per_arm() -> None:
    compliant = AxolConfig(left_stiffness=0.0, right_stiffness=1.0).resolved()
    assert compliant.left.shoulder_1.kp < compliant.right.shoulder_1.kp
    with pytest.raises(ValueError):
        AxolConfig(left_stiffness=1.1).resolved()


def test_vr_coordinate_and_quaternion_helpers() -> None:
    position, rotation = _vr_to_flu_np(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(position, [3.0, 2.0, -1.0])
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-7)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    identity = _quat_xyzw_to_matrix(0.0, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(identity, np.eye(3), atol=1e-7)
    np.testing.assert_allclose(_scale_rotation_np(identity, 0.5), np.eye(3), atol=1e-7)
