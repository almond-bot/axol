from __future__ import annotations

import math

import pytest

from almond_axol.motor.config import (
    DAMIAO_PARAMS,
    MYACTUATOR_PARAMS,
    Access,
    DamiaoParam,
    MyActuatorParam,
)
from almond_axol.motor.damiao import _float_to_uint as dm_float_to_uint
from almond_axol.motor.damiao import _uint_to_float as dm_uint_to_float
from almond_axol.motor.firmware import FirmwareUpdater, _crc16
from almond_axol.motor.myactuator import _float_to_uint as ma_float_to_uint
from almond_axol.motor.myactuator import (
    _model_max_torque,
    _uint_to_float,
    decode_control_reply,
    force_position_frame,
    position_torque_ff_frame,
)


@pytest.mark.parametrize("encoder", [ma_float_to_uint, dm_float_to_uint])
def test_protocol_float_encoding_clamps_to_wire_range(encoder) -> None:  # type: ignore[no-untyped-def]
    assert encoder(-100.0, -1.0, 1.0, 12) == 0
    assert encoder(100.0, -1.0, 1.0, 12) == 4095
    mid = encoder(0.0, -1.0, 1.0, 12)
    assert mid in (2047, 2048)


def test_protocol_round_trip_is_within_quantization() -> None:
    encoded = ma_float_to_uint(3.25, -12.5, 12.5, 16)
    assert _uint_to_float(encoded, -12.5, 12.5, 16) == pytest.approx(3.25, abs=4e-4)
    assert dm_uint_to_float(encoded, -12.5, 12.5, 16) == pytest.approx(3.25, abs=4e-4)


def test_motor_parameter_tables_remain_distinct_and_typed() -> None:
    assert MyActuatorParam.OVER_VOLTAGE in MYACTUATOR_PARAMS
    assert DamiaoParam.TIMEOUT in DAMIAO_PARAMS
    assert DAMIAO_PARAMS[DamiaoParam.HW_VER].access is Access.READ_ONLY
    assert DAMIAO_PARAMS[DamiaoParam.TIMEOUT].scale == 0.05


def test_crc16_xmodem_known_vector_and_firmware_id_validation() -> None:
    assert _crc16(b"123456789") == 0x31C3
    with pytest.raises(ValueError, match="motor_id"):
        FirmwareUpdater(None, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="motor_id"):
        FirmwareUpdater(None, 0x20)  # type: ignore[arg-type]


def test_known_myactuator_torque_models() -> None:
    assert _model_max_torque("RMD-X6") > 0
    assert _model_max_torque(None) > 0


def test_force_position_frame_matches_the_vendor_example() -> None:
    # The manual's 0xA9 example: 60 % of rated current, 500 dps, 360.00 deg.
    assert force_position_frame(math.radians(360.0), math.radians(500.0), 60) == bytes(
        [0xA9, 0x3C, 0xF4, 0x01, 0xA0, 0x8C, 0x00, 0x00]
    )
    # Out-of-range inputs saturate into their fields instead of raising a
    # struct error mid-stream.
    over = force_position_frame(0.0, 1e6, 900)
    assert over[1] == 255
    assert over[2:4] == b"\xff\xff"
    with pytest.raises(ValueError):
        force_position_frame(math.nan, 1.0, 60)


def test_position_torque_ff_frame_shares_the_layout_and_clamps_the_int8() -> None:
    a9 = force_position_frame(1.25, 2.0, 60)
    tf = position_torque_ff_frame(1.25, 2.0, 60)
    # Same speed limit and same 0.01 deg/LSB target; only the command byte
    # and the meaning of byte 1 differ.
    assert (a9[0], tf[0]) == (0xA9, 0x73)
    assert a9[2:] == tf[2:]
    assert position_torque_ff_frame(0.0, 0.0, 400)[1] == 127
    assert position_torque_ff_frame(0.0, 0.0, -400)[1] == 0x80  # -128
    # Rounds half away from zero, matching the core's Rust port.
    assert position_torque_ff_frame(0.0, 0.0, 2.5)[1] == 3
    assert position_torque_ff_frame(0.0, 0.0, -2.5)[1] == (-3 & 0xFF)


def test_control_reply_decodes_coarsely_and_in_amps() -> None:
    # The manual's 0xA9 reply example: 50 degC, 1 A, 500 dps, 45 deg.
    pos, vel, current, temp = decode_control_reply(
        bytes([0xA9, 0x32, 0x64, 0x00, 0xF4, 0x01, 0x2D, 0x00])
    )
    assert pos == pytest.approx(math.radians(45.0))
    assert vel == pytest.approx(math.radians(500.0))
    assert current == pytest.approx(1.0)
    assert temp == 50.0
    # Position resolution is a whole degree — 45x coarser than the MIT
    # feedback frame, which is why the wire modes are experiments.
    nudged = decode_control_reply(
        bytes([0xA9, 0x32, 0x64, 0x00, 0xF4, 0x01, 0x2E, 0x00])
    )
    assert nudged[0] - pos == pytest.approx(math.radians(1.0))
    # Negative angles and currents are signed.
    neg = decode_control_reply(bytes([0xA9, 0x00, 0x9C, 0xFF, 0x00, 0x00, 0xFF, 0xFF]))
    assert neg[0] == pytest.approx(math.radians(-1.0))
    assert neg[2] == pytest.approx(-1.0)
