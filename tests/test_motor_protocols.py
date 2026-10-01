from __future__ import annotations

import pytest

from almond_axol.motor.config import (
    DAMIAO_PARAMS,
    MYACTUATOR_FW_V44,
    MYACTUATOR_PARAMS,
    PARAM_SWEEP_RANGE,
    Access,
    DamiaoParam,
    MyActuatorParam,
)
from almond_axol.motor.damiao import _float_to_uint as dm_float_to_uint
from almond_axol.motor.damiao import _uint_to_float as dm_uint_to_float
from almond_axol.motor.firmware import FirmwareUpdater, _crc16
from almond_axol.motor.myactuator import _float_to_uint as ma_float_to_uint
from almond_axol.motor.myactuator import _model_max_torque, _uint_to_float


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


def test_myactuator_indices_match_fleet_firmware() -> None:
    # GUI-verified 0xC0 indices for the fleet firmware (2025070202 / 2026042402),
    # cross-checked against a raw dump of all five left-arm motors. These are the
    # anchors most likely to regress if the table is ever "corrected" back to the
    # old (mis-indexed) layout.
    expected = {
        "OVER_VOLTAGE": 0x13,
        "LOW_VOLTAGE": 0x14,
        "STALL_TIME_LIMIT": 0x15,
        "MOTOR_POSITION_ZERO": 0x21,
        "MAX_CURRENT": 0x23,
        "STALL_CURRENT": 0x24,
        "MAX_SPEED": 0x27,
        "NOMINAL_SPEED": 0x28,
        "KT_OUT": 0x3E,
        "ENCODER2_ABNORMAL_VALUE": 0x3F,
        "ENCODER2_ABNORMAL_SPEED": 0x40,
        "MOTOR_NUMBER": 0x01,
        "FACTORY_TIME": 0x02,
        "ENABLE_2ND_ENCODER": 0x29,
        "SECOND_ENCODER_RESOLUTION": 0x3C,
        "ENABLE_ETHERCAT": 0x3B,
        "THERMISTOR": 0x3D,
        "MAX_TORQUE": 0x46,
    }
    for name, index in expected.items():
        assert MyActuatorParam[name] == index, name

    # Every named parameter is reachable by the raw sweep and carries a spec.
    for param in MyActuatorParam:
        assert param in PARAM_SWEEP_RANGE
        assert param in MYACTUATOR_PARAMS

    # The undervoltage suppression the driver applies on enable() targets the
    # real low-voltage threshold, in volts, with no scaling.
    lv = MYACTUATOR_PARAMS[MyActuatorParam.LOW_VOLTAGE]
    assert lv.access is Access.READ_WRITE and lv.unit == "V" and lv.scale == 1.0

    named = {int(p) for p in MyActuatorParam}
    # 0x32 ("Enable CAN Filter") is deliberately unnamed: its readback
    # contradicts the GUI on every fleet motor. 0x54/0x55 (the GUI's MIT KP/KD
    # ceilings) are unimplemented on every fleet firmware.
    assert not named & {0x32, 0x54, 0x55}

    # 0x46-0x4A only exist from firmware 2026042402 on; everything else is
    # implemented by both fleet firmware lines.
    gated = {int(p) for p, spec in MYACTUATOR_PARAMS.items() if spec.min_firmware}
    assert gated == {0x46, 0x47, 0x48, 0x49, 0x4A}
    assert {spec.min_firmware for spec in MYACTUATOR_PARAMS.values()} == {
        None,
        MYACTUATOR_FW_V44,
    }


def test_crc16_xmodem_known_vector_and_firmware_id_validation() -> None:
    assert _crc16(b"123456789") == 0x31C3
    with pytest.raises(ValueError, match="motor_id"):
        FirmwareUpdater(None, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="motor_id"):
        FirmwareUpdater(None, 0x20)  # type: ignore[arg-type]


def test_known_myactuator_torque_models() -> None:
    assert _model_max_torque("RMD-X6") > 0
    assert _model_max_torque(None) > 0
