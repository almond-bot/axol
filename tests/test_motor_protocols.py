from __future__ import annotations

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


def test_crc16_xmodem_known_vector_and_firmware_id_validation() -> None:
    assert _crc16(b"123456789") == 0x31C3
    with pytest.raises(ValueError, match="motor_id"):
        FirmwareUpdater(None, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="motor_id"):
        FirmwareUpdater(None, 0x20)  # type: ignore[arg-type]


def test_known_myactuator_torque_models() -> None:
    assert _model_max_torque("RMD-X6") > 0
    assert _model_max_torque(None) > 0
