from unittest.mock import Mock

import pytest

from almond_axol.cli.plan_dagger_control import disconnect_plan_robot


def test_clean_quit_parks_before_motor_disable():
    calls = []
    robot = Mock()
    robot.disconnect.side_effect = lambda: calls.append("disable")
    reset = Mock()
    reset.park.side_effect = lambda *a, **kw: calls.append("park") or True
    disconnect_plan_robot(
        robot, reset, park=True, torque_threshold=4.0, stopped=lambda: False
    )
    assert calls == ["park", "disable"]
    robot.disconnect_preserving_position.assert_not_called()


def test_abort_preserves_position_without_starting_another_move():
    robot, reset = Mock(), Mock()
    disconnect_plan_robot(
        robot, reset, park=False, torque_threshold=4.0, stopped=lambda: True
    )
    robot.disconnect_preserving_position.assert_called_once_with()
    robot.disconnect.assert_not_called()
    reset.park.assert_not_called()


@pytest.mark.parametrize(
    "result", [False, RuntimeError("contact"), KeyboardInterrupt()]
)
def test_failed_or_interrupted_park_never_removes_motor_support(result):
    robot, reset = Mock(), Mock()
    if isinstance(result, BaseException):
        reset.park.side_effect = result
    else:
        reset.park.return_value = result
    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        disconnect_plan_robot(
            robot, reset, park=True, torque_threshold=4.0, stopped=lambda: False
        )
    robot.disconnect_preserving_position.assert_called_once_with()
    robot.disconnect.assert_not_called()
