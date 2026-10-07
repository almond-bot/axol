"""VRTeleopCore.block_engage: grips can't engage while collect-data saves."""

from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace

from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.core import VRTeleopCore


def _frame(left: bool, right: bool, release_id: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        l_lock=left,
        r_lock=right,
        l_grip=0.5,
        r_grip=0.5,
        lock_release_id=release_id,
    )


def _box_frame(left: bool, right: bool) -> SimpleNamespace:
    """A frame with centred sticks, as box mode's engage reads them."""
    frame = _frame(left, right)
    frame.l_stick_x = frame.l_stick_y = frame.r_stick_x = frame.r_stick_y = 0.0
    frame.l_stick_click = frame.r_stick_click = False
    return frame


SQUEEZE = _frame(True, True)
RELEASE = _frame(False, False)


class EngageBlockTest(unittest.TestCase):
    def _core(self, **config: object) -> tuple[VRTeleopCore, list[dict]]:
        sent: list[dict] = []
        core = VRTeleopCore(
            VRTeleopConfig(**{"gripper": "parcel", **config}),
            logging.getLogger(__name__),
            lambda _enabled: None,
            broadcast_json=sent.append,
        )
        return core, sent

    def test_squeeze_while_blocked_does_not_engage(self) -> None:
        core, _ = self._core()
        core.block_engage()
        core.update_engage(RELEASE)
        core.update_engage(SQUEEZE)
        self.assertFalse(core.teleop_enabled)

    def test_grip_held_through_unblock_needs_release_first(self) -> None:
        for hold in (False, True):
            with self.subTest(hold_to_engage=hold):
                core, _ = self._core(hold_to_engage=hold)
                core.block_engage()
                core.update_engage(SQUEEZE)
                core.unblock_engage()
                # Still squeezing from the saving window: no engage.
                core.update_engage(SQUEEZE)
                self.assertFalse(core.teleop_enabled)
                core.update_engage(RELEASE)
                core.update_engage(SQUEEZE)
                self.assertTrue(core.teleop_enabled)

    def test_release_during_block_does_not_satisfy_gate(self) -> None:
        # Only a release after the block lifts counts, so a held-down grip at
        # unblock can't engage the dead-man scheme on its first frame.
        core, _ = self._core(hold_to_engage=True)
        core.block_engage()
        core.update_engage(RELEASE)
        core.update_engage(SQUEEZE)
        core.unblock_engage()
        core.update_engage(SQUEEZE)
        self.assertFalse(core.teleop_enabled)

    def test_block_disengages_an_engaged_session(self) -> None:
        core, _ = self._core()
        core.update_engage(SQUEEZE)
        self.assertTrue(core.teleop_enabled)
        core.block_engage()
        core.update_engage(RELEASE)
        self.assertFalse(core.teleop_enabled)

    def test_box_mode_is_blocked_too(self) -> None:
        # Neither a box-mode switch's auto-engage nor a box-mode grip gets
        # past the block.
        core, _ = self._core()
        core.block_engage()
        core.set_box_mode(True)
        core.update_engage(_box_frame(False, False))
        self.assertTrue(core.box_mode)
        self.assertFalse(core.teleop_enabled)
        core.update_engage(_box_frame(False, True))
        self.assertFalse(core.teleop_enabled)
        # Unblocked: the right grip leads the pair again after a release.
        core.unblock_engage()
        core.update_engage(_box_frame(False, False))
        core.update_engage(_box_frame(False, True))
        self.assertTrue(core.teleop_enabled)
        self.assertEqual(core._box_leader, "right")

    def test_block_freezes_a_leading_pair(self) -> None:
        core, _ = self._core()
        core.set_box_mode(True)
        core.update_engage(_box_frame(False, False))
        self.assertTrue(core.teleop_enabled)
        core.block_engage()
        core.update_engage(_box_frame(False, False))
        self.assertFalse(core.teleop_enabled)
        self.assertIsNone(core._box_leader)

    def test_lock_release_still_acknowledged_while_blocked(self) -> None:
        core, sent = self._core()
        core.block_engage()
        core.update_engage(_frame(False, False, release_id=7))
        self.assertEqual(sent, [{"type": "lock_release", "value": 7}])


if __name__ == "__main__":
    unittest.main()
