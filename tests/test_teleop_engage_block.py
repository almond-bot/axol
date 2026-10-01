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


SQUEEZE = _frame(True, True)
RELEASE = _frame(False, False)


class EngageBlockTest(unittest.TestCase):
    def _core(self, **config: object) -> tuple[VRTeleopCore, list[dict]]:
        sent: list[dict] = []
        core = VRTeleopCore(
            VRTeleopConfig(**config),
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

    def test_lock_release_still_acknowledged_while_blocked(self) -> None:
        core, sent = self._core()
        core.block_engage()
        core.update_engage(_frame(False, False, release_id=7))
        self.assertEqual(sent, [{"type": "lock_release", "value": 7}])


if __name__ == "__main__":
    unittest.main()
