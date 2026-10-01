"""quiet_noisy_loggers is a cap that follows the root level, never a floor."""

from __future__ import annotations

import logging
import unittest

from almond_axol.utils.logquiet import NOISY_LOGGERS, quiet_noisy_loggers


class QuietNoisyLoggersTest(unittest.TestCase):
    def setUp(self) -> None:
        self._root_level = logging.getLogger().level
        self._levels = {n: logging.getLogger(n).level for n in NOISY_LOGGERS}

    def tearDown(self) -> None:
        logging.getLogger().setLevel(self._root_level)
        for name, level in self._levels.items():
            logging.getLogger(name).setLevel(level)

    def test_debug_root_caps_the_noisy_packages_at_info(self) -> None:
        logging.getLogger().setLevel(logging.DEBUG)
        quiet_noisy_loggers()
        for name in NOISY_LOGGERS:
            self.assertEqual(logging.getLogger(name).getEffectiveLevel(), logging.INFO)
        # Children inherit the cap through the package root.
        self.assertFalse(
            logging.getLogger("aiortc.rtcrtpsender").isEnabledFor(logging.DEBUG)
        )
        self.assertTrue(
            logging.getLogger("aiortc.rtcrtpsender").isEnabledFor(logging.WARNING)
        )

    def test_stricter_root_is_not_undercut(self) -> None:
        # Bugbot on #304: pinning INFO under a WARNING root would let INFO
        # records from these packages *through* the stricter root.
        logging.getLogger().setLevel(logging.WARNING)
        quiet_noisy_loggers()
        for name in NOISY_LOGGERS:
            self.assertFalse(logging.getLogger(name).isEnabledFor(logging.INFO), name)
            self.assertTrue(logging.getLogger(name).isEnabledFor(logging.WARNING), name)

    def test_cap_follows_the_root_in_both_directions(self) -> None:
        # The serve panel sets the root per op and calls this each time: a
        # DEBUG op after a WARNING op must get INFO from these packages again,
        # and a WARNING op after a DEBUG op must not get their INFO lines.
        logging.getLogger().setLevel(logging.WARNING)
        quiet_noisy_loggers()
        self.assertEqual(
            logging.getLogger("aiortc").getEffectiveLevel(), logging.WARNING
        )
        logging.getLogger().setLevel(logging.DEBUG)
        quiet_noisy_loggers()
        self.assertEqual(logging.getLogger("aiortc").getEffectiveLevel(), logging.INFO)
        logging.getLogger().setLevel(logging.ERROR)
        quiet_noisy_loggers()
        self.assertEqual(logging.getLogger("aiortc").getEffectiveLevel(), logging.ERROR)


if __name__ == "__main__":
    unittest.main()
