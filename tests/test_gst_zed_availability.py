from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from almond_axol.video import gst_zed


def _fake_gst(factory: object) -> Mock:
    gst = Mock()
    gst.ElementFactory.find.return_value = factory
    return gst


class ZedGstElementAvailabilityTest(unittest.TestCase):
    """``zed_gst_available`` must answer the way ``parse_launch`` will behave."""

    def setUp(self) -> None:
        gst_zed._stale_plugin_warned.clear()  # noqa: SLF001

    def test_unregistered_element_is_unavailable(self) -> None:
        with patch.object(
            gst_zed, "_require_gst", return_value=(_fake_gst(None), None)
        ):
            self.assertFalse(gst_zed._element_available("zedxonesrc"))  # noqa: SLF001

    def test_registered_and_loadable_element_is_available(self) -> None:
        factory = Mock()
        factory.load.return_value = factory
        with patch.object(
            gst_zed, "_require_gst", return_value=(_fake_gst(factory), None)
        ):
            self.assertTrue(gst_zed._element_available("zedxonesrc"))  # noqa: SLF001
        factory.load.assert_called_once_with()

    def test_registered_but_unloadable_plugin_is_unavailable_with_fix(self) -> None:
        # The registry cache still lists zedxonesrc after an in-place ZED SDK
        # upgrade broke the plugin's symbols; loading the feature fails just as
        # parse_launch would ("no element zedxonesrc"), so we must fall back.
        factory = Mock()
        factory.load.return_value = None
        with (
            patch.object(
                gst_zed, "_require_gst", return_value=(_fake_gst(factory), None)
            ),
            self.assertLogs(gst_zed._logger, level="WARNING") as logs,  # noqa: SLF001
        ):
            self.assertFalse(gst_zed._element_available("zedxonesrc"))  # noqa: SLF001
            # Probed once per camera and fps attempt; the guidance is said once.
            self.assertFalse(gst_zed._element_available("zedxonesrc"))  # noqa: SLF001

        self.assertEqual(len(logs.output), 1)
        self.assertIn("axol gst.build-zed", logs.output[0])
        self.assertIn("ZED SDK upgrade", logs.output[0])

    def test_loader_exception_is_treated_as_unavailable(self) -> None:
        factory = Mock()
        factory.load.side_effect = RuntimeError("dlopen failed")
        with patch.object(
            gst_zed, "_require_gst", return_value=(_fake_gst(factory), None)
        ):
            self.assertFalse(gst_zed._element_available("zedsrc"))  # noqa: SLF001

    def test_available_gate_requires_loadable_plugin(self) -> None:
        factory = Mock()
        factory.load.return_value = None
        with (
            patch.object(gst_zed, "_gi_available", return_value=True),
            patch.object(gst_zed, "hw_h264_available", return_value=True),
            patch.object(
                gst_zed, "_require_gst", return_value=(_fake_gst(factory), None)
            ),
        ):
            self.assertFalse(gst_zed.zed_gst_available())
            self.assertFalse(gst_zed.zed_stereo_gst_available())


if __name__ == "__main__":
    unittest.main()
