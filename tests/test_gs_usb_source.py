"""Guards on the vendored gs_usb kernel module source.

The module can only be compiled against the robot's kernel headers, so the
suite cannot build it. These checks pin the properties `axol can.driver`
relies on and the hardening that keeps a corrupt RX frame from panicking the
host (see almond_axol/cli/can/gs_usb/README.md, items 3-6).
"""

from __future__ import annotations

import re
import unittest

from almond_axol.cli.can import driver

_SOURCE = (driver._SRC_DIR / "gs_usb.c").read_text()


def _function_body(name: str) -> str:
    """The text of a top-level C function, from its signature to its closing brace."""
    match = re.search(rf"^static .*\b{re.escape(name)}\(", _SOURCE, re.MULTILINE)
    assert match, f"{name} not found"
    start = match.start()
    end = _SOURCE.index("\n}\n", start)
    return _SOURCE[start:end]


class VendoredModuleVersionTest(unittest.TestCase):
    def test_module_version_matches_installer_marker(self) -> None:
        # can.driver decides whether the installed module is current by
        # comparing modinfo's version to this marker; a source change that
        # bumps one without the other is silently never rolled out (or
        # rebuilt on every run).
        match = re.search(r'^MODULE_VERSION\("([^"]+)"\);', _SOURCE, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), driver._VENDORED_MODULE_VERSION)


class ReceiveCallbackHardeningTest(unittest.TestCase):
    body = _function_body("gs_usb_receive_bulk_callback")

    def test_rx_urb_is_reanchored_before_every_resubmit(self) -> None:
        # Without this, URBs that completed once escape usb_kill_anchored_urbs()
        # in gs_can_close(), and their buffers are freed underneath them.
        anchor = self.body.index("usb_anchor_urb(urb, &usbcan->rx_submitted);")
        submit = self.body.index("usb_submit_urb(urb, GFP_ATOMIC);")
        self.assertLess(anchor, submit)
        # ...and a URB whose resubmit failed must not stay anchored, or the
        # kill loop in gs_can_close() never terminates.
        self.assertIn("usb_unanchor_urb(urb);", self.body[submit:])

    def test_channel_is_validated_before_dereference(self) -> None:
        guard = self.body.index(
            "hf->channel >= GS_MAX_INTF || !usbcan->canch[hf->channel]"
        )
        deref = self.body.index("dev = usbcan->canch[hf->channel];")
        self.assertLess(guard, deref)
        # A bad channel drops the frame; it must not detach the interfaces.
        self.assertNotIn("goto device_detach", self.body)

    def test_short_transfers_are_dropped_before_parsing(self) -> None:
        runt = self.body.index("urb->actual_length < sizeof(*hf)")
        first_field_use = self.body.index("hf->channel")
        self.assertLess(runt, first_field_use)

    def test_frames_for_closed_channels_are_not_processed(self) -> None:
        running = self.body.index("if (!netif_running(netdev))")
        processing = self.body.index("if (hf->echo_id == -1)")
        self.assertLess(running, processing)


class OpenCloseOwnershipTest(unittest.TestCase):
    def test_rx_buffers_belong_to_the_device_not_a_channel(self) -> None:
        gs_can = _SOURCE[
            _SOURCE.index("struct gs_can {") : _SOURCE.index("struct gs_usb {")
        ]
        gs_usb = _SOURCE[_SOURCE.index("struct gs_usb {") :]
        gs_usb = gs_usb[: gs_usb.index("\n};")]
        self.assertNotIn("rxbuf", gs_can)
        self.assertIn("void *rxbuf[GS_MAX_RX_URBS];", gs_usb)

    def test_failed_open_unwinds_and_closes_candev(self) -> None:
        body = _function_body("gs_can_open")
        self.assertIn("out_kill_rx_urbs:", body)
        self.assertIn("close_candev(netdev);", body[body.index("out_kill_rx_urbs:") :])
        # No early return may skip the unwind once open_candev() succeeded.
        after_open = body[body.index("rc = open_candev(netdev);") :]
        returns = re.findall(r"^\s*return (.*);", after_open, re.MULTILINE)
        self.assertEqual(returns, ["rc", "0", "rc"])

    def test_parent_is_set_before_register_candev(self) -> None:
        body = _function_body("gs_make_candev")
        self.assertLess(
            body.index("dev->parent = parent;"),
            body.index("register_candev(dev->netdev);"),
        )
        self.assertNotIn("->parent = dev;", _function_body("gs_usb_probe"))


if __name__ == "__main__":
    unittest.main()
