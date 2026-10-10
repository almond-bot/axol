"""``axol teleop`` stores the tracking state for headsets that join late.

The core's engage toggle reaches the headset through
``VRServer.broadcast_tracking``, which remembers the state so a headset that
connects mid-session is told whether the arms are engaged — even when the
toggle happened with no headset connected.
"""

from __future__ import annotations

import asyncio
import threading
import unittest

from almond_axol.teleop.config import VRTeleopConfig
from almond_axol.teleop.teleop import VRTeleop
from almond_axol.vr.config import VRServerConfig


class _Robot:
    left = None
    right = None


class TrackingAnnounceTest(unittest.TestCase):
    def test_tracking_is_stored_with_no_headset_connected(self) -> None:
        teleop = VRTeleop(
            _Robot(),
            config=VRTeleopConfig(),
            kinematics_config=object(),
            vr_server_config=VRServerConfig(),
        )
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            teleop._vr_loop = loop
            self.assertFalse(teleop._vr_server.connected)
            teleop._broadcast_tracking(True)
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(2)
            self.assertIs(teleop._vr_server._tracking, True)
            teleop._broadcast_tracking(False)
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(2)
            self.assertIs(teleop._vr_server._tracking, False)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(2)
            loop.close()


if __name__ == "__main__":
    unittest.main()
