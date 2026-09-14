"""The relay's ``gstshm+pyshm`` transport as seen from the control process.

The recorder keeps the encoded (gsth264) meta per source; the control
process gets a :class:`RawFrameReader` over the policy ring for the same
source instead of the dims-only stub the plain gstshm transport attaches.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from almond_axol.video import video_proc


class _Reader:
    def __init__(self, shm_name: str, width: int, height: int, fps: int, cond: object):
        self.shm_name = shm_name
        self.width = width
        self.height = height
        self.fps = fps
        self.cond = cond


def _relay() -> video_proc.VideoRelayProcess:
    relay = video_proc.VideoRelayProcess.__new__(video_proc.VideoRelayProcess)
    relay.raw_cameras = {}
    relay._raw_cond = object()
    return relay


class PolicyRingAttachTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("almond_axol.video.shm_frames.RawFrameReader", _Reader)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_gstshm_only_sources_get_stubs(self) -> None:
        relay = _relay()
        relay._attach_raw_readers(
            {"cam": video_proc._gsth264_meta("/tmp/cam.sock", 960, 600, 60, 60, 1.0)}
        )
        self.assertIsInstance(relay.raw_cameras["cam"], video_proc._RawCameraStub)
        self.assertEqual(relay.readable_raw_cameras, {})

    def test_policy_ring_source_gets_a_reader_over_the_ring(self) -> None:
        relay = _relay()
        raw_meta = {
            "overhead_left": video_proc._gsth264_meta(
                "/tmp/ol.sock", 960, 600, 60, 60, 1.0
            ),
            "overhead_right": video_proc._gsth264_meta(
                "/tmp/or.sock", 960, 600, 60, 60, 1.0
            ),
        }
        policy_meta = {
            "overhead_left": video_proc._pyshm_meta("ring-ol", 960, 600, 60),
        }
        relay._attach_raw_readers(raw_meta, policy_meta)

        reader = relay.raw_cameras["overhead_left"]
        self.assertIsInstance(reader, _Reader)
        self.assertEqual(reader.shm_name, "ring-ol")
        self.assertIs(reader.cond, relay._raw_cond)
        # The recorder's transport for that source is untouched.
        self.assertEqual(raw_meta["overhead_left"]["transport"], "gstshm-h264")
        # A source without a ring stays a stub, and only readers are "readable".
        self.assertIsInstance(
            relay.raw_cameras["overhead_right"], video_proc._RawCameraStub
        )
        self.assertEqual(list(relay.readable_raw_cameras), ["overhead_left"])

    def test_pyshm_fallback_sources_are_readable(self) -> None:
        relay = _relay()
        relay._attach_raw_readers({"cam": video_proc._pyshm_meta("blk", 960, 600, 60)})
        self.assertIsInstance(relay.raw_cameras["cam"], _Reader)
        self.assertEqual(list(relay.readable_raw_cameras), ["cam"])


if __name__ == "__main__":
    unittest.main()
