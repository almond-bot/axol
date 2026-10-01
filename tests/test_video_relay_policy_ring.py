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


class _FakeGstCamera:
    """Stands in for ZedGstCamera inside _open_gst_camera_raw."""

    instances: list[_FakeGstCamera] = []

    def __init__(self, serial: int, resolution: str, fps: int, **kwargs: object):
        self.serial = serial
        self.fps = fps
        self.kwargs = kwargs
        self.pts_perf_offset_s = 0.5
        _FakeGstCamera.instances.append(self)

    def connect(self) -> None:
        pass


class _FakeWriter:
    name = "ring-cam"

    @classmethod
    def create(cls, w: int, h: int, cond: object) -> _FakeWriter:
        return cls()

    def publish(self, *args: object) -> None:
        pass

    def close(self) -> None:
        pass


class PolicyRingRateTest(unittest.TestCase):
    """``policy_fps`` in a spec thins only the control-process ring."""

    def setUp(self) -> None:
        _FakeGstCamera.instances.clear()
        for target, value in (
            ("almond_axol.video.gst_zed.ZedGstCamera", _FakeGstCamera),
            ("almond_axol.video.gst_zed.zed_gst_available", lambda **kw: True),
            ("almond_axol.video.shm_frames.RawFrameWriter", _FakeWriter),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _open(self, spec: dict) -> tuple[dict, dict]:
        result = video_proc._open_gst_camera_raw(
            "cam",
            {"serial": 7, "resolution": "SVGA", "fps": 60, **spec},
            object(),
            "/tmp",
        )
        assert result is not None
        _cam, _sources, _writers, raw_meta, policy_meta = result
        return raw_meta, policy_meta

    def test_combined_transport_thins_the_ring_to_policy_fps(self) -> None:
        raw_meta, policy_meta = self._open(
            {"raw_transport": "gstshm+pyshm", "policy_fps": 20}
        )
        camera = _FakeGstCamera.instances[-1]
        self.assertEqual(camera.kwargs["policy_fps"], 20)
        self.assertEqual(camera.kwargs["dataset_fps"], 60)
        # The reader paces its freshness rules on the ring's rate; the
        # recorder's encoded branch stays at the dataset rate.
        self.assertEqual(policy_meta["cam"]["fps"], 20)
        self.assertEqual(raw_meta["cam"]["fps"], 60)
        self.assertEqual(raw_meta["cam"]["transport"], "gstshm-h264")

    def test_policy_fps_is_capped_at_capture_rate(self) -> None:
        _raw_meta, policy_meta = self._open(
            {"raw_transport": "gstshm+pyshm", "policy_fps": 90}
        )
        self.assertEqual(_FakeGstCamera.instances[-1].kwargs["policy_fps"], 60)
        self.assertEqual(policy_meta["cam"]["fps"], 60)

    def test_without_policy_fps_the_ring_runs_at_capture_rate(self) -> None:
        _raw_meta, policy_meta = self._open({"raw_transport": "gstshm+pyshm"})
        self.assertEqual(_FakeGstCamera.instances[-1].kwargs["policy_fps"], 60)
        self.assertEqual(policy_meta["cam"]["fps"], 60)

    def test_plain_pyshm_fallback_ignores_policy_fps(self) -> None:
        # On plain pyshm the ring is the recorder's input too: never thinned.
        raw_meta, policy_meta = self._open({"raw_transport": "pyshm", "policy_fps": 20})
        self.assertNotIn("policy_fps", _FakeGstCamera.instances[-1].kwargs)
        self.assertEqual(raw_meta["cam"]["transport"], "pyshm")
        self.assertEqual(raw_meta["cam"]["fps"], 60)
        self.assertEqual(policy_meta, {})


if __name__ == "__main__":
    unittest.main()
