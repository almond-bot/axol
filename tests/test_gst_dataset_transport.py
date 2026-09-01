from __future__ import annotations

import unittest

from almond_axol.video.gst_zed import ZedGstCamera, ZedGstStereoCamera


_H264_OUTPUT = (
    "video/x-h264,stream-format=byte-stream,alignment=au ! "
    "queue leaky=downstream max-size-buffers=2 ! gdppay ! shmsink"
)


class GstDatasetTransportTest(unittest.TestCase):
    def assert_dataset_branch_is_backpressure_safe(
        self, pipeline: str, valve_name: str, encoder_name: str, socket_path: str
    ) -> None:
        start = pipeline.index(f"valve name={valve_name} drop=false")
        sink = f"shmsink socket-path={socket_path} wait-for-connection=true"
        end = pipeline.index(sink, start) + len(sink)
        branch = pipeline[start:end]

        self.assertIn(f"name={encoder_name}", branch)
        self.assertIn(_H264_OUTPUT, branch)
        self.assertIn(sink, branch)

    def test_mono_dataset_encoder_drains_before_recorder_connects(self) -> None:
        camera = ZedGstCamera(
            serial=1,
            resolution="SVGA",
            raw_socket_path="/tmp/mono-dataset.sock",
        )

        pipeline = camera._pipeline_str()

        self.assertIn("valve name=rawvalve drop=false", pipeline)
        self.assert_dataset_branch_is_backpressure_safe(
            pipeline, "rawvalve", "dsenc", "/tmp/mono-dataset.sock"
        )
        self.assertEqual(camera._raw_gates(), (("rawvalve", "dsenc"),))

    def test_both_stereo_dataset_encoders_drain_before_recorder_connects(
        self,
    ) -> None:
        camera = ZedGstStereoCamera(
            serial=2,
            resolution="SVGA",
            left_raw_socket_path="/tmp/left-dataset.sock",
            right_raw_socket_path="/tmp/right-dataset.sock",
        )

        pipeline = camera._pipeline_str()

        self.assertIn("valve name=rawvalve_l drop=false", pipeline)
        self.assertIn("valve name=rawvalve_r drop=false", pipeline)
        self.assert_dataset_branch_is_backpressure_safe(
            pipeline, "rawvalve_l", "dsenc_l", "/tmp/left-dataset.sock"
        )
        self.assert_dataset_branch_is_backpressure_safe(
            pipeline, "rawvalve_r", "dsenc_r", "/tmp/right-dataset.sock"
        )
        self.assertEqual(
            camera._raw_gates(),
            (("rawvalve_l", "dsenc_l"), ("rawvalve_r", "dsenc_r")),
        )


if __name__ == "__main__":
    unittest.main()
