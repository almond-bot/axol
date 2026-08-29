from __future__ import annotations

import multiprocessing

import numpy as np
import pytest

from almond_axol.video.gst_zed import _split_nals
from almond_axol.video.hw_video import (
    _bitrate_for,
    _drain_complete_nals,
    _gst_argv,
    _strip_start_code,
    dataset_vbr_bitrate,
)
from almond_axol.video.shm_frames import (
    RawFrameReader,
    RawFrameWriter,
    _au_has_coded_slice,
    _block_size,
)
from almond_axol.video.video_proc import _eye_plan, _plan, _pyshm_meta, _raw_plan


def test_bitrate_budgets_are_bounded() -> None:
    assert _bitrate_for(320, 240, 15) == 4_000_000
    assert _bitrate_for(8000, 8000, 120) == 20_000_000
    target, peak = dataset_vbr_bitrate(960, 600, 60)
    assert 3_000_000 <= target <= 16_000_000
    assert peak == target * 2


def test_gstreamer_command_is_shell_free_and_dimensioned() -> None:
    argv = _gst_argv(640, 480, 30, 5_000_000)
    assert argv[:2] == ["gst-launch-1.0", "-q"]
    assert "width=640" in argv
    assert "height=480" in argv
    assert "bitrate=5000000" in argv
    assert "!" in argv


def test_annex_b_nal_helpers_handle_three_and_four_byte_prefixes() -> None:
    assert _strip_start_code(b"\x00\x00\x01\x65abc") == b"\x65abc"
    assert _strip_start_code(b"\x00\x00\x00\x01\x67abc") == b"\x67abc"
    assert _strip_start_code(b"plain") is None

    buf = bytearray(b"\x00\x00\x00\x01\x67aa\x00\x00\x01\x68bb\x00\x00\x01\x65tail")
    assert _drain_complete_nals(buf) == [b"\x67aa", b"\x68bb"]
    assert bytes(buf).endswith(b"\x00\x00\x01\x65tail")

    assert _split_nals(b"\x00\x00\x01\x67aa\x00\x00\x00\x01\x68bb") == [
        b"\x67aa",
        b"\x68bb",
    ]
    assert _au_has_coded_slice(b"\x00\x00\x01\x65frame")
    assert not _au_has_coded_slice(b"\x00\x00\x01\x67sps")


def test_camera_eye_plans_keep_stream_and_record_independent() -> None:
    assert _plan("head", ["left", "right"], True) == [
        ("left", "head_left"),
        ("right", "head_right"),
    ]
    spec = {
        "stream_eyes": ["left", "right"],
        "stream_suffix": True,
        "record_eyes": ["left"],
        "record_suffix": False,
    }
    assert _eye_plan("head", spec) == [("left", "head_left"), ("right", "head_right")]
    assert _raw_plan("head", spec) == [("left", "head")]
    assert _pyshm_meta("block", 10, 20, 30)["transport"] == "pyshm"


def test_shared_memory_raw_frame_round_trip() -> None:
    assert _block_size(2, 3) > 2 * 3 * 3
    condition = multiprocessing.Condition()
    writer = RawFrameWriter.create(2, 3, condition)
    reader = RawFrameReader(writer.name, 2, 3, 60, condition)
    rgba = np.arange(3 * 2 * 4, dtype=np.uint8).reshape(3, 2, 4)
    try:
        with pytest.raises(RuntimeError, match="no frames"):
            reader.read_latest_with_ts()
        writer.publish(rgba, cap_ts=10.0, recv_ts=11.0)
        frame, cap, recv = reader.read_latest_with_ts()
        np.testing.assert_array_equal(frame, rgba[:, :, :3])
        assert (cap, recv) == (10.0, 11.0)
    finally:
        reader.close()
        writer.close()
