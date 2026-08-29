from __future__ import annotations

import multiprocessing
import queue
from types import SimpleNamespace

import numpy as np
import pytest

from almond_axol.video import gst_zed
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


def test_gst_availability_checks_optional_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gst_zed, "_require_gst", lambda: (_ for _ in ()).throw(ImportError())
    )
    assert not gst_zed._gi_available()
    assert not gst_zed._element_available("zedsrc")

    class Factory:
        @staticmethod
        def find(name: str) -> object | None:
            return object() if name == "zedsrc" else None

    class Gst:
        ElementFactory = Factory

    monkeypatch.setattr(gst_zed, "_require_gst", lambda: (Gst, object()))
    assert gst_zed._gi_available()
    assert gst_zed._element_available("zedsrc")
    assert not gst_zed._element_available("zedxonesrc")

    monkeypatch.setattr(gst_zed, "_gi_available", lambda: True)
    monkeypatch.setattr(gst_zed, "hw_h264_available", lambda: True)
    monkeypatch.setattr(gst_zed, "_element_available", lambda name: name == "zedsrc")
    assert not gst_zed.zed_gst_available()
    assert gst_zed.zed_stereo_gst_available()

    monkeypatch.setattr(gst_zed, "_gi_available", lambda: False)
    assert not gst_zed.zed_gst_available()
    assert not gst_zed.zed_stereo_gst_available()


def test_gst_typelib_path_adds_only_existing_directories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = "/custom/typelibs"
    monkeypatch.setenv("GI_TYPELIB_PATH", existing)
    monkeypatch.setattr(
        gst_zed.os.path,
        "isdir",
        lambda path: path.endswith("x86_64-linux-gnu/girepository-1.0"),
    )
    gst_zed._set_typelib_path()
    parts = gst_zed.os.environ["GI_TYPELIB_PATH"].split(gst_zed.os.pathsep)
    assert parts == [existing, "/usr/lib/x86_64-linux-gnu/girepository-1.0"]

    gst_zed._set_typelib_path()
    assert gst_zed.os.environ["GI_TYPELIB_PATH"].split(gst_zed.os.pathsep) == parts


def test_encoded_channel_fans_out_and_drops_stale_units() -> None:
    channel = gst_zed._AUChannel(lambda: True)
    first = channel.subscribe()
    second = channel.subscribe()
    assert channel.alive

    for value in range(gst_zed._SUBSCRIBER_QUEUE_DEPTH + 2):
        channel.broadcast([bytes([value])])

    assert channel.first_au.is_set()
    assert first.qsize() == gst_zed._SUBSCRIBER_QUEUE_DEPTH
    assert first.get_nowait() == [b"\x02"]
    assert second.get_nowait() == [b"\x02"]

    while not first.empty():
        first.get_nowait()
    channel.unsubscribe(first)
    channel.unsubscribe(first)
    channel.broadcast([b"latest"])
    with pytest.raises(queue.Empty):
        first.get_nowait()


def test_raw_gst_buffer_returns_rgb_and_enforces_freshness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = gst_zed._RawBuffer(width=2, height=1)
    with pytest.raises(RuntimeError, match="has not captured"):
        raw.read_latest_with_ts()
    with pytest.raises(TimeoutError, match="waiting for frame"):
        raw.read_at_or_after(1.0, timeout_ms=0)

    rgba = np.arange(8, dtype=np.uint8).reshape(1, 2, 4)
    raw.set(rgba, cap_ts=4.0, recv_ts=5.0)
    frame, cap, recv = raw.read_at_or_after(3.0)
    np.testing.assert_array_equal(frame, rgba[:, :, :3])
    assert frame.flags.c_contiguous
    assert (cap, recv) == (4.0, 5.0)

    monkeypatch.setattr(gst_zed.time, "perf_counter", lambda: 5.01)
    np.testing.assert_array_equal(raw.read_latest(max_age_ms=20), frame)
    with pytest.raises(TimeoutError, match="old"):
        raw.read_latest(max_age_ms=5)


def test_gst_stream_consumer_delegates_to_available_branches() -> None:
    consumer = gst_zed._GstStreamConsumer()
    consumer._alive_fn = lambda: True
    consumer._enc = None
    consumer._raw = None
    assert consumer.alive
    with pytest.raises(RuntimeError, match="no encoded"):
        consumer.subscribe()
    with pytest.raises(RuntimeError, match="no raw"):
        consumer.read_latest()
    with pytest.raises(RuntimeError, match="no raw"):
        consumer.read_latest_with_ts()
    with pytest.raises(RuntimeError, match="no raw"):
        consumer.read_at_or_after(0.0)

    channel = gst_zed._AUChannel(lambda: True)
    raw = gst_zed._RawBuffer(1, 1)
    raw.set(np.array([[[1, 2, 3, 4]]], dtype=np.uint8), 1.0, 1.0)
    consumer._enc = channel
    consumer._raw = raw
    subscribed = consumer.subscribe()
    channel.broadcast([b"au"])
    assert subscribed.get_nowait() == [b"au"]
    consumer.unsubscribe(subscribed)
    np.testing.assert_array_equal(consumer.read(), np.array([[[1, 2, 3]]]))
    assert consumer.read_latest_with_ts()[1:] == (1.0, 1.0)


def test_gst_pipeline_fragments_include_transport_and_encoder_settings() -> None:
    assert "name=encoded" in gst_zed._enc_appsink("encoded")
    assert "drop=true" in gst_zed._raw_appsink("raw")
    assert "socket-path=/tmp/raw.sock" in gst_zed._raw_shmsink("/tmp/raw.sock")

    dataset = gst_zed._dataset_enc_shmsink("/tmp/data.sock", 960, 600, 60, "data")
    assert "name=data" in dataset
    assert "idrinterval=15" in dataset
    assert "socket-path=/tmp/data.sock" in dataset
    assert "peak-bitrate=" in dataset

    encoded = gst_zed._enc_branch(4_000_000, 30, "head")
    assert "name=head" in encoded
    assert "bitrate=4000000" in encoded
    assert "idrinterval=30" in encoded


def test_gst_pipeline_timestamps_latency_and_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = gst_zed._GstPipelineBase()
    assert not base.alive
    assert not base.is_connected

    class State:
        NULL = "null"

    class Query:
        @staticmethod
        def new_latency() -> object:
            return object()

    Gst = SimpleNamespace(CLOCK_TIME_NONE=-1, State=State, Query=Query)

    class Clock:
        def get_time(self) -> int:
            return 6_000_000_000

    class Pipeline:
        state = "playing"

        def get_state(self, timeout: int) -> tuple[None, str, None]:
            return None, self.state, None

        def get_base_time(self) -> int:
            return 1_000_000_000

        def query(self, query: object) -> bool:
            return True

        def set_state(self, state: str) -> None:
            self.state = state

    base._gst = Gst
    base._pipeline = Pipeline()
    base._clock = Clock()
    assert base.alive
    assert base.is_connected
    assert base._cap_perf_from_pts(4_000_000_000, 10.0) == 9.0
    assert base._cap_perf_from_pts(Gst.CLOCK_TIME_NONE, 10.0) == 10.0

    query_result = SimpleNamespace(parse_latency=lambda: (True, 20_000_000, None))
    monkeypatch.setattr(Query, "new_latency", lambda: query_result)
    assert base._measure_raw_latency_s(60) == 0.02
    base._pipeline.query = lambda query: False  # type: ignore[method-assign]
    assert base._measure_raw_latency_s(50) == 0.02
    assert base._measure_raw_latency_s(0) == 0.0

    thread = SimpleNamespace(join=lambda timeout: None)
    base._threads.append(thread)  # type: ignore[arg-type]
    base.disconnect()
    assert base._pipeline is None
    assert base._threads == []
    assert not base.is_connected


def test_gst_buffer_handlers_unmap_and_publish_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Gst:
        class MapFlags:
            READ = "read"

    class Buffer:
        def __init__(self, data: bytes, *, ok: bool = True) -> None:
            self.data = data
            self.ok = ok
            self.pts = 4
            self.unmapped = 0

        def map(self, flags: str) -> tuple[bool, SimpleNamespace]:
            return self.ok, SimpleNamespace(data=self.data, size=len(self.data))

        def unmap(self, mapinfo: SimpleNamespace) -> None:
            self.unmapped += 1

    base = gst_zed._GstPipelineBase()
    base._gst = Gst
    base._cap_perf_from_pts = lambda pts, recv: recv - 0.1  # type: ignore[method-assign]
    channel = gst_zed._AUChannel(lambda: True)
    subscriber = channel.subscribe()
    monkeypatch.setattr(gst_zed.time, "perf_counter", lambda: 1.0)
    encoded = base._make_au_handler(channel, "head")

    rejected = Buffer(b"ignored", ok=False)
    encoded(rejected, 1.0)
    with pytest.raises(queue.Empty):
        subscriber.get_nowait()

    accepted = Buffer(b"\x00\x00\x01\x65frame")
    encoded(accepted, 1.0)
    assert subscriber.get_nowait() == [b"\x65frame"]
    assert accepted.unmapped == 1

    received: list[tuple[np.ndarray, float, float]] = []
    raw_handler = base._make_raw_handler(
        lambda rgba, cap, recv: received.append((rgba.copy(), cap, recv)), 2, 1
    )
    short = Buffer(b"\x00" * 7)
    raw_handler(short, 5.0)
    assert received == []
    assert short.unmapped == 1

    complete = Buffer(bytes(range(8)))
    raw_handler(complete, 5.0)
    np.testing.assert_array_equal(received[0][0], np.arange(8).reshape(1, 2, 4))
    assert received[0][1:] == (4.9, 5.0)
    assert complete.unmapped == 1

    raw = gst_zed._RawBuffer(2, 1)
    sink = base._buffer_sink(raw)
    source = np.arange(8, dtype=np.uint8).reshape(1, 2, 4)
    sink(source, 2.0, 3.0)
    source.fill(0)
    np.testing.assert_array_equal(
        raw.read_latest_with_ts()[0], [[[0, 1, 2], [4, 5, 6]]]
    )


def test_encoder_bitrate_update_is_best_effort() -> None:
    encoder = SimpleNamespace(
        values=[], set_property=lambda name, value: encoder.values.append((name, value))
    )
    pipeline = SimpleNamespace(
        get_by_name=lambda name: encoder if name == "present" else None
    )
    gst_zed._set_enc_bitrate(pipeline, "present", 4.5)
    gst_zed._set_enc_bitrate(pipeline, "missing", 10)
    assert encoder.values == [("bitrate", 4)]
