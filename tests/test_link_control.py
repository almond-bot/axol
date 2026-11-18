"""Headset video on a slow link: live-edge frame skipping + adaptive bitrate."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from almond_axol.video import gst_zed
from almond_axol.video.link_control import (
    LIVE_EDGE_PERSIST_S,
    LIVE_EDGE_SKIP_IDR_S,
    LIVE_EDGE_SKIP_S,
    BitrateController,
    LinkSample,
    RembState,
    transport_backlog_bytes,
)
from almond_axol.video.video import PrecodedVideoTrack, WebRTCManager

CEILING = 16_000_000

# --- BitrateController -------------------------------------------------------


def test_clean_link_stays_at_the_configured_rate() -> None:
    ctl = BitrateController()
    for t in range(20):
        assert ctl.update(t * 0.5, CEILING, [LinkSample(loss=0.0)]) == 1.0
    # Never above the configured rate, and no peers leaves it alone.
    assert ctl.update(11.0, CEILING, []) == 1.0


def test_queued_video_cuts_the_rate_at_most_every_interval() -> None:
    ctl = BitrateController()
    # 100 ms of video queued at the full rate, seen on two ticks in a row.
    full = [LinkSample(backlog_bytes=int(CEILING * 0.1 / 8))]
    assert ctl.update(0.0, CEILING, full) == 1.0  # one tick: could be an IDR
    assert ctl.update(0.5, CEILING, full) == pytest.approx(0.7)
    assert ctl.reason is not None and "queued" in ctl.reason
    # Too soon for another cut.
    assert ctl.update(0.7, CEILING, full) == pytest.approx(0.7)
    assert ctl.update(1.0, CEILING, full) == pytest.approx(0.49)


def test_a_single_idr_burst_is_not_congestion() -> None:
    ctl = BitrateController()
    burst = [LinkSample(backlog_bytes=int(CEILING * 0.2 / 8))]
    clear = [LinkSample(backlog_bytes=0, loss=0.0)]
    for t in range(0, 40, 2):
        assert ctl.update(t * 0.5, CEILING, burst) == 1.0
        assert ctl.update((t + 1) * 0.5, CEILING, clear) == 1.0


def test_loss_cuts_and_the_floor_holds() -> None:
    ctl = BitrateController(min_scale=0.2)
    for t in range(20):
        scale = ctl.update(t * 0.5, CEILING, [LinkSample(loss=0.2)])
    assert scale == pytest.approx(0.2)
    assert ctl.reason == "loss 20.0%"


def test_recovers_slowly_after_the_hold() -> None:
    ctl = BitrateController()
    ctl.update(0.0, CEILING, [LinkSample(loss=0.2)])
    assert ctl.scale == pytest.approx(0.7)
    # Inside the hold: no increase yet.
    assert ctl.update(2.0, CEILING, [LinkSample(loss=0.0)]) == pytest.approx(0.7)
    assert ctl.update(3.0, CEILING, [LinkSample(loss=0.0)]) == pytest.approx(0.756)
    # At most one step per second.
    assert ctl.update(3.5, CEILING, [LinkSample(loss=0.0)]) == pytest.approx(0.756)
    t = 4.0
    while ctl.scale < 1.0 and t < 60:
        ctl.update(t, CEILING, [LinkSample(loss=0.0)])
        t += 1.0
    assert ctl.scale == 1.0


def test_worst_peer_decides() -> None:
    ctl = BitrateController()
    good = LinkSample(loss=0.0)
    bad = LinkSample(loss=0.3)
    assert ctl.update(0.0, CEILING, [good, bad]) == pytest.approx(0.7)


def test_receiver_estimate_caps_the_rate_immediately() -> None:
    ctl = BitrateController()
    assert ctl.update(0.0, CEILING, [LinkSample(remb_bps=4_000_000)]) == pytest.approx(
        0.25
    )
    assert ctl.reason == "receiver estimate 4.0 Mbps"
    # An estimate above the ceiling never raises it.
    ctl.reset()
    assert ctl.update(0.0, CEILING, [LinkSample(remb_bps=40_000_000)]) == 1.0


def test_remb_warms_up_and_goes_stale() -> None:
    remb = RembState()
    assert remb.usable(0.0) is None
    remb.record(3_000_000, 0.0)
    assert remb.usable(1.0) is None  # still warming up
    remb.record(3_500_000, 6.0)
    assert remb.usable(6.0) == 3_500_000
    assert remb.usable(20.0) is None  # receiver stopped reporting


# --- transport_backlog_bytes -------------------------------------------------


class _FakeDatagramTransport:
    def __init__(self, buffered: int) -> None:
        self.buffered = buffered

    def get_write_buffer_size(self) -> int:
        return self.buffered

    def get_extra_info(self, name: str) -> Any:
        return None  # no socket: kernel queue reads as 0


def _fake_pc(*transports: Any) -> Any:
    transceivers = []
    for transport in transports:
        pair = SimpleNamespace(protocol=SimpleNamespace(transport=transport))
        connection = SimpleNamespace(_nominated={1: pair})
        ice = SimpleNamespace(_connection=connection)
        sender = SimpleNamespace(transport=SimpleNamespace(transport=ice))
        transceivers.append(SimpleNamespace(sender=sender))
    return SimpleNamespace(getTransceivers=lambda: transceivers)


def test_backlog_counts_each_bundled_transport_once() -> None:
    shared = _FakeDatagramTransport(50_000)
    assert transport_backlog_bytes(_fake_pc(shared, shared, shared)) == 50_000
    assert (
        transport_backlog_bytes(_fake_pc(shared, _FakeDatagramTransport(7))) == 50_007
    )


def test_backlog_is_zero_before_ice_connects_or_on_a_closed_peer() -> None:
    unconnected = SimpleNamespace(
        getTransceivers=lambda: [
            SimpleNamespace(
                sender=SimpleNamespace(
                    transport=SimpleNamespace(
                        transport=SimpleNamespace(
                            _connection=SimpleNamespace(_nominated={})
                        )
                    )
                )
            )
        ]
    )
    assert transport_backlog_bytes(unconnected) == 0

    def closed() -> list[Any]:
        raise RuntimeError("closed")

    assert transport_backlog_bytes(SimpleNamespace(getTransceivers=closed)) == 0


# --- encoder binding + GOP gaps ----------------------------------------------


class _FakeEncoder:
    def __init__(self, bitrate: int) -> None:
        self.props = {"bitrate": bitrate}
        self.sets: list[int] = []

    def get_property(self, name: str) -> int:
        return self.props[name]

    def set_property(self, name: str, value: int) -> None:
        self.props[name] = value
        self.sets.append(value)


def test_channel_scales_its_encoder_without_exceeding_the_configured_rate() -> None:
    channel = gst_zed._AUChannel(lambda: True)
    assert channel.set_bitrate_scale(0.5) == 0  # nothing bound yet
    encoder = _FakeEncoder(16_588_800)
    channel.bind_encoder(encoder)
    assert channel.bitrate_ceiling == 16_588_800

    assert channel.set_bitrate_scale(0.5) == 8_294_400
    assert encoder.props["bitrate"] == 8_294_400
    # Tiny changes don't poke NVENC.
    channel.set_bitrate_scale(0.505)
    assert encoder.sets == [8_294_400]
    # Never above the configured rate.
    assert channel.set_bitrate_scale(3.0) == 16_588_800

    consumer = gst_zed._GstStreamConsumer()
    consumer._enc = channel
    assert consumer.stream_bitrate_ceiling == 16_588_800
    assert consumer.set_stream_bitrate_scale(0.25) == 4_147_200
    assert consumer.stream_bitrate == 4_147_200
    consumer._enc = None
    assert consumer.stream_bitrate_ceiling == 0
    assert consumer.set_stream_bitrate_scale(0.25) == 0


def test_channel_ignores_a_missing_or_broken_encoder() -> None:
    channel = gst_zed._AUChannel(lambda: True)
    channel.bind_encoder(None)
    assert channel.bitrate_ceiling == 0

    class Broken:
        def get_property(self, name: str) -> int:
            raise TypeError(name)

    channel.bind_encoder(Broken())
    assert channel.bitrate_ceiling == 0


def test_dropping_an_au_marks_the_subscriber_gap() -> None:
    channel = gst_zed._AUChannel(lambda: True)
    q = channel.subscribe()
    for value in range(gst_zed._SUBSCRIBER_QUEUE_DEPTH):
        channel.broadcast([bytes([value])])
    assert not q.gap
    channel.broadcast([b"overflow"])
    assert q.gap


# --- PrecodedVideoTrack live edge ---------------------------------------------

_IDR = bytes([0x65, 0x88])
_P = bytes([0x41, 0x9A])


class _FakeSource:
    alive = True

    def __init__(self) -> None:
        self.channel = gst_zed._AUChannel(lambda: True)
        self.q = self.channel.subscribe()

    def subscribe(self) -> Any:
        return self.q

    def unsubscribe(self, q: Any) -> None:
        pass


def _ticking_clock(step: float) -> Any:
    """A clock that advances ``step`` s per read (one read per AU over the limit)."""
    state = {"t": 0.0}

    def clock() -> float:
        state["t"] += step
        return state["t"]

    return clock


def _recv_all(track: PrecodedVideoTrack, count: int) -> list[bytes]:
    async def run() -> list[bytes]:
        out = []
        for _ in range(count):
            packet = await track.recv()
            out.append(bytes(packet)[4:6])
        return out

    return asyncio.run(run())


def test_track_skips_to_the_next_idr_while_the_link_stays_backed_up() -> None:
    source = _FakeSource()
    track = PrecodedVideoTrack(source)
    track._clock = _ticking_clock(LIVE_EDGE_PERSIST_S)  # each AU: one persist step
    queued = {"s": 0.0}
    track.backlog_s = lambda: queued["s"]

    source.q.put([_IDR])
    source.q.put([_P])
    assert _recv_all(track, 2) == [_IDR, _P]

    # Between the two limits, persisting: P-frames are skipped, the next IDR
    # goes out and decoding resumes from it (never from a skipped GOP).
    queued["s"] = (LIVE_EDGE_SKIP_S + LIVE_EDGE_SKIP_IDR_S) / 2
    source.q.put([_P])  # first read over the limit starts the clock: sent
    source.q.put([_P])  # still over after the persist time: skipped
    source.q.put([_P])  # skipped
    source.q.put([_IDR])  # below the IDR limit: sent
    assert _recv_all(track, 2) == [_P, _IDR]
    assert track.frames_skipped == 2

    # Past the IDR limit even an IDR is skipped.
    queued["s"] = LIVE_EDGE_SKIP_IDR_S + 0.1
    assert track._should_skip(is_idr=True)

    # Drained: everything flows again.
    queued["s"] = 0.0
    source.q.put([_P])
    assert _recv_all(track, 1) == [_P]


def test_a_brief_backlog_spike_skips_nothing() -> None:
    source = _FakeSource()
    track = PrecodedVideoTrack(source)
    track._clock = _ticking_clock(LIVE_EDGE_PERSIST_S / 4)
    readings = iter([0.0, LIVE_EDGE_SKIP_S + 0.05, LIVE_EDGE_SKIP_S + 0.05, 0.0])
    track.backlog_s = lambda: next(readings)
    for au in ([_IDR], [_P], [_P], [_P]):
        source.q.put(au)
    assert _recv_all(track, 4) == [_IDR, _P, _P, _P]
    assert track.frames_skipped == 0


def test_track_resyncs_on_a_gop_gap() -> None:
    source = _FakeSource()
    track = PrecodedVideoTrack(source)
    source.q.put([_IDR])
    assert _recv_all(track, 1) == [_IDR]
    # The camera dropped an AU for this subscriber: P-frames after it are
    # undecodable, so the track waits for the next IDR.
    source.q.gap = True
    source.q.put([_P])
    source.q.put([_IDR])
    assert _recv_all(track, 1) == [_IDR]
    assert not source.q.gap


def test_track_without_link_info_never_skips() -> None:
    track = PrecodedVideoTrack(_FakeSource())
    assert not track._should_skip(is_idr=False)

    def broken() -> float:
        raise OSError("gone")

    track.backlog_s = broken
    assert not track._should_skip(is_idr=False)


# --- manager wiring ------------------------------------------------------------


class _AdjustableSource(_FakeSource):
    def __init__(self, ceiling: int) -> None:
        super().__init__()
        self.stream_bitrate_ceiling = ceiling
        self.scales: list[float] = []

    def set_stream_bitrate_scale(self, scale: float) -> int:
        self.scales.append(scale)
        return int(self.stream_bitrate_ceiling * scale)


def test_manager_scales_every_encoder_and_resets_when_the_last_viewer_leaves() -> None:
    sbs = _AdjustableSource(16_000_000)
    wrist = _AdjustableSource(8_000_000)
    manager = WebRTCManager({"overhead_sbs": sbs, "wrist": wrist, "other": object()})
    assert manager._ceiling_bps() == 24_000_000

    pc = SimpleNamespace(connectionState="connected", getTransceivers=lambda: [])
    from almond_axol.video.video import _PeerLink

    link = _PeerLink(pc=pc)
    manager._links[1] = link

    async def stats() -> dict[str, Any]:
        return {
            "r": SimpleNamespace(
                type="remote-inbound-rtp", fractionLost=64, roundTripTime=0.08
            ),
            "o": SimpleNamespace(type="outbound-rtp", bytesSent=1000),
        }

    pc.getStats = stats
    asyncio.run(manager._link_tick(0.0))
    assert link.last.loss == pytest.approx(0.25)
    assert link.last.rtt_s == pytest.approx(0.08)
    assert sbs.scales == [pytest.approx(0.7)]
    assert wrist.scales == [pytest.approx(0.7)]

    # The last viewer leaves: the next one starts at the full rate.
    asyncio.run(manager.close(1))
    assert sbs.scales[-1] == 1.0
    assert manager._controller.scale == 1.0


def test_manager_records_remb_from_the_sender_rtcp_path() -> None:
    from aiortc.rtp import RTCP_PSFB_APP, RtcpPsfbPacket, pack_remb_fci

    from almond_axol.video.video import _PeerLink

    manager = WebRTCManager({})
    manager._links[3] = _PeerLink(pc=SimpleNamespace())
    seen: list[Any] = []

    async def original(packet: Any) -> None:
        seen.append(packet)

    sender = SimpleNamespace(_handle_rtcp_packet=original)
    manager._hook_remb(3, sender)  # type: ignore[arg-type]
    packet = RtcpPsfbPacket(
        fmt=RTCP_PSFB_APP, ssrc=1, media_ssrc=0, fci=pack_remb_fci(5_000_000, [42])
    )
    asyncio.run(sender._handle_rtcp_packet(packet))
    assert seen == [packet]  # aiortc still gets it
    assert manager._links[3].remb.bps == 5_000_000
