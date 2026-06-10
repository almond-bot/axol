"""Hardware H.264 encoding for the WebRTC headset relay (NVIDIA Jetson).

aiortc encodes video in software (libvpx VP8 / libx264), which cannot keep
up with high-resolution camera relays on the Jetson's ARM cores — at HD1200
the encoder falls behind the 30 fps track pacing and frames queue up as
latency. Jetsons ship a hardware encoder (NVENC) exposed through GStreamer's
``nvv4l2h264enc``; this module routes aiortc's H.264 encoding through it.

How it plugs in:

- :func:`install_hw_encoder` monkeypatches ``aiortc``'s ``get_encoder`` so
  H.264 senders get a :class:`JetsonH264Encoder`. It is a no-op (returning
  ``False``) when the GStreamer element isn't present, in which case aiortc
  behaves exactly as before — callers should only prefer H.264 in the SDP
  when this returns ``True`` (see ``WebRTCManager.create_offer``).
- :class:`JetsonH264Encoder` feeds RGBA frames to a ``gst-launch-1.0``
  subprocess (``nvvidconv`` does the colorspace conversion on the GPU,
  ``nvv4l2h264enc`` encodes) and packetizes the returned Annex-B byte
  stream with aiortc's own RTP packetizer. If the pipeline dies (e.g. a
  Jetson model without NVENC), it falls back to aiortc's software
  ``H264Encoder`` for the rest of the session.

The encoder cannot force keyframes mid-stream (``gst-launch`` offers no
runtime control), so the pipeline sets a 1-second IDR interval instead —
a lost-packet (PLI) recovery worst case of ~1 s on an otherwise reliable
LAN link.
"""

from __future__ import annotations

import functools
import logging
import os
import queue
import select
import shutil
import subprocess
import threading
from collections.abc import Callable
from typing import Any

import av
from aiortc.codecs.h264 import H264Encoder
from aiortc.mediastreams import VIDEO_TIME_BASE, convert_timebase

from ..utils.jetson import pin_engine_clocks

_logger = logging.getLogger(__name__)

# The relay tracks pace at ~30 fps (aiortc VideoStreamTrack.next_timestamp).
_FPS = 30

# Once the encoder's stdout has been quiet this long, the buffered tail is
# the final NAL of the current access unit (the encoder writes a whole AU
# per frame, then idles ~33 ms until the next one). Flushing on idle avoids
# waiting for the next AU's delimiter, saving a frame of framing latency.
# Buffers belonging to one frame arrive back-to-back (<1 ms apart), so a
# 2 ms quiet window is decisively past the end of the AU; encode() blocks
# on this flush every frame, so it's direct glass-to-glass latency.
_IDLE_FLUSH_S = 0.002

# How long encode() waits for the current frame's AU before skipping it.
# Steady-state NVENC latency is ~10 ms; anything past this is a hiccup and
# the late AU gets drained on the next call.
_AU_TIMEOUT_S = 0.1

_NAL_TYPE_AUD = 9


def _bitrate_for(width: int, height: int, fps: int) -> int:
    """Encoder bitrate (bits/s) for a resolution: ~0.12 bpp, clamped 4–20 Mbps."""
    return max(4_000_000, min(20_000_000, int(width * height * fps * 0.12)))


def _gst_argv(width: int, height: int, fps: int, bitrate: int) -> list[str]:
    """``gst-launch-1.0`` argv: RGB24 on stdin -> H.264 byte stream on stdout.

    The input is RGB24 (not RGBA): the relay's frames are already ``rgb24``
    numpy arrays, so this makes the Python-side handoff a plain buffer copy
    and the pipe carry 25% less data. ``videoconvert`` expands to RGBA on
    the pipeline side (multi-threaded, overlapped with the next frame's
    write — measured faster end-to-end than converting in Python), since
    ``nvvidconv`` accepts RGBA but not RGB24. ``nvvidconv`` then converts to
    NV12 on the VIC and ``nvv4l2h264enc`` is the hardware encoder.
    ``insert-aud=true`` delimits access units so the reader can frame the
    byte stream; ``idrinterval`` bounds keyframe spacing to 1 s since
    ``gst-launch`` offers no runtime keyframe control (PLI recovery worst
    case).
    """
    pipeline = [
        f"fdsrc fd=0 blocksize={width * height * 3}",
        (
            "rawvideoparse use-sink-caps=false format=rgb "
            f"width={width} height={height} framerate={fps}/1"
        ),
        "videoconvert n-threads=4",
        "video/x-raw,format=RGBA",
        "nvvidconv",
        "video/x-raw(memory:NVMM),format=NV12",
        (
            f"nvv4l2h264enc control-rate=1 bitrate={bitrate} preset-level=1 "
            f"insert-sps-pps=true insert-aud=true idrinterval={fps} "
            "maxperf-enable=true"
        ),
        "video/x-h264,stream-format=byte-stream",
        "fdsink fd=1 sync=false",
    ]
    # gst-launch takes each property and each "!" separator as its own argv
    # item; no element string above contains spaces inside a value, so a
    # flat whitespace split yields the right argv (no shell needed).
    return ["gst-launch-1.0", "-q", *" ! ".join(pipeline).split()]


@functools.cache
def hw_h264_available() -> bool:
    """True when GStreamer's ``nvv4l2h264enc`` element is installed."""
    inspect = shutil.which("gst-inspect-1.0")
    if not inspect or not shutil.which("gst-launch-1.0"):
        return False
    try:
        # A full inspect, not `--exists`: nvvideo4linux2 registers its
        # elements by probing the V4L2 device when the plugin loads, so the
        # registry-cache-only `--exists` check misses them on some JetPacks.
        ok = (
            subprocess.run(
                [inspect, "nvv4l2h264enc"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).returncode
            == 0
        )
    except Exception:  # noqa: BLE001 - inspection failure → no hardware path
        return False
    if ok:
        _logger.info("Jetson hardware H.264 encoder (nvv4l2h264enc) available")
    return ok


class _GstH264Pipeline:
    """One ``gst-launch-1.0`` NVENC pipeline: RGBA frames in, H.264 AUs out.

    Frames are written to the subprocess's stdin; a reader thread splits the
    Annex-B byte stream on stdout into NAL units, groups them into access
    units (delimited by the AUD NALs the encoder is told to insert, or by
    the idle-flush heuristic), and queues them for :meth:`pop_au`.
    """

    def __init__(self, width: int, height: int, fps: int = _FPS) -> None:
        self.width = width
        self.height = height
        self.frames_written = 0
        bitrate = _bitrate_for(width, height, fps)
        self._proc = subprocess.Popen(
            _gst_argv(width, height, fps, bitrate),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        _logger.info(
            "gst h264 pipeline started: %dx%d@%dfps %.1f Mbps (pid %d)",
            width,
            height,
            fps,
            bitrate / 1e6,
            self._proc.pid,
        )
        self._aus: queue.Queue[list[bytes]] = queue.Queue()
        self._reader = threading.Thread(
            target=self._read_loop, name="gst-h264-reader", daemon=True
        )
        self._reader.start()

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def write_frame(self, rgb: bytes | memoryview) -> None:
        """Feed one packed RGB24 frame; raises if the pipeline has died."""
        assert self._proc.stdin is not None
        self._proc.stdin.write(rgb)
        self.frames_written += 1

    def pop_au(self, timeout: float | None = None) -> list[bytes] | None:
        """Oldest completed access unit (list of NALs), or ``None``.

        With a ``timeout`` (seconds), blocks up to that long for one to
        arrive; otherwise returns immediately.
        """
        try:
            if timeout is None:
                return self._aus.get_nowait()
            return self._aus.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=2.0)  # reap — no zombies
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        for stream in (self._proc.stdin, self._proc.stdout):
            if stream is not None:
                try:
                    stream.close()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass

    def __del__(self) -> None:
        self.close()

    # -- byte-stream framing -------------------------------------------------

    def _read_loop(self) -> None:
        """Split stdout's Annex-B byte stream into NALs and queue AUs."""
        assert self._proc.stdout is not None
        read_annexb_aus(self._proc.stdout.fileno(), self._aus.put, lambda: self.alive)


def read_annexb_aus(
    fd: int,
    emit_au: Callable[[list[bytes]], None],
    alive: Callable[[], bool],
) -> None:
    """Blocking loop: frame ``fd``'s Annex-B byte stream into access units.

    Each completed AU — a list of NALs with start codes stripped, delimited
    by the AUD NALs the encoder is told to insert or by the idle-flush
    heuristic (see ``_IDLE_FLUSH_S``) — is passed to ``emit_au``. Returns on
    EOF, on the fd being closed, or once ``alive()`` goes false while idle.
    Shared by the frame-fed NVENC pipeline here and the camera-native
    pipelines in ``gst_zed``.
    """
    buf = bytearray()
    current_au: list[bytes] = []

    def emit_nal(nal: bytes) -> None:
        nonlocal current_au
        if not nal:
            return
        if (nal[0] & 0x1F) == _NAL_TYPE_AUD and current_au:
            emit_au(current_au)
            current_au = []
        current_au.append(nal)

    def flush_au() -> None:
        nonlocal current_au
        if current_au:
            emit_au(current_au)
            current_au = []

    while True:
        try:
            ready, _, _ = select.select([fd], [], [], _IDLE_FLUSH_S)
            chunk = os.read(fd, 1 << 20) if ready else None
        except (OSError, ValueError):  # fd closed under us — clean teardown
            return
        if chunk is None:
            # Encoder is idle between frames: the buffered tail (if any)
            # is the last NAL of the current AU — emit and flush.
            tail = _strip_start_code(bytes(buf))
            if tail is not None:
                emit_nal(tail)
                buf.clear()
            flush_au()
            if not alive():
                return
            continue
        if not chunk:  # EOF — pipeline exited
            tail = _strip_start_code(bytes(buf))
            if tail is not None:
                emit_nal(tail)
            flush_au()
            return
        buf.extend(chunk)
        for nal in _drain_complete_nals(buf):
            emit_nal(nal)


def _strip_start_code(data: bytes) -> bytes | None:
    """``data`` minus its leading Annex-B start code, or ``None`` if absent."""
    if data.startswith(b"\x00\x00\x00\x01"):
        return data[4:] or None
    if data.startswith(b"\x00\x00\x01"):
        return data[3:] or None
    return None


def _drain_complete_nals(buf: bytearray) -> list[bytes]:
    """Pop every NAL in ``buf`` whose end (the next start code) is known.

    The bytes from the last start code onward stay in ``buf`` — that NAL's
    end is unknown until more data (or an idle flush) arrives.
    """
    nals: list[bytes] = []
    start = buf.find(b"\x00\x00\x01")
    if start == -1:
        return nals
    while True:
        nal_start = start + 3
        nxt = buf.find(b"\x00\x00\x01", nal_start)
        if nxt == -1:
            break
        end = nxt - 1 if buf[nxt - 1] == 0 else nxt
        nals.append(bytes(buf[nal_start:end]))
        start = nxt
    del buf[: start - 1 if start > 0 and buf[start - 1] == 0 else start]
    return nals


class JetsonH264Encoder:
    """aiortc-compatible H.264 encoder backed by the Jetson NVENC pipeline.

    Mirrors ``aiortc.codecs.h264.H264Encoder``'s interface (``encode`` /
    ``pack`` / ``target_bitrate``) and reuses its RTP packetizer. The
    pipeline is started lazily from the first frame's dimensions and
    restarted if they change; on any pipeline failure the encoder degrades
    to aiortc's software ``H264Encoder`` for the rest of the session.
    """

    def __init__(self) -> None:
        self._pipeline: _GstH264Pipeline | None = None
        self._fallback: H264Encoder | None = None
        self._packer: H264Encoder | None = None
        self._target_bitrate = 0  # informational only; NVENC bitrate is fixed

    def encode(
        self, frame: av.frame.Frame, force_keyframe: bool = False
    ) -> tuple[list[bytes], int]:
        assert isinstance(frame, av.VideoFrame)
        if self._fallback is not None:
            return self._fallback.encode(frame, force_keyframe)
        try:
            return self._hw_encode(frame)
        except Exception as exc:  # noqa: BLE001 - degrade, don't kill the track
            _logger.warning(
                "hardware H.264 encode failed (%s); falling back to software", exc
            )
            if self._pipeline is not None:
                self._pipeline.close()
                self._pipeline = None
            self._fallback = H264Encoder()
            return self._fallback.encode(frame, force_keyframe)

    def _hw_encode(self, frame: av.VideoFrame) -> tuple[list[bytes], int]:
        timestamp = convert_timebase(frame.pts, frame.time_base, VIDEO_TIME_BASE)

        if self._pipeline is not None and (
            not self._pipeline.alive
            or self._pipeline.width != frame.width
            or self._pipeline.height != frame.height
        ):
            if not self._pipeline.alive:
                raise RuntimeError("gst pipeline exited")
            self._pipeline.close()
            self._pipeline = None
        if self._pipeline is None:
            self._pipeline = _GstH264Pipeline(frame.width, frame.height)

        # The track's frames are already rgb24, so this is a plain unpadded
        # copy (no colorspace work); the ndarray is written via the buffer
        # protocol to skip a tobytes() round trip.
        pipeline = self._pipeline
        first_frame = not pipeline.frames_written
        pipeline.write_frame(frame.to_ndarray(format="rgb24"))

        # Encode synchronously, like aiortc's software encoder: block until
        # *this* frame's AU is out (NVENC delivers in ~10 ms; pipeline
        # startup makes the first one slower). Never returning early keeps
        # the queue empty — returning [] here would let a backlog build that
        # never drains, showing up as a permanent multi-frame latency.
        au = pipeline.pop_au(timeout=2.0 if first_frame else _AU_TIMEOUT_S)
        if au is None:
            if not pipeline.alive:
                raise RuntimeError("gst pipeline exited")
            # Transient encoder hiccup: skip this frame; the late AU is
            # drained (below) on the next call, so latency still can't build.
            _logger.debug("hw encoder missed the AU deadline; skipping a frame")
            return [], timestamp
        payloads = H264Encoder._packetize(au)
        # Drain anything else queued (startup burst, recovered hiccup) so
        # queued latency never persists; extra AUs ride along under this
        # frame's timestamp, which momentarily speeds up playback rather
        # than lagging it.
        while (extra := pipeline.pop_au()) is not None:
            payloads += H264Encoder._packetize(extra)
        return payloads, timestamp

    def pack(self, packet: Any) -> tuple[list[bytes], int]:
        """Pre-encoded passthrough — used by ``PrecodedVideoTrack`` senders.

        The gst-native camera path (``gst_zed``) delivers already-encoded
        access units; aiortc routes them here for RTP packetization. Reuses
        one software encoder instance purely as a packetizer (it never
        encodes).
        """
        if self._packer is None:
            self._packer = H264Encoder()
        return self._packer.pack(packet)

    @property
    def target_bitrate(self) -> int:
        return self._target_bitrate

    @target_bitrate.setter
    def target_bitrate(self, bitrate: int) -> None:
        # REMB feedback can't retune a gst-launch pipeline; the LAN link
        # comfortably carries the fixed NVENC bitrate, so just record it.
        self._target_bitrate = bitrate


_install_lock = threading.Lock()
_installed = False


def install_hw_encoder() -> bool:
    """Route aiortc's H.264 encoding through NVENC when available.

    Returns ``True`` when the hardware encoder is active (callers should
    then prefer H.264 in their SDP offers); ``False`` leaves aiortc
    untouched. Safe to call repeatedly.
    """
    if not hw_h264_available():
        return False
    global _installed
    with _install_lock:
        # Re-check the clocks on every install (they reset at reboot, and a
        # long-lived serve process spans teleop sessions).
        pin_engine_clocks()
        if _installed:
            return True
        import aiortc.codecs
        import aiortc.rtcrtpsender

        original = aiortc.codecs.get_encoder

        def patched(codec: Any) -> Any:
            if codec.mimeType.lower() == "video/h264":
                return JetsonH264Encoder()
            return original(codec)

        aiortc.codecs.get_encoder = patched
        aiortc.rtcrtpsender.get_encoder = patched
        _installed = True
        return True
