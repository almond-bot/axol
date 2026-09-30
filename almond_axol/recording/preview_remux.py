"""Cut a quick-starting, lighter preview of one episode's video by copying packets.

The dataset preview plays every camera of an episode at once. Axol records
every dataset frame as an IDR at ~0.6 bpp (``hw_video.dataset_intra_vbr_bitrate``:
~21 Mbps per SVGA camera at 60 fps), so four cameras need ~85 Mbps — more than
a Wi-Fi or Tailscale link to the station carries, and playback stalls. Because
every frame is independently decodable, keeping every ``step``-th packet is a
valid, lower-rate stream: no decode, no encode, a fraction of the bytes.

The output holds only the episode's span (a stock-LeRobot mp4 is shared by
several episodes), rebased to start at 0, with its index first (``faststart``)
— which is also why the full-rate view goes through here (``step`` 1). A source with predicted frames is
copied whole-span instead of decimated (dropping a reference frame would break
the frames after it); LeRobot starts every episode on a keyframe, so the span
still decodes from its first packet.

Runs as a child process (``python -m almond_axol.recording.preview_remux``)
from the serve backend: under ``axol serve`` the control loop is a thread of
the server process, and the demux/mux loop holds the GIL per packet. The
child lowers its priority and keeps to the background cores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

# Packets within this much of the span's edges count as inside it.
_EDGE_S = 1e-3


def _span_packets(stream: Any, container: Any, start: float, end: float):
    """Demuxed packets of ``stream`` whose presentation time is in the span."""
    tb = stream.time_base
    for packet in container.demux(stream):
        if packet.pts is None or packet.size == 0:
            continue  # the demuxer's flush packet
        t = float(packet.pts * tb)
        if start - _EDGE_S <= t < end - _EDGE_S:
            yield packet


def remux_preview(
    src: Path, dst: Path, start: float, end: float, step: int
) -> dict[str, Any]:
    """Write ``src``'s ``[start, end)`` to ``dst``, keeping every ``step``-th frame.

    Returns ``{"frames": kept, "source_frames": n, "decimated": bool}``;
    ``decimated`` is False when the span has predicted frames (copied whole).
    """
    import av

    with av.open(str(src)) as probe:
        stream = probe.streams.video[0]
        all_intra = all(p.is_keyframe for p in _span_packets(stream, probe, start, end))
    step = step if all_intra and step > 1 else 1

    kept = total = 0
    # faststart: the index (moov) goes first. LeRobot's and the recorder's mp4s
    # keep it at the end, so a browser on a slow link reads nearly the whole
    # file before it can show a frame; with it first, playback starts at once.
    with (
        av.open(str(src)) as source,
        av.open(str(dst), "w", format="mp4", options={"movflags": "+faststart"}) as out,
    ):
        in_stream = source.streams.video[0]
        # opaque: a stream copy, so no codec is looked up (or needs to exist).
        out_stream = out.add_stream_from_template(in_stream, opaque=True)
        base: int | None = None
        for packet in _span_packets(in_stream, source, start, end):
            index = total
            total += 1
            if index % step:
                continue
            if base is None:
                base = packet.dts if packet.dts is not None else packet.pts
            packet.pts -= base
            if packet.dts is not None:
                packet.dts -= base
            if packet.duration:
                packet.duration *= step
            packet.stream = out_stream
            out.mux(packet)
            kept += 1
    return {"frames": kept, "source_frames": total, "decimated": step > 1}


def _lower_priority() -> None:
    try:
        os.nice(10)
    except OSError:
        pass
    try:
        from ..utils import affinity

        affinity.pin_background()
    except Exception:  # best-effort: an unpinned preview is still correct
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("start", type=Fraction)
    parser.add_argument("end", type=Fraction)
    parser.add_argument("step", type=int)
    args = parser.parse_args(argv)
    _lower_priority()
    result = remux_preview(
        args.src, args.dst, float(args.start), float(args.end), args.step
    )
    json.dump(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
