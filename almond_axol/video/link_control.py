"""Keep the headset video live on a slow or lossy link.

The relay's encoders run at a fixed bitrate and aiortc sends every packet it
is given: it has no pacer and never waits for the network. When the link is
slower than the stream, nothing upstream notices. The kernel's socket buffer
fills, then asyncio's datagram transport buffers the rest in an *unbounded*
deque, so headset latency grows for as long as the link stays slow (the
"laggier the longer I'm in VR" report). Two mechanisms fix that:

* **Live edge** (:class:`~almond_axol.video.video.PrecodedVideoTrack`): before
  handing an access unit to aiortc the track asks how many seconds of video
  are already queued for its peer (:func:`transport_backlog_bytes`). Past a
  limit it skips frames and resumes on the next IDR, so the queue drains
  instead of growing and the picture never decodes a P-frame whose reference
  was skipped.
* **Adaptive bitrate** (:class:`BitrateController`): every tick the relay
  reads, for each peer, the queued bytes, the receiver-reported loss and the
  receiver's bandwidth estimate (REMB), and scales every stream encoder's
  bitrate between ``min_scale`` and its configured value. Congestion cuts the
  rate quickly; a clean link raises it slowly back to the configured value,
  which is also the ceiling (it never encodes above what it was built with).
  The encoders are shared by every peer, so the worst peer decides.

Nothing here imports aiortc, so the policy is unit-testable on its own.
"""

from __future__ import annotations

import fcntl
import struct
import termios
from dataclasses import dataclass
from typing import Any

# Frame skipping (live edge). Queued video beyond these many seconds is
# latency the operator sees; a P-frame is skipped past the first limit, and
# even an IDR past the second (sending a large IDR into a full queue only
# makes it longer; the next one comes within a GOP). An IDR is several times
# a P-frame, so on a link that carries the average rate it still queues for
# tens of ms; skipping on that spike would cost a whole GOP. The backlog must
# therefore stay over the limit for LIVE_EDGE_PERSIST_S before anything is
# skipped.
LIVE_EDGE_SKIP_S = 0.15
LIVE_EDGE_SKIP_IDR_S = 0.30
LIVE_EDGE_PERSIST_S = 0.10

# A REMB older than this is ignored (the receiver stopped reporting).
_REMB_FRESH_S = 5.0
# Chrome's receive-side estimate starts low and ramps up over the first
# seconds, so it would throttle a healthy link at session start. Only trust it
# after it has been reporting for this long.
_REMB_WARMUP_S = 5.0


def transport_backlog_bytes(pc: Any) -> int:
    """Bytes queued for ``pc``'s media transport but not yet on the air.

    The sum of asyncio's datagram write buffer (unbounded; grows only once the
    kernel refuses more) and the kernel's per-socket send queue (``TIOCOUTQ``:
    bytes the socket has handed down that the qdisc / NIC driver has not yet
    transmitted). Both are zero on a link that keeps up. Walks aiortc/aioice
    internals defensively and returns 0 if any of them is missing.
    """
    total = 0
    seen: set[int] = set()
    for transceiver in _transceivers(pc):
        sender = getattr(transceiver, "sender", None)
        dtls = getattr(sender, "transport", None)
        ice = getattr(dtls, "transport", None)
        connection = getattr(ice, "_connection", None)
        nominated = getattr(connection, "_nominated", None)
        if not isinstance(nominated, dict):
            continue
        pair = nominated.get(1)
        protocol = getattr(pair, "protocol", None)
        transport = getattr(protocol, "transport", None)
        if transport is None or id(transport) in seen:
            continue  # BUNDLE: every transceiver shares one transport
        seen.add(id(transport))
        try:
            total += int(transport.get_write_buffer_size())
        except Exception:  # noqa: BLE001 - best-effort introspection
            pass
        total += _kernel_send_queue_bytes(transport)
    return total


def _transceivers(pc: Any) -> list[Any]:
    try:
        return list(pc.getTransceivers())
    except Exception:  # noqa: BLE001 - closed / half-built peer
        return []


def _kernel_send_queue_bytes(transport: Any) -> int:
    """Bytes in the socket's kernel send queue (Linux ``TIOCOUTQ``), else 0."""
    try:
        sock = transport.get_extra_info("socket")
        if sock is None:
            return 0
        buf = fcntl.ioctl(sock.fileno(), termios.TIOCOUTQ, struct.pack("i", 0))
        return max(0, struct.unpack("i", buf)[0])
    except Exception:  # noqa: BLE001 - not Linux, or the socket is gone
        return 0


@dataclass
class LinkSample:
    """One peer's link health for one controller tick."""

    backlog_bytes: int = 0
    # Receiver-reported fraction lost over its last report interval (0..1).
    loss: float | None = None
    rtt_s: float | None = None
    # The receiver's bandwidth estimate for everything sent to it (bits/s),
    # or None when it is stale or still warming up.
    remb_bps: int | None = None


class RembState:
    """Latest REMB for one peer, with warm-up and freshness."""

    def __init__(self) -> None:
        self.bps: int | None = None
        self.at: float | None = None
        self.first_at: float | None = None

    def record(self, bps: int, now: float) -> None:
        self.bps = int(bps)
        self.at = now
        if self.first_at is None:
            self.first_at = now

    def usable(self, now: float) -> int | None:
        if self.bps is None or self.at is None or self.first_at is None:
            return None
        if now - self.at > _REMB_FRESH_S or now - self.first_at < _REMB_WARMUP_S:
            return None
        return self.bps


class BitrateController:
    """Scale the shared stream encoders to what the worst peer's link carries.

    ``scale`` multiplies every stream encoder's configured bitrate. It starts
    at 1.0 (the configured rate is the default and the ceiling) and only goes
    down when a peer shows congestion:

    * queued video above ``backlog_limit_s`` at the current rate on two
      ticks in a row (a single IDR burst is not congestion), or
    * receiver-reported loss above ``loss_limit``,

    each of which cuts the scale by ``decrease`` at most every
    ``decrease_interval_s``. A usable REMB below the current rate caps the
    scale at the receiver's estimate immediately. After ``hold_s`` without
    congestion the scale grows by ``increase`` per second back toward 1.0.
    """

    def __init__(
        self,
        *,
        min_scale: float = 0.15,
        decrease: float = 0.7,
        increase: float = 1.08,
        decrease_interval_s: float = 0.5,
        hold_s: float = 3.0,
        increase_interval_s: float = 1.0,
        backlog_limit_s: float = 0.05,
        loss_limit: float = 0.05,
        loss_ok: float = 0.02,
    ) -> None:
        self.min_scale = min_scale
        self.decrease = decrease
        self.increase = increase
        self.decrease_interval_s = decrease_interval_s
        self.hold_s = hold_s
        self.increase_interval_s = increase_interval_s
        self.backlog_limit_s = backlog_limit_s
        self.loss_limit = loss_limit
        self.loss_ok = loss_ok
        self.reset()

    def reset(self) -> None:
        self.scale = 1.0
        self._last_decrease: float | None = None
        self._last_increase: float | None = None
        self._backlog_over_last = False
        self.reason: str | None = None

    def update(
        self, now: float, ceiling_bps: float, samples: list[LinkSample]
    ) -> float:
        """Return the new scale for this tick (``ceiling_bps`` = scale 1.0)."""
        if not samples or ceiling_bps <= 0:
            return self.scale
        current_bps = max(ceiling_bps * self.scale, 1.0)
        backlog_s = max(8.0 * s.backlog_bytes / current_bps for s in samples)
        losses = [s.loss for s in samples if s.loss is not None]
        loss = max(losses) if losses else 0.0
        rembs = [s.remb_bps for s in samples if s.remb_bps is not None]

        # A full queue counts only when two ticks in a row see it: an IDR
        # burst queues briefly even on a link that carries the average rate.
        backlog_over = backlog_s > self.backlog_limit_s
        queue_congested = backlog_over and self._backlog_over_last
        self._backlog_over_last = backlog_over
        congested = queue_congested or loss > self.loss_limit
        if congested:
            if (
                self._last_decrease is None
                or now - self._last_decrease >= self.decrease_interval_s
            ):
                self.scale *= self.decrease
                self._last_decrease = now
                self.reason = (
                    f"queued {1e3 * backlog_s:.0f} ms"
                    if queue_congested
                    else f"loss {100.0 * loss:.1f}%"
                )
        elif (
            loss <= self.loss_ok
            and not backlog_over
            and (
                self._last_decrease is None or now - self._last_decrease >= self.hold_s
            )
            and (
                self._last_increase is None
                or now - self._last_increase >= self.increase_interval_s
            )
            and self.scale < 1.0
        ):
            self.scale *= self.increase
            self._last_increase = now

        if rembs:
            cap = min(rembs) / ceiling_bps
            if cap < self.scale:
                self.scale = cap
                self._last_decrease = now
                self.reason = f"receiver estimate {min(rembs) / 1e6:.1f} Mbps"

        self.scale = min(1.0, max(self.min_scale, self.scale))
        return self.scale
