"""Async SocketCAN bus shared across all Motor instances on one physical interface."""

from __future__ import annotations

import asyncio
import errno
import logging
import socket
import struct
import time
from pathlib import Path
from typing import Callable

import can

from ..constants import CAN_BRINGUP_SCRIPT
from ..utils.sudo import run_root

_logger = logging.getLogger(__name__)

# CAN_RAW socket option constants (missing from the socket module on some
# Python builds; values are stable Linux ABI).
_SOL_CAN_RAW = getattr(socket, "SOL_CAN_RAW", 101)
_CAN_RAW_FILTER = getattr(socket, "CAN_RAW_FILTER", 1)

# Socket failures with these errnos mean the interface went away or lost its
# link mid-session. On the Axol hub that's a USB disconnect — EMI from the
# arms' motor drivers can kick the adapter off the bus — after which the
# gs_usb netdevs are destroyed and recreated when the adapter re-enumerates.
_IFACE_LOST_ERRNOS = {
    errno.ENODEV,  # netdev unregistered (adapter dropped off the USB bus)
    errno.ENXIO,
    errno.ENETDOWN,  # netdev exists but its link is (still) down
    errno.ENETUNREACH,
    errno.EBADF,  # socket torn down under us
}

# Cadence for probing /sys while waiting for the interface to come back; the
# hotplug bring-up installed by `axol can.setup` normally has a re-enumerated
# adapter configured and up again within a second or two.
_RECONNECT_POLL_S = 0.2
# A long wait means the adapter is really unplugged or the hotplug bring-up
# is missing — remind operators at this cadence rather than spamming.
_RECONNECT_REMIND_S = 30.0

# A send failing with ENOBUFS means the interface TX queue is full. Two very
# different situations produce it:
#
#  - Transient host-side congestion: both channels share one dual-channel
#    gs_usb adapter, so a brief stall of its USB pipe (e.g. camera traffic
#    during data collection) backs the queue up while the bus itself is
#    healthy and draining. The frame is simply dropped, like a send during
#    a lost interface.
#  - A dead bus: no node is ACKing frames — on the Axol that's the e-stop
#    cutting motor power mid-stream. Everything queued behind the dead bus is
#    a stale motion command that would replay all at once when power returns
#    (the arm suddenly snapping back to its pre-e-stop position), so the
#    queue must be purged and sends held off until the bus is proven alive.
#
# The two are told apart by duration: at 1 Mbit/s a healthy full queue
# (txqueuelen 512) drains in ~65 ms, so ENOBUFS persisting across sends for
# longer than this means nothing is draining — the bus is dead.
_STALL_DETECT_S = 1.0

# Transient overflows come in bursts at telemetry rates; warn at most this
# often so congestion doesn't flood the log with one line per dropped frame.
_TX_FULL_WARN_INTERVAL_S = 5.0

# Arbitration ID of the aliveness probe frame. Unused by both motor protocols
# (Damiao: motor IDs 0x01-0x08 plus 0x100/0x200/0x300 offsets, feedback
# 0x11-0x18, register access 0x7FF; MyActuator: 0x140/0x240/0x400/0x500 + ID),
# so every device on the bus ignores it — but any powered CAN node still ACKs
# it at the link layer, which is exactly the signal we need.
_PROBE_ID = 0x7F0
_PROBE_POLL_S = 0.25
# One flush of the bring-up script flaps *every* channel, so when both arms'
# buses stall together (they share the e-stop), the second one can skip it.
_FLUSH_DEDUPE_S = 3.0

_flush_lock = asyncio.Lock()
_last_flush_monotonic = 0.0


def _error_code(exc: BaseException) -> int | None:
    """Extract the errno from a python-can or OS-level exception."""
    code = getattr(exc, "error_code", None)  # python-can CanOperationError
    if code is None:
        code = getattr(exc, "errno", None)
    return code


def _iface_lost(exc: BaseException) -> bool:
    """True when *exc* is a socket failure meaning the interface went away."""
    return _error_code(exc) in _IFACE_LOST_ERRNOS


def _tx_queue_full(exc: BaseException) -> bool:
    """True when *exc* means the interface's TX queue is full (``ENOBUFS``).

    python-can surfaces this two ways: ``sendto`` failing with ``ENOBUFS``
    (errno attached), or ``select`` never reporting the socket writable, in
    which case it raises a bare ``CanOperationError("Transmit buffer full")``
    with no error code. Whether that's transient host-side congestion or a
    dead bus is decided by how long it persists — see :data:`_STALL_DETECT_S`.
    """
    code = _error_code(exc)
    if code == errno.ENOBUFS:
        return True
    return (
        code is None
        and isinstance(exc, can.CanOperationError)
        and "buffer full" in str(exc).lower()
    )


def _iface_is_up(channel: str) -> bool:
    """True when *channel* exists and is administratively up (IFF_UP)."""
    try:
        flags = int((Path("/sys/class/net") / channel / "flags").read_text(), 16)
    except (OSError, ValueError):
        return False
    return bool(flags & 0x1)


class CanBus:
    """Async wrapper around a python-can SocketCAN bus.

    A single instance is shared between all Motor objects on the same physical
    interface.  The background reader task dispatches every incoming frame to
    registered listeners.

    A lost interface — the Axol hub dropping off the USB bus and re-enumerating
    — is survived transparently: the reader loop reopens the socket once the
    interface is back up, keeping every registered listener attached. Sends
    during the gap are dropped, so request/response commands fall into their
    usual timeout path (``MotorError``) and resume after the reconnect.
    A momentarily full TX queue (``ENOBUFS`` under host-side USB congestion)
    is handled the same way, minus the reconnect: the frame is dropped and
    the next command cycle proceeds normally.

    A stalled bus — the e-stop cutting motor power so nothing ACKs frames and
    the kernel TX queue fills (``ENOBUFS`` persisting past
    :data:`_STALL_DETECT_S`, past any transient congestion) — is handled the
    same way, with two extra steps: the queued (now stale) motion commands are purged by
    flapping the interface, and sends stay dropped until a probe frame is
    actually ACKed on the wire again. Without the purge, up to ``txqueuelen``
    stale position commands replay the instant the arm is powered back on,
    snapping it to its pre-e-stop pose.

    Use as an async context manager:

        async with CanBus("can_alm_axol_l") as bus:
            motor = Motor(bus, Joint.SHOULDER_1)
            ...
    """

    def __init__(self, channel: str) -> None:
        """Open a SocketCAN socket on the given interface.

        The background reader loop is not started until :meth:`start` (or
        ``async with``) is called.

        Args:
            channel: SocketCAN interface name, e.g. ``"can_alm_axol_l"``.
        """
        self._channel = channel
        self._bus: can.BusABC | None = can.Bus(channel=channel, bustype="socketcan")
        self._listeners: list[Callable[[can.Message], None]] = []
        self._reader_task: asyncio.Task | None = None
        # Set while the interface is gone (USB drop): sends are dropped and
        # the reader loop is waiting to reopen the socket.
        self._lost = False
        # Set while the bus is stalled (e-stop cut motor power, nothing ACKs):
        # sends are dropped until a probe frame is ACKed on the wire again.
        self._stalled = False
        # Poked by socket readability or a failed send, so the idle reader
        # wakes both for frames and for lost-interface handling.
        self._wake = asyncio.Event()
        # Monotonic deadline for the next TX-queue-full warning (rate limit).
        self._next_tx_full_warn = 0.0
        # Loop time of the first ENOBUFS in the current burst (None outside
        # one); a successful send resets it. Overflow persisting longer than
        # _STALL_DETECT_S means the queue isn't draining — bus dead (e-stop).
        self._enobufs_since: float | None = None

    async def start(self) -> None:
        """Start the background frame-dispatch loop. Idempotent.

        A second call while the loop is already running is a no-op, so flows
        that compose bus-starting steps (``Axol.connect`` querying state and
        then delegating to ``attach``/``enable``) don't spawn duplicate
        reader tasks.
        """
        if self._reader_task is not None and not self._reader_task.done():
            return
        self._reader_task = asyncio.create_task(
            self._reader_loop(),
            name=f"can_reader:{self._channel}",
        )

    async def close(self) -> None:
        """Stop the reader loop and shut down the socket."""
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._bus is not None:
            self._bus.shutdown()
            self._bus = None

    async def __aenter__(self) -> CanBus:
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    def mute_rx(self) -> None:
        """Stop receiving frames at the kernel: zero-length CAN_RAW_FILTER.

        Used by the rt path once the Rust core owns the bus: SocketCAN
        broadcasts every frame to every open socket, so at 240 Hz Python
        would otherwise dispatch ~7,700 frames/s through python-can and
        asyncio just to fill caches it now gets from the core's telemetry
        packets. With zero filters the kernel delivers nothing — the reader
        task stays parked with no per-frame work. TX is unaffected.
        Reversed by :meth:`unmute_rx` (needed before any request/reply
        exchange, e.g. the belt-and-braces disable).
        """
        if self._bus is not None:
            self._bus.socket.setsockopt(_SOL_CAN_RAW, _CAN_RAW_FILTER, b"")

    def unmute_rx(self) -> None:
        """Restore the match-everything CAN_RAW_FILTER after :meth:`mute_rx`."""
        if self._bus is not None:
            self._bus.socket.setsockopt(
                _SOL_CAN_RAW, _CAN_RAW_FILTER, struct.pack("=II", 0, 0)
            )

    def _add_listener(self, listener: Callable[[can.Message], None]) -> None:
        self._listeners.append(listener)

    async def _send(self, arbitration_id: int, data: bytes) -> None:
        if self._lost or self._stalled or self._bus is None:
            # Interface is gone (USB drop) or the bus is stalled (e-stop) —
            # drop the frame; motor commands time out upstream and resume
            # once the reader has recovered the bus.
            return
        msg = can.Message(
            arbitration_id=arbitration_id, data=data, is_extended_id=False
        )
        # Send synchronously — SocketCAN send is fast non-blocking at OS level.
        # This prevents asyncio thread pool starvation at high telemetry rates.
        try:
            self._bus.send(msg)
        except (can.CanError, OSError) as exc:
            if _tx_queue_full(exc):
                self._on_tx_queue_full(exc)
            elif _iface_lost(exc):
                self._mark_lost(exc)
            else:
                raise
        else:
            self._enobufs_since = None

    def _on_tx_queue_full(self, exc: BaseException) -> None:
        """Classify an ``ENOBUFS`` send: transient congestion or a dead bus.

        The frame is dropped either way — commands time out upstream
        (``MotorError``) and resume. Transient host-side congestion drains
        within milliseconds, so overflow persisting across sends for
        :data:`_STALL_DETECT_S` means no node is ACKing (e-stop) and stall
        recovery (purge + probe) takes over.
        """
        now = asyncio.get_running_loop().time()
        if self._enobufs_since is None:
            self._enobufs_since = now
        elif now - self._enobufs_since >= _STALL_DETECT_S:
            self._enobufs_since = None
            self._mark_stalled(exc)
            return
        if now >= self._next_tx_full_warn:
            self._next_tx_full_warn = now + _TX_FULL_WARN_INTERVAL_S
            _logger.warning(
                "CAN %s: TX queue full (%s) — dropping frame(s); "
                "consider raising txqueuelen if this persists",
                self._channel,
                exc,
            )

    def _mark_lost(self, cause: BaseException) -> None:
        """Flag the interface as gone and wake the reader to start reconnecting."""
        if self._lost:
            return
        self._lost = True
        _logger.warning(
            "CAN %s: interface lost (%s) — waiting for it to come back",
            self._channel,
            cause,
        )
        self._wake.set()

    def _mark_stalled(self, cause: BaseException) -> None:
        """Flag the bus as stalled and wake the reader to purge the TX queue."""
        if self._stalled:
            return
        self._stalled = True
        # Also flag lost so the pump exits and hands control to the recovery
        # path; _reconnect clears it once the socket is reopened, while
        # _stalled keeps sends dropped until the bus is proven alive.
        self._lost = True
        _logger.warning(
            "CAN %s: TX queue full (%s) — no node is ACKing frames, so the "
            "motors are most likely unpowered (e-stop?). Dropping commands and "
            "purging the queued ones so the arm doesn't replay them when power "
            "returns.",
            self._channel,
            cause,
        )
        self._wake.set()

    async def _reader_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self._pump(loop)
            if self._stalled:
                await self._flush_tx_queue()
            await self._reconnect(loop)
            if self._stalled:
                await self._wait_bus_alive(loop)

    async def _pump(self, loop: asyncio.AbstractEventLoop) -> None:
        """Dispatch frames until the interface is lost (or the task is cancelled)."""
        assert self._bus is not None
        fd = self._bus.fileno()
        loop.add_reader(fd, self._wake.set)
        try:
            while not self._lost:
                try:
                    msg: can.Message | None = self._bus.recv(timeout=0)
                except Exception as exc:  # noqa: BLE001 - classified below
                    if _iface_lost(exc):
                        self._mark_lost(exc)
                        return
                    _logger.warning("CAN reader loop warning: %s", exc)
                    await asyncio.sleep(0.01)
                    continue
                if msg is not None:
                    for listener in self._listeners:
                        try:
                            listener(msg)
                        except Exception as e:
                            _logger.error(
                                "CAN listener %s error: %s", listener.__name__, e
                            )
                    continue
                # No data — wait until the socket is readable instead of
                # polling on a fixed 1ms timer, cutting up to 1ms of response
                # latency per round-trip. The timeout is a safety net: a
                # failed send can flag the interface lost without the socket
                # ever becoming readable again.
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=0.5)
                except TimeoutError:
                    pass
        finally:
            loop.remove_reader(fd)

    async def _reconnect(self, loop: asyncio.AbstractEventLoop) -> None:
        """Reopen the socket once the interface is back up; waits indefinitely."""
        started = loop.time()
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except Exception as exc:  # noqa: BLE001 - dead-socket teardown is best-effort
                _logger.debug("CAN %s: socket shutdown failed: %s", self._channel, exc)
            self._bus = None
        next_reminder = started + _RECONNECT_REMIND_S
        while True:
            if _iface_is_up(self._channel):
                try:
                    self._bus = can.Bus(channel=self._channel, bustype="socketcan")
                    break
                except (can.CanError, OSError) as exc:
                    # Raced the interface going away again — keep waiting.
                    _logger.debug("CAN %s: reopen failed: %s", self._channel, exc)
            if loop.time() >= next_reminder:
                _logger.warning(
                    "CAN %s: still waiting for the interface (%.0fs) — is the "
                    "adapter plugged in, and the hotplug bring-up installed "
                    "(re-run `axol can.setup` once to add it)?",
                    self._channel,
                    loop.time() - started,
                )
                next_reminder = loop.time() + _RECONNECT_REMIND_S
            await asyncio.sleep(_RECONNECT_POLL_S)
        self._lost = False
        _logger.warning(
            "CAN %s: interface is back after %.1fs — resuming",
            self._channel,
            loop.time() - started,
        )

    async def _flush_tx_queue(self) -> None:
        """Purge the stale frames queued behind a dead (e-stopped) bus.

        Closing the socket does not help: frames already handed to the
        interface queue survive socket close and transmit the moment a
        powered-on motor ACKs again. Only taking the interface down drops
        them. Robots configured by ``axol can.setup`` have the bring-up
        script, which flaps both arm-hub channels together (flapping one at
        a time can wedge the adapter's RX path) and leaves them configured
        and up; without it, flap just this channel.
        """
        global _last_flush_monotonic
        async with _flush_lock:
            script_exists = CAN_BRINGUP_SCRIPT.exists()
            if (
                script_exists
                and time.monotonic() - _last_flush_monotonic < _FLUSH_DEDUPE_S
            ):
                # The other arm's bus just ran the script, which flapped this
                # channel too — the queue is already empty.
                return
            try:
                if script_exists:
                    await asyncio.to_thread(
                        run_root, ["bash", str(CAN_BRINGUP_SCRIPT)], check=True
                    )
                else:
                    await asyncio.to_thread(
                        run_root,
                        ["ip", "link", "set", self._channel, "down"],
                        check=True,
                    )
                    await asyncio.to_thread(
                        run_root, ["ip", "link", "set", self._channel, "up"], check=True
                    )
            except (RuntimeError, OSError) as exc:
                _logger.error(
                    "CAN %s: could not purge the TX queue (%s) — stale motion "
                    "commands will replay when the motors power back on. Flap "
                    "the interface (`axol can.setup`, or ip link set %s "
                    "down/up) before re-powering the arm.",
                    self._channel,
                    exc,
                    self._channel,
                )
                return
            _last_flush_monotonic = time.monotonic()
            _logger.warning(
                "CAN %s: purged the stale TX queue — commands stay disabled "
                "until the motors are powered back on",
                self._channel,
            )

    async def _wait_bus_alive(self, loop: asyncio.AbstractEventLoop) -> None:
        """Hold sends off until a probe frame is ACKed on the wire again.

        A CAN frame only completes once *some* powered node ACKs it, so a
        single probe on an ID no motor listens to (:data:`_PROBE_ID`) detects
        power returning without commanding anything: it sits at the head of
        the (freshly purged) TX queue, silently retrying, and its own-message
        echo arrives the moment the transmission actually completes. Only
        then are application sends re-enabled — everything commanded while
        the bus was dead stays dropped instead of becoming the next stale
        replay.
        """
        started = loop.time()
        next_reminder = started + _RECONNECT_REMIND_S
        try:
            probe_bus = can.Bus(
                channel=self._channel,
                bustype="socketcan",
                receive_own_messages=True,
                can_filters=[{"can_id": _PROBE_ID, "can_mask": 0x7FF}],
            )
        except (can.CanError, OSError) as exc:
            self._mark_lost(exc)
            return
        # 8 zero bytes rather than an empty payload: motor-driver listeners
        # index into msg.data, and this frame loops back to their socket too.
        probe = can.Message(
            arbitration_id=_PROBE_ID, data=bytes(8), is_extended_id=False
        )
        sent = False
        try:
            while True:
                try:
                    if not sent:
                        probe_bus.send(probe)
                        sent = True
                    if sent and probe_bus.recv(timeout=0) is not None:
                        break
                except (can.CanError, OSError) as exc:
                    if _iface_lost(exc):
                        self._mark_lost(exc)
                        return
                    if not _tx_queue_full(exc):
                        raise
                    # Queue still full (the purge failed, e.g. no root) — the
                    # probe doesn't fit yet; keep trying so recovery is still
                    # detected once the stale frames drain on power-up.
                if loop.time() >= next_reminder:
                    _logger.warning(
                        "CAN %s: still waiting for a motor to ACK (%.0fs) — "
                        "commands stay disabled until the arm is powered on",
                        self._channel,
                        loop.time() - started,
                    )
                    next_reminder = loop.time() + _RECONNECT_REMIND_S
                await asyncio.sleep(_PROBE_POLL_S)
        finally:
            probe_bus.shutdown()
        self._stalled = False
        _logger.warning(
            "CAN %s: bus is ACKing again after %.1fs — resuming commands",
            self._channel,
            loop.time() - started,
        )
