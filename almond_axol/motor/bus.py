"""Async SocketCAN bus shared across all Motor instances on one physical interface."""

from __future__ import annotations

import asyncio
import errno
import logging
from pathlib import Path
from typing import Callable

import can

_logger = logging.getLogger(__name__)

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

# TX-queue overflows come in bursts at telemetry rates; warn at most this
# often so a stall doesn't flood the log with one line per dropped frame.
_TX_FULL_WARN_INTERVAL_S = 5.0


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
    """True when *exc* means the interface's TX queue is momentarily full.

    Happens under host-side congestion only (``sendto`` returns ``ENOBUFS``):
    on the Axol hub both CAN channels share a single dual-channel gs_usb
    adapter, so a brief stall of its USB pipe — e.g. USB contention from the
    cameras during data collection — backs the txqueue up. The bus itself is
    healthy, so this is transient and the frame can be dropped like a send
    during a lost interface.
    """
    return _error_code(exc) == errno.ENOBUFS


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
        # Poked by socket readability or a failed send, so the idle reader
        # wakes both for frames and for lost-interface handling.
        self._wake = asyncio.Event()
        # Monotonic deadline for the next TX-queue-full warning (rate limit).
        self._next_tx_full_warn = 0.0

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

    def _add_listener(self, listener: Callable[[can.Message], None]) -> None:
        self._listeners.append(listener)

    async def _send(self, arbitration_id: int, data: bytes) -> None:
        if self._lost or self._bus is None:
            # Interface is gone (USB drop) — drop the frame; motor commands
            # time out upstream and resume once the reader has reconnected.
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
                # Transient host-side overflow — drop the frame just like a
                # send during a lost interface: commands time out upstream
                # (MotorError) and the next cycle resumes. No reconnect
                # needed; the queue drains on its own within milliseconds.
                now = asyncio.get_running_loop().time()
                if now >= self._next_tx_full_warn:
                    self._next_tx_full_warn = now + _TX_FULL_WARN_INTERVAL_S
                    _logger.warning(
                        "CAN %s: TX queue full (%s) — dropping frame(s); "
                        "consider raising txqueuelen if this persists",
                        self._channel,
                        exc,
                    )
                return
            if not _iface_lost(exc):
                raise
            self._mark_lost(exc)

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

    async def _reader_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self._pump(loop)
            await self._reconnect(loop)

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
