from __future__ import annotations

import asyncio
import logging
from typing import Callable

import can

_logger = logging.getLogger(__name__)


class CanBus:
    """
    Async wrapper around a python-can SocketCAN bus.

    A single instance is shared between all Motor objects on the same physical
    interface.  The background reader task dispatches every incoming frame to
    registered listeners.

    Use as an async context manager:

        async with CanBus("can_alm_axol_l") as bus:
            motor = Motor(bus, Joint.SHOULDER_1)
            ...
    """

    def __init__(self, channel: str) -> None:
        self._bus = can.Bus(channel=channel, bustype="socketcan")
        self._listeners: list[Callable[[can.Message], None]] = []
        self._reader_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background frame-dispatch loop."""
        self._reader_task = asyncio.create_task(
            self._reader_loop(),
            name=f"can_reader:{self._bus.channel_info}",
        )

    async def close(self) -> None:
        """Stop the reader loop and shut down the socket."""
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        self._bus.shutdown()

    async def __aenter__(self) -> CanBus:
        await self.start()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    def _add_listener(self, listener: Callable[[can.Message], None]) -> None:
        self._listeners.append(listener)

    async def _send(self, arbitration_id: int, data: bytes) -> None:
        msg = can.Message(
            arbitration_id=arbitration_id, data=data, is_extended_id=False
        )
        # Retry on ENOBUFS: the gs_usb kernel driver has only 10 tx_context slots.
        # A burst of 8 motor frames can transiently exhaust them if USB echo latency
        # drifts slightly beyond the cycle period.  Waiting 2ms lets pending echoes
        # drain before the next attempt; 5 retries covers any realistic jitter.
        for attempt in range(5):
            try:
                self._bus.send(msg)
                return
            except can.CanOperationError:
                if attempt == 4:
                    raise
                _logger.debug(
                    "CAN TX buffer full (attempt %d/5), retrying", attempt + 1
                )
                await asyncio.sleep(0.002)

    async def _reader_loop(self) -> None:
        while True:
            try:
                # Non-blocking read directly in the async loop saves overhead
                # and avoids threadpool contention with 'send' calls.
                msg: can.Message | None = self._bus.recv(timeout=0)
                if msg is not None:
                    for listener in self._listeners:
                        try:
                            listener(msg)
                        except Exception as e:
                            _logger.error(
                                "CAN listener %s error: %s", listener.__name__, e
                            )
                else:
                    # Give control back to event loop if no data
                    await asyncio.sleep(0.001)
            except asyncio.CancelledError:
                break
            except Exception as e:
                _logger.warning("CAN reader loop warning: %s", e)
                await asyncio.sleep(0.01)
