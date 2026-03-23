from __future__ import annotations

import asyncio
from typing import Callable

import can


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
        await asyncio.to_thread(self._bus.send, msg)

    async def _reader_loop(self) -> None:
        while True:
            msg: can.Message | None = await asyncio.to_thread(self._bus.recv, 0.02)
            if msg is not None:
                for listener in self._listeners:
                    listener(msg)
