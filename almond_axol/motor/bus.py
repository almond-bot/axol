"""Async CAN transport backed exclusively by the Rust ``axol-rt`` proxy."""

from __future__ import annotations

import asyncio
import logging
import os
import struct
import subprocess
from collections.abc import Callable

import can

_logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT_S = 5.0
_READY_TIMEOUT_S = 5.0
_FRAME = struct.Struct("<IB8sQ")  # arbitration id, DLC, data, Unix timestamp ns


class CanBus:
    """Frame-level CAN API whose SocketCAN owner is a Rust subprocess.

    Existing Python motor protocol implementations remain useful for
    calibration, firmware, and diagnostics, but Python never opens a CAN
    socket. ``axol-rt proxy`` owns the interface and forwards raw frames over
    a length-prefixed Unix socket. Production control closes this maintenance
    proxy before the realtime core takes ownership, then reopens it only after
    the core has disarmed.
    """

    def __init__(self, channel: str) -> None:
        self._channel = channel
        self._socket_path = f"/tmp/axol-can-{os.getpid()}-{id(self):x}.sock"
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._listeners: list[Callable[[can.Message], None]] = []
        self._ready = asyncio.Event()
        self._closed_reason: str | None = None

    async def start(self) -> None:
        """Start the Rust transport and wait until it owns the CAN socket."""
        if self._reader_task is not None and not self._reader_task.done():
            return

        # Import lazily: rt.robot imports this module, so resolving the binary
        # at module import time would create a package cycle.
        from ..rt.link import find_binary

        self._ready.clear()
        self._closed_reason = None
        try:
            self._proc = subprocess.Popen(
                [
                    find_binary(),
                    "proxy",
                    "--socket",
                    self._socket_path,
                    "--iface",
                    self._channel,
                ]
            )
        except OSError as exc:
            raise can.CanInitializationError(
                f"could not start axol-rt proxy for {self._channel}: {exc}"
            ) from exc
        deadline = asyncio.get_running_loop().time() + _CONNECT_TIMEOUT_S
        while True:
            try:
                self._reader, self._writer = await asyncio.open_unix_connection(
                    self._socket_path
                )
                break
            except (ConnectionRefusedError, FileNotFoundError):
                if self._proc.poll() is not None:
                    returncode = self._proc.returncode
                    await self.close()
                    raise can.CanInitializationError(
                        f"axol-rt proxy exited while opening {self._channel} "
                        f"(code {returncode})"
                    ) from None
                if asyncio.get_running_loop().time() >= deadline:
                    await self.close()
                    raise can.CanInitializationError(
                        f"timed out starting axol-rt proxy for {self._channel}"
                    ) from None
                await asyncio.sleep(0.05)

        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"can_proxy_reader:{self._channel}"
        )
        try:
            await asyncio.wait_for(self._ready.wait(), _READY_TIMEOUT_S)
        except TimeoutError:
            await self.close()
            raise can.CanInitializationError(
                f"axol-rt proxy for {self._channel} did not become ready"
            ) from None
        if self._closed_reason is not None:
            reason = self._closed_reason
            await self.close()
            raise can.CanInitializationError(reason)

    async def close(self) -> None:
        """Close the proxy connection and reap its Rust process."""
        if self._writer is not None and not self._writer.is_closing():
            try:
                self._send_message(b"Q")
                await self._writer.drain()
            except (ConnectionError, RuntimeError):
                pass
            self._writer.close()
            try:
                await self._writer.wait_closed()
            except ConnectionError:
                pass
            self._writer = None
        if self._reader_task is not None:
            if self._reader_task is not asyncio.current_task():
                try:
                    await asyncio.wait_for(self._reader_task, 1.0)
                except TimeoutError:
                    self._reader_task.cancel()
                    try:
                        await self._reader_task
                    except asyncio.CancelledError:
                        pass
            self._reader_task = None
        self._reader = None
        if self._proc is not None:
            if self._proc.poll() is None:
                try:
                    await asyncio.to_thread(self._proc.wait, 1.0)
                except subprocess.TimeoutExpired:
                    self._proc.terminate()
                    try:
                        await asyncio.to_thread(self._proc.wait, 2.0)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                        await asyncio.to_thread(self._proc.wait)
            self._proc = None
        try:
            os.unlink(self._socket_path)
        except OSError:
            pass

    async def __aenter__(self) -> CanBus:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    def _add_listener(self, listener: Callable[[can.Message], None]) -> None:
        self._listeners.append(listener)

    async def _send(self, arbitration_id: int, data: bytes) -> None:
        """Forward one standard CAN frame to the Rust-owned socket."""
        if self._writer is None or self._writer.is_closing():
            raise can.CanOperationError(
                f"axol-rt proxy for {self._channel} is not connected"
            )
        if not 0 <= arbitration_id <= 0x7FF:
            raise ValueError(
                f"standard CAN arbitration id out of range: {arbitration_id:#x}"
            )
        if len(data) > 8:
            raise ValueError(f"classic CAN payload exceeds 8 bytes: {len(data)}")
        payload = (
            b"S"
            + struct.pack("<IB", arbitration_id, len(data))
            + data.ljust(8, b"\x00")
        )
        self._send_message(payload)
        try:
            await self._writer.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise can.CanOperationError(
                f"axol-rt proxy for {self._channel} disconnected"
            ) from exc

    def _send_message(self, payload: bytes) -> None:
        if self._writer is None or self._writer.is_closing():
            raise RuntimeError(f"CAN proxy for {self._channel} is not connected")
        self._writer.write(struct.pack("<I", len(payload)) + payload)

    async def _read_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                header = await self._reader.readexactly(4)
                (size,) = struct.unpack("<I", header)
                if size <= 0 or size > 1024:
                    raise RuntimeError(f"invalid axol-rt proxy message size {size}")
                payload = await self._reader.readexactly(size)
                if payload == b"R":
                    self._ready.set()
                    continue
                if payload[:1] == b"E":
                    _logger.warning(
                        "axol-rt proxy: %s",
                        payload[1:].decode("utf-8", errors="replace"),
                    )
                    continue
                if payload[:1] != b"F" or len(payload) != 1 + _FRAME.size:
                    _logger.warning("axol-rt proxy: invalid message tag/size")
                    continue
                arbitration_id, dlc, raw, timestamp_ns = _FRAME.unpack(payload[1:])
                if dlc > 8:
                    _logger.warning("axol-rt proxy: invalid CAN DLC %d", dlc)
                    continue
                message = can.Message(
                    timestamp=timestamp_ns / 1e9,
                    arbitration_id=arbitration_id,
                    data=raw[:dlc],
                    is_extended_id=False,
                )
                for listener in self._listeners:
                    try:
                        listener(message)
                    except Exception as exc:  # noqa: BLE001 - isolate listeners
                        name = getattr(listener, "__name__", repr(listener))
                        _logger.error("CAN listener %s error: %s", name, exc)
        except (asyncio.IncompleteReadError, ConnectionResetError) as exc:
            if self._proc is not None and self._proc.poll() is not None:
                self._closed_reason = (
                    f"axol-rt proxy for {self._channel} exited "
                    f"(code {self._proc.returncode})"
                )
            else:
                self._closed_reason = (
                    f"axol-rt proxy for {self._channel} disconnected: {exc}"
                )
            self._ready.set()
        except asyncio.CancelledError:
            raise
