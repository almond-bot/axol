"""Process + socket link to the ``axol-rt`` realtime core.

Owns the ``axol-rt serve`` subprocess and the length-prefixed Unix-socket
protocol (see ``rust/axol-rt/src/serve.rs`` for the wire format and the
core's safety semantics).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import struct
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

_logger = logging.getLogger(__name__)

# One telemetry entry per slot: pos (rad), vel (rad/s), tau (Nm), age_us.
_FEEDBACK_SLOTS = 8
_FEEDBACK_FMT = struct.Struct("<BB" + "3dI" * _FEEDBACK_SLOTS)

#: One parsed telemetry slot: (position, velocity, torque, receive_ts) —
#: receive_ts is the frame's CAN receive time reconstructed on this host's
#: ``time.time()`` clock from the packet's per-slot age.
FeedbackSlot = tuple[float, float, float, float]

_CONNECT_TIMEOUT_S = 5.0
# PREP includes the MyActuator reset settle (~2.2 s per bus, run serially).
_PREP_TIMEOUT_S = 15.0
_ARM_TIMEOUT_S = 15.0


def find_binary() -> str:
    """Locate ``axol-rt``: env override, this checkout's build, then PATH."""
    env = os.environ.get("AXOL_RT_BIN")
    if env:
        return env
    repo_build = (
        Path(__file__).resolve().parents[2]
        / "rust"
        / "axol-rt"
        / "target"
        / "release"
        / "axol-rt"
    )
    if repo_build.exists():
        return str(repo_build)
    on_path = shutil.which("axol-rt")
    if on_path:
        return on_path
    raise FileNotFoundError(
        "axol-rt binary not found — build it with "
        "`cargo build --release` in rust/axol-rt, put it on PATH, or set "
        "AXOL_RT_BIN"
    )


class RtLinkError(RuntimeError):
    """The realtime core reported a fault or went away."""


class RtLink:
    """One ``axol-rt serve`` subprocess and its socket connection."""

    def __init__(self, binary: str | None = None) -> None:
        self._binary = binary or find_binary()
        self._socket_path = f"/tmp/axol-rt-{os.getpid()}.sock"
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._states: asyncio.Queue[str] = asyncio.Queue()
        # Called for each telemetry packet: (side, {slot: FeedbackSlot}).
        self.on_feedback: Callable[[int, dict[int, FeedbackSlot]], None] | None = None

    async def start(self) -> None:
        """Launch the core and connect. The core is idle until configured."""
        # stdout/stderr inherit the console: the core logs little, and what
        # it does log (bring-up, faults) belongs in the teleop output.
        self._proc = subprocess.Popen(
            [self._binary, "serve", "--socket", self._socket_path]
        )
        deadline = asyncio.get_running_loop().time() + _CONNECT_TIMEOUT_S
        while True:
            try:
                self._reader, self._writer = await asyncio.open_unix_connection(
                    self._socket_path
                )
                break
            except (ConnectionRefusedError, FileNotFoundError):
                if self._proc.poll() is not None:
                    raise RtLinkError(
                        f"axol-rt exited during startup (code {self._proc.returncode})"
                    ) from None
                if asyncio.get_running_loop().time() > deadline:
                    raise RtLinkError("timed out connecting to axol-rt") from None
                await asyncio.sleep(0.05)
        self._reader_task = asyncio.create_task(
            self._read_loop(), name="rt-link-reader"
        )

    async def _read_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                header = await self._reader.readexactly(4)
                (size,) = struct.unpack("<I", header)
                payload = await self._reader.readexactly(size)
                tag = payload[:1]
                if tag == b"F":
                    if self.on_feedback is not None:
                        side, slots = self._parse_feedback(payload)
                        self.on_feedback(side, slots)
                    continue
                body = payload[1:].decode("utf-8", errors="replace")
                if tag == b"S":
                    _logger.info("axol-rt: %s", body)
                    self._states.put_nowait(body)
                elif tag == b"L":
                    _logger.info("axol-rt: %s", body)
                else:
                    _logger.warning("axol-rt: unknown message tag %r", tag)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            _logger.info("axol-rt: connection closed")
        except asyncio.CancelledError:
            raise

    @staticmethod
    def _parse_feedback(payload: bytes) -> tuple[int, dict[int, FeedbackSlot]]:
        """Decode one `F` telemetry packet (layout: see ``build_feedback``
        in ``serve.rs`` and its ``feedback_packet_layout`` test).

        Per-slot receive timestamps are reconstructed from the packet's
        age_us fields against ``time.time()`` now — within socket transit
        (~0.1 ms) of the frames' true CAN receive times, and mutually
        consistent across slots, which is all downstream differentiation
        needs.
        """
        vals = _FEEDBACK_FMT.unpack(payload[1:])
        side, mask = vals[0], vals[1]
        now = time.time()
        slots: dict[int, FeedbackSlot] = {}
        for i in range(_FEEDBACK_SLOTS):
            if not mask & (1 << i):
                continue
            pos, vel, tau, age_us = vals[2 + i * 4 : 6 + i * 4]
            slots[i] = (pos, vel, tau, now - age_us / 1e6)
        return side, slots

    def _send(self, payload: bytes) -> None:
        if self._writer is None or self._writer.is_closing():
            raise RtLinkError("axol-rt link is not connected")
        self._writer.write(struct.pack("<I", len(payload)) + payload)

    async def _await_state(self, expected: str, timeout: float) -> None:
        """Wait for a state message; a ``fault:`` state raises."""
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise RtLinkError(f"timed out waiting for {expected!r} from axol-rt")
            try:
                state = await asyncio.wait_for(self._states.get(), remaining)
            except asyncio.TimeoutError:
                raise RtLinkError(
                    f"timed out waiting for {expected!r} from axol-rt"
                ) from None
            if state == expected:
                return
            if state.startswith("fault:"):
                raise RtLinkError(f"axol-rt: {state}")
            # Unrelated state (e.g. a stats line routed as state) — keep waiting.

    async def configure(self, config_text: str) -> None:
        self._send(b"C" + config_text.encode())
        await self._await_state("config-ok", 5.0)

    async def prep(self) -> None:
        self._send(b"P")
        await self._await_state("prepped", _PREP_TIMEOUT_S)

    async def arm(self) -> None:
        self._send(b"A")
        await self._await_state("armed", _ARM_TIMEOUT_S)

    async def disarm(self) -> None:
        self._send(b"D")
        await self._await_state("disarmed", 5.0)

    def send_target(
        self,
        side: int,
        seq: int,
        cmds: list[tuple[float, ...]],
    ) -> None:
        """Ship one arm's 8 slot tuples (p_des, mode, kp, kd, gravity t_ff,
        kd_host, damp_w0, damp_q, j_eff; the gripper slot repurposes the
        first three as target/speed/torque). mode ≥ 0.5 runs the core's
        tracker + friction/inertia terms, 0 is passthrough (gravity comp).
        Sync — safe to call from the event loop; the write is buffered."""
        payload = struct.pack("<cBI", b"T", side, seq & 0xFFFFFFFF) + b"".join(
            struct.pack("<9d", *cmd) for cmd in cmds
        )
        self._send(payload)

    async def close(self) -> None:
        """Tear down the link and the core process."""
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()  # SIGTERM → the core disables and exits
                try:
                    await asyncio.to_thread(self._proc.wait, 5.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    await asyncio.to_thread(self._proc.wait)
            self._proc = None
        try:
            os.unlink(self._socket_path)
        except OSError:
            pass
