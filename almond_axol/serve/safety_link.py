"""Always-on link to Jelly's safety board for the control panel's software E-stop.

Some Jelly builds carry the Almond Safety board on the lift controller's CAN
bus (see :mod:`almond_axol.robot.safety`). This link keeps a
:class:`~almond_axol.robot.safety.SafetyBoard` open on that bus for the whole
life of the server, so the panel can show the E-stop chain and cut power at
any moment, including in the middle of an operation.

Unlike the arm and Jelly links it is not handed over to tasks. It never
commands a motor, and a software stop has to work most of all while a task
is running. Its traffic (a status poll at 5 Hz, on 0x440/0x441) is clear of
every ID a task uses, so it coexists with whatever owns the bus. The one
exception is CAN discovery, which renames and resets adapters:
:meth:`SafetyLink.pause` closes the board for it and :meth:`SafetyLink.resume`
lets the supervisor reopen it.

Nothing needs connecting by hand. A supervisor task opens the board whenever
the lift bus exists and is up, and reopens it after the interface goes away
(an adapter unplugged, a bus flapped). On a Jelly without a safety board the
link polls once a second, nothing answers, and :meth:`SafetyLink.status`
reports ``detected: False``, which keeps the panel's control hidden.

The link runs on its own asyncio event loop in a daemon thread, like the
other idle links.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from ..robot.safety import (
    SafetyBoard,
    SafetyError,
    SafetyStatus,
    resolve_safety_channel,
)

_logger = logging.getLogger(__name__)

# Supervisor cadence: how soon a new or returning bus gets its board opened.
_SUPERVISE_S = 1.0
# The board counts as answering while its last status is this fresh (it is
# polled at 5 Hz once it answers).
_REACHABLE_S = 1.0
# Upper bound on one stop / start request, on top of the driver's own
# confirmation timeout.
_COMMAND_TIMEOUT_S = 5.0


def status_payload(status: SafetyStatus) -> dict[str, Any]:
    """One status frame as the panel's JSON."""
    return {
        "estopOk": status.estop_ok,
        "latched": status.latched,
        "outputOn": status.output_on,
        "softwareEnable": status.sw_enable,
        "remoteStartActive": status.remote_start_active,
        "heartbeatOn": status.heartbeat_on,
        "supplyOk": status.supply_ok,
        "supplyVolts": round(status.supply_volts, 2),
        "outputVolts": round(status.output_volts, 2),
        "stopCause": status.stop_cause,
        "firmware": status.firmware_version,
        "faults": {
            "relay": status.relay_fault,
            "k2Stuck": status.k2_stuck,
            "k2NoClose": status.k2_no_close,
            "k1PoleB": status.k1_pole_b,
            "remoteStartFailed": status.remote_start_failed,
            "heartbeatLost": status.heartbeat_lost,
        },
    }


class SafetyLink:
    """Owns the server's connection to Jelly's safety board."""

    def __init__(self, *, start: bool = True) -> None:
        self._board: SafetyBoard | None = None
        self._channel = resolve_safety_channel()
        # Channels a board has answered on since the server started. A board
        # that goes quiet stays "detected", so the panel says it stopped
        # answering instead of hiding the control.
        self._detected: set[str] = set()
        self._paused = False
        self._error: str | None = None
        self._lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="axol-safety-link", daemon=True
        )
        self._thread.start()
        # Serializes the supervisor's open/close with pause() and commands.
        self._board_lock = asyncio.Lock()
        self._supervisor: asyncio.Future[Any] | None = None
        if start:
            self._supervisor = asyncio.run_coroutine_threadsafe(
                self._supervise(), self._loop
            )

    # -- thread plumbing ----------------------------------------------------

    def _submit(self, coro: Any, timeout: float = _COMMAND_TIMEOUT_S) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except BaseException:
            future.cancel()
            raise

    # -- public API ---------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """The board's latest state for the panel."""
        board = self._board
        status = board.status if board is not None else None
        age = board.status_age if board is not None else None
        with self._lock:
            channel = self._channel
            detected = channel in self._detected
            paused = self._paused
            error = self._error
        return {
            "channel": channel,
            "connected": board is not None and board.running,
            "paused": paused,
            "detected": detected,
            "reachable": board is not None and board.status_is_fresh(_REACHABLE_S),
            "ageSeconds": None if age is None else round(age, 1),
            "status": None if status is None else status_payload(status),
            "error": error,
        }

    def stop(self) -> dict[str, Any]:
        """Software E-stop. Raises :class:`SafetyError` unless confirmed."""
        self._submit(self._command("stop"))
        return self.status()

    def start(self) -> dict[str, Any]:
        """Remote start: power back on. Raises :class:`SafetyError` with why not."""
        self._submit(self._command("start"))
        return self.status()

    def pause(self) -> None:
        """Close the board and keep it closed until :meth:`resume`."""
        self._submit(self._pause())

    def resume(self) -> None:
        """Let the supervisor reopen the board."""
        with self._lock:
            self._paused = False

    def shutdown(self) -> None:
        """Close the board and stop the loop thread (server shutdown)."""
        try:
            if self._supervisor is not None:
                self._supervisor.cancel()
            self._submit(self._close_board())
        except Exception as exc:  # noqa: BLE001 - shutdown continues
            _logger.warning("safety link shutdown: %s", exc)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)

    # -- loop-side coroutines ----------------------------------------------

    async def _command(self, action: str) -> SafetyStatus:
        async with self._board_lock:
            board = self._board
            with self._lock:
                detected = self._channel in self._detected
            # A stop is always attempted: a board that never answered may
            # still take it. A start needs a board known to be there.
            if action != "stop" and not detected:
                raise SafetyError("This Jelly has no safety board.")
            if board is None or not board.running:
                raise SafetyError(
                    "Jelly's safety board is not connected. Press the E-Stop."
                )
            if action == "stop":
                _logger.warning("software E-stop requested from the control panel")
                try:
                    status = await board.stop()
                except SafetyError as exc:
                    _logger.error("software E-stop NOT confirmed: %s", exc)
                    raise
                _logger.warning("software E-stop confirmed: power is off")
            else:
                _logger.info("remote start requested from the control panel")
                status = await board.remote_start()
                _logger.info("remote start: power is on")
            return status

    async def _pause(self) -> None:
        with self._lock:
            self._paused = True
        await self._close_board()

    async def _close_board(self) -> None:
        async with self._board_lock:
            board, self._board = self._board, None
            if board is not None:
                await board.close()

    async def _supervise(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep supervising
                _logger.debug("safety link supervisor: %s", exc)
            await asyncio.sleep(_SUPERVISE_S)

    async def _tick(self) -> None:
        """Open, keep, or drop the board to match the bus that exists now."""
        from ..cli.can.setup import iface_up

        channel = resolve_safety_channel()
        with self._lock:
            self._channel = channel
            paused = self._paused
        async with self._board_lock:
            board = self._board
            if board is not None and (
                paused or not board.running or board.channel != channel
            ):
                if board.failure is not None:
                    _logger.info(
                        "safety link: bus %s went away (%s); reopening",
                        board.channel,
                        board.failure,
                    )
                self._board = None
                await board.close()
                board = None
            if paused:
                return
            if board is None:
                if not iface_up(channel):
                    return
                board = SafetyBoard(channel)
                try:
                    await board.start(bring_up=False)
                except Exception as exc:  # noqa: BLE001 - retried next tick
                    with self._lock:
                        self._error = str(exc) or type(exc).__name__
                    await board.close()
                    return
                self._board = board
                with self._lock:
                    self._error = None
            if board.status is not None:
                with self._lock:
                    if channel not in self._detected:
                        _logger.info(
                            "safety board found on %s (firmware %s)",
                            channel,
                            board.status.firmware_version,
                        )
                    self._detected.add(channel)
