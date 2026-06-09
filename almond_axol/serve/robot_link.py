"""Persistent in-process robot connection for the web control panel.

Unlike the four operations (which open the robot themselves for the duration
of a task), this module keeps a *detached* link to the robot alive while the
panel is idle: it brings up the CAN interfaces and pings all 16 motors once a
second so the UI can show a live connected / disconnected indicator and
per-motor health.

The link runs on its own asyncio event loop in a dedicated thread so the CAN
reader loops and the ping timer never touch uvicorn's loop. While a task runs
the buses are released (see :meth:`RobotLink.release`) — there is exactly one
owner of the CAN bus at a time, matching "ping the motors every second unless
we are running a task".
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any

from ..motor import CanBus, Joint, Motor, MotorError
from ..utils.shared import CAN_LEFT, CAN_RIGHT
from ..utils.sudo import SudoPasswordIncorrect, SudoPasswordRequired, run_root

_logger = logging.getLogger(__name__)

# Ping cadence + per-motor read timeout. One full sweep reads 16 motors; the
# timeout is generous so a momentarily-busy bus doesn't flap the indicator.
_PING_INTERVAL_S = 1.0
_PING_TIMEOUT_S = 0.5

# State machine surfaced to the UI.
#   disconnected -> connecting -> connected
#   connected    -> busy (a task owns the bus)  -> connected
#   any          -> error
STATE_DISCONNECTED = "disconnected"
STATE_CONNECTING = "connecting"
STATE_CONNECTED = "connected"
STATE_BUSY = "busy"
STATE_ERROR = "error"

# IFF_UP flag in /sys/class/net/<iface>/flags (administratively up).
_IFF_UP = 0x1


class _ArmLink:
    """One arm's CAN bus plus its eight motors, kept open for pinging."""

    def __init__(self, channel: str, side: str) -> None:
        self.channel = channel
        self.side = side
        self._bus: CanBus | None = None
        self._motors: dict[Joint, Motor] = {}
        # joint name -> {"reachable": bool, "status": str | None}
        self.health: dict[str, dict[str, Any]] = {}

    async def open(self) -> None:
        self._bus = CanBus(self.channel)
        await self._bus.start()
        self._motors = {joint: Motor(self._bus, joint) for joint in Joint}

    async def close(self) -> None:
        if self._bus is not None:
            try:
                await self._bus.close()
            except Exception as exc:  # noqa: BLE001 - teardown is best-effort
                _logger.debug("closing %s bus failed: %s", self.channel, exc)
        self._bus = None
        self._motors = {}

    async def ping(self) -> None:
        """Read each motor's status; record reachability without raising."""
        for joint, motor in self._motors.items():
            reachable = True
            status: str | None = None
            try:
                code = await asyncio.wait_for(
                    motor.get_error_code(), timeout=_PING_TIMEOUT_S
                )
                status = getattr(code, "name", str(code))
            except (MotorError, asyncio.TimeoutError, Exception):  # noqa: BLE001
                reachable = False
            self.health[joint.name] = {"reachable": reachable, "status": status}


class RobotLink:
    """Owns the idle-time robot connection (CAN + 1 Hz motor ping)."""

    def __init__(
        self,
        left_channel: str | None = CAN_LEFT,
        right_channel: str | None = CAN_RIGHT,
    ) -> None:
        self._arms: list[_ArmLink] = []
        if left_channel:
            self._arms.append(_ArmLink(left_channel, "left"))
        if right_channel:
            self._arms.append(_ArmLink(right_channel, "right"))

        self._state = STATE_DISCONNECTED
        self._error: str | None = None
        self._last_ping: float | None = None
        # True when the last connect attempt needs a sudo password from the UI.
        self._needs_sudo = False

        # Dedicated event loop running in a daemon thread.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="axol-robot-link", daemon=True
        )
        self._thread.start()
        self._ping_task: asyncio.Task[Any] | None = None
        self._lock = threading.Lock()

    # -- thread plumbing ----------------------------------------------------

    def _submit(self, coro: Any, timeout: float = 30.0) -> Any:
        """Run a coroutine on the link loop from any thread and wait for it."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # -- public API ---------------------------------------------------------

    def connect(self, sudo_password: str | None = None) -> dict[str, Any]:
        """Bring up CAN, open the buses, and start the ping loop.

        ``sudo_password`` is only used (and never stored) if CAN is down and
        passwordless sudo is unavailable. If a password is needed but absent,
        the returned status has ``needsSudo`` set so the UI can prompt.
        """
        with self._lock:
            if self._state in (STATE_CONNECTED, STATE_BUSY):
                return self.status()
            self._state = STATE_CONNECTING
            self._error = None
        try:
            self._enable_can(sudo_password)
        except SudoPasswordRequired:
            with self._lock:
                self._state = STATE_DISCONNECTED
                self._needs_sudo = True
                self._error = "CAN bring-up needs a sudo password."
            return self.status()
        except SudoPasswordIncorrect:
            with self._lock:
                self._state = STATE_ERROR
                self._needs_sudo = False
                self._error = "Incorrect sudo password."
            return self.status()
        except Exception as exc:  # noqa: BLE001 - report any bring-up failure
            with self._lock:
                self._state = STATE_ERROR
                self._needs_sudo = False
                self._error = f"{type(exc).__name__}: {exc}"
            _logger.warning("robot connect failed: %s", self._error)
            return self.status()
        try:
            self._submit(self._open_and_start())
        except Exception as exc:  # noqa: BLE001 - report any bring-up failure
            with self._lock:
                self._state = STATE_ERROR
                self._needs_sudo = False
                self._error = f"{type(exc).__name__}: {exc}"
            _logger.warning("robot connect failed: %s", self._error)
            return self.status()
        with self._lock:
            self._state = STATE_CONNECTED
            self._needs_sudo = False
        return self.status()

    def disconnect(self) -> dict[str, Any]:
        """Stop pinging and close the buses."""
        try:
            self._submit(self._stop_and_close())
        except Exception as exc:  # noqa: BLE001
            _logger.debug("robot disconnect cleanup failed: %s", exc)
        with self._lock:
            self._state = STATE_DISCONNECTED
            self._error = None
            self._needs_sudo = False
            self._last_ping = None
        for arm in self._arms:
            arm.health = {}
        return self.status()

    def release(self) -> None:
        """Hand the CAN bus to a task: stop pinging and close the buses.

        No-op unless currently connected. The prior state is remembered so
        :meth:`reacquire` only reconnects if the link was up before the task.
        """
        with self._lock:
            if self._state not in (STATE_CONNECTED,):
                return
            self._state = STATE_BUSY
        try:
            self._submit(self._stop_and_close())
        except Exception as exc:  # noqa: BLE001
            _logger.debug("robot release cleanup failed: %s", exc)

    def reacquire(self) -> None:
        """Re-open the buses + ping loop after a task releases the bus."""
        with self._lock:
            if self._state != STATE_BUSY:
                return
        try:
            self._submit(self._open_and_start())
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._state = STATE_ERROR
                self._error = f"{type(exc).__name__}: {exc}"
            _logger.warning("robot reacquire failed: %s", self._error)
            return
        with self._lock:
            self._state = STATE_CONNECTED

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = self._state
            error = self._error
            last_ping = self._last_ping
            needs_sudo = self._needs_sudo
        motors: list[dict[str, Any]] = []
        for arm in self._arms:
            for joint in Joint:
                h = arm.health.get(joint.name, {})
                motors.append(
                    {
                        "arm": arm.side,
                        "joint": joint.name,
                        "reachable": bool(h.get("reachable", False)),
                        "status": h.get("status"),
                    }
                )
        reachable = sum(1 for m in motors if m["reachable"])
        return {
            "state": state,
            "connected": state in (STATE_CONNECTED, STATE_BUSY),
            "error": error,
            "needsSudo": needs_sudo,
            "lastPing": last_ping,
            "motors": motors,
            "motorCount": len(motors),
            "reachableCount": reachable,
        }

    def shutdown(self) -> None:
        """Tear down the link and stop the loop thread (server shutdown)."""
        try:
            self.disconnect()
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)

    # -- loop-side coroutines ----------------------------------------------

    async def _open_and_start(self) -> None:
        for arm in self._arms:
            await arm.open()
        if self._ping_task is None or self._ping_task.done():
            self._ping_task = asyncio.ensure_future(self._ping_loop())

    async def _stop_and_close(self) -> None:
        if self._ping_task is not None:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
            self._ping_task = None
        for arm in self._arms:
            await arm.close()

    async def _ping_loop(self) -> None:
        while True:
            start = self._loop.time()
            try:
                await asyncio.gather(*(arm.ping() for arm in self._arms))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                _logger.debug("ping sweep error: %s", exc)
            with self._lock:
                self._last_ping = time.time()
            elapsed = self._loop.time() - start
            await asyncio.sleep(max(0.0, _PING_INTERVAL_S - elapsed))

    # -- CAN bring-up -------------------------------------------------------

    def _can_already_up(self) -> bool:
        """True when every CAN interface is administratively up (no sudo needed)."""
        if not self._arms:
            return False
        for arm in self._arms:
            try:
                flags = int(
                    Path(f"/sys/class/net/{arm.channel}/flags").read_text().strip(),
                    16,
                )
            except (OSError, ValueError):
                return False
            if not (flags & _IFF_UP):
                return False
        return True

    def _enable_can(self, sudo_password: str | None) -> None:
        """Bring up the CAN interfaces, asking the UI for a password if needed.

        Order of attempts:
          1. If the interfaces are already up, do nothing (common case: cron
             brought them up at boot).
          2. If CAN was never configured on this machine, run the full
             ``can.setup`` (udev rules, persistent names, @reboot bring-up)
             non-interactively — it may not have been set up before.
          3. Otherwise just run the persisted startup script.

        Privileged steps escalate through :func:`run_root`, which raises
        :class:`SudoPasswordRequired` when a password is needed but absent (the
        UI then prompts) or :class:`SudoPasswordIncorrect` when it's wrong.
        """
        if self._can_already_up():
            _logger.info("CAN interfaces already up; skipping bring-up.")
            return

        from ..cli.can.setup import _CRON_SCRIPT, ensure_setup, is_configured

        if not is_configured():
            _logger.info("CAN not configured yet; running can.setup.")
            ensure_setup(password=sudo_password)
            _logger.info("CAN setup complete; interfaces brought up.")
            return

        run_root(["bash", str(_CRON_SCRIPT)], password=sudo_password, check=True)
        _logger.info("CAN interfaces brought up.")
