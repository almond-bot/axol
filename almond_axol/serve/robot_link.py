"""Persistent in-process robot connection for the web control panel.

Unlike the four operations (which open the robot themselves for the duration
of a task), this module keeps a *detached* link to the robot alive while the
panel is idle: it brings up the CAN interfaces, pings all 16 motors once a
second (reachability, status, temperature, voltage), and samples position /
velocity / torque at :data:`~.telemetry.SAMPLE_HZ` into the telemetry hub for
the diagnostics dashboard.

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

from ..constants import CAN_LEFT, CAN_RIGHT
from ..motor import CanBus, Joint, Motor, MotorError
from .telemetry import SAMPLE_HZ, TelemetryHub, motor_key

_logger = logging.getLogger(__name__)

# Ping cadence + per-motor read timeout. One full sweep reads 16 motors; the
# timeout is generous so a momentarily-busy bus doesn't flap the indicator.
_PING_INTERVAL_S = 1.0
_PING_TIMEOUT_S = 0.5

# Per-read timeout for the fast telemetry sweep. Tighter than the ping's: a
# skipped sample is invisible on a chart, a flapping health dot is not.
_SAMPLE_TIMEOUT_S = 0.2

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


def _format_error(exc: BaseException) -> str:
    """Short, human-readable error for the UI status pill.

    ``RuntimeError``s raised by the bring-up path are already written for
    humans ("Robot not detected"), so show them as-is; anything else keeps
    the exception type as context. Multi-line errors (e.g. a driver build
    failure dumping compiler output) are reduced to their first line.
    """
    if isinstance(exc, RuntimeError) and str(exc):
        text = str(exc)
    else:
        text = f"{type(exc).__name__}: {exc}"
    return text.strip().splitlines()[0] if text.strip() else type(exc).__name__


class _ArmLink:
    """One arm's CAN bus plus its eight motors, kept open for pinging."""

    def __init__(self, channel: str, side: str) -> None:
        self.channel = channel
        self.side = side
        self._bus: CanBus | None = None
        self._motors: dict[Joint, Motor] = {}
        # Serializes reads to one motor between the ping and sample loops, so
        # two in-flight requests to the same CAN ID can't mismatch replies.
        self._locks: dict[Joint, asyncio.Lock] = {}
        # joint name -> {"reachable": bool, "status": str | None, ...}
        self.health: dict[str, dict[str, Any]] = {}

    @property
    def motors(self) -> dict[Joint, Motor]:
        return self._motors

    def lock(self, joint: Joint) -> asyncio.Lock:
        return self._locks[joint]

    async def open(self) -> None:
        self._bus = CanBus(self.channel)
        await self._bus.start()
        self._motors = {joint: Motor(self._bus, joint) for joint in Joint}
        self._locks = {joint: asyncio.Lock() for joint in Joint}

    async def close(self) -> None:
        if self._bus is not None:
            try:
                await self._bus.close()
            except Exception as exc:  # noqa: BLE001 - teardown is best-effort
                _logger.debug("closing %s bus failed: %s", self.channel, exc)
        self._bus = None
        self._motors = {}
        self._locks = {}

    async def ping(self) -> dict[str, dict[str, Any]]:
        """Read each motor's status/temperature/voltage; never raises.

        Returns the slow-telemetry sweep keyed by ``arm:JOINT`` for the hub.
        """
        sweep: dict[str, dict[str, Any]] = {}
        for joint, motor in self._motors.items():
            reachable = True
            status: str | None = None
            temperature: float | None = None
            voltage: float | None = None
            try:
                async with self._locks[joint]:
                    code = await asyncio.wait_for(
                        motor.get_error_code(), timeout=_PING_TIMEOUT_S
                    )
                    status = getattr(code, "name", str(code))
                    temperature = await asyncio.wait_for(
                        motor.get_temperature(), timeout=_PING_TIMEOUT_S
                    )
                    voltage = await asyncio.wait_for(
                        motor.get_voltage(), timeout=_PING_TIMEOUT_S
                    )
            except (MotorError, asyncio.TimeoutError, Exception):  # noqa: BLE001
                # A failed temperature/voltage read after a good status read
                # still counts as reachable; a failed status read does not.
                reachable = status is not None
            self.health[joint.name] = {
                "reachable": reachable,
                "status": status,
                "temperature": temperature,
                "voltage": voltage,
            }
            sweep[motor_key(self.side, joint.name)] = self.health[joint.name]
        return sweep

    async def sample(self) -> dict[str, list[float]]:
        """One fast sweep: position / velocity / torque for every motor."""

        async def read(joint: Joint, motor: Motor) -> tuple[str, list[float] | None]:
            try:
                async with self._locks[joint]:
                    pos = await asyncio.wait_for(
                        motor.get_position(), timeout=_SAMPLE_TIMEOUT_S
                    )
                    vel = await asyncio.wait_for(
                        motor.get_velocity(), timeout=_SAMPLE_TIMEOUT_S
                    )
                    torque = await asyncio.wait_for(
                        motor.get_torque(), timeout=_SAMPLE_TIMEOUT_S
                    )
            except (MotorError, asyncio.TimeoutError, Exception):  # noqa: BLE001
                return motor_key(self.side, joint.name), None
            return motor_key(self.side, joint.name), [pos, vel, torque]

        results = await asyncio.gather(
            *(read(joint, motor) for joint, motor in self._motors.items())
        )
        return {key: values for key, values in results if values is not None}


class RobotLink:
    """Owns the idle-time robot connection (CAN + ping + telemetry sampling)."""

    def __init__(
        self,
        left_channel: str | None = CAN_LEFT,
        right_channel: str | None = CAN_RIGHT,
        hub: TelemetryHub | None = None,
    ) -> None:
        self._arms: list[_ArmLink] = []
        if left_channel:
            self._arms.append(_ArmLink(left_channel, "left"))
        if right_channel:
            self._arms.append(_ArmLink(right_channel, "right"))

        self.hub = hub if hub is not None else TelemetryHub()

        self._state = STATE_DISCONNECTED
        self._error: str | None = None
        self._last_ping: float | None = None

        # Dedicated event loop running in a daemon thread.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="axol-robot-link", daemon=True
        )
        self._thread.start()
        self._ping_task: asyncio.Task[Any] | None = None
        self._sample_task: asyncio.Task[Any] | None = None
        self._lock = threading.Lock()

    # -- thread plumbing ----------------------------------------------------

    def _submit(self, coro: Any, timeout: float = 30.0) -> Any:
        """Run a coroutine on the link loop from any thread and wait for it."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def _set_state(self, state: str, error: str | None = None) -> None:
        with self._lock:
            self._state = state
            self._error = error
        self.hub.push_state(state)

    # -- public API ---------------------------------------------------------

    def connect(self) -> dict[str, Any]:
        """Bring up CAN, open the buses, and start the ping loop."""
        with self._lock:
            if self._state in (STATE_CONNECTED, STATE_BUSY):
                return self.status()
        self._set_state(STATE_CONNECTING)
        try:
            self._enable_can()
        except Exception as exc:  # noqa: BLE001 - report any bring-up failure
            self._set_state(STATE_ERROR, _format_error(exc))
            _logger.warning("robot connect failed: %s", exc)
            return self.status()
        try:
            self._submit(self._open_and_start())
        except Exception as exc:  # noqa: BLE001 - report any bring-up failure
            self._set_state(STATE_ERROR, _format_error(exc))
            _logger.warning("robot connect failed: %s", exc)
            return self.status()
        self._set_state(STATE_CONNECTED)
        return self.status()

    def disconnect(self) -> dict[str, Any]:
        """Stop pinging and close the buses."""
        try:
            self._submit(self._stop_and_close())
        except Exception as exc:  # noqa: BLE001
            _logger.debug("robot disconnect cleanup failed: %s", exc)
        self._set_state(STATE_DISCONNECTED)
        with self._lock:
            self._last_ping = None
        for arm in self._arms:
            arm.health = {}
        self.hub.clear_slow()
        return self.status()

    def release(self) -> None:
        """Hand the CAN bus to a task: stop pinging and close the buses.

        No-op unless currently connected. The prior state is remembered so
        :meth:`reacquire` only reconnects if the link was up before the task.
        """
        with self._lock:
            if self._state not in (STATE_CONNECTED,):
                return
        self._set_state(STATE_BUSY)
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
            self._set_state(STATE_ERROR, _format_error(exc))
            _logger.warning("robot reacquire failed: %s", exc)
            return
        self._set_state(STATE_CONNECTED)

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = self._state
            error = self._error
            last_ping = self._last_ping
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
            "lastPing": last_ping,
            "motors": motors,
            "motorCount": len(motors),
            "reachableCount": reachable,
        }

    def motor_details(self, arm: str, joint_name: str) -> dict[str, Any]:
        """Full one-motor readout (the ``motor.info`` set) for the dashboard.

        Raises ``RuntimeError`` unless the link currently owns the bus, and
        ``KeyError`` for an unknown arm/joint.
        """
        with self._lock:
            if self._state != STATE_CONNECTED:
                raise RuntimeError(f"robot link is {self._state}")
        arm_link = next((a for a in self._arms if a.side == arm), None)
        if arm_link is None:
            raise KeyError(arm)
        joint = Joint[joint_name]
        return self._submit(_read_motor_details(arm_link, joint))

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
        if self._sample_task is None or self._sample_task.done():
            self._sample_task = asyncio.ensure_future(self._sample_loop())

    async def _stop_and_close(self) -> None:
        for task in (self._ping_task, self._sample_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._ping_task = None
        self._sample_task = None
        for arm in self._arms:
            await arm.close()

    async def _ping_loop(self) -> None:
        while True:
            start = self._loop.time()
            try:
                sweeps = await asyncio.gather(*(arm.ping() for arm in self._arms))
                slow: dict[str, dict[str, Any]] = {}
                for sweep in sweeps:
                    slow.update(sweep)
                if slow:
                    self.hub.push_slow(slow)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                _logger.debug("ping sweep error: %s", exc)
            with self._lock:
                self._last_ping = time.time()
            elapsed = self._loop.time() - start
            await asyncio.sleep(max(0.0, _PING_INTERVAL_S - elapsed))

    async def _sample_loop(self) -> None:
        interval = 1.0 / SAMPLE_HZ
        while True:
            start = self._loop.time()
            try:
                sweeps = await asyncio.gather(*(arm.sample() for arm in self._arms))
                motors: dict[str, list[float]] = {}
                for sweep in sweeps:
                    motors.update(sweep)
                if motors:
                    self.hub.push_frame(motors)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                _logger.debug("telemetry sweep error: %s", exc)
            elapsed = self._loop.time() - start
            await asyncio.sleep(max(0.0, interval - elapsed))

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

    def _enable_can(self) -> None:
        """Bring up the CAN interfaces.

        1. If the interfaces are already up, do nothing (common case: cron
           brought them up at boot).
        2. Otherwise run the full ``can.setup`` (driver, udev rules, persistent
           names, @reboot bring-up, then bring-up) non-interactively.

        We always run the full setup rather than just the persisted startup
        script (``can.enable``): on a fresh axol the script doesn't exist yet,
        and on a partially-configured one the driver may be unloaded or the
        interfaces unnamed, so the bare bring-up script can't connect. The
        whole setup is idempotent (see :func:`ensure_setup`), so re-running it
        on an already-configured machine is safe and cheap.

        ``axol serve`` runs as root under the hosted install, so the privileged
        steps inside :func:`ensure_setup` run without a sudo prompt.
        """
        if self._can_already_up():
            _logger.info("CAN interfaces already up; skipping bring-up.")
            return

        from ..cli.can.setup import ensure_setup

        _logger.info("CAN interfaces down; running can.setup.")
        ensure_setup()
        _logger.info("CAN setup complete; interfaces brought up.")


async def _read_motor_details(arm_link: _ArmLink, joint: Joint) -> dict[str, Any]:
    """The ``motor.info`` read set against a link-owned motor."""
    motor = arm_link.motors[joint]

    async def read(coro: Any) -> Any:
        try:
            return await asyncio.wait_for(coro, timeout=_PING_TIMEOUT_S)
        except (MotorError, asyncio.TimeoutError, Exception):  # noqa: BLE001
            return None

    async with arm_link.lock(joint):
        status = await read(motor.get_error_code())
        mode = await read(motor.get_control_mode())
        gains = await read(motor.get_gains())
        return {
            "arm": arm_link.side,
            "joint": joint.name,
            "model": await read(motor.get_model()),
            "firmware": await read(motor.get_firmware_version()),
            "status": getattr(status, "name", None),
            "mode": getattr(mode, "name", None),
            "position": await read(motor.get_position()),
            "velocity": await read(motor.get_velocity()),
            "torque": await read(motor.get_torque()),
            "temperature": await read(motor.get_temperature()),
            "voltage": await read(motor.get_voltage()),
            "gains": vars(gains) if gains is not None else None,
        }
