"""Idle-time links to Jelly's wheel bus and lift controller for the control panel.

The :class:`~.robot_link.RobotLink` keeps a detached connection to the arm
hub while the panel is idle so the operator can see, before starting
anything, that the motors are reachable and healthy. Jelly's two extra
devices ride on their own single-channel adapters — the wheel bus
(:data:`~almond_axol.constants.CAN_BASE`, four Damiao motors at IDs 1–4) and
the jelly_legs lift controller (its own chest bus, or shared with the
wheels) — and this module gives each of them the same treatment:

- **wheels**: open the wheel bus and ping the four motors once a second
  (status, temperature, bus voltage). Never enables or commands a motor.
- **lift**: open the lift's bus and poll the jelly_legs board's status
  (homed, height, moving, stall/driver faults) and power telemetry (the 24 V
  rail, i.e. Jelly's battery — see :mod:`almond_axol.robot.battery`) once a
  second. Never sends a motion opcode — only ``SET_RATE 0`` (quiet the
  board's broadcast), ``GET_STATUS`` and ``GET_POWER``. The last battery
  estimate outlives a hand-over to a task, reported with its age, since the
  pack drains over hours and "as of a few minutes ago" beats nothing.

Each device is its own state machine (``disconnected`` → ``connecting`` →
``connected``, ``busy`` while a task owns the bus, ``error``), so the panel
shows two tiles that connect and disconnect independently. Exactly one
process may command a CAN bus at a time: when an operation or diagnostic
that opens the Jelly buses starts, :meth:`JellyLink.release` closes every
connected device (marking it ``busy``) and :meth:`JellyLink.reacquire`
reopens them when the task ends — the same handover the arm link performs.

The link runs on its own asyncio event loop in a daemon thread so the CAN
reader loops never touch uvicorn's loop.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any, Literal

from ..constants import CAN_BASE
from ..motor import CanBus, MotorError
from ..motor.damiao import DamiaoMotor
from ..robot.battery import BatteryEstimator, BatteryStatus
from ..robot.jelly import WHEELS
from ..robot.lift import (
    _ID_CMD,
    _ID_POWER,
    _ID_STATUS,
    _OP_GET_POWER,
    _OP_GET_STATUS,
    _OP_SET_RATE,
    LiftStatus,
    decode_power,
    decode_status,
    power_under_load,
    resolve_lift_channel,
)
from .robot_link import (
    STATE_BUSY,
    STATE_CONNECTED,
    STATE_CONNECTING,
    STATE_DISCONNECTED,
    STATE_ERROR,
    _format_error,
)

_logger = logging.getLogger(__name__)

JellyDevice = Literal["wheels", "lift"]
JELLY_DEVICES: tuple[JellyDevice, ...] = ("wheels", "lift")

# Ping cadence and per-read timeout, matching the arm link's idle sweep.
_PING_INTERVAL_S = 1.0
_PING_TIMEOUT_S = 0.5
# A lift status frame older than this (with GET_STATUS polled every second)
# means the board stopped answering: unpowered, unplugged, or not on this bus.
_LIFT_FRESH_S = 3.0
# Same for the power frame (GET_POWER polled every second).
_POWER_FRESH_S = 3.0

# Damiao feedback (MST) ID = 0x10 + motor ID, the factory convention Jelly's
# wheels follow (see ``almond_axol.motor.motor.make_driver``).
_DAMIAO_FEEDBACK_BASE = 0x10

_SYS_NET = Path("/sys/class/net")


def wheels_channel() -> str:
    """The SocketCAN interface carrying Jelly's wheel motors."""
    return CAN_BASE


def lift_channel() -> str:
    """The SocketCAN interface carrying the jelly_legs lift controller.

    Re-resolved on every call (the chest adapter may be plugged in after the
    server started): the chest bus when it exists, else the shared wheel bus.
    """
    return resolve_lift_channel()


def device_presence(device: JellyDevice) -> dict[str, Any]:
    """Whether ``device``'s pinned CAN interface exists (and is up) on this host.

    Presence follows the interface, exactly as ``detect_jelly`` does for
    teleop: ``axol can.setup`` only creates ``can_alm_axol_b`` / ``_c`` for
    adapters it positively identified as the wheel or lift bus. The lift on a
    shared wheel bus has no interface of its own, so it reports the wheel
    bus; whether a board actually answers there is what connecting shows.
    """
    channel = wheels_channel() if device == "wheels" else lift_channel()
    path = _SYS_NET / channel
    up = False
    try:
        up = bool(int(path.joinpath("flags").read_text().strip(), 16) & 0x1)
    except (OSError, ValueError):
        pass
    return {"channel": channel, "present": path.exists(), "up": up}


class _WheelsDevice:
    """The wheel bus: four Damiao motors, pinged but never commanded."""

    name: JellyDevice = "wheels"

    def __init__(self) -> None:
        self.channel = wheels_channel()
        self._bus: CanBus | None = None
        self._motors: dict[int, DamiaoMotor] = {}
        self._locks: dict[int, asyncio.Lock] = {}
        # motor id -> {"reachable": bool, "status": str | None, ...}
        self.health: dict[int, dict[str, Any]] = {}

    async def open(self) -> None:
        self.channel = wheels_channel()
        self._bus = CanBus(self.channel)
        await self._bus.start()
        self._motors = {
            wheel.motor_id: DamiaoMotor(
                self._bus,
                wheel.motor_id,
                feedback_id=_DAMIAO_FEEDBACK_BASE + wheel.motor_id,
            )
            for wheel in WHEELS
        }
        self._locks = {wheel.motor_id: asyncio.Lock() for wheel in WHEELS}

    async def close(self) -> None:
        if self._bus is not None:
            await self._bus.close()
        self._bus = None
        self._motors = {}
        self._locks = {}

    async def ping(self) -> None:
        """Read each wheel's status / temperature / bus voltage; never raises."""
        for motor_id, motor in self._motors.items():
            reachable = True
            status: str | None = None
            temperature: float | None = None
            voltage: float | None = None
            try:
                async with self._locks[motor_id]:
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
            self.health[motor_id] = {
                "reachable": reachable,
                "status": status,
                "temperature": temperature,
                "voltage": voltage,
            }

    def snapshot(self) -> dict[str, Any]:
        motors: list[dict[str, Any]] = []
        for wheel in WHEELS:
            h = self.health.get(wheel.motor_id, {})
            reachable = h.get("reachable")
            motors.append(
                {
                    "name": wheel.name,
                    "id": wheel.motor_id,
                    # No health means nobody is reading this motor (a task
                    # owns the bus, or the link is down): unknown, not a claim.
                    "reachable": None if reachable is None else bool(reachable),
                    "status": h.get("status"),
                    "temperature": h.get("temperature"),
                    "voltage": h.get("voltage"),
                }
            )
        return {
            "motors": motors,
            "motorCount": len(motors),
            "reachableCount": sum(1 for m in motors if m["reachable"] is True),
        }


class _LiftDevice:
    """The lift bus: listens for jelly_legs status, polls it, never moves it."""

    name: JellyDevice = "lift"

    def __init__(self) -> None:
        self.channel = lift_channel()
        self._bus: CanBus | None = None
        self.status: LiftStatus | None = None
        self.last_status_monotonic: float | None = None
        self._estimator = BatteryEstimator()
        # Last estimate and when it was measured; kept across close/open.
        self.battery: BatteryStatus | None = None
        self.last_power_monotonic: float | None = None

    def _on_message(self, msg) -> None:  # noqa: ANN001 - can.Message, typed lazily
        if msg.arbitration_id == _ID_STATUS and len(msg.data) >= 6:
            self.status = decode_status(bytes(msg.data))
            self.last_status_monotonic = time.monotonic()
        elif msg.arbitration_id == _ID_POWER and len(msg.data) >= 8:
            power = decode_power(bytes(msg.data))
            self.battery = self._estimator.update(
                power.supply_volts, under_load=power_under_load(power, self.status)
            )
            self.last_power_monotonic = time.monotonic()

    async def open(self) -> None:
        self.channel = lift_channel()
        self.status = None
        self.last_status_monotonic = None
        # Keep showing the last battery estimate until a new frame lands, but
        # start the average afresh: the pack may have drained while a task
        # owned the bus.
        self._estimator.reset()
        self._bus = CanBus(self.channel)
        self._bus._add_listener(self._on_message)
        await self._bus.start()
        # The board broadcasts status every 50 ms once it has seen any frame,
        # and that stream starves the CANable's TX path; keep it quiet and
        # poll instead (the same sequence the lift driver and can.setup use).
        await self._bus._send(_ID_CMD, bytes([_OP_SET_RATE, 0x00, 0x00]))
        await self._bus._send(_ID_CMD, bytes([_OP_GET_STATUS]))
        await self._bus._send(_ID_CMD, bytes([_OP_GET_POWER]))

    async def close(self) -> None:
        if self._bus is not None:
            await self._bus.close()
        self._bus = None

    async def ping(self) -> None:
        """Solicit one status and one power frame; never raises."""
        if self._bus is None:
            return
        for op in (_OP_GET_STATUS, _OP_GET_POWER):
            try:
                await asyncio.wait_for(
                    self._bus._send(_ID_CMD, bytes([op])),
                    timeout=_PING_TIMEOUT_S,
                )
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                _logger.debug(
                    "lift poll 0x%02x on %s failed: %s", op, self.channel, exc
                )

    def reachable(self) -> bool:
        return (
            self.last_status_monotonic is not None
            and time.monotonic() - self.last_status_monotonic <= _LIFT_FRESH_S
        )

    def battery_snapshot(self, *, polling: bool) -> dict[str, Any] | None:
        battery = self.battery
        if battery is None or self.last_power_monotonic is None:
            return None
        age = max(0.0, time.monotonic() - self.last_power_monotonic)
        return {
            "voltage": round(battery.voltage, 2),
            "percent": round(battery.percent, 1),
            "charging": battery.charging,
            "underLoad": battery.under_load,
            # Server-side age: the browser's clock need not match the robot's.
            "ageSeconds": round(age, 1),
            # False while a task owns the bus or the board stopped answering:
            # the numbers are the last known ones.
            "live": polling and age <= _POWER_FRESH_S,
        }

    def snapshot(self, *, polling: bool) -> dict[str, Any]:
        status = self.status
        return {
            "battery": self.battery_snapshot(polling=polling),
            # Unknown while nobody polls the board (a task owns the bus).
            "reachable": self.reachable() if polling else None,
            "status": (
                None
                if status is None
                else {
                    "homed": status.homed,
                    "heightPercent": status.height_percent,
                    "moving": status.moving,
                    "homing": status.homing,
                    "stallFault": status.stall_fault,
                    "atLower": status.at_lower,
                    "atUpper": status.at_upper,
                    "driversEnabled": status.drivers_enabled,
                    "vmPresent": status.vm_present,
                    "driverFaultMask": status.driver_fault_mask,
                }
            ),
        }


class _DeviceLink:
    """One device's state machine plus its open/close bookkeeping."""

    def __init__(self, device: _WheelsDevice | _LiftDevice) -> None:
        self.device = device
        self.state = STATE_DISCONNECTED
        self.error: str | None = None
        self.last_ping: float | None = None
        # True from before the bus opens until close verifiably completes
        # ("open or uncertain"): a failed close must never be mistaken for a
        # released bus.
        self.bus_may_be_open = False
        self.ping_task: asyncio.Task[Any] | None = None

    @property
    def name(self) -> JellyDevice:
        return self.device.name


class JellyLink:
    """Owns the idle-time links to Jelly's wheel bus and lift controller."""

    def __init__(self) -> None:
        self._devices: dict[JellyDevice, _DeviceLink] = {
            "wheels": _DeviceLink(_WheelsDevice()),
            "lift": _DeviceLink(_LiftDevice()),
        }
        self._lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="axol-jelly-link", daemon=True
        )
        self._thread.start()
        # Serializes open/close lifecycles on the owning loop so a retry can
        # never overlap a still-unwinding teardown.
        self._lifecycle_lock = asyncio.Lock()

    # -- thread plumbing ----------------------------------------------------

    def _submit(self, coro: Any, timeout: float = 30.0) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except BaseException:
            future.cancel()
            raise

    def _link(self, device: str) -> _DeviceLink:
        try:
            return self._devices[device]  # type: ignore[index]
        except KeyError:
            raise KeyError(f"unknown Jelly device: {device}") from None

    def _set_state(
        self, link: _DeviceLink, state: str, error: str | None = None
    ) -> None:
        with self._lock:
            link.state = state
            link.error = error

    # -- public API ---------------------------------------------------------

    def connect(self, device: str) -> dict[str, Any]:
        """Bring up the device's CAN interface, open the bus, start pinging."""
        link = self._link(device)
        with self._lock:
            already_active = link.state in (STATE_CONNECTED, STATE_BUSY)
            cleanup_needed = link.bus_may_be_open
        if already_active:
            return self.status()
        self._set_state(link, STATE_CONNECTING)
        if cleanup_needed:
            try:
                self._submit(self._close(link))
            except Exception as exc:  # noqa: BLE001
                self._set_state(link, STATE_ERROR, _format_error(exc))
                _logger.warning("Jelly %s pre-connect cleanup failed: %s", device, exc)
                return self.status()
        try:
            self._enable_can(link)
            self._submit(self._open(link))
        except Exception as exc:  # noqa: BLE001 - report any bring-up failure
            self._set_state(link, STATE_ERROR, _format_error(exc))
            _logger.warning("Jelly %s connect failed: %s", device, exc)
            return self.status()
        self._set_state(link, STATE_CONNECTED)
        return self.status()

    def disconnect(self, device: str) -> dict[str, Any]:
        """Stop pinging and close the device's bus."""
        link = self._link(device)
        with self._lock:
            if link.state == STATE_BUSY:
                raise RuntimeError(
                    f"cannot disconnect the Jelly {device} while a task owns its bus"
                )
        try:
            self._submit(self._close(link))
        except Exception as exc:  # noqa: BLE001
            self._set_state(link, STATE_ERROR, _format_error(exc))
            _logger.warning("Jelly %s disconnect cleanup failed: %s", device, exc)
            return self.status()
        self._set_state(link, STATE_DISCONNECTED)
        with self._lock:
            link.last_ping = None
        return self.status()

    def release(self) -> None:
        """Hand the Jelly buses to a task: close every connected device.

        Each connected device becomes ``busy`` so :meth:`reacquire` knows
        which to reopen. A device that fails to close is left in ``error``
        and the failure is raised — an operation must never open a bus this
        link may still hold. Nothing then borrows the buses, so any sibling
        that did close is reopened first: a device is ``busy`` only while a
        task actually owns it, never stranded there by an aborted hand-over.
        """
        failures: list[str] = []
        for link in self._devices.values():
            with self._lock:
                if link.state != STATE_CONNECTED and not link.bus_may_be_open:
                    continue
            self._set_state(link, STATE_BUSY)
            try:
                self._submit(self._close(link))
            except Exception as exc:  # noqa: BLE001
                self._set_state(link, STATE_ERROR, _format_error(exc))
                _logger.warning("Jelly %s release failed: %s", link.name, exc)
                failures.append(f"{link.name}: {exc}")
        if failures:
            self.reacquire()
            raise RuntimeError(
                "could not release the Jelly link (" + "; ".join(failures) + ")"
            )

    def reacquire(self) -> bool:
        """Reopen every device a task borrowed. True when anything reconnected."""
        reconnected = False
        for link in self._devices.values():
            with self._lock:
                if link.state != STATE_BUSY:
                    continue
            try:
                self._enable_can(link)
                self._submit(self._open(link))
            except Exception as exc:  # noqa: BLE001
                self._set_state(link, STATE_ERROR, _format_error(exc))
                _logger.warning("Jelly %s reacquire failed: %s", link.name, exc)
                continue
            self._set_state(link, STATE_CONNECTED)
            reconnected = True
        return reconnected

    def disconnect_all(self) -> dict[str, Any]:
        """Disconnect every device that is not busy (CAN discovery, shutdown)."""
        for device in JELLY_DEVICES:
            with self._lock:
                busy = self._devices[device].state == STATE_BUSY
            if not busy:
                self.disconnect(device)
        return self.status()

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for device, link in self._devices.items():
            with self._lock:
                state = link.state
                error = link.error
                last_ping = link.last_ping
            polling = state == STATE_CONNECTED
            entry: dict[str, Any] = {
                "state": state,
                "connected": state in (STATE_CONNECTED, STATE_BUSY),
                "error": error,
                "lastPing": last_ping,
                "channel": link.device.channel,
            }
            if isinstance(link.device, _LiftDevice):
                entry.update(link.device.snapshot(polling=polling))
            else:
                entry.update(link.device.snapshot())
            out[device] = entry
        return out

    def shutdown(self) -> None:
        """Tear down every device and stop the loop thread (server shutdown)."""
        try:
            for link in self._devices.values():
                with self._lock:
                    busy = link.state == STATE_BUSY
                if busy:
                    _logger.warning(
                        "server stopped while a task still owned the Jelly %s bus; "
                        "skipping its detached-link cleanup",
                        link.name,
                    )
                    continue
                self.disconnect(link.name)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)

    # -- loop-side coroutines ----------------------------------------------

    async def _open(self, link: _DeviceLink) -> None:
        async with self._lifecycle_lock:
            with self._lock:
                link.bus_may_be_open = True
                link.last_ping = None
            if isinstance(link.device, _WheelsDevice):
                link.device.health = {}
            await link.device.open()
            if link.ping_task is None or link.ping_task.done():
                link.ping_task = asyncio.ensure_future(self._ping_loop(link))

    async def _close(self, link: _DeviceLink) -> None:
        async with self._lifecycle_lock:
            task = link.ping_task
            link.ping_task = None
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:  # noqa: BLE001 - teardown continues
                    _logger.debug("Jelly %s ping loop ended: %s", link.name, exc)
            await link.device.close()
            if isinstance(link.device, _WheelsDevice):
                link.device.health = {}
            with self._lock:
                link.bus_may_be_open = False

    async def _ping_loop(self, link: _DeviceLink) -> None:
        while True:
            start = self._loop.time()
            try:
                await link.device.ping()
                with self._lock:
                    link.last_ping = time.time()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                _logger.debug("Jelly %s ping error: %s", link.name, exc)
            elapsed = self._loop.time() - start
            await asyncio.sleep(max(0.0, _PING_INTERVAL_S - elapsed))

    # -- CAN bring-up -------------------------------------------------------

    def _enable_can(self, link: _DeviceLink) -> None:
        """Make sure the device's interface exists and is up.

        Jelly's adapters are plain single-channel gs_usb devices already named
        by ``axol can.setup``; a missing interface is reported by name and an
        interface that is down is configured and brought up (the same path
        the Jelly and lift drivers take at enable time).
        """
        from ..cli.can.setup import bring_up_interfaces, iface_up

        channel = wheels_channel() if link.name == "wheels" else lift_channel()
        link.device.channel = channel
        if not (_SYS_NET / channel).exists():
            raise RuntimeError(
                f"Jelly {link.name} bus {channel} not found — is its CAN adapter "
                "plugged in? Run CAN discovery or `axol can.setup`."
            )
        if not iface_up(channel):
            bring_up_interfaces([channel])
