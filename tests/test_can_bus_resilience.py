from __future__ import annotations

import asyncio
import errno
from pathlib import Path

import can
import pytest

from almond_axol.motor import bus


class _FakeBus:
    def __init__(self, *, send_error: BaseException | None = None) -> None:
        self.send_error = send_error
        self.sent: list[can.Message] = []
        self.shutdown_calls = 0
        self.received: list[can.Message | None] = []

    def send(self, message: can.Message) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(message)

    def recv(self, timeout=0):
        return self.received.pop(0) if self.received else None

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def fileno(self) -> int:
        return 10


class _Loop:
    def __init__(self, *times: float) -> None:
        self.times = iter(times or (0.0, 0.1))
        self.readers: list[int] = []

    def time(self) -> float:
        return next(self.times)

    def add_reader(self, fd, callback) -> None:
        self.readers.append(fd)

    def remove_reader(self, fd) -> None:
        self.readers.remove(fd)


def test_can_error_classification_and_interface_flags(
    tmp_path: Path, monkeypatch
) -> None:
    assert bus._error_code(OSError(errno.ENODEV, "gone")) == errno.ENODEV
    assert bus._iface_lost(OSError(errno.ENETDOWN, "down"))
    assert bus._tx_queue_full(OSError(errno.ENOBUFS, "full"))
    assert not bus._iface_lost(RuntimeError("other"))

    root = tmp_path / "net"
    (root / "can0").mkdir(parents=True)
    (root / "can0" / "flags").write_text("1")

    monkeypatch.setattr(
        bus, "Path", lambda value: root if value == "/sys/class/net" else Path(value)
    )
    assert bus._iface_is_up("can0")
    assert not bus._iface_is_up("missing")


def test_can_send_success_drop_and_error_paths(monkeypatch) -> None:
    fake = _FakeBus()
    monkeypatch.setattr(bus.can, "Bus", lambda **kwargs: fake)
    can_bus = bus.CanBus("can0")

    async def exercise() -> None:
        await can_bus._send(1, b"abc")
        assert fake.sent[0].arbitration_id == 1
        assert can_bus._enobufs_since is None

        can_bus._lost = True
        await can_bus._send(2, b"drop")
        assert len(fake.sent) == 1
        can_bus._lost = False

        fake.send_error = OSError(errno.ENOBUFS, "full")
        await can_bus._send(3, b"full")
        assert can_bus._enobufs_since is not None

        fake.send_error = OSError(errno.ENODEV, "gone")
        await can_bus._send(4, b"gone")
        assert can_bus._lost

        can_bus._lost = False
        fake.send_error = OSError(errno.EINVAL, "bad")
        with pytest.raises(OSError):
            await can_bus._send(5, b"bad")

    asyncio.run(exercise())


def test_persistent_full_queue_marks_bus_stalled(monkeypatch) -> None:
    fake = _FakeBus()
    monkeypatch.setattr(bus.can, "Bus", lambda **kwargs: fake)
    can_bus = bus.CanBus("can0")
    error = OSError(errno.ENOBUFS, "full")

    class Clock:
        values = iter([1.0, 2.1])

        def time(self):
            return next(self.values)

    async def exercise() -> None:
        clock = Clock()
        monkeypatch.setattr(asyncio, "get_running_loop", lambda: clock)
        can_bus._on_tx_queue_full(error)
        can_bus._on_tx_queue_full(error)
        assert can_bus._stalled
        assert can_bus._lost
        assert can_bus._wake.is_set()
        can_bus._mark_lost(error)
        can_bus._mark_stalled(error)

    asyncio.run(exercise())


def test_reconnect_flush_and_probe_recovery(tmp_path: Path, monkeypatch) -> None:
    initial = _FakeBus()
    reopened = _FakeBus()
    probe = _FakeBus()
    probe.received = [can.Message(arbitration_id=bus._PROBE_ID, data=b"")]
    created = iter([initial, reopened, probe])
    monkeypatch.setattr(bus.can, "Bus", lambda **kwargs: next(created))
    monkeypatch.setattr(bus, "_iface_is_up", lambda channel: True)
    can_bus = bus.CanBus("can0")
    can_bus._lost = True

    commands: list[list[str]] = []
    monkeypatch.setattr(bus, "run_root", lambda argv, check: commands.append(argv))
    monkeypatch.setattr(bus, "CAN_BRINGUP_SCRIPT", tmp_path / "missing-script")

    async def exercise() -> None:
        await can_bus._reconnect(_Loop(1.0, 1.1))  # type: ignore[arg-type]
        assert can_bus._bus is reopened
        assert not can_bus._lost
        assert initial.shutdown_calls == 1

        await can_bus._flush_tx_queue()
        assert commands == [
            ["ip", "link", "set", "can0", "down"],
            ["ip", "link", "set", "can0", "up"],
        ]

        can_bus._stalled = True
        await can_bus._wait_bus_alive(_Loop(2.0, 2.1))  # type: ignore[arg-type]
        assert not can_bus._stalled
        assert probe.sent[0].arbitration_id == bus._PROBE_ID
        assert probe.shutdown_calls == 1

    asyncio.run(exercise())


def test_listener_dispatch_and_close(monkeypatch) -> None:
    fake = _FakeBus()
    message = can.Message(arbitration_id=1, data=b"x")
    fake.received = [message]
    monkeypatch.setattr(bus.can, "Bus", lambda **kwargs: fake)
    can_bus = bus.CanBus("can0")
    received: list[can.Message] = []
    can_bus._add_listener(received.append)

    class StopAfterDispatch:
        def __init__(self) -> None:
            self.readers: list[int] = []

        def add_reader(self, fd, callback) -> None:
            self.readers.append(fd)

        def remove_reader(self, fd) -> None:
            self.readers.remove(fd)

    async def exercise() -> None:
        async def stop_wait(awaitable, timeout):
            awaitable.close()
            can_bus._lost = True
            raise TimeoutError

        monkeypatch.setattr(bus.asyncio, "wait_for", stop_wait)
        await can_bus._pump(StopAfterDispatch())  # type: ignore[arg-type]
        assert received == [message]
        await can_bus.close()
        assert fake.shutdown_calls == 1
        assert can_bus._bus is None

    asyncio.run(exercise())
