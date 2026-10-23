"""A passive observer whose proxy died under it is reopened, not left dark.

``axol serve`` keeps one :class:`BusObserver` per arm open across tasks. A
panel ``can.setup`` USB-resets the arm hub, the interface goes down, and the
observer's proxy exits on ENETDOWN; these pin that the observer can replace
the dead proxy and that the robot link's watchdog does so once the interface
is back up.
"""

from __future__ import annotations

import asyncio
import threading
import unittest
from unittest.mock import patch

from almond_axol.constants import Joint
from almond_axol.motor import bus as bus_mod
from almond_axol.motor import observer as observer_mod
from almond_axol.motor.bus import CanBus, stalled_channels
from almond_axol.motor.observer import BusObserver
from almond_axol.serve import robot_link
from almond_axol.serve.robot_link import RobotLink


class _FakeBus:
    """Stands in for the Rust proxy: open until ``die()``."""

    instances: list[_FakeBus] = []
    fail_start: BaseException | None = None
    fail_observer_mode = False

    def __init__(self, channel: str, *, publishes_stall: bool = True) -> None:
        self.channel = channel
        self.publishes_stall = publishes_stall
        self.is_open = False
        self.started = False
        self.closed = False
        _FakeBus.instances.append(self)

    def _add_listener(self, _listener: object) -> None:
        pass

    async def start(self) -> None:
        # Like the real proxy, the process exists before start() can fail.
        self.started = True
        if _FakeBus.fail_start is not None:
            raise _FakeBus.fail_start
        self.is_open = True

    def enable_observer_mode(self) -> None:
        if _FakeBus.fail_observer_mode:
            raise ConnectionResetError("proxy went away")

    def die(self) -> None:
        self.is_open = False

    async def close(self) -> None:
        self.closed = True
        self.is_open = False


class BusObserverReopenTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _FakeBus.instances = []
        _FakeBus.fail_start = None
        _FakeBus.fail_observer_mode = False
        patcher = patch.object(observer_mod, "CanBus", _FakeBus)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_start_is_a_noop_while_the_proxy_lives(self) -> None:
        obs = BusObserver("can_test", [Joint.SHOULDER_1])
        await obs.start()
        await obs.start()
        self.assertEqual(len(_FakeBus.instances), 1)
        self.assertTrue(obs.alive)

    async def test_start_replaces_a_dead_proxy_and_keeps_synced_ranges(self) -> None:
        obs = BusObserver("can_test", [Joint.SHOULDER_1, Joint.WRIST_3])
        await obs.start()
        obs.set_myactuator_ranges(Joint.SHOULDER_1, 12.5, 30.0)
        obs.set_damiao_ranges(Joint.WRIST_3, 12.5, 30.0, 10.0)
        first = _FakeBus.instances[0]

        first.die()
        self.assertTrue(obs.running)
        self.assertFalse(obs.alive)

        await obs.start()
        self.assertTrue(first.closed, "the dead proxy is reaped")
        self.assertEqual(len(_FakeBus.instances), 2)
        self.assertTrue(obs.alive)
        self.assertTrue(obs.ranges_synced(Joint.SHOULDER_1))
        self.assertTrue(obs.ranges_synced(Joint.WRIST_3))

    async def test_failed_reopen_leaves_it_retryable(self) -> None:
        obs = BusObserver("can_test", [Joint.SHOULDER_1])
        await obs.start()
        _FakeBus.instances[0].die()

        _FakeBus.fail_start = OSError("Network is down")
        with self.assertRaises(OSError):
            await obs.start()
        self.assertFalse(obs.running)
        self.assertFalse(obs.alive)

        _FakeBus.fail_start = None
        await obs.start()
        self.assertTrue(obs.alive)

    async def test_cancelled_or_failed_start_reaps_the_spawned_proxy(self) -> None:
        for failure in ("cancelled", "observer_mode"):
            with self.subTest(failure=failure):
                _FakeBus.instances = []
                _FakeBus.fail_start = (
                    asyncio.CancelledError() if failure == "cancelled" else None
                )
                _FakeBus.fail_observer_mode = failure == "observer_mode"
                obs = BusObserver("can_test", [Joint.SHOULDER_1])
                with self.assertRaises((asyncio.CancelledError, ConnectionError)):
                    await obs.start()
                (bus,) = _FakeBus.instances
                self.assertTrue(bus.closed, "a half-started proxy is reaped")
                self.assertFalse(obs.running)
        _FakeBus.fail_start = None
        _FakeBus.fail_observer_mode = False

    async def test_observer_bus_stays_out_of_the_stall_record(self) -> None:
        obs = BusObserver("can_test", [Joint.SHOULDER_1])
        await obs.start()
        self.assertFalse(_FakeBus.instances[0].publishes_stall)


class ObserverBusStallRecordTest(unittest.IsolatedAsyncioTestCase):
    """A non-publishing CanBus can't clear a stall the channel's owner reported."""

    def tearDown(self) -> None:
        bus_mod.set_channel_stalled("can_stall_test", False)

    async def test_open_and_close_leave_the_owners_stall_flagged(self) -> None:
        bus_mod.set_channel_stalled("can_stall_test", True)
        passive = CanBus("can_stall_test", publishes_stall=False)
        await passive.close()
        self.assertIn("can_stall_test", stalled_channels())

    async def test_a_publishing_bus_still_clears_it(self) -> None:
        bus_mod.set_channel_stalled("can_stall_test", True)
        CanBus("can_stall_test")
        self.assertNotIn("can_stall_test", stalled_channels())


class _FakeObserver:
    def __init__(self, *, alive: bool) -> None:
        self.alive = alive
        self.start_calls = 0
        self.started = asyncio.Event()

    async def start(self) -> None:
        self.start_calls += 1
        self.alive = True
        self.started.set()


class _Arm:
    def __init__(self, channel: str, observer: _FakeObserver | None) -> None:
        self.channel = channel
        self.observer = observer


def _link(arms: list[_Arm]) -> RobotLink:
    link = object.__new__(RobotLink)
    link._lock = threading.Lock()
    link._lifecycle_lock = asyncio.Lock()
    link._arms = arms  # type: ignore[assignment]
    return link


class ObserverWatchLoopTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        patcher = patch.object(robot_link, "_OBSERVER_REOPEN_INTERVAL_S", 0.0)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def _run_ticks(self, link: RobotLink, ticks: int = 5) -> None:
        task = asyncio.ensure_future(link._observer_watch_loop())
        for _ in range(ticks):
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_reopens_a_dead_observer_once_its_interface_is_up(self) -> None:
        dead = _FakeObserver(alive=False)
        live = _FakeObserver(alive=True)
        link = _link([_Arm("can_l", dead), _Arm("can_r", live)])
        with patch("almond_axol.cli.can.setup.iface_up", return_value=True):
            await self._run_ticks(link)
        self.assertEqual(dead.start_calls, 1)
        self.assertTrue(dead.alive)
        self.assertEqual(live.start_calls, 0)

    async def test_waits_while_the_interface_is_down(self) -> None:
        dead = _FakeObserver(alive=False)
        link = _link([_Arm("can_l", dead)])
        with patch("almond_axol.cli.can.setup.iface_up", return_value=False):
            await self._run_ticks(link)
        self.assertEqual(dead.start_calls, 0)

    async def test_leaves_an_observer_torn_down_meanwhile_alone(self) -> None:
        dead = _FakeObserver(alive=False)
        arm = _Arm("can_l", dead)
        link = _link([arm])
        with patch("almond_axol.cli.can.setup.iface_up", return_value=True):
            await link._lifecycle_lock.acquire()
            task = asyncio.ensure_future(link._observer_watch_loop())
            for _ in range(3):
                await asyncio.sleep(0)
            arm.observer = None  # disconnect tore it down under the lock
            link._lifecycle_lock.release()
            for _ in range(5):
                await asyncio.sleep(0)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(dead.start_calls, 0)


if __name__ == "__main__":
    unittest.main()
