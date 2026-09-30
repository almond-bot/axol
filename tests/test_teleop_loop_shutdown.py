"""The synchronous teleop bridge joins async finalizers before closing its loop."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from almond_axol.lerobot.teleop import teleop_vr


@pytest.fixture
def teleop_loop():
    # Construct only the async bridge: no robot, Jelly, camera, or IK startup.
    teleop = object.__new__(teleop_vr.AxolVRTeleop)
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    errors = []
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    teleop._loop = loop
    teleop._loop_thread = thread
    teleop._startup_done_event = None
    teleop._cleanup_pending = False
    teleop._disconnect_async = AsyncMock()
    thread.start()
    loop.call_soon_threadsafe(ready.set)
    assert ready.wait(timeout=1)
    try:
        yield SimpleNamespace(teleop=teleop, loop=loop, thread=thread, errors=errors)
    finally:
        if thread.is_alive():
            asyncio.run_coroutine_threadsafe(teleop._drain_loop_tasks(), loop).result(3)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=1)
        if not loop.is_closed():
            loop.close()


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_disconnect_joins_pending_finalizers_even_after_cleanup_failure(
    teleop_loop, cleanup_failure
):
    session = teleop_loop
    finished = threading.Event()

    async def background():
        try:
            await asyncio.Future()
        finally:
            # Cancelling a future is not proof that this awaited work ran.
            await asyncio.sleep(0.01)
            finished.set()

    async def start():
        task = asyncio.create_task(background(), name="test-vr-connection")
        await asyncio.sleep(0)
        return task

    task = asyncio.run_coroutine_threadsafe(start(), session.loop).result(1)
    if cleanup_failure:
        session.teleop._disconnect_async.side_effect = ValueError("VR cleanup failed")
        with pytest.raises(RuntimeError, match="asynchronous cleanup") as failure:
            session.teleop.disconnect()
        assert isinstance(failure.value.__cause__, ValueError)
    else:
        session.teleop.disconnect()

    assert finished.is_set()
    assert task.done()
    assert not session.thread.is_alive()
    assert session.loop.is_closed()
    assert session.teleop._loop is None
    assert session.teleop._cleanup_pending is cleanup_failure
    assert session.errors == []


def test_timed_out_disconnect_finishes_cancellation_before_loop_close(
    teleop_loop, monkeypatch
):
    session = teleop_loop
    monkeypatch.setattr(teleop_vr, "_ASYNC_DISCONNECT_TIMEOUT_S", 0.02)
    finished = threading.Event()
    cleanup_tasks = []

    async def cleanup():
        cleanup_tasks.append(asyncio.current_task())
        try:
            await asyncio.Future()
        finally:
            await asyncio.sleep(0.02)
            finished.set()

    session.teleop._disconnect_async.side_effect = cleanup
    with pytest.raises(RuntimeError, match="asynchronous cleanup") as failure:
        session.teleop.disconnect()

    assert isinstance(failure.value.__cause__, TimeoutError)
    assert finished.is_set()
    assert cleanup_tasks[0].done()
    assert not session.thread.is_alive()
    assert session.loop.is_closed()
    assert session.teleop._cleanup_pending
    assert session.errors == []


@pytest.mark.parametrize("spawn_followup", [False, True])
def test_disconnect_finalizes_async_generators_on_live_loop(
    teleop_loop, spawn_followup
):
    session = teleop_loop
    finished = threading.Event()
    followup_finished = threading.Event()
    followup_tasks = []

    async def followup():
        try:
            await asyncio.Future()
        finally:
            await asyncio.sleep(0)
            followup_finished.set()

    async def stream():
        try:
            yield "frame"
        finally:
            if spawn_followup:
                followup_tasks.append(asyncio.create_task(followup()))
            await asyncio.sleep(0)
            finished.set()

    async def start():
        generator = stream()
        assert await generator.__anext__() == "frame"
        return generator

    generator = asyncio.run_coroutine_threadsafe(start(), session.loop).result(1)
    session.teleop.disconnect()

    assert finished.is_set()
    assert generator.ag_frame is None
    if spawn_followup:
        assert followup_finished.is_set()
        assert followup_tasks[0].done()
    assert session.loop.is_closed()
    assert session.errors == []
