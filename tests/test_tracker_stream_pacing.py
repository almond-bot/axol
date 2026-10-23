"""A sender stall must not turn into a burst of duplicate current poses."""

from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import patch

from almond_axol.tracker.bridge import StopEventControls, TrackerBridge


def test_stalled_send_skips_missed_slots_and_preserves_control_pulse_spacing():
    stop = threading.Event()
    bridge = TrackerBridge(
        object(), left="left", right="right", hz=100, controls=StopEventControls(stop)
    )
    now = [100.0]
    sent = []
    sleeps = []

    def compose():
        return {"t": now[0] * 1000, "l_lock": len(sent) < 3}

    async def sleep(delay):
        sleeps.append(delay)
        now[0] += delay

    class Socket:
        async def send(self, payload):
            sent.append((now[0], json.loads(payload)))
            if len(sent) == 1:
                now[0] += 0.4
            if len(sent) == 5:
                stop.set()

    async def drain(_socket):
        await asyncio.Future()

    with (
        patch.object(bridge, "compose_frame", side_effect=compose),
        patch.object(bridge, "_drain", side_effect=drain),
        patch(
            "almond_axol.tracker.bridge.time.perf_counter", side_effect=lambda: now[0]
        ),
        patch("almond_axol.tracker.bridge.asyncio.sleep", side_effect=sleep),
    ):
        asyncio.run(bridge._stream(Socket()))

    assert len(sent) == 5
    assert all(delay >= 0.0099 for delay in sleeps)
    assert sent[1][0] >= 100.41 - 1e-9
    assert all(b[0] - a[0] >= 0.0099 for a, b in zip(sent[1:], sent[2:]))
    assert all(payload["t"] == stamp * 1000 for stamp, payload in sent)
    assert [payload["l_lock"] for _, payload in sent] == [
        True,
        True,
        True,
        False,
        False,
    ]
