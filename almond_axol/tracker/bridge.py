"""Tracker → VRServer bridge: streams tracker poses as VRFrame JSON.

The bridge is a WebSocket *client* of the existing VR server — exactly
what a Quest headset is — so teleop, IK, and collect-data run unchanged.
It composes a :class:`~almond_axol.vr.models.VRFrame` at the configured
rate from the latest tracker poses, stamps ``t`` (monotonic ms, for the
server's pose interpolator) and ``seq``, and ships it over WSS (the
server's self-signed certificate is not verified).

Engage/reset control is a stopgap until the rig's button PCB exists (see
:class:`StdinControls`): Enter toggles tracking engage, ``r`` triggers a
reset. The toggle is realised as a short pulse of both lock bits — the
shared teleop core enables on a rising edge of both locks together and
disables on a rising edge of either, so one pulse toggles either way.

A side whose tracker stops reporting (occlusion, SLAM relocalising)
holds its last good pose rather than going quiet, so IK never chases a
glitch and the operator can recover by re-engaging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import sys
import threading
import time

import numpy as np

from ..utils.ports import VR_PORT
from .base import TrackerPose, TrackerSource

_logger = logging.getLogger(__name__)

# A pose older than this is stale: hold the last streamed pose and warn.
_STALE_S = 0.5
# Lock/reset pulses span this many frames so the server-side edge
# detection can't miss them across interpolation or a dropped frame.
_PULSE_FRAMES = 10

# Placeholder pose streamed for a side with no tracker assigned (teleop can
# run one-armed, but VRFrame carries both sides).
_DEFAULT_POSE = {
    "left": (np.array([0.2, 1.0, -0.4]), np.array([0.0, 0.0, 0.0, 1.0])),
    "right": (np.array([-0.2, 1.0, -0.4]), np.array([0.0, 0.0, 0.0, 1.0])),
}


class StdinControls:
    """Line-based stdin control surface (stopgap until the button PCB).

    Reads stdin on a daemon thread: an empty line (Enter) requests an
    engage toggle, ``r`` a reset, ``q`` a quit. The PCB input will replace
    this class with the same three request bits.
    """

    def __init__(self) -> None:
        self._toggle_requests = 0
        self._reset_requests = 0
        self.quit = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name="tracker-stdin"
        )

    def start(self) -> None:
        self._thread.start()

    def _read_loop(self) -> None:
        for line in sys.stdin:
            cmd = line.strip().lower()
            with self._lock:
                if cmd == "":
                    self._toggle_requests += 1
                elif cmd == "r":
                    self._reset_requests += 1
                elif cmd == "q":
                    self.quit.set()
                    return
        self.quit.set()  # stdin closed (e.g. running under a supervisor)

    def consume(self) -> tuple[bool, bool]:
        """Return and clear ``(toggle_requested, reset_requested)``."""
        with self._lock:
            toggle = self._toggle_requests > 0
            reset = self._reset_requests > 0
            self._toggle_requests = 0
            self._reset_requests = 0
        return toggle, reset


class TrackerBridge:
    """Composes and streams VRFrames from a :class:`TrackerSource`.

    Args:
        source: Started tracker backend.
        left:   Device key bound to the left rig side, or ``None``.
        right:  Device key bound to the right rig side, or ``None``.
        host:   VR server host (the teleop machine; usually localhost).
        port:   VR server port.
        hz:     Frame streaming rate.
        controls: Engage/reset input; defaults to :class:`StdinControls`.
    """

    def __init__(
        self,
        source: TrackerSource,
        *,
        left: str | None,
        right: str | None,
        host: str = "localhost",
        port: int = VR_PORT,
        hz: float = 120.0,
        controls: StdinControls | None = None,
    ) -> None:
        if left is None and right is None:
            raise ValueError(
                "no tracker is bound to either side — run `axol tracker.identify` first"
            )
        self._source = source
        self._keys = {"left": left, "right": right}
        self._host = host
        self._port = port
        self._hz = hz
        self._controls = controls or StdinControls()

        self._seq = 0
        self._engaged = False
        self._lock_pulse = 0
        self._reset_pulse = 0
        self._held: dict[str, TrackerPose] = {}
        self._warned_stale: dict[str, bool] = {"left": False, "right": False}

    # -- Frame composition ---------------------------------------------------

    def _side_pose(self, side: str, now: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(pos, quat)`` to stream for one side.

        The last good pose wins over anything stale/untracked; a side with
        no data yet (or no tracker assigned) streams the fixed placeholder.
        """
        key = self._keys[side]
        if key is not None:
            sample = self._source.poses().get(key)
            if sample is not None and sample.tracking and now - sample.t <= _STALE_S:
                self._held[side] = sample
                if self._warned_stale[side]:
                    self._warned_stale[side] = False
                    _logger.info("%s tracker (%s) is tracking again", side, key)
            elif not self._warned_stale[side]:
                self._warned_stale[side] = True
                _logger.warning(
                    "%s tracker (%s) is %s — holding its last pose",
                    side,
                    key,
                    "not tracking" if sample is not None else "not reporting",
                )
        held = self._held.get(side)
        if held is not None:
            return held.pos, held.quat
        return _DEFAULT_POSE[side]

    def compose_frame(self) -> dict:
        """Build one VRFrame JSON object from the latest tracker poses."""
        toggle, reset = self._controls.consume()
        if toggle and self._lock_pulse == 0:
            self._engaged = not self._engaged
            self._lock_pulse = _PULSE_FRAMES
            _logger.info(
                "engage toggle → %s", "engaged" if self._engaged else "disengaged"
            )
        if reset and self._reset_pulse == 0:
            self._reset_pulse = _PULSE_FRAMES
            _logger.info("reset requested")

        now = time.perf_counter()
        frame: dict = {}
        for side, ee_key, elbow_key in (
            ("left", "l_ee", "l_elbow"),
            ("right", "r_ee", "r_elbow"),
        ):
            pos, quat = self._side_pose(side, now)
            frame[ee_key] = {
                "position": {
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2]),
                },
                "quaternion": {
                    "x": float(quat[0]),
                    "y": float(quat[1]),
                    "z": float(quat[2]),
                    "w": float(quat[3]),
                },
            }
            # Elbow hints are ignored in absolute (UMI) mode; stream the
            # tracker position so the field is well-formed.
            frame[elbow_key] = {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
            }

        lock = self._lock_pulse > 0
        if self._lock_pulse > 0:
            self._lock_pulse -= 1
        frame["l_lock"] = lock
        frame["r_lock"] = lock

        frame["reset"] = self._reset_pulse > 0
        if self._reset_pulse > 0:
            self._reset_pulse -= 1

        # Grip placeholders (open) until the analog pot PCB exists.
        frame["l_grip"] = 1.0
        frame["r_grip"] = 1.0

        self._seq += 1
        frame["seq"] = self._seq
        frame["t"] = now * 1000.0  # monotonic ms, like performance.now()
        return frame

    # -- Streaming -----------------------------------------------------------

    async def run(self) -> None:
        """Stream frames until stdin quits, reconnecting on socket loss."""
        import websockets

        self._controls.start()
        uri = f"wss://{self._host}:{self._port}/ws"
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE  # the VR server's cert is self-signed

        print(
            "Streaming tracker poses. Controls: Enter = engage/disengage, "
            "r = reset, q = quit."
        )
        while not self._controls.quit.is_set():
            try:
                async with websockets.connect(uri, ssl=ssl_ctx, max_queue=4) as ws:
                    _logger.info("connected to %s", uri)
                    await self._stream(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on any drop
                if self._controls.quit.is_set():
                    break
                _logger.warning("connection to %s lost (%s); retrying in 2s", uri, exc)
                await asyncio.sleep(2.0)

    async def _stream(self, ws) -> None:
        """Send frames at the configured rate over one connection."""
        drain = asyncio.create_task(self._drain(ws))
        interval = 1.0 / self._hz
        deadline = time.perf_counter()
        try:
            while not self._controls.quit.is_set():
                deadline += interval
                await ws.send(json.dumps(self.compose_frame()))
                await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
        finally:
            drain.cancel()

    @staticmethod
    async def _drain(ws) -> None:
        """Consume server → client broadcasts (mode/state/episode messages)."""
        try:
            async for msg in ws:
                _logger.debug("server: %s", msg)
        except Exception:  # noqa: BLE001 - connection teardown ends the drain
            pass
