"""Time-based pose interpolation (jitter buffer) for VR teleoperation.

VR pose frames are produced by the headset at a steady rate (~72–90 Hz) but
arrive at the robot **batched and jittered** — over a relayed/Funnel path a
whole burst can land together after a ~150 ms gap. The old "latest-wins"
ingestion threw away every frame in a burst except the newest, so the IK target
jumped once per burst and sat still in between: the move-pause-jerk that makes
teleop feel jittery. The downstream One Euro / EMA / trapezoid filters can't fix
that because the jumps are a *network* artefact, not real hand motion — they're
driven by an irregular, lossy sample stream.

:class:`PoseInterpolator` reconstructs the original smooth motion: it buffers
recent frames stamped with the headset's **capture** time (``VRFrame.t``) and,
when the consumer asks for the current target, renders the pose at a playout
time held slightly in the past (``now - delay``) by interpolating between the
two buffered frames that bracket it (lerp for positions/grips, slerp for
orientation). A whole 150 ms burst then plays back as the smooth stream it
originally was.

``delay`` is **adaptive**: it tracks the observed arrival jitter (clamped to
``[min_delay, max_delay]``), so a clean LAN adds almost no latency while a
jittery relay adds just enough to stay ahead of the bursts. Control-state fields
(locks / reset / session state) are taken from the *latest* received frame, not
the delayed playout, so engage/disengage/reset stay responsive while only the
motion is smoothed.

Transports that don't stamp ``t`` (the USB link, or an older web build) degrade
gracefully to the original latest-wins behaviour — capture time falls back to
server arrival time, where bursts can't be reconstructed but nothing breaks.
"""

from __future__ import annotations

import bisect
import threading
import time

import numpy as np

from .models import VRFrame, VRPose, VRPosition, VRQuaternion


class PoseInterpolator:
    """Adaptive playout buffer that renders a smooth pose from jittery frames.

    Thread-safe: :meth:`push` is called from the VR server's asyncio thread as
    frames arrive; :meth:`sample` is called from the IK loop thread at its own
    cadence.

    Args:
        enabled: When ``False``, :meth:`sample` just returns the latest frame
            (pure latest-wins, the original behaviour).
        min_delay_s: Floor on the playout delay (seconds).
        max_delay_s: Cap on the playout delay (seconds) — bounds the added
            latency. Bursts longer than this still cause a small catch-up.
        window_s: Sliding window over which arrival jitter is measured.
        max_frames: Hard cap on buffered frames (safety bound).
        pos_eps: Position change (metres) below which a re-render is considered
            unchanged, so the consumer's identity check can skip redundant IK.
    """

    def __init__(
        self,
        enabled: bool = True,
        min_delay_s: float = 0.0,
        max_delay_s: float = 0.1,
        window_s: float = 2.0,
        max_frames: int = 512,
        pos_eps: float = 1e-4,
    ) -> None:
        self.enabled = enabled
        self._min_delay = float(min_delay_s)
        self._max_delay = float(max_delay_s)
        self._window = float(window_s)
        self._max_frames = int(max_frames)
        self._pos_eps = float(pos_eps)

        self._lock = threading.Lock()
        # Buffer of (capture_time_s, frame), kept sorted by capture time.
        self._caps: list[float] = []
        self._frames: list[VRFrame] = []
        # Recent (local_recv_s, transit_s) for jitter/offset estimation.
        self._transits: list[tuple[float, float]] = []
        self._clock_offset: float | None = None
        self._delay: float = float(min_delay_s)
        # True once we've seen client timestamps; flipping source resets state.
        self._t_is_client: bool | None = None
        # Latest received frame (for responsive control-state passthrough).
        self._latest: VRFrame | None = None

        # Identity-stable output: return the same object when nothing moved, so
        # the IK loop's `frame is last_frame` check can skip redundant solves.
        self._last_out: VRFrame | None = None
        self._last_pos: np.ndarray | None = None

    def reset(self) -> None:
        """Drop all buffered state (e.g. on reconnect)."""
        with self._lock:
            self._caps.clear()
            self._frames.clear()
            self._transits.clear()
            self._clock_offset = None
            self._delay = self._min_delay
            self._t_is_client = None
            self._latest = None
            self._last_out = None
            self._last_pos = None

    def push(self, frame: VRFrame, now: float | None = None) -> None:
        """Ingest a freshly received frame.

        Args:
            frame: The received pose frame.
            now: Local receive time (``time.perf_counter()`` seconds). Injectable
                for testing; defaults to the current monotonic clock.
        """
        local_recv = time.perf_counter() if now is None else now
        is_client = frame.t is not None
        cap_t = (frame.t / 1000.0) if is_client else local_recv

        with self._lock:
            # Reset estimation if the capture-time source changes (e.g. the
            # client transport switched between USB and network).
            if self._t_is_client is not None and is_client != self._t_is_client:
                self._caps.clear()
                self._frames.clear()
                self._transits.clear()
                self._clock_offset = None
                self._delay = self._min_delay
                self._last_out = None
                self._last_pos = None
            self._t_is_client = is_client
            self._latest = frame

            # Jitter / clock-offset estimation over the sliding window.
            transit = local_recv - cap_t
            self._transits.append((local_recv, transit))
            cutoff = local_recv - self._window
            while len(self._transits) > 1 and self._transits[0][0] < cutoff:
                self._transits.pop(0)
            ts = [t for _, t in self._transits]
            self._clock_offset = min(ts)
            # Host-clock estimate of when this frame's poses were captured
            # (biased late by the minimum one-way transit, which the min-filter
            # can't separate from the clock offset — negligible on USB).
            frame.t_host = cap_t + self._clock_offset
            jitter = max(ts) - self._clock_offset
            target_delay = min(max(jitter, self._min_delay), self._max_delay)
            # Grow the delay immediately (don't let the buffer run dry), shrink
            # it slowly so we don't reintroduce jitter on a brief calm patch.
            if target_delay > self._delay:
                self._delay = target_delay
            else:
                self._delay += 0.05 * (target_delay - self._delay)

            # Insert in capture-time order (the datachannel may reorder).
            i = bisect.bisect_right(self._caps, cap_t)
            self._caps.insert(i, cap_t)
            self._frames.insert(i, frame)

            # Prune: keep a little history behind the current playout point.
            play = (local_recv - self._clock_offset) - self._delay
            keep_before = play - 0.5
            drop = 0
            while drop < len(self._caps) - 2 and self._caps[drop] < keep_before:
                drop += 1
            if drop:
                del self._caps[:drop]
                del self._frames[:drop]
            extra = len(self._caps) - self._max_frames
            if extra > 0:
                del self._caps[:extra]
                del self._frames[:extra]

    def sample(self, now: float | None = None) -> VRFrame | None:
        """Render the current interpolated target frame.

        Returns ``None`` only before any frame has been received. The returned
        object is *identity-stable*: when the rendered pose hasn't moved beyond
        ``pos_eps`` and the control state is unchanged, the previous object is
        returned so the IK loop can skip a redundant solve.
        """
        if now is None:
            now = time.perf_counter()
        with self._lock:
            latest = self._latest
            if latest is None:
                return None
            if not self.enabled or self._clock_offset is None or len(self._caps) < 2:
                # Passthrough: behave like latest-wins.
                if self._last_out is latest:
                    return self._last_out
                self._last_out = latest
                self._last_pos = None
                return latest

            play = (now - self._clock_offset) - self._delay
            caps = self._caps
            frames = self._frames
            if play <= caps[0]:
                a = b = frames[0]
                alpha = 0.0
            elif play >= caps[-1]:
                a = b = frames[-1]
                alpha = 0.0
            else:
                j = bisect.bisect_right(caps, play)
                a, b = frames[j - 1], frames[j]
                span = caps[j] - caps[j - 1]
                alpha = (play - caps[j - 1]) / span if span > 1e-9 else 0.0
            last_out = self._last_out
            last_pos = self._last_pos
            # The rendered pose corresponds to headset-time ``play``; map it
            # back onto the host clock for consumers that align to capture time.
            play_host = play + self._clock_offset

        rendered, pos = _interpolate(a, b, alpha, latest, play_host)

        # Identity-stable: reuse the previous object when nothing moved and the
        # control state matches, so the consumer's `is` check skips the solve.
        if (
            last_out is not None
            and last_pos is not None
            and _same_control(last_out, rendered)
            and float(np.max(np.abs(pos - last_pos))) < self._pos_eps
        ):
            return last_out

        with self._lock:
            self._last_out = rendered
            self._last_pos = pos
        return rendered


def _lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return a + alpha * (b - a)


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-path quaternion interpolation; ``q`` is ``[x, y, z, w]``."""
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:  # nearly colinear — nlerp is numerically safer
        q = q0 + alpha * (q1 - q0)
        n = np.linalg.norm(q)
        return q / n if n > 1e-12 else q0
    theta0 = np.arccos(d)
    theta = theta0 * alpha
    q2 = q1 - q0 * d
    n2 = np.linalg.norm(q2)
    if n2 < 1e-12:
        return q0
    q2 = q2 / n2
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def _pos(p: VRPosition) -> np.ndarray:
    return np.array([p.x, p.y, p.z], dtype=np.float64)


def _quat(q: VRQuaternion) -> np.ndarray:
    return np.array([q.x, q.y, q.z, q.w], dtype=np.float64)


def _interpolate(
    a: VRFrame, b: VRFrame, alpha: float, latest: VRFrame, t_host: float | None = None
) -> tuple[VRFrame, np.ndarray]:
    """Interpolate motion between ``a`` and ``b``; take control state from
    ``latest``. Returns ``(frame, pos_vector)`` where ``pos_vector`` is the
    concatenated EE+elbow positions used for the change/identity check."""
    l_ee_p = _lerp(_pos(a.l_ee.position), _pos(b.l_ee.position), alpha)
    r_ee_p = _lerp(_pos(a.r_ee.position), _pos(b.r_ee.position), alpha)
    l_ee_q = _slerp(_quat(a.l_ee.quaternion), _quat(b.l_ee.quaternion), alpha)
    r_ee_q = _slerp(_quat(a.r_ee.quaternion), _quat(b.r_ee.quaternion), alpha)
    l_el = _lerp(_pos(a.l_elbow), _pos(b.l_elbow), alpha)
    r_el = _lerp(_pos(a.r_elbow), _pos(b.r_elbow), alpha)
    l_grip = float(a.l_grip + alpha * (b.l_grip - a.l_grip))
    r_grip = float(a.r_grip + alpha * (b.r_grip - a.r_grip))

    frame = VRFrame(
        l_ee=VRPose(
            position=VRPosition(x=l_ee_p[0], y=l_ee_p[1], z=l_ee_p[2]),
            quaternion=VRQuaternion(x=l_ee_q[0], y=l_ee_q[1], z=l_ee_q[2], w=l_ee_q[3]),
        ),
        r_ee=VRPose(
            position=VRPosition(x=r_ee_p[0], y=r_ee_p[1], z=r_ee_p[2]),
            quaternion=VRQuaternion(x=r_ee_q[0], y=r_ee_q[1], z=r_ee_q[2], w=r_ee_q[3]),
        ),
        l_elbow=VRPosition(x=l_el[0], y=l_el[1], z=l_el[2]),
        r_elbow=VRPosition(x=r_el[0], y=r_el[1], z=r_el[2]),
        l_grip=l_grip,
        r_grip=r_grip,
        # Control state is responsive: always the latest received, never delayed.
        l_lock=latest.l_lock,
        r_lock=latest.r_lock,
        reset=latest.reset,
        state=latest.state,
        t=latest.t,
        seq=latest.seq,
        t_host=t_host,
    )
    pos = np.concatenate([l_ee_p, r_ee_p, l_el, r_el])
    return frame, pos


def _same_control(a: VRFrame, b: VRFrame) -> bool:
    return (
        a.l_lock == b.l_lock
        and a.r_lock == b.r_lock
        and a.reset == b.reset
        and a.state == b.state
    )
