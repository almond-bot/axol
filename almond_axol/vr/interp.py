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
time held slightly in the past (``now - delay``). A whole 150 ms burst then
plays back as the smooth stream it originally was.

On top of the jitter buffer, the playout point is rendered as a **fixed-lag
smoother**: a Gaussian-weighted average over every buffered frame within
``smooth_window_s`` of the playout time (weighted mean for positions / grips,
weighted quaternion mean for orientation). Because the window extends into the
*future* relative to the playout point, this is zero-phase smoothing whose lag
is fixed (``delay + smooth_window_s / 2``) rather than growing with hand speed
the way a causal low-pass filter's does.

The buffered lookahead also enables **glitch rejection**, which no causal
filter can do (at the instant of a jump it cannot distinguish a tracking
glitch from the start of a fast intentional move). Before averaging, a Hampel
test — computed over a wider window (``_REJECT_WINDOW_MULT`` x the smoothing
window) so a glitch burst stays a minority for the median — drops frames whose
EE/elbow positions deviate from the window median by more than ``outlier_k``
robust standard deviations (with an absolute floor so a still hand's
micro-noise never flags). A tracking excursion that jumps away and comes back
within roughly the smoothing window is erased entirely instead of being
chased; while it lasts, the nearest clean frames dominate the renormalized
weights so the output bridges smoothly across it. A *sustained* jump
eventually wins the median and is followed normally after the fixed lag. When
the window holds too few frames (startup, sparse or unstamped streams),
rendering falls back to plain lerp/slerp between the two bracketing frames.

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
        smooth_window_s: Width (seconds) of the Gaussian fixed-lag smoothing
            window centred on the playout point. Adds ``smooth_window_s / 2``
            of fixed latency in exchange for zero-phase smoothing and glitch
            rejection. ``0`` disables smoothing (plain two-frame lerp).
        outlier_k: Hampel threshold in robust standard deviations (MAD-based).
            Frames whose EE/elbow positions deviate from the window median by
            more than this are excluded from the average, so transient
            tracking glitches are dropped rather than followed. ``<= 0``
            disables rejection.
        outlier_floor_m: Absolute floor (metres) on the Hampel threshold so a
            perfectly still hand's micro-noise is never flagged as outliers.
    """

    def __init__(
        self,
        enabled: bool = True,
        min_delay_s: float = 0.0,
        max_delay_s: float = 0.15,
        window_s: float = 2.0,
        max_frames: int = 512,
        pos_eps: float = 1e-4,
        smooth_window_s: float = 0.12,
        outlier_k: float = 4.0,
        outlier_floor_m: float = 0.02,
    ) -> None:
        self.enabled = enabled
        self._min_delay = float(min_delay_s)
        self._max_delay = float(max_delay_s)
        self._window = float(window_s)
        self._max_frames = int(max_frames)
        self._pos_eps = float(pos_eps)
        self._half_smooth = max(0.0, float(smooth_window_s)) / 2.0
        self._outlier_k = float(outlier_k)
        self._outlier_floor = float(outlier_floor_m)

        self._lock = threading.Lock()
        # Buffer of (capture_time_s, frame), kept sorted by capture time.
        self._caps: list[float] = []
        self._frames: list[VRFrame] = []
        # Per-frame numeric vector (see _frame_vec), parallel to _caps/_frames.
        # Precomputed at push time (~90 Hz) so the smoothing render in sample()
        # is pure vectorized numpy — walking the pydantic models per sample was
        # ~0.8 ms/call on Jetson, enough to slow the IK dispatch loop it runs
        # on (in series with every solve) and cost more smoothness through a
        # coarser IK staircase than the smoothing itself bought.
        self._vecs: list[np.ndarray] = []
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
            self._vecs.clear()
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
                self._vecs.clear()
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
            self._vecs.insert(i, _frame_vec(frame))

            # Prune: keep a little history behind the current playout point
            # (which is held an extra half smoothing-window in the past).
            play = (local_recv - self._clock_offset) - self._delay - self._half_smooth
            keep_before = play - 0.5
            drop = 0
            while drop < len(self._caps) - 2 and self._caps[drop] < keep_before:
                drop += 1
            if drop:
                del self._caps[:drop]
                del self._frames[:drop]
                del self._vecs[:drop]
            extra = len(self._caps) - self._max_frames
            if extra > 0:
                del self._caps[:extra]
                del self._frames[:extra]
                del self._vecs[:extra]

    def sample(self, now: float | None = None) -> VRFrame | None:
        """Render the current smoothed target frame.

        Renders a Gaussian-weighted, outlier-rejected average of the buffered
        frames within the smoothing window around the playout point; falls back
        to plain lerp/slerp between the two bracketing frames when the window
        is too sparse (startup, unstamped transports, ``smooth_window_s == 0``).

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

            # The extra half-window playout shift only applies when smoothing
            # is active (client-stamped streams); unstamped streams keep the
            # undelayed playout point so they still render the latest frame.
            smoothing = self._half_smooth > 0.0 and bool(self._t_is_client)
            play = (now - self._clock_offset) - self._delay
            if smoothing:
                play -= self._half_smooth
            caps = self._caps
            frames = self._frames

            # Fixed-lag smoothing: snapshot the window slices under the lock
            # (frames are immutable models; slicing copies only the refs).
            # The slice spans the *rejection* window — wider than the Gaussian
            # smoothing window so the Hampel median sees enough clean frames to
            # out-vote a glitch burst, at no extra latency (the history side is
            # already buffered; the future side is capped by what has arrived).
            # Only client-stamped streams are smoothed: unstamped frames carry
            # arrival timestamps, so a delivery burst collapses distinct poses
            # onto near-identical time coordinates and the weighted average
            # would blend them into a false target. Unstamped streams fall
            # through to the lerp path, which renders the latest frame.
            win_caps: list[float] | None = None
            win_vecs: list[np.ndarray] | None = None
            if smoothing:
                half_rej = self._half_smooth * _REJECT_WINDOW_MULT
                lo = bisect.bisect_left(caps, play - half_rej)
                hi = bisect.bisect_right(caps, play + half_rej)
                if hi - lo >= 3:
                    win_caps = caps[lo:hi]
                    win_vecs = self._vecs[lo:hi]

            if win_vecs is None:
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

        if win_vecs is not None and win_caps is not None:
            rendered, pos = _smooth_window(
                win_vecs,
                win_caps,
                play,
                self._half_smooth,
                self._outlier_k,
                self._outlier_floor,
                latest,
            )
        else:
            rendered, pos = _interpolate(a, b, alpha, latest)

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


def _quat_weighted_mean(quats: np.ndarray, w: np.ndarray, ref_i: int) -> np.ndarray:
    """Weighted mean of ``(n, 4)`` quaternions, hemisphere-aligned to row ``ref_i``.

    Sign-aligning every quaternion to the reference before the weighted sum
    makes the normalized result a good mean for the small angular spreads seen
    within a smoothing window.
    """
    ref = quats[ref_i]
    sign = np.where(quats @ ref < 0.0, -1.0, 1.0)
    q = (w * sign) @ quats
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-12 else ref.copy()


# The Hampel rejection window is this multiple of the smoothing window, so a
# glitch burst as long as the whole smoothing window is still a minority for
# the median and gets rejected. Costs no latency: the extra frames come from
# history already buffered (and whatever future frames have arrived).
_REJECT_WINDOW_MULT = 2.5


# _frame_vec layout: 4 position streams, 2 quaternions, 2 grips.
_VEC_POS = slice(0, 12)  # l_ee, r_ee, l_elbow, r_elbow (3 each)
_VEC_LQ = slice(12, 16)
_VEC_RQ = slice(16, 20)
_VEC_GRIP = slice(20, 22)


def _frame_vec(f: VRFrame) -> np.ndarray:
    """Flatten a frame's motion fields into a (22,) float64 vector.

    Computed once per received frame so the smoothing render never walks the
    pydantic models on the hot sampling path.
    """
    lp, rp, le, re = f.l_ee.position, f.r_ee.position, f.l_elbow, f.r_elbow
    lq, rq = f.l_ee.quaternion, f.r_ee.quaternion
    return np.array(
        [
            lp.x, lp.y, lp.z,
            rp.x, rp.y, rp.z,
            le.x, le.y, le.z,
            re.x, re.y, re.z,
            lq.x, lq.y, lq.z, lq.w,
            rq.x, rq.y, rq.z, rq.w,
            f.l_grip, f.r_grip,
        ],
        dtype=np.float64,
    )  # fmt: skip


def _smooth_window(
    vecs: list[np.ndarray],
    caps: list[float],
    play: float,
    half_window: float,
    outlier_k: float,
    outlier_floor: float,
    latest: VRFrame,
) -> tuple[VRFrame, np.ndarray]:
    """Render the Gaussian-weighted, outlier-rejected mean pose of a window.

    ``vecs`` (per-frame :func:`_frame_vec` vectors) spans the wide *rejection*
    window; the Gaussian sigma comes from the narrower smoothing
    ``half_window``, so frames beyond the smoothing window carry negligible
    weight — *unless* the frames near the playout point are rejected as
    glitches, in which case the nearest clean frames dominate the renormalized
    weights and the output bridges smoothly across the glitch.

    Hampel rejection runs first: a frame whose EE or elbow position deviates
    from the window's component-wise median by more than
    ``outlier_k * 1.4826 * MAD`` (floored at ``outlier_floor`` metres) is
    excluded from the average entirely — orientation included, since a glitched
    tracking sample corrupts position and orientation together. The MAD scale
    grows with genuine motion spread, so fast intentional moves are never
    flagged; only a minority cluster far from the median is. Control state
    comes from ``latest``, exactly like :func:`_interpolate`.
    """
    V = np.stack(vecs)  # (n, 22)
    t = np.array(caps, dtype=np.float64)
    n = len(V)

    inlier = np.ones(n, dtype=bool)
    if outlier_k > 0.0:
        # (n, 4, 3): the four position streams of every frame. Medians use
        # np.partition (upper median for even n) — np.median's generality is
        # ~10x the cost on windows this small, and robust statistics don't
        # care about the midpoint averaging.
        P = V[:, _VEC_POS].reshape(n, 4, 3)
        mid = n // 2
        med = np.partition(P, mid, axis=0)[mid]  # (4, 3)
        diff = P - med
        d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))  # (n, 4)
        mad = np.partition(d, mid, axis=0)[mid]  # (4,)
        thresh = np.maximum(outlier_k * 1.4826 * mad, outlier_floor)
        inlier = np.all(d <= thresh, axis=1)
        if int(inlier.sum()) < 2:
            # Degenerate (e.g. bimodal window mid-jump): rejecting almost
            # everything would leave a meaningless average, so keep all frames.
            inlier[:] = True

    # Gaussian weights centred on the playout point; smoothing-window edges sit
    # at ~2 sigma so the average is dominated by frames near the playout time.
    sigma = max(half_window / 2.0, 1e-6)
    w = np.exp(-0.5 * ((t - play) / sigma) ** 2)
    w[~inlier] = 0.0
    w /= w.sum()

    ref_i = int(np.argmax(w))
    mean = w @ V  # (22,) — correct for all linear fields; quats fixed below
    l_qm = _quat_weighted_mean(V[:, _VEC_LQ], w, ref_i)
    r_qm = _quat_weighted_mean(V[:, _VEC_RQ], w, ref_i)
    return _build_frame(
        mean[0:3],
        l_qm,
        mean[3:6],
        r_qm,
        mean[6:9],
        mean[9:12],
        float(mean[20]),
        float(mean[21]),
        latest,
    )


def _interpolate(
    a: VRFrame, b: VRFrame, alpha: float, latest: VRFrame
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
    return _build_frame(
        l_ee_p, l_ee_q, r_ee_p, r_ee_q, l_el, r_el, l_grip, r_grip, latest
    )


def _build_frame(
    l_ee_p: np.ndarray,
    l_ee_q: np.ndarray,
    r_ee_p: np.ndarray,
    r_ee_q: np.ndarray,
    l_el: np.ndarray,
    r_el: np.ndarray,
    l_grip: float,
    r_grip: float,
    latest: VRFrame,
) -> tuple[VRFrame, np.ndarray]:
    """Assemble a rendered frame; motion from the args, control state from
    ``latest``. Returns ``(frame, pos_vector)`` where ``pos_vector`` is the
    concatenated EE+elbow positions used for the change/identity check.

    Uses ``model_construct`` (no validation) with explicit ``float()``
    conversion: this runs per sample on the IK dispatch thread and the fields
    are numerics we computed ourselves, so validation is pure overhead.
    """
    frame = VRFrame.model_construct(
        l_ee=VRPose.model_construct(
            position=VRPosition.model_construct(
                x=float(l_ee_p[0]), y=float(l_ee_p[1]), z=float(l_ee_p[2])
            ),
            quaternion=VRQuaternion.model_construct(
                x=float(l_ee_q[0]),
                y=float(l_ee_q[1]),
                z=float(l_ee_q[2]),
                w=float(l_ee_q[3]),
            ),
        ),
        r_ee=VRPose.model_construct(
            position=VRPosition.model_construct(
                x=float(r_ee_p[0]), y=float(r_ee_p[1]), z=float(r_ee_p[2])
            ),
            quaternion=VRQuaternion.model_construct(
                x=float(r_ee_q[0]),
                y=float(r_ee_q[1]),
                z=float(r_ee_q[2]),
                w=float(r_ee_q[3]),
            ),
        ),
        l_elbow=VRPosition.model_construct(
            x=float(l_el[0]), y=float(l_el[1]), z=float(l_el[2])
        ),
        r_elbow=VRPosition.model_construct(
            x=float(r_el[0]), y=float(r_el[1]), z=float(r_el[2])
        ),
        l_grip=float(l_grip),
        r_grip=float(r_grip),
        # Control state is responsive: always the latest received, never delayed.
        l_lock=latest.l_lock,
        r_lock=latest.r_lock,
        reset=latest.reset,
        state=latest.state,
        t=latest.t,
        seq=latest.seq,
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
