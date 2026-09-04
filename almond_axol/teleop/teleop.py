"""
VR teleoperation for the Axol robot.

VRTeleop connects a VRServer (headset input) and a MotionControl implementation
(Axol hardware or Sim visualizer) into a single runnable teleop session. IK
runs in a separate subprocess so JAX never blocks the asyncio event loop.

Typical usage::

    from almond_axol.robot import Sim
    from almond_axol.teleop import VRTeleop

    async def main():
        sim = Sim()
        async with VRTeleop(sim) as teleop:
            await teleop.run()

Or with custom components::

    async with VRTeleop(
        Axol(),
        config=VRTeleopConfig(teleop_max_vel=2.0),
        vr_server_config=VRServerConfig(port=9000),
    ) as teleop:
        await teleop.run()
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import multiprocessing
import multiprocessing.connection
import multiprocessing.context
import os
import signal
import threading
import time

import numpy as np

from ..kinematics import KinematicsConfig
from ..robot.base import (
    HardwareCleanupError,
    RobotBase,
    mark_hardware_cleanup_uncertain,
)
from ..robot.cart import Cart
from ..robot.control import ContactWatchdog
from ..utils.jetson_diag import TegraStatsDiag
from ..utils.proc_diag import SystemDiag
from ..vr.config import VRServerConfig
from ..vr.server import VRServer
from .config import VRTeleopConfig
from .core import VRTeleopCore
from .worker import run_ik_worker

_logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _defer_sigint_during_cleanup():
    """Keep a repeated Ctrl-C from interrupting ownership teardown.

    ``asyncio.run`` turns the first SIGINT into task cancellation and a second
    one into an immediate ``KeyboardInterrupt``. The latter can otherwise land
    inside a process/thread join and skip the remaining terminate/kill checks.
    Once shutdown starts, finish its bounded teardown before restoring the
    caller's handler.

    ``VRTeleop`` can also be used from a non-main thread, where Python forbids
    changing signal handlers. In that case teardown still runs, just without
    the main-thread-only guard.
    """

    previous_handler: object | None = None
    handler_installed = False
    try:
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        handler_installed = True
    except (OSError, ValueError):
        pass
    try:
        yield
    finally:
        if handler_installed:
            try:
                signal.signal(signal.SIGINT, previous_handler)
            except (OSError, ValueError):
                _logger.warning(
                    "could not restore the SIGINT handler after teleop cleanup",
                    exc_info=True,
                )


class VRTeleop:
    """Connects a VR headset and robot into a teleoperation session.

    IK runs in a dedicated subprocess; the main process handles frame
    dispatch, smoothing, reset trajectory playback, and robot I/O.

    Args:
        robot:             Hardware or simulation target implementing :class:`MotionControl`.
        config:            Teleop session parameters (rest poses, loop frequency).
        kinematics_config: IK solver parameters forwarded to the subprocess.
        vr_server_config:  VR WebSocket server parameters (port, TLS certs).
        cart:              Powered cart (x-drive base + lift) for robots that
                           have one; ``None`` for a static base.
    """

    def __init__(
        self,
        robot: RobotBase,
        *,
        config: VRTeleopConfig = VRTeleopConfig(),
        kinematics_config: KinematicsConfig = KinematicsConfig(),
        vr_server_config: VRServerConfig = VRServerConfig(),
        cart: Cart | None = None,
    ) -> None:
        """Construct the teleoperation session.

        No network connections or subprocesses are started until :meth:`enable`
        (or ``async with``) is called.

        Args:
            robot:             Hardware or simulation target implementing :class:`RobotBase`.
            config:            Teleop loop parameters (rest poses, frequency, velocity limits).
            kinematics_config: IK solver cost weights forwarded to the IK subprocess.
            vr_server_config:  VR WebSocket server parameters (port, TLS certs).
            cart:              Powered cart (x-drive base + telescoping lift),
                               or ``None`` for a robot on a static base. When
                               present, the headset thumbsticks drive it: left
                               stick translates, right stick x rotates, stick
                               clicks run the lift (left down / right up).
                               Deflection is the deadman — the cart is active
                               whenever a stick leaves its deadzone,
                               independent of the arm engage toggle.
        """
        self._robot = robot
        self._cart = cart
        self._config = config
        self._kinematics_config = kinematics_config
        self._vr_server = VRServer(vr_server_config)
        self._vr_server.set_on_frame(self._on_vr_frame)
        # Lock the headset HUD to plain teleop: no data-collection toggle and no
        # recording controls, since this flow never records.
        self._vr_server.set_mode("teleop")
        self._vr_server.set_pose_mode(
            "absolute" if config.absolute_mode else "relative"
        )

        # Engage toggle, EMA/trapezoidal smoothing, and reset handling all live
        # in the shared core so this flow and `axol collect-data` (AxolVRTeleop)
        # cannot drift apart.
        self._core = VRTeleopCore(
            config, _logger, self._broadcast_tracking, self._broadcast_json
        )

        self._parent_conn: multiprocessing.connection.Connection | None = None
        self._ik_process: multiprocessing.context.SpawnProcess | None = None
        self._ik_thread: threading.Thread | None = None
        self._ik_stop: threading.Event = threading.Event()
        # Stashed only so the CPU diag can label the relay subprocess.
        self._video_manager: object | None = None

        self._ik_loop_times: list[float] = []
        self._ik_loop_times_lock: threading.Lock = threading.Lock()
        self._vr_frame_times: list[float] = []
        self._vr_frame_times_lock: threading.Lock = threading.Lock()

        self._vr_thread: threading.Thread | None = None
        self._vr_stop: threading.Event = threading.Event()
        self._vr_ready: threading.Event = threading.Event()
        # Set by the VR thread before ``_vr_ready`` is signalled when server
        # startup fails.  Without this side channel the caller waits forever
        # when certificate creation or socket binding raises before the normal
        # ready signal.  The value is reset for every enable attempt so a
        # cleaned-up instance remains retryable.
        self._vr_start_error: BaseException | None = None
        # A failure while undoing a partial VR startup is distinct from the
        # startup error itself: enable() preserves the useful primary error,
        # while disable() reports uncertain background cleanup after it has
        # attempted all other teardown.
        self._vr_cleanup_error: BaseException | None = None
        # Event loop of the VR server thread, captured so the IK thread can
        # broadcast tracking-state changes to the headset.
        self._vr_loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _run_vr_thread(self) -> None:
        """Run the VR WebSocket server in its own asyncio event loop.

        Keeping VR in a dedicated thread prevents burst WebSocket callbacks
        from contending with the IK thread for the GIL or CPU time.
        """

        async def _serve() -> None:
            self._vr_loop = asyncio.get_running_loop()
            try:
                await self._vr_server.enable()
            except BaseException as startup_error:
                self._vr_start_error = startup_error
                # Wake enable() before attempting cleanup.  Teardown is meant
                # to be bounded, but a third-party WebRTC close must not be
                # able to turn a known bind/certificate failure back into an
                # unbounded startup wait.
                self._vr_ready.set()
                # enable() may have opened a socket or created a background
                # server task before a later setup step failed.  Release any
                # partial state on this same event loop before the thread exits.
                try:
                    await self._vr_server.disable()
                except BaseException as cleanup_error:
                    self._vr_cleanup_error = cleanup_error
                    startup_error.add_note(
                        "VR server cleanup after failed startup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                return
            self._vr_ready.set()
            try:
                while not self._vr_stop.is_set():
                    await asyncio.sleep(0.05)
            finally:
                await self._vr_server.disable()

        try:
            asyncio.run(_serve())
        except BaseException as thread_error:
            # A failure after readiness is a shutdown/serve failure.  A startup
            # failure is handled inside _serve so its original exception is not
            # replaced by a cleanup exception.
            self._vr_cleanup_error = thread_error
        finally:
            # This must be unconditional: it is the completion signal for both
            # successful startup and every failure before the listener is live.
            self._vr_ready.set()

    def _broadcast_tracking(self, enabled: bool) -> None:
        """Push the engage-toggle state to the headset (fire-and-forget).

        The VR app uses it to allow screen repositioning (trigger grabs) only
        while the robot isn't being controlled. Safe to call from any thread.
        """
        if self._vr_loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._vr_server.broadcast_tracking(enabled), self._vr_loop
            )
        except RuntimeError:
            pass  # VR loop already shut down

    def _broadcast_json(self, obj: dict) -> None:
        """Push an arbitrary JSON message to the headset (fire-and-forget).

        Used by the shared core for the URDF overlay state in absolute (Mantis)
        mode. Safe to call from any thread.
        """
        if self._vr_loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._vr_server.broadcast_text(json.dumps(obj)), self._vr_loop
            )
        except RuntimeError:
            pass  # VR loop already shut down

    async def enable(self) -> None:
        """Start the VR server, robot, and IK subprocess."""
        if any(
            resource is not None
            for resource in (
                self._ik_thread,
                self._parent_conn,
                self._ik_process,
                self._vr_thread,
            )
        ):
            raise HardwareCleanupError(
                "previous teleop cleanup is incomplete; retry disable() before "
                "starting another session"
            )
        self._vr_stop.clear()
        self._vr_ready.clear()
        self._vr_start_error = None
        self._vr_cleanup_error = None
        self._vr_thread = threading.Thread(
            target=self._run_vr_thread, daemon=True, name="vr-server"
        )
        self._vr_thread.start()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._vr_ready.wait)
        if self._vr_start_error is not None:
            raise self._vr_start_error

        await self._robot.enable()
        if self._cart is not None:
            # After the arms so a cart failure (missing CAN interface, GPIO
            # chip) surfaces before the IK worker spins up; its command task
            # runs on this event loop.
            await self._cart.enable()

        pos_l, pos_r = await self._robot.get_positions()
        self._core.set_initial_grips(
            pos_l[7] if pos_l is not None else None,
            pos_r[7] if pos_r is not None else None,
        )

        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._parent_conn = parent_conn

        try:
            process = ctx.Process(
                target=run_ik_worker,
                args=(
                    child_conn,
                    self._config,
                    self._kinematics_config,
                    pos_l[:7] if pos_l is not None else None,
                    pos_r[:7] if pos_r is not None else None,
                ),
                daemon=True,
            )
        except BaseException as process_error:
            try:
                child_conn.close()
            except BaseException as close_error:
                process_error.add_note(
                    "additional IK child-pipe close failure: "
                    f"{type(close_error).__name__}: {close_error}"
                )
            raise
        # Retain the exact object before start: start() may create the child
        # and then raise, and the following local pipe close can also fail.
        # __aenter__'s unwind must be able to reap either partial startup.
        self._ik_process = process
        try:
            process.start()
        except BaseException as start_error:
            try:
                child_conn.close()
            except BaseException as close_error:
                start_error.add_note(
                    "additional IK child-pipe close failure: "
                    f"{type(close_error).__name__}: {close_error}"
                )
            raise
        child_conn.close()

        loop = asyncio.get_running_loop()
        msg = await loop.run_in_executor(None, parent_conn.recv)
        assert isinstance(msg, tuple) and msg[0] == "ready"
        _, q_init, left_indices, right_indices, startup_traj = msg
        self._core.set_solution(q_init, left_indices, right_indices)

        cur_l, cur_r = await self._robot.get_positions()
        self._core.seed_filters(cur_l, cur_r)
        self._core.set_startup_trajectory(startup_traj)

        self._ik_stop.clear()
        self._ik_thread = threading.Thread(
            target=self._ik_loop, daemon=True, name="ik-loop"
        )
        self._ik_thread.start()

    async def disable(self) -> None:
        """Disable motors, stop IK subprocess, and stop VR server."""
        with _defer_sigint_during_cleanup():
            await self._disable()

    async def _disable(self) -> None:
        """Teardown implementation, called with repeated SIGINT deferred."""
        cleanup_failures: list[tuple[str, BaseException]] = []
        hardware_failures: list[tuple[str, BaseException]] = []

        # Stop new work first, then turn the hardware off before waiting on
        # background workers.  A wedged IK reader must not delay torque-off.
        if self._ik_thread is not None:
            self._ik_stop.set()
        if self._vr_thread is not None:
            self._vr_stop.set()

        if self._cart is not None:
            try:
                await self._cart.disable()
            except BaseException as exc:  # finish the arm shutdown too
                _logger.exception("cart disable failed")
                hardware_failures.append(("cart", exc))
        try:
            await self._robot.disable()
        except BaseException as exc:
            _logger.exception("robot disable failed")
            hardware_failures.append(("robot", exc))

        thread = self._ik_thread
        if thread is not None:
            try:
                thread.join(timeout=3.0)
            except BaseException as exc:
                cleanup_failures.append(("IK thread join", exc))
            try:
                thread_alive = thread.is_alive()
            except BaseException as exc:
                cleanup_failures.append(("IK thread liveness check", exc))
                thread_alive = True
            if thread_alive:
                cleanup_failures.append(
                    ("IK thread", RuntimeError("IK dispatch thread did not stop"))
                )
            else:
                self._ik_thread = None

        if self._parent_conn is not None:
            # Only send the graceful sentinel after the dispatch thread has
            # released its end of the pipe. Closing still happens when it is
            # stuck, which unblocks a pending recv without racing two readers.
            if self._ik_thread is None:
                try:
                    while self._parent_conn.poll(0):
                        self._parent_conn.recv()
                    self._parent_conn.send(None)
                except BaseException:
                    # EOF/worker death is acceptable; process termination below
                    # is the authoritative ownership check.
                    pass
            try:
                # Drain in-flight worker responses before closing: the dispatch
                # thread can stop mid solve, and closing a connection with
                # unread data RSTs the peer — the worker's blocking recv then
                # dies with ConnectionResetError instead of reading the None
                # shutdown sentinel.
                self._parent_conn.close()
            except BaseException as exc:
                cleanup_failures.append(("IK pipe close", exc))
            else:
                self._parent_conn = None

        process = self._ik_process
        if process is not None:
            # A control signal interrupting one blocking join is not evidence
            # that the child remains live. Continue through liveness probes
            # and stronger shutdown actions; report the interruption only if
            # final ownership still cannot be proved.
            process_interruptions: list[tuple[str, BaseException]] = []

            def process_join(label: str, timeout: float) -> None:
                try:
                    process.join(timeout=timeout)
                except (KeyboardInterrupt, asyncio.CancelledError) as exc:
                    process_interruptions.append((label, exc))
                except BaseException as exc:
                    cleanup_failures.append((label, exc))

            def process_is_alive(label: str) -> bool:
                try:
                    return bool(process.is_alive())
                except (KeyboardInterrupt, asyncio.CancelledError) as exc:
                    process_interruptions.append((label, exc))
                    return True
                except BaseException as exc:
                    cleanup_failures.append((label, exc))
                    return True

            try:
                process_started = process.pid is not None
            except BaseException as exc:
                cleanup_failures.append(("IK process pid check", exc))
                process_started = True

            if process_started:
                process_join("IK process graceful join", 3.0)
                process_alive = process_is_alive("IK process post-join liveness check")
            else:
                # multiprocessing keeps pid at None until start succeeds. An
                # unstarted child needs no join (which itself raises), but
                # still probe rather than assuming a custom Process is inert.
                process_alive = process_is_alive("unstarted IK process liveness check")

            if process_alive:
                try:
                    process.terminate()
                except (KeyboardInterrupt, asyncio.CancelledError) as exc:
                    process_interruptions.append(("IK process terminate", exc))
                except BaseException as exc:
                    cleanup_failures.append(("IK process terminate", exc))
                process_join("IK process post-terminate join", 2.0)
                process_alive = process_is_alive(
                    "IK process post-terminate liveness check"
                )
            if process_alive:
                try:
                    process.kill()
                except (KeyboardInterrupt, asyncio.CancelledError) as exc:
                    process_interruptions.append(("IK process kill", exc))
                except BaseException as exc:
                    cleanup_failures.append(("IK process kill", exc))
                process_join("IK process post-kill join", 2.0)
                process_alive = process_is_alive("IK process post-kill liveness check")

            if process_alive:
                cleanup_failures.append(
                    ("IK process", RuntimeError("IK worker process did not stop"))
                )
                cleanup_failures.extend(process_interruptions)
            else:
                self._ik_process = None

        vr_thread = self._vr_thread
        if vr_thread is not None:
            try:
                vr_thread.join(timeout=5.0)
            except BaseException as exc:
                cleanup_failures.append(("VR thread join", exc))
            try:
                vr_alive = vr_thread.is_alive()
            except BaseException as exc:
                cleanup_failures.append(("VR thread liveness check", exc))
                vr_alive = True
            if vr_alive:
                cleanup_failures.append(
                    ("VR thread", RuntimeError("VR server thread did not stop"))
                )
            else:
                self._vr_loop = None
                vr_cleanup_error = getattr(self, "_vr_cleanup_error", None)
                if vr_cleanup_error is not None:
                    # The thread's same-loop cleanup failed.  Retry after the
                    # thread has exited; if this still fails, retain its
                    # reference so another disable() is required before enable
                    # may start a replacement server against uncertain state.
                    try:
                        await self._vr_server.disable()
                    except BaseException as retry_error:
                        vr_cleanup_error.add_note(
                            "VR server cleanup retry also failed: "
                            f"{type(retry_error).__name__}: {retry_error}"
                        )
                        cleanup_failures.append(("VR server cleanup", vr_cleanup_error))
                    else:
                        self._vr_thread = None
                        self._vr_start_error = None
                        self._vr_cleanup_error = None
                else:
                    self._vr_thread = None
                    self._vr_start_error = None
                    self._vr_cleanup_error = None

        if hardware_failures:
            first_label, first_failure = hardware_failures[0]
            for extra_label, extra in hardware_failures[1:] + cleanup_failures:
                first_failure.add_note(
                    f"additional teleop {extra_label} cleanup failure: "
                    f"{type(extra).__name__}: {extra}"
                )
            if isinstance(first_failure, HardwareCleanupError):
                raise first_failure
            raise HardwareCleanupError(
                f"teleop {first_label} disable failed; hardware ownership is uncertain"
            ) from first_failure
        if cleanup_failures:
            label, first_failure = cleanup_failures[0]
            for extra_label, extra in cleanup_failures[1:]:
                first_failure.add_note(
                    f"additional teleop {extra_label} cleanup failure: "
                    f"{type(extra).__name__}: {extra}"
                )
            raise RuntimeError(
                f"teleop {label} failed; background ownership is uncertain"
            ) from first_failure

    async def __aenter__(self) -> VRTeleop:
        # If enable() is interrupted partway — e.g. the operation is stopped
        # while the IK worker is still compiling JAX, cancelling enable()'s wait
        # for its "ready" message — __aexit__ never runs (the context never
        # finished entering). Without cleanup here the already-started VR server
        # thread leaks and keeps holding its WebSocket port, so the next teleop
        # fails to bind it ("address already in use"). Tear down what enable()
        # started before propagating.
        try:
            await self.enable()
        except BaseException as setup_error:
            try:
                await self.disable()
            except BaseException as cleanup_error:
                _logger.exception("teleop startup cleanup failed")
                mark_hardware_cleanup_uncertain(setup_error, cleanup_error)
            raise
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disable()

    def set_video_expected(self, expected: bool) -> None:
        """Declare that camera video will be registered once it finishes starting.

        Call before ``async with`` / :meth:`enable` when cameras are
        configured: the VR server starts accepting headsets long before the
        video relay finishes opening the cameras, and an early video request
        would otherwise be answered "unavailable" — hiding the camera screens
        for the whole session. With this set, early requests wait
        (``webrtc-pending``) and receive their offer the moment
        :meth:`set_video_manager` / :meth:`set_video_sources` lands. See
        :meth:`almond_axol.vr.server.VRServer.set_video_expected`.
        """
        self._vr_server.set_video_expected(expected)

    def set_video_sources(self, sources: dict[str, object] | None) -> None:
        """Stream camera frames to the headset via WebRTC.

        Each value is a connected ``ZedCamera`` / stereo eye (registered
        directly) or any raw-frame source exposing ``width`` / ``height`` /
        ``fps`` + ``wait_next``; see
        :meth:`almond_axol.vr.server.VRServer.set_video_sources`. Must be
        called after :meth:`enable` (so the VR server exists). Safe to call
        from any thread. Requires the GStreamer NVENC stack (``axol
        gst.install``); without it video is silently disabled.
        """
        self._vr_server.set_video_sources(sources)  # type: ignore[arg-type]

    def set_video_manager(self, manager: object | None) -> None:
        """Stream camera video via a pre-built WebRTC manager.

        Used with the out-of-process video relay
        (:class:`almond_axol.video.video_proc.VideoRelayProcess`) so encoding
        and RTP traffic never contend with the teleop control loops. Must
        be called after :meth:`enable`. Safe to call from any thread.
        """
        self._video_manager = manager
        self._vr_server.set_video_manager(manager)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run the teleop control loop until cancelled."""
        interval = 1.0 / self._config.frequency
        loop_times: list[float] = []
        last_log = time.perf_counter()
        # Per-section timing, matching `axol collect-data` so the two flows can
        # be compared directly: `step` is the smoothing, `send` is the CAN
        # motion_control round-trip. A `send` that stays flat here but inflates
        # under collect-data points at cross-thread GIL contention, not the bus.
        sect = {"step": 0.0, "send": 0.0}
        max_gap = 0.0  # worst loop-iteration spacing within the window
        max_slip = 0.0  # worst lateness past the absolute deadline
        prev_iter = 0.0

        # Same /proc CPU sampler as collect-data, so the two flows' per-core
        # saturation + hottest process/thread breakdowns line up exactly.
        diag_labels: dict[int, str] = {os.getpid(): "main"}
        if self._ik_process is not None and getattr(self._ik_process, "pid", None):
            diag_labels[self._ik_process.pid] = "ik"
        relay_proc = getattr(self._video_manager, "_proc", None)
        if getattr(relay_proc, "pid", None):
            diag_labels[relay_proc.pid] = "relay"
        diag = SystemDiag(diag_labels, _logger)
        diag.start()
        # Jetson GPU / EMC / NVENC / per-core-freq / thermal sampler. This is the
        # A/B baseline for collect-data: teleop runs with the relay raw branch
        # closed, so its diag/tegra lines isolate the record-phase delta.
        tegra = TegraStatsDiag(_logger)
        tegra.start()

        # Guarded return-to-rest needs torque feedback and gravity comp —
        # hardware (Axol) only. The Sim target has neither (and nothing
        # physical to protect), so its resets play through the normal path
        # below.
        guard = all(
            hasattr(self._robot, attr)
            for attr in (
                "torque_residuals",
                "gravity_compensate",
                "reset_command_state",
            )
        )

        async def _guard_send_step() -> None:
            left, right = self.step()
            if left is not None:
                await self._robot.motion_control(left=left, right=right)

        async def _guard_gravity_step() -> None:
            await self._robot.gravity_compensate(  # type: ignore[attr-defined]
                kd=self._config.reset_gravity_comp_kd
            )

        def _guard_positions() -> tuple[np.ndarray | None, np.ndarray | None]:
            left_arm = getattr(self._robot, "left", None)
            right_arm = getattr(self._robot, "right", None)
            return (
                left_arm.positions if left_arm is not None else None,
                right_arm.positions if right_arm is not None else None,
            )

        def _guard_vr_alive() -> bool:
            # Frames within the last ~2s: a Y-exit ends the XR session (the
            # stream stops instantly), so a contact hold on the way out has
            # no headset left to press reset — the engine settles it instead
            # of waiting forever.
            with self._vr_frame_times_lock:
                last = self._vr_frame_times[-1] if self._vr_frame_times else None
            return last is not None and (time.perf_counter() - last) < 2.0

        # Tracking-phase contact watchdog (hardware only, opt-in — the
        # threshold defaults to 0 = off): the same sustained-torque trip the
        # guarded return uses, but active while the operator drives (or
        # holds) the arms. On a trip tracking disengages and the arms go
        # limp until reset — see VRTeleopCore.contact_hold.
        track_watchdog = (
            ContactWatchdog(self._config.teleop_torque_threshold)
            if guard and self._config.teleop_torque_threshold > 0
            else None
        )

        _logger.info("VRTeleop loop started at %.0f Hz", self._config.frequency)
        # Track an absolute deadline so late wakeups are corrected in the next
        # cycle rather than accumulating as permanent drift.
        deadline = time.perf_counter()
        try:
            while True:
                # Every rest move — the startup trajectory, an X reset, the
                # Y-exit reset — plays through the shared guarded engine on
                # hardware: torque watchdog, and a limp gravity-comp hold on
                # contact (reset replans from wherever the arms are left).
                # See VRTeleopCore.guarded_return.
                if guard and self._core.is_resetting:
                    await self._core.guarded_return(
                        send_step=_guard_send_step,
                        gravity_step=_guard_gravity_step,
                        torque_residuals=self._robot.torque_residuals,  # type: ignore[attr-defined]
                        reset_command_state=self._robot.reset_command_state,  # type: ignore[attr-defined]
                        get_positions=_guard_positions,
                        stopped=lambda: False,  # unwound by task cancellation
                        announce=_logger.info,
                        vr_alive=_guard_vr_alive,
                    )
                    # Re-anchor pacing after the excursion so the next cycle
                    # doesn't try to catch up on the elapsed time.
                    deadline = time.perf_counter()
                    prev_iter = 0.0
                    continue
                deadline += interval
                t_start = time.perf_counter()
                left, right = self.step()
                t_step = time.perf_counter()
                await self._robot.motion_control(left=left, right=right)

                if track_watchdog is not None and left is not None:
                    tripped = track_watchdog.update(
                        self._robot.torque_residuals()  # type: ignore[attr-defined]
                    )
                    if tripped is not None:
                        joint, residual = tripped
                        _logger.warning(
                            "teleop contact: %s torque residual %.1f exceeds "
                            "%.1f — going limp",
                            joint,
                            residual,
                            self._config.teleop_torque_threshold,
                        )
                        await self._core.contact_hold(
                            gravity_step=_guard_gravity_step,
                            reset_command_state=self._robot.reset_command_state,  # type: ignore[attr-defined]
                            get_positions=_guard_positions,
                            stopped=lambda: False,  # unwound by task cancellation
                            announce=_logger.info,
                            vr_alive=_guard_vr_alive,
                        )
                        track_watchdog = ContactWatchdog(
                            self._config.teleop_torque_threshold
                        )
                        deadline = time.perf_counter()
                        prev_iter = 0.0
                        continue

                now = time.perf_counter()
                sect["step"] += t_step - t_start
                sect["send"] += now - t_step
                if prev_iter:
                    max_gap = max(max_gap, now - prev_iter)
                prev_iter = now
                loop_times.append(now)
                if now - last_log >= 1.0 and len(loop_times) > 1:
                    # Surface silent freeze modes: the control loop keeps
                    # commanding the last smoothed target when either of these
                    # dies, which reads as "teleop froze" with no explanation.
                    if self._ik_process is not None and not self._ik_process.is_alive():
                        _logger.error(
                            "IK worker process died (exit code %s) — arms are "
                            "holding the last target; restart teleop",
                            self._ik_process.exitcode,
                        )
                    elif self._ik_thread is not None and not self._ik_thread.is_alive():
                        _logger.error(
                            "IK dispatch thread died — arms are holding the "
                            "last target; restart teleop"
                        )
                    total = loop_times[-1] - loop_times[0]
                    rate = (len(loop_times) - 1) / total
                    with self._ik_loop_times_lock:
                        ik_times_snap = list(self._ik_loop_times)
                    if len(ik_times_snap) >= 2:
                        ik_total = ik_times_snap[-1] - ik_times_snap[0]
                        ik_hz = (
                            (len(ik_times_snap) - 1) / ik_total if ik_total > 0 else 0.0
                        )
                        with self._vr_frame_times_lock:
                            vr_times_snap = list(self._vr_frame_times)
                        if len(vr_times_snap) >= 2:
                            vr_total = vr_times_snap[-1] - vr_times_snap[0]
                            vr_hz = (
                                (len(vr_times_snap) - 1) / vr_total
                                if vr_total > 0
                                else 0.0
                            )
                            _logger.info(
                                "loop: %.1f Hz  vr: %.1f Hz  ik: %.1f Hz",
                                rate,
                                vr_hz,
                                ik_hz,
                            )
                        else:
                            _logger.info("loop: %.1f Hz  ik: %.1f Hz", rate, ik_hz)
                    else:
                        _logger.info("loop: %.1f Hz", rate)
                    n = len(loop_times)
                    _logger.debug(
                        "loop sections (mean ms): step=%.2f send=%.2f  "
                        "maxgap=%.1fms maxslip=%.1fms",
                        1e3 * sect["step"] / n,
                        1e3 * sect["send"] / n,
                        1e3 * max_gap,
                        1e3 * max_slip,
                    )
                    sect = {"step": 0.0, "send": 0.0}
                    max_gap = 0.0
                    max_slip = 0.0
                    loop_times.clear()
                    last_log = now

                await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
                slip = time.perf_counter() - deadline
                if slip > max_slip:
                    max_slip = slip
        except asyncio.CancelledError:
            pass
        finally:
            diag.stop()
            tegra.stop()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the latest smoothed joint positions.

        Returns ``(None, None)`` until the IK subprocess is ready. Once ready,
        always returns positions so the robot actively holds its commanded pose.
        The smoothing itself lives in :meth:`VRTeleopCore.compute_output` so it
        stays identical to the ``collect-data`` flow.

        Returns:
            Tuple ``(left, right)`` where each is a shape (8,) float32 array
            of joint positions in radians (Joint enum order), or ``None``
            if not yet ready.
        """
        out = self._core.compute_output()
        if out is None:
            return None, None
        return out[:8], out[8:]

    # ------------------------------------------------------------------
    # VR frame callback (runs on every incoming frame)
    # ------------------------------------------------------------------

    def _on_vr_frame(self, frame) -> None:
        """Latch the reset rising edge as soon as the frame arrives.

        Called from the VR server thread; uses a lock for the frame-time list.
        """
        now = time.perf_counter()
        with self._vr_frame_times_lock:
            self._vr_frame_times.append(now)
            while (
                len(self._vr_frame_times) > 1
                and self._vr_frame_times[-1] - self._vr_frame_times[0] > 2.0
            ):
                self._vr_frame_times.pop(0)
        self._core.note_frame_reset(frame.reset)
        if self._cart is not None:
            # The stick → cart mapping lives on Cart (shared with the
            # collect-data flow). Resets force a stop so the base doesn't
            # creep while the arms replay their return-to-rest trajectory.
            self._cart.apply_vr_frame(frame, resetting=self._core.is_resetting)

    # ------------------------------------------------------------------
    # IK loop (daemon thread)
    # ------------------------------------------------------------------

    def _note_ik_sample(self, now: float) -> None:
        with self._ik_loop_times_lock:
            self._ik_loop_times.append(now)
            while (
                len(self._ik_loop_times) > 1
                and self._ik_loop_times[-1] - self._ik_loop_times[0] > 2.0
            ):
                self._ik_loop_times.pop(0)

    def _ik_loop(self) -> None:
        """Dispatch VR frames to the IK subprocess via the shared core.

        Runs in a dedicated daemon thread so asyncio event-loop activity (e.g.
        VR WebSocket burst callbacks) cannot delay IK scheduling. All the engage
        / reset / dispatch logic lives in :meth:`VRTeleopCore.run_ik_loop`, so
        it stays identical to the ``collect-data`` flow.
        """
        assert self._parent_conn is not None
        self._core.run_ik_loop(
            self._parent_conn,
            self._vr_server.get_render_frame,
            self._ik_stop,
            lambda: self._ik_process is None or self._ik_process.is_alive(),
            self._note_ik_sample,
        )
