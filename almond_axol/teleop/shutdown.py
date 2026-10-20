"""Shutdown park: rest, then zero, before a flow torques the arms off.

Disabling raised arms drops them under gravity, so every hardware flow parks
them first. The park is two legs of the same guarded, collision-aware return
every rest move uses (:meth:`VRTeleopCore.guarded_return`): home to the rest
pose, then — with ``VRTeleopConfig.park_to_zero`` — on to the all-zero joint
pose, where the arms hang straight down and nothing moves when the torque
comes off. The grippers hold the grasp that was commanded when the park began
through both legs (:meth:`VRTeleopCore.begin_park`).

Native ``axol teleop`` (``VRTeleop._park_arms``) and ``collect-data``
(``_park_before_disconnect``) both run :func:`park_arms`, so the sequencing
cannot drift between them. Each caller supplies its own robot hooks and
decides what to log; every outcome other than ``"done"`` leaves the arms
where they are for the torque-off that follows, which is what happened before
the park existed.
"""

from __future__ import annotations

import contextlib
import signal
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from typing import TYPE_CHECKING

import numpy as np

from ..constants import PARK_TIMEOUT_S, PARK_ZERO_TIMEOUT_S

if TYPE_CHECKING:
    from .core import VRTeleopCore

# What each non-``done`` outcome means for the operator, worded for the
# warning the flows log before torquing the arms off where they are.
OUTCOME_MESSAGES: dict[str, str] = {
    "contact": "the park stopped on contact; disabling the arms where they are",
    "timeout": "the park ran out of time; disabling the arms where they are",
    "aborted": "the park was aborted (second Ctrl+C); disabling the arms where they are",
    "failed": "the park's plan was refused; disabling the arms where they are",
}


@contextlib.contextmanager
def abort_on_sigint(abort: threading.Event) -> Iterator[None]:
    """Turn a Ctrl+C during the park into an abort request, not an exception.

    Teardown runs with SIGINT ignored so a repeated Ctrl+C can't abandon the
    arms mid-move or skip the torque-off; inside the park a second Ctrl+C is
    still meaningful — the operator wants the arms to stop where they are —
    so it sets ``abort`` instead. Main thread only (Python forbids handler
    changes elsewhere, and under ``axol serve`` there is no Ctrl+C to catch);
    off the main thread this is a no-op.
    """
    previous: object | None = None
    installed = False
    try:
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda _signum, _frame: abort.set())
        installed = True
    except (OSError, ValueError):
        pass
    try:
        yield
    finally:
        if installed:
            with contextlib.suppress(OSError, ValueError):
                signal.signal(signal.SIGINT, previous)


async def park_arms(
    core: VRTeleopCore,
    *,
    park_to_zero: bool,
    send_step: Callable[[], Awaitable[object]],
    gravity_step: Callable[[], Awaitable[object]],
    torque_residuals: Callable[[], tuple[object, object]],
    reset_command_state: Callable[[], None],
    get_positions: Callable[[], tuple[np.ndarray, np.ndarray]],
    announce: Callable[[str], None],
    abort: Callable[[], bool] | None = None,
    rest_timeout_s: float = PARK_TIMEOUT_S,
    zero_timeout_s: float = PARK_ZERO_TIMEOUT_S,
) -> str:
    """Play the shutdown park on ``core``: ``done`` or why it stopped early.

    The caller has already decided a park is possible (hardware target,
    position-controlled arms, a live IK worker and dispatch thread, a bus
    that reports positions). Legs: the return to rest (skipped when the arms
    are already home and no reset is playing — a stop that lands mid-reset
    finishes that move), then the zero leg when ``park_to_zero``. A leg
    that ends on contact, its cap, an abort, or a refused plan ends the park
    there: nobody is left to hand-guide the arms and press reset, so a limp
    hold would only delay the same torque-off. Tracking is blocked for the
    whole park; the grips and the headset's reset button are ignored.

    Returns ``"done"``, ``"contact"``, ``"timeout"``, ``"aborted"`` or
    ``"failed"`` (see :data:`OUTCOME_MESSAGES`). Hook arguments are those of
    :meth:`VRTeleopCore.guarded_return`; ``abort`` is polled every cycle.
    """
    legs: list[tuple[str, float]] = []
    if not (core.at_rest and not core.is_resetting):
        legs.append(("rest", rest_timeout_s))
    if park_to_zero:
        legs.append(("zero", zero_timeout_s))
    if not legs:
        return "done"
    core.block_engage()
    core.begin_park()
    try:
        for goal, timeout in legs:
            contact = False
            deadline = time.perf_counter() + timeout

            def _on_contact() -> None:
                nonlocal contact
                contact = True

            def _stopped() -> bool:
                return (
                    contact
                    or time.perf_counter() >= deadline
                    or (abort is not None and abort())
                )

            core.reset_error = None
            if goal == "zero":
                announce("Parking the arms at zero before torque-off.")
                core.request_zero()
            else:
                announce("Returning the arms to rest before torque-off.")
                if not core.is_resetting:
                    core.request_reset()
            await core.guarded_return(
                send_step=send_step,
                gravity_step=gravity_step,
                torque_residuals=torque_residuals,
                reset_command_state=reset_command_state,
                get_positions=get_positions,
                stopped=_stopped,
                # The hold's "press reset" prompt has no one to read it here.
                announce=lambda msg: None if contact else announce(msg),
                on_contact=_on_contact,
                move_timeout_s=timeout,
            )
            if contact:
                return "contact"
            if abort is not None and abort():
                return "aborted"
            if time.perf_counter() >= deadline:
                return "timeout"
            if core.reset_error is not None:
                return "failed"
        return "done"
    finally:
        core.unblock_engage()
