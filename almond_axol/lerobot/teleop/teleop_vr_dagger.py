"""VR teleoperator wired for DAgger interventions (``axol collect-dagger``).

:class:`DaggerVRTeleop` is :class:`~.teleop_vr.AxolVRTeleop` with the shared
core swapped for :class:`~almond_axol.teleop.dagger.DaggerTeleopCore` — the
freeze / take-over / hand-back grip-button state machine, robot-pose sync on
engage, and trigger-adopting grippers — plus the small intervention API the
DAgger control loop drives.

Everything else (VR server, IK subprocess, the ``sync`` message it needs — see
:func:`~almond_axol.teleop.worker.run_ik_worker` — smoothing, episode events)
is the stock adapter: episode boundaries via ``get_teleop_events()`` (record
button start/terminate, reset+stop rerecord) behave exactly like
``collect-data``.

:meth:`set_position_source` must be called before :meth:`connect` so the
engage-time sync can read the robot's measured positions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from ...teleop.dagger import DaggerTeleopCore
from .config_vr import AxolVRTeleopConfig
from .teleop_vr import AxolVRTeleop

_logger = logging.getLogger(__name__)


class DaggerVRTeleop(AxolVRTeleop):
    """``AxolVRTeleop`` wired for DAgger interventions.

    Differences from the base adapter:

    - the core is a :class:`DaggerTeleopCore` (freeze/engage/hand-back
      buttons, robot-pose sync, trigger-adopting grippers);
    - :meth:`set_position_source` must be called before :meth:`connect` so
      the engage-time sync can read the robot's measured positions.

    Episode events (``get_teleop_events``: record button start/terminate,
    reset+stop rerecord) keep the base behaviour, so the CLI drives episode
    boundaries exactly like ``collect-data``.
    """

    def __init__(self, config: AxolVRTeleopConfig) -> None:
        super().__init__(config)
        # Replace the stock core built by the base __init__ before anything
        # runs (connect starts the threads that use it).
        self._core: DaggerTeleopCore = DaggerTeleopCore(
            config.vr_teleop_config, _logger, self._broadcast_tracking
        )
        self._position_source: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None

    def set_position_source(
        self, get_positions: Callable[[], tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Provide the robot's cached-position getter (``lambda: robot.positions``).

        Must be called before :meth:`connect`; the engage-time robot sync
        reads it on the IK thread.
        """
        self._position_source = get_positions

    def connect(
        self,
        calibrate: bool = True,
        q_start_left: np.ndarray | None = None,
        q_start_right: np.ndarray | None = None,
    ) -> None:
        """Connect the stock stack, then wire the core's engage-time sync.

        Attaching after the base connect is safe: the grip buttons are inert
        (``intervention_allowed`` starts cleared) until the CLI arms them,
        which happens only after ``connect()`` returns — so nothing can reach
        the sync path before the pipe and position source are attached.
        """
        if self._position_source is None:
            raise RuntimeError(
                "DaggerVRTeleop.set_position_source() must be called before "
                "connect() so takeovers can sync to the robot's pose."
            )
        super().connect(calibrate, q_start_left, q_start_right)
        assert self._parent_conn is not None
        self._core.attach(self._parent_conn, self._position_source)

    # -- Intervention state exposed to the CLI --------------------------------

    @property
    def teleop_engaged(self) -> bool:
        """Whether the operator currently has control (both grips engaged)."""
        return self._core.teleop_enabled

    def consume_freeze(self) -> bool:
        """Return-and-clear the freeze latch (either grip pressed alone)."""
        return self._core.consume_freeze()

    def force_disengage(self) -> None:
        """Disengage teleop, e.g. when an episode ends mid-intervention."""
        self._core.force_disengage()

    def set_intervention_allowed(self, allowed: bool) -> None:
        """Arm/disarm the grip buttons (armed only while an episode is live)."""
        if allowed:
            self._core.consume_freeze()  # drop presses latched while disarmed
            self._core.intervention_allowed.set()
        else:
            self._core.intervention_allowed.clear()
            self._core.consume_freeze()

    def set_idle_reset_armed(self, armed: bool) -> None:
        """Arm/disarm the VR reset button's idle home request (idle phase only)."""
        if armed:
            self._core.consume_idle_reset()  # drop presses latched while disarmed
            self._core.idle_reset_armed.set()
        else:
            self._core.idle_reset_armed.clear()
            self._core.consume_idle_reset()

    def consume_idle_reset(self) -> bool:
        """Return-and-clear the idle home request (VR reset button)."""
        return self._core.consume_idle_reset()
