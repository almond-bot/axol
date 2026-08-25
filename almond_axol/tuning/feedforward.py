"""Per-run feedforward matching the production ``motion_control`` path."""

from __future__ import annotations

import math

from ..robot.control import (
    DAMP_BP_W0,
    VEL_CUTOFF_FREQ,
    BandPass,
    Differentiator,
    compute_friction,
)

FF_MODES = ("full", "gravity", "friction", "none")


class FeedForward:
    """Per-run feedforward matching the production ``motion_control`` path.

    ``compute(q_target, meas)`` returns ``(v_des, t_ff)`` where::

        t_ff = gravity(q_target) + friction(v_des) + j_eff · a_des
               + host_kd · (v_des − v_meas)

    ``gravity_fn`` evaluates the full-chain URDF model with the *other*
    joints at their real (measured) positions, not an assumed zero pose —
    shoulder_2 / wrist_2 are deliberately never parked at 0 (base
    collision), so assuming zeros there skews the model torque.

    ``host_kd`` adds host-side velocity damping. ``v_meas`` is the cached
    feedback position differentiated against the frame's CAN receive
    timestamp — matching production: frame-timestamped differentiation is
    jitter-free, and it stays *fresh*, unlike the motor-reported velocity
    (MyActuator's firmware estimate lags too much to damp the shoulders'
    ~2.3 Hz resonance — the same reason firmware kd underdelivers there).
    ``host_kd_hz`` sets the band-pass centre (Hz) confining that damping
    to the joint's resonance band, matching the production per-joint
    ``kd_host_hz``; ``None`` uses the shared shoulder default
    (:data:`~almond_axol.robot.control.DAMP_BP_W0` — 20 rad/s ≈ 3.2 Hz).

    Construct one instance per candidate run: the differentiators are
    stateful low-pass filters and must not leak between runs. For step
    references pass ``differentiate_target=False`` — differentiating a
    discontinuous target would fire a one-sample velocity/accel spike into
    the motor; production never sees that because ``max_step_rad`` keeps
    commanded steps small.
    """

    def __init__(
        self,
        gravity_fn,
        fc: float,
        k: float,
        fv: float,
        fo: float,
        j_eff: float,
        differentiate_target: bool = True,
        host_kd: float = 0.0,
        host_kd_hz: float | None = None,
    ) -> None:
        self.gravity_fn = gravity_fn
        self._fric = (fc, k, fv, fo)
        self._j_eff = j_eff
        self._differentiate_target = differentiate_target
        self._host_kd = host_kd
        # Mirrors production motion_control: motor-facing v_des/a_des keep the
        # slow pole; the host damping term uses fast differentiators feeding a
        # band-pass centred on the shoulder resonance (see BandPass in
        # robot.control for the design).
        self._v_des_diff = Differentiator(1)
        self._a_des_diff = Differentiator(1)
        self._v_des_fast_diff = Differentiator(1, cutoff=VEL_CUTOFF_FREQ)
        self._v_meas_diff = Differentiator(1, cutoff=VEL_CUTOFF_FREQ)
        self._damp_bp = BandPass(
            1, 2 * math.pi * host_kd_hz if host_kd_hz is not None else DAMP_BP_W0
        )

    def compute(
        self, q_target: float, meas: tuple[float, float] | None = None
    ) -> tuple[float, float]:
        """Feedforward for one cycle.

        Args:
            q_target: Commanded position (rad).
            meas:     ``(position, feedback_ts)`` from the motor's feedback
                      cache, or ``None`` (collapses the host term to 0).
        """
        if self._differentiate_target:
            v_des = self._v_des_diff.differentiate([q_target])[0]
            a_des = self._a_des_diff.differentiate([v_des])[0]
            v_des_fast = self._v_des_fast_diff.differentiate([q_target])[0]
        else:
            v_des = a_des = v_des_fast = 0.0
        t_ff = (
            self.gravity_fn(q_target)
            + compute_friction(v_des, *self._fric)
            + self._j_eff * a_des
        )
        if self._host_kd and meas is not None:
            q_meas, ts = meas
            v_meas = self._v_meas_diff.differentiate([q_meas], [ts])[0]
            t_ff += self._host_kd * self._damp_bp.update([v_des_fast - v_meas])[0]
        return v_des, t_ff
