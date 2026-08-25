"""Deterministic tuning library.

Reusable pieces behind the ``tune.*`` CLI commands and the diagnostics UI:
single-joint sine/step runners with their safety geometry (:mod:`.runner`),
the production-matching feedforward (:mod:`.feedforward`), tracking-accuracy
and smoothness metrics shared by every suite (:mod:`.metrics`), the
noise-injection filter-stack test (:mod:`.filtering`), joint-frame motor
access (:mod:`.joint_frame`), and the persisted run-artifact store
(:mod:`.runs`).
"""

from .feedforward import FF_MODES, FeedForward
from .filtering import (
    filter_noise_analysis,
    inject_ik_noise,
    inject_noise,
    replay_filter_stack,
)
from .joint_frame import JointFrameMotor, joint_frame_motors
from .metrics import (
    BAND_HIGH,
    BAND_LOW,
    band_rms,
    chatter_metrics,
    ring_frequency,
    sine_metrics,
    step_metrics,
    tracking_lag_ms,
    tracking_metrics,
)
from .runner import (
    BASE_COLLISION_JOINTS,
    DEFAULT_AMP_RAD,
    RAMP_SPEED,
    HolderMonitor,
    cached_meas,
    cached_torque,
    make_target_noise,
    ramp_impedance,
    ramp_joints_to,
    ramp_others_to_zero,
    run_sine,
    run_step,
    safe_amplitude,
    safe_outboard_direction,
    sine_center,
)
from .runs import (
    TUNING_RUNS_DIR,
    clear_runs,
    delete_run,
    list_runs,
    load_run,
    log_to_series,
    save_run,
)

__all__ = [
    "BAND_HIGH",
    "BAND_LOW",
    "BASE_COLLISION_JOINTS",
    "DEFAULT_AMP_RAD",
    "FF_MODES",
    "FeedForward",
    "HolderMonitor",
    "JointFrameMotor",
    "RAMP_SPEED",
    "TUNING_RUNS_DIR",
    "band_rms",
    "cached_meas",
    "cached_torque",
    "chatter_metrics",
    "clear_runs",
    "delete_run",
    "filter_noise_analysis",
    "inject_ik_noise",
    "inject_noise",
    "joint_frame_motors",
    "list_runs",
    "load_run",
    "log_to_series",
    "make_target_noise",
    "ramp_impedance",
    "ramp_joints_to",
    "ramp_others_to_zero",
    "replay_filter_stack",
    "ring_frequency",
    "run_sine",
    "run_step",
    "safe_amplitude",
    "safe_outboard_direction",
    "save_run",
    "sine_center",
    "sine_metrics",
    "step_metrics",
    "tracking_lag_ms",
    "tracking_metrics",
]
