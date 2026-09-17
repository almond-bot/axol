"""Shared experiment/config plumbing for the replay-style tuning commands.

``tune.motion`` and ``tune.repeatability`` drive the robot through the
production ``motion_control`` path, which means the realtime core applies
whatever :class:`~almond_axol.robot.config.ControlExperiments` the config
carries. Both used to build a bare ``AxolConfig()``, so the experiments were
always at their defaults and a replay could not A/B them — which is exactly
what the deterministic replay harness is for (ad-hoc teleop was the only way
to turn a flag on, and ad-hoc teleop is not repeatable).

This module gives both commands one base config and one override flag:

* the base is the robot's shared settings (``~/.almond/settings.json``, the
  file the control panel edits) over the calibrated defaults — the same
  config ``axol teleop`` runs, which is what ``tune.motion``'s docstring
  always promised and what an operator who sets an ``experiments`` block in
  the panel expects a replay to honour;
* ``--experiment name=value`` overrides individual fields per run, so a
  CLI A/B sweep does not have to edit the settings file between runs.

The resolved set is printed at startup and stored on the run artifact, so a
saved run is never ambiguous about which control law produced it.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from ...robot.config import WIRE_MODES, AxolConfig, ControlExperiments


def add_experiment_argument(parser: Any) -> None:
    """Register the shared ``--experiment`` flag on a tuning parser."""
    parser.add_argument(
        "--experiment",
        action="append",
        metavar="NAME=VALUE",
        help=(
            "Override one ControlExperiments field for this run, e.g. "
            "--experiment friction_k_max=400 --experiment friction_slew=30. "
            "Repeatable. Applied over the robot's shared settings; every "
            "other field keeps its saved value."
        ),
    )


def _coerce(field_name: str, raw: str) -> Any:
    """Parse one ``--experiment`` value against its dataclass field type."""
    spec = {f.name: f for f in fields(ControlExperiments)}.get(field_name)
    if spec is None:
        known = ", ".join(f.name for f in fields(ControlExperiments))
        raise SystemExit(f"--experiment: unknown field {field_name!r} (known: {known})")
    if spec.type in ("bool", bool):
        lowered = raw.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        raise SystemExit(f"--experiment: {field_name} wants a boolean, got {raw!r}")
    if spec.type in ("str", str):
        return raw.strip()
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"--experiment: {field_name} wants a number, got {raw!r}")


def parse_experiment_overrides(specs: list[str] | None) -> dict[str, Any]:
    """Parse ``--experiment NAME=VALUE`` specs into a field → value mapping."""
    out: dict[str, Any] = {}
    for spec in specs or []:
        name, sep, raw = spec.partition("=")
        if not sep:
            raise SystemExit(f"--experiment: want NAME=VALUE, got {spec!r}")
        out[name.strip()] = _coerce(name.strip(), raw)
    return out


def base_config(
    *,
    stiffness: float,
    has_gripper: bool,
    experiment_overrides: dict[str, Any] | None = None,
    settings: bool = True,
) -> AxolConfig:
    """The ``AxolConfig`` a replay run should drive the robot with.

    Starts from the robot's shared settings (``settings=False`` falls back to
    the calibrated defaults, for a run that must ignore whatever the panel
    has saved), then applies the run's stiffness / gripper choice and any
    ``--experiment`` overrides. Raises ``SystemExit`` on an experiment
    combination the core would refuse, so a bad flag fails before the arm
    moves rather than at core-configure time.
    """
    config = AxolConfig()
    if settings:
        try:
            from ...settings import load_store, shared_axol_config

            config = shared_axol_config(load_store())
        except Exception as err:  # pragma: no cover - unreadable settings file
            print(f"  (shared settings unavailable, using calibrated defaults: {err})")
            config = AxolConfig()
    config = replace(
        config,
        left_stiffness=stiffness,
        right_stiffness=stiffness,
        has_gripper=has_gripper,
    )
    if experiment_overrides:
        config = replace(
            config, experiments=replace(config.experiments, **experiment_overrides)
        )
    try:
        config.experiments.validate()
    except ValueError as err:
        raise SystemExit(f"--experiment: {err}")
    return config


def describe(experiments: ControlExperiments) -> dict[str, Any]:
    """Non-default experiment fields, for printing and run provenance."""
    default = ControlExperiments()
    return {
        f.name: getattr(experiments, f.name)
        for f in fields(ControlExperiments)
        if getattr(experiments, f.name) != getattr(default, f.name)
    }


def announce(experiments: ControlExperiments) -> dict[str, Any]:
    """Print the active experiments and return them for the run artifact."""
    active = describe(experiments)
    if not active:
        print("  experiments: none (production control law)")
        return active
    print("  experiments: " + ", ".join(f"{k}={v}" for k, v in active.items()))
    if experiments.wire_mode in WIRE_MODES and experiments.wire_mode != "mit":
        print(
            f"  NOTE: wire_mode {experiments.wire_mode} replaces the MIT frame on the "
            "MyActuator joints — measured position drops to 1 deg/LSB and the "
            "torque channel becomes current, so this run's tracking metrics are "
            "quantised by the feedback, not just the controller."
        )
    return active
