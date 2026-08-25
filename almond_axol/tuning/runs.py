"""Persisted tuning runs: full time series (NPZ) + metrics/metadata (JSON).

Every tuning suite — sine/step PID probes, reference-motion replays, offline
analyses — saves its result through :func:`save_run` so the diagnostics UI
(and offline scripts) can chart, rank, and A/B-compare runs with one loader.

Layout (one directory per run)::

    ~/.almond/diagnostics/tuning/<run-id>/
        meta.json     kind, side, joint(s), gains, params, metrics, label
        series.npz    named float arrays (t, target, actual, error, torque,
                      or per-joint variants like target/<joint>)

The full-resolution series lives in the NPZ (compact, fast to load); the
JSON stays small so listing hundreds of runs is cheap.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

TUNING_RUNS_DIR = Path.home() / ".almond" / "diagnostics" / "tuning"


def _json_safe(value: Any) -> Any:
    """Recursively replace NaN/Inf with None so the JSON stays standard."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_run(
    kind: str,
    series: dict[str, np.ndarray],
    metrics: dict[str, Any],
    *,
    side: str | None = None,
    joint: str | None = None,
    gains: dict[str, float] | None = None,
    params: dict[str, Any] | None = None,
    label: str | None = None,
    group: str | None = None,
    runs_dir: Path = TUNING_RUNS_DIR,
) -> str:
    """Persist one tuning run; returns its id.

    Args:
        kind:    Suite name, e.g. ``"sine"``, ``"step"``, ``"motion"``,
                 ``"filtering"``, ``"kinematics"``, ``"wifi"``.
        series:  Named time-series arrays (full resolution).
        metrics: The run's scorecard (from :mod:`.metrics`); NaNs are
                 stored as null.
        side:    ``"left"`` / ``"right"`` for hardware runs.
        joint:   Joint name for single-joint runs.
        gains:   The gain overrides under test (kp/kd/kd_host/...).
        params:  Everything else needed to reproduce the run (amplitude,
                 frequency, motion name, source recording, ...).
        label:   Free-form operator note shown in listings.
        group:   Shared id linking the runs of one sweep/session so the UI
                 can offer them as an A/B set.
    """
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "series.npz",
        **{k: np.asarray(v) for k, v in series.items()},
    )
    meta = {
        "id": run_id,
        "kind": kind,
        "side": side,
        "joint": joint,
        "gains": gains or {},
        "params": _json_safe(params or {}),
        "metrics": _json_safe(metrics),
        "label": label,
        "group": group,
        "startedAt": time.time(),
        "seriesKeys": sorted(series),
        "samples": max((len(v) for v in series.values()), default=0),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return run_id


def list_runs(runs_dir: Path = TUNING_RUNS_DIR) -> list[dict[str, Any]]:
    """All run metadata, newest first. Unreadable entries are skipped."""
    if not runs_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for meta_path in runs_dir.glob("*/meta.json"):
        try:
            out.append(json.loads(meta_path.read_text()))
        except (OSError, ValueError):
            continue
    out.sort(key=lambda m: m.get("startedAt") or 0, reverse=True)
    return out


def load_run(
    run_id: str, runs_dir: Path = TUNING_RUNS_DIR
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    """Load one run's ``(meta, series)``, or ``None`` if it doesn't exist."""
    run_dir = runs_dir / run_id
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text())
    series: dict[str, np.ndarray] = {}
    npz_path = run_dir / "series.npz"
    if npz_path.is_file():
        with np.load(npz_path) as data:
            series = {k: data[k] for k in data.files}
    return meta, series


def log_to_series(log: list[dict]) -> dict[str, np.ndarray]:
    """Convert a runner's per-sample dict log into named arrays for saving."""
    if not log:
        return {}
    return {key: np.array([r[key] for r in log], dtype=float) for key in log[0].keys()}
