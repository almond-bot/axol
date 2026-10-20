"""Guided Lighthouse consistency check, optionally rebuilding station geometry."""

from __future__ import annotations

import argparse
import signal
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..tracker.base import TRACKER_POSE_MAX_AGE_S, valid_tracker_pose
from ..tracker.config import load_tracker_config, save_tracker_config
from ..tracker.lighthouse_survey import save_lighthouse_survey
from ..tracker.lighthouse_validation import (
    LIGHTHOUSE_ACQUISITION_TIMEOUT_S,
    VALIDATION_MAX_AGE_S,
    acquire_lighthouse_runtime_lock,
    binding_problems,
    collect_stationary_check,
    collect_workspace_check,
    is_lighthouse_serial,
    lighthouse_bindings,
    new_validation_report,
    release_lighthouse_runtime_lock,
    save_validation_report,
    validation_fingerprint,
    validation_snapshot,
)
from ..tracker.survive_config import parse_survive_config
from ..utils.state_files import (
    secure_atomic_write_bytes,
    secure_open_new_text,
    secure_read_text,
    secure_unlink,
)

# Serial broadcasts can need several uninterrupted packets even when optical
# tracking is already healthy. Keep the same native decoder running throughout
# acquisition; restarting it would throw away partially received broadcasts.
_ACQUISITION_PROGRESS_INTERVAL_S = 5.0
_CALIBRATION_HOLD_SECONDS = 8.0


class _PhaseCheckFailed(RuntimeError):
    """Keep phase failures structured in the report and brief in the command log."""

    def __init__(self, phase: str, problems: list[str]) -> None:
        labels = {
            "stationaryStart": "Initial stationary check",
            "workspace": "Workspace sweep",
            "stationaryEnd": "Final stationary check",
        }
        self.phase = phase
        self.problems = [
            f"{labels[phase]} failed. Calibration was not replaced.",
            *problems,
        ]
        detail = "; ".join(problems[:2])
        if len(problems) > 2:
            detail += f"; {len(problems) - 2} additional failure summaries in the saved report"
        super().__init__(f"{self.problems[0]} {detail}")


def _record_phase(
    report: dict[str, Any],
    phase: str,
    metrics: dict[str, Any],
    problems: list[str],
    diagnostics: dict[str, Any],
) -> None:
    report.setdefault("phases", {})[phase] = metrics
    report.setdefault("phaseProblems", {})[phase] = problems
    report.setdefault("phaseDiagnostics", {})[phase] = diagnostics
    if problems:
        report["failurePhase"] = phase
        raise _PhaseCheckFailed(phase, problems)


def add_parser(subparsers: Any) -> None:
    for command, calibrate, help_text in (
        (
            "tracker.lighthouse.validate",
            False,
            "Check Lighthouse stationary consistency for this session.",
        ),
        (
            "tracker.lighthouse.calibrate",
            True,
            "Recalibrate Lighthouse stations and check stationary consistency.",
        ),
    ):
        parser = subparsers.add_parser(command, help=help_text)
        parser.add_argument(
            "--expected_stations",
            type=int,
            choices=range(1, 5),
            default=None,
            help="Number of installed base stations (1–4; default: saved count).",
        )
        parser.add_argument("--web-prompts", action="store_true")
        parser.set_defaults(func=run, recalibrate=calibrate)


def _prompt(message: str, web_prompts: bool) -> None:
    if web_prompts:
        print(f"[prompt] {message}", flush=True)
        if sys.stdin.readline() == "":
            raise KeyboardInterrupt("The session prompt was closed.")
    else:
        try:
            input(f"{message} Press Enter when ready ... ")
        except EOFError:
            raise KeyboardInterrupt("The session prompt was closed.") from None


def _check_native_config(contents: bytes, path: Path) -> None:
    try:
        parse_survive_config(contents.decode("utf-8"))
    except ValueError as exc:
        raise RuntimeError(
            f"Cannot read native Lighthouse configuration {path}: {exc}. "
            "The existing calibration was not replaced."
        ) from exc


def _prepare_calibration(
    source: Any, report: dict[str, Any], *, web_prompts: bool
) -> None:
    """Supply stationary views at multiple positions before grading consistency.

    libsurvive's global scene solver acquires new geometry observations when
    trackers pause after movement, not during a continuous sweep. These holds
    give it input; they do not establish convergence or replace the scored checks.
    """
    holds: list[dict[str, Any]] = []
    report["calibrationPreparation"] = {"holds": holds, "completed": False}
    print(
        "Preparing calibration with three stationary workspace views before the "
        "tracking checks. Calibration may adjust during these preparation holds.",
        flush=True,
    )
    for index in range(1, 4):
        _prompt(
            f"Calibration position {index} of 3: move both rigs to a different part "
            "of your normal working area and vary their orientation. Rest them on "
            "a stable surface with tracker sensors uncovered and visible to the "
            "stations. Continue when both are still; leave them untouched for "
            f"{_CALIBRATION_HOLD_SECONDS:.0f} seconds.",
            web_prompts,
        )
        print(
            f"Calibration position {index}/3: keep both rigs still for "
            f"{_CALIBRATION_HOLD_SECONDS:.0f} seconds…",
            flush=True,
        )
        started = time.perf_counter()
        deadline = started + _CALIBRATION_HOLD_SECONDS
        while time.perf_counter() < deadline:
            source.poses()  # propagate native crashes; never approve on a timer
            time.sleep(min(0.1, max(0.0, deadline - time.perf_counter())))
        holds.append(
            {
                "position": index,
                "durationSeconds": time.perf_counter() - started,
                "trackingQuality": source.tracking_quality(),
            }
        )
    report["calibrationPreparation"]["completed"] = True
    print(
        "Calibration preparation holds complete. Waiting for tracking to settle before the "
        "stationary and workspace checks; those checks must all pass.",
        flush=True,
    )


@contextmanager
def _cancel_on_terminate() -> Iterator[None]:
    """The web Stop action must unwind native ownership before the process exits."""
    previous = signal.getsignal(signal.SIGTERM)

    def cancel(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt("Lighthouse check cancelled.")

    signal.signal(signal.SIGTERM, cancel)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _wait_for_ready(
    source: Any,
    expected: int,
    timeout_s: float = LIGHTHOUSE_ACQUISITION_TIMEOUT_S,
) -> None:
    started = time.perf_counter()
    deadline = started + timeout_s
    next_progress = started
    previous_progress = None
    problems = ["Waiting for Lighthouse tracking"]
    identity_pending = False
    while time.perf_counter() < deadline:
        source.poses()  # surface native crashes promptly
        survey = source.lighthouse_survey()
        survey.expected = expected
        problems = survey.problems(require_serials=True)
        quality = source.tracking_quality()
        status = quality.get("status", "unknown")
        if status != "ready":
            problems.extend(quality.get("reasons") or ["Tracking has not stabilized"])
        if quality.get("calibrationRequired"):
            raise RuntimeError(
                "Lighthouse calibration must be repeated: "
                + "; ".join(quality.get("reasons") or ["Station geometry changed"])
                + ". Run Recalibrate Lighthouse before recording. "
                "The existing calibration was not replaced."
            )
        channels = tuple(
            (channel, tuple(sorted(serials)))
            for channel, serials in sorted(survey.channels.items())
        )
        identified = {serial for _, serials in channels for serial in serials}
        pending = [str(channel + 1) for channel, serials in channels if not serials]
        progress = channels
        now = time.perf_counter()
        if progress != previous_progress or now >= next_progress:
            channel_names = ", ".join(str(channel + 1) for channel, _ in channels)
            detail = (
                f"Lighthouse acquisition {now - started:.0f}/{timeout_s:.0f} s: "
                f"{len(channels)}/{expected} station channels seen"
                f" ({channel_names or 'none'}); "
                f"{len(identified)}/{expected} identities decoded"
            )
            if pending:
                detail += f"; identity pending on channels {', '.join(pending)}"
            if len(channels) < expected:
                detail += "; waiting for missing stations' optical signals"
            detail += f"; tracking {status}"
            if status != "ready" and quality.get("reasons"):
                detail += ": " + "; ".join(quality["reasons"][:2])
            print(detail, flush=True)
            previous_progress = progress
            next_progress = now + _ACQUISITION_PROGRESS_INTERVAL_S
        if not problems:
            return
        identity_pending = (
            status == "ready"
            and len(channels) == expected
            and bool(pending)
            and len(identified) + len(pending) == expected
            and not survey.clashing_channels()
        )
        time.sleep(min(0.1, max(0.0, deadline - now)))
    if identity_pending:
        raise RuntimeError(
            f"Lighthouse identity broadcast decoding timed out after {timeout_s:.0f} s: "
            f"all {expected} station channels were seen and live tracking was ready, "
            f"but only {len(identified)} of {expected} station identities decoded "
            f"(pending channels {', '.join(pending)}). "
            "The existing calibration was not replaced. Reposition the rigs so "
            "both trackers have a clear view of every station, keep them still, "
            "then retry the check."
        )
    raise RuntimeError(
        f"Lighthouse acquisition did not stabilize after {timeout_s:.0f} s: "
        + "; ".join(problems)
        + ". The existing calibration was not replaced."
    )


def _capture_failure_diagnostics(source: Any, report: dict[str, Any]) -> None:
    """Retain acquisition evidence before native shutdown, without hiding its failure."""
    errors: list[str] = []
    try:
        survey = source.lighthouse_survey()
        report["baseStations"] = survey.to_dict()
        observed = "; ".join(
            f"{channel}: {', '.join(serials) or 'identity pending'}"
            for channel, serials in report["baseStations"]["channels"].items()
        )
        print(f"Observed station channels: {observed or 'none'}", flush=True)
        saved = "; ".join(
            f"{channel}: {', '.join(serials) or 'identity unknown'}"
            for channel, serials in report["baseStations"]["savedChannels"].items()
        )
        if saved:
            print(
                f"Saved station identities (not live observations): {saved}",
                flush=True,
            )
    except Exception as exc:
        errors.append(f"Station diagnostics unavailable: {exc}")
    try:
        quality = source.tracking_quality()
        report["trackingQuality"] = quality
        report["trackers"] = quality.get("trackers", {})
        print(f"Live tracking status: {quality.get('status', 'unknown')}", flush=True)
        for reason in quality.get("reasons", [])[-6:]:
            print(f"Tracking check: {reason}", flush=True)
    except Exception as exc:
        errors.append(f"Tracking diagnostics unavailable: {exc}")
    try:
        warnings = source.warnings()[-6:]
        report["nativeWarnings"] = warnings
        for warning in warnings:
            print(f"Backend warning: {warning}", flush=True)
    except Exception as exc:
        errors.append(f"Native warnings unavailable: {exc}")
    try:
        get_identity_diagnostics = getattr(source, "identity_diagnostics", None)
        if callable(get_identity_diagnostics):
            diagnostics = get_identity_diagnostics()
            if isinstance(diagnostics, dict):
                report["identityDiagnostics"] = diagnostics
                print(
                    "Identity packet evidence: "
                    f"{diagnostics.get('acceptedPacketCount', 0)} accepted of "
                    f"{diagnostics.get('packetCount', 0)} observed packet files",
                    flush=True,
                )
                for packet in diagnostics.get("packets", []):
                    if packet.get("status") != "accepted":
                        print(
                            f"Identity packet {packet.get('name', 'unknown')}: "
                            f"{packet.get('reason') or packet.get('status', 'unknown')}",
                            flush=True,
                        )
    except Exception as exc:
        errors.append(f"Identity packet diagnostics unavailable: {exc}")
    if errors:
        report["diagnosticErrors"] = errors


def _discover_calibration_trackers(
    source: Any, timeout_s: float = 35.0
) -> dict[str, str]:
    """Find a physical pair without guessing or persisting handedness.

    Initialization can mark poses untrusted while station geometry converges.
    Discovery accepts fresh finite observations; the later stationary and
    workspace checks still require fully trustworthy tracking for both devices.
    """
    deadline = time.perf_counter() + timeout_s
    pair: tuple[str, ...] = ()
    stable_since: float | None = None
    live: list[str] = []
    while time.perf_counter() < deadline:
        poses = source.poses()
        now = time.perf_counter()
        discovered = getattr(source, "discovered_tracker_serials", None)
        identities = set(poses) | (set(discovered()) if callable(discovered) else set())
        physical = sorted(key for key in identities if is_lighthouse_serial(key))
        if len(physical) > 2:
            raise RuntimeError(
                "More than two trackers were detected ("
                + ", ".join(physical)
                + "). Power off trackers outside the two Mantis rigs, then recalibrate again."
            )
        live = [
            key
            for key in physical
            if key in poses
            and valid_tracker_pose(poses[key])
            and 0 <= now - poses[key].t <= TRACKER_POSE_MAX_AGE_S
        ]
        current = tuple(live)
        if len(current) != 2 or current != pair:
            pair = current
            stable_since = now if len(current) == 2 else None
        if stable_since is not None and now - stable_since >= 1.0:
            return {f"tracker{index}": key for index, key in enumerate(pair, 1)}
        time.sleep(0.1)
    seen = ", ".join(live) or "none"
    raise RuntimeError(
        f"Only {len(live)} of two trackers supplied fresh poses ({seen}). "
        "Power both trackers and connect their paired dongles; keep their sensors "
        "visible to the stations, then retry. Recalibration needs both rigs."
    )


def run(args: argparse.Namespace) -> None:
    """Commit native calibration only after a complete passing consistency check."""
    from ..tracker.survive import SurviveSource, default_survive_config_path
    from .mantis_bridge import require_mantis_tracker_readiness

    config = load_tracker_config()
    if args.expected_stations is not None:
        config.expected_base_stations = args.expected_stations
        save_tracker_config(config)
    previous = validation_snapshot(config)
    report = new_validation_report(config, calibrated=args.recalibrate)
    if previous.get("calibrationRequired"):
        report["calibrationRequired"] = True
    save_validation_report(
        report
    )  # an interrupted check cannot leave an old green badge
    source: Any = None
    stopped = False
    committed = False
    try:
        require_mantis_tracker_readiness("lighthouse")
        if report.get("calibrationRequired") and not args.recalibrate:
            raise RuntimeError(
                "A station/world-frame change requires Recalibrate Lighthouse "
                "before Validate tracking can pass."
            )
        problems = binding_problems(config)
        needs_identification = bool(problems)
        if needs_identification and not args.recalibrate:
            raise RuntimeError("; ".join(problems))
        measured_bindings = lighthouse_bindings(config)
        if needs_identification:
            print(
                "Recalibrating before side identification. Power only the two rig "
                "trackers; they will be checked by serial without assigning left/right sides.",
                flush=True,
            )
        native_path = default_survive_config_path()
        try:
            original = secure_read_text(native_path).encode("utf-8")
        except FileNotFoundError:
            original = None
        if original is not None:
            _check_native_config(original, native_path)
            backup = native_path.with_name(
                f"{native_path.name}.{time.time_ns()}.backup"
            )
            secure_atomic_write_bytes(backup, original)
            report["backupPath"] = str(backup)
            print(f"Preserved native calibration: {backup}", flush=True)

        _prompt(
            f"Confirm all {config.expected_base_stations} base stations are fixed, "
            "powered and on distinct channels. Set both rigs down with their tracker "
            f"sensors visible to at least {min(2, config.expected_base_stations)} "
            "station(s) each. Reposition them if visibility is blocked, then keep "
            "them still while tracking initializes.",
            args.web_prompts,
        )
        # Ensure the directory exists via a no-follow state writer; do not
        # change the mode/owner of the operator's existing libsurvive directory.
        marker = native_path.parent / f".axol-check-{uuid.uuid4().hex}"
        with secure_open_new_text(marker):
            pass
        secure_unlink(marker)
        with tempfile.TemporaryDirectory(
            prefix=".axol-check-", dir=native_path.parent
        ) as directory:
            candidate = Path(directory) / "config.json"
            secure_atomic_write_bytes(
                candidate, original if original is not None else b"{}\n"
            )
            source = SurviveSource(
                force_calibrate=args.recalibrate,
                config_path=candidate,
                expected_stations=config.expected_base_stations,
            )
            if not needs_identification:
                source.set_required_trackers(
                    {key for key in measured_bindings.values() if key}
                )
            with _cancel_on_terminate():
                cleanup_started = False
                try:
                    source.start()
                    if needs_identification:
                        measured_bindings = _discover_calibration_trackers(source)
                        source.set_required_trackers(set(measured_bindings.values()))
                        report["discoveredTrackers"] = measured_bindings
                        print(
                            "Found the two rig trackers: "
                            + ", ".join(measured_bindings.values())
                            + ". Left/right sides still need identification.",
                            flush=True,
                        )
                    print(
                        "Waiting for every station and both trackers to stabilize…",
                        flush=True,
                    )
                    _wait_for_ready(source, config.expected_base_stations)
                    if args.recalibrate:
                        _prepare_calibration(
                            source, report, web_prompts=args.web_prompts
                        )
                        _wait_for_ready(source, config.expected_base_stations)
                        # GSS is selected only at native startup. Stop the
                        # calibration process and score its candidate in a new
                        # process using exactly the collection solver flags.
                        # Timer-based preparation never approves the candidate.
                        print(
                            "Calibration preparation complete. Restarting tracking "
                            "with station refinement disabled for validation and recording…",
                            flush=True,
                        )
                        cleanup_started = True
                        source.stop()
                        stopped = True
                        source = SurviveSource(
                            force_calibrate=False,
                            config_path=candidate,
                            expected_stations=config.expected_base_stations,
                        )
                        source.set_required_trackers(
                            {key for key in measured_bindings.values() if key}
                        )
                        cleanup_started = False
                        stopped = False
                        source.start()
                        _wait_for_ready(source, config.expected_base_stations)
                    _prompt(
                        "Keep both rigs resting still with clear tracker visibility. "
                        "Continue to measure five seconds of stationary consistency; "
                        "do not touch the rigs during this measurement.",
                        args.web_prompts,
                    )
                    metrics, problems, survey, quality = collect_stationary_check(
                        source,
                        measured_bindings,
                        expected_stations=config.expected_base_stations,
                    )
                    report.update(
                        metrics=metrics,
                        baseStations=survey.to_dict(),
                        trackers=quality.get("trackers", {}),
                        trackingQuality=quality,
                    )
                    _record_phase(
                        report,
                        "stationaryStart",
                        metrics,
                        problems,
                        quality.get("validationDiagnostics", {}),
                    )
                    _prompt(
                        "Pick up both rigs. After Continue, slowly move each through "
                        "the working area for 12 seconds, including wrist turns "
                        "and the positions used for recording (at least 10 cm and 15° per rig).",
                        args.web_prompts,
                    )
                    print(
                        "Sweep both rigs slowly now — observing tracking for 12 seconds…",
                        flush=True,
                    )
                    workspace, sweep_problems = collect_workspace_check(
                        source,
                        measured_bindings,
                        expected_stations=config.expected_base_stations,
                    )
                    _record_phase(
                        report,
                        "workspace",
                        workspace,
                        sweep_problems,
                        workspace.get("diagnostics", {}),
                    )
                    report["trackingQuality"] = source.tracking_quality()
                    _prompt(
                        "Set both rigs down at another working position with sensors visible. "
                        "Keep them still, then Continue for the final five-second hold.",
                        args.web_prompts,
                    )
                    _wait_for_ready(source, config.expected_base_stations)
                    final_metrics, final_problems, survey, quality = (
                        collect_stationary_check(
                            source,
                            measured_bindings,
                            expected_stations=config.expected_base_stations,
                        )
                    )
                    metrics = final_metrics
                    report.update(
                        metrics=metrics,
                        baseStations=survey.to_dict(),
                        trackers=quality.get("trackers", {}),
                        trackingQuality=quality,
                    )
                    _record_phase(
                        report,
                        "stationaryEnd",
                        final_metrics,
                        final_problems,
                        quality.get("validationDiagnostics", {}),
                    )
                    save_lighthouse_survey(survey)
                except BaseException:
                    _capture_failure_diagnostics(source, report)
                    raise
                finally:
                    if not cleanup_started:
                        cleanup_started = True
                        source.stop()
                        stopped = True
                # stop() releases the source's reservation. Reacquire before
                # publication: if another tracker session won the race, leave
                # the original calibration intact rather than changing it live.
                reservation = acquire_lighthouse_runtime_lock()
                try:
                    updated = secure_read_text(candidate).encode("utf-8")
                    _check_native_config(updated, candidate)
                    try:
                        current = secure_read_text(native_path).encode("utf-8")
                    except FileNotFoundError:
                        current = None
                    if current != original:
                        raise RuntimeError(
                            "Native calibration changed during the check; leaving that change intact. Run the check again."
                        )
                    secure_atomic_write_bytes(native_path, updated)
                    committed = True
                    report["fingerprint"] = validation_fingerprint(config, native_path)
                    report.update(
                        status="needs_identification"
                        if needs_identification
                        else "passed",
                        valid=not needs_identification,
                        problems=[
                            "Calibration saved. Identify the two rig sides, then run Validate tracking before recording."
                        ]
                        if needs_identification
                        else [],
                        calibrated=args.recalibrate,
                        calibrationCommitted=True,
                        calibrationRequired=False,
                    )
                    report["checkedAt"] = time.time()
                    report["expiresAt"] = report["checkedAt"] + VALIDATION_MAX_AGE_S
                    save_validation_report(report)
                finally:
                    release_lighthouse_runtime_lock(reservation)
        print(
            "Stationary and workspace consistency checks passed. Absolute accuracy was not measured.",
            flush=True,
        )
        if needs_identification:
            print(
                "Calibration saved. Identify sides next, then run Validate tracking; "
                "recording remains blocked until those steps pass.",
                flush=True,
            )
        elif args.recalibrate:
            print(
                "Recalibration and tracking validation passed. No separate Validate "
                "tracking check is needed now; start a new collection session.",
                flush=True,
            )
        for side, metrics in report["metrics"].items():
            print(
                f"  {side}: position p95 {metrics['positionP95Mm']:.2f} mm; "
                f"orientation p95 {metrics['angleP95Deg']:.2f}°",
                flush=True,
            )
    except BaseException as exc:
        # A native cleanup failure keeps ownership inside SurviveSource; do
        # not retry a potentially unsafe native close here or publish its state.
        report.update(
            status="failed",
            valid=False,
            calibrationCommitted=committed,
            problems=exc.problems
            if isinstance(exc, _PhaseCheckFailed)
            else [str(exc) or "Session check cancelled."],
        )
        if source is not None and not stopped:
            report["problems"].append(
                "Native cleanup did not complete; restart the tracking session before retrying."
            )
        save_validation_report(report)
        if isinstance(exc, KeyboardInterrupt):
            detail = (
                "The validated calibration was saved, but its session report did not finish; run the check again."
                if committed
                else "Original calibration was retained."
            )
            raise SystemExit(f"Lighthouse check cancelled. {detail}") from None
        if isinstance(exc, (RuntimeError, OSError, ValueError)):
            raise SystemExit(str(exc)) from None
        raise
