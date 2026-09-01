"""
axol tracker.lighthouse.check

Survey the Lighthouse base stations and trackers libsurvive can see. The one
setup fault libsurvive only reports in its log — two Base Station 2.0 units on
the same channel — is surfaced here as a failing check, and the survey is
persisted so the control panel's Mantis readiness badges can show it.
"""

from __future__ import annotations

import time

from ..tracker.base import TRACKER_POSE_MAX_AGE_S, valid_tracker_pose
from ..tracker.lighthouse_survey import (
    LIGHTHOUSE_SURVEY_FILE,
    LighthouseSurvey,
    display_channel,
    save_lighthouse_survey,
)

# The channel-clash warning repeats about once a second and the first LH_POSE
# (which names a station) follows the initial solve a few seconds in, so hold
# the survey open long enough for both before deciding.
_SURVEY_S = 12.0
_EXPECTED_TRACKERS = 2


def add_parser(subparsers) -> None:  # type: ignore[type-arg]
    """Register the ``tracker.lighthouse.check`` subcommand."""
    subparsers.add_parser(
        "tracker.lighthouse.check",
        help="Check Lighthouse base-station channels and tracker visibility.",
    ).set_defaults(func=run_check)


def _result(level: str, label: str, detail: str) -> None:
    print(f"{level:<4} {label:<18} {detail}", flush=True)


def survey_lighthouses(source, window_s: float = _SURVEY_S) -> LighthouseSurvey:  # type: ignore[no-untyped-def]
    """Run a started ``SurviveSource`` for ``window_s`` and return its survey."""
    deadline = time.perf_counter() + window_s
    while time.perf_counter() < deadline:
        source.poses()  # surfaces a backend failure promptly
        time.sleep(0.2)
    survey = source.lighthouse_survey()
    now = time.perf_counter()
    survey.trackers = {
        key
        for key, sample in source.poses().items()
        if valid_tracker_pose(sample) and now - sample.t <= TRACKER_POSE_MAX_AGE_S
    }
    return survey


def report_survey(survey: LighthouseSurvey) -> int:
    """Print the survey as check lines; return the number of failures."""
    failures = 0
    if not survey.channels:
        _result("FAIL", "Base stations", "none detected")
        failures += 1
    for channel, serials in sorted(survey.channels.items()):
        names = (
            ", ".join(s.upper() for s in sorted(serials)) or "serial not yet decoded"
        )
        if channel in survey.clashing_channels():
            _result(
                "FAIL",
                f"Channel {display_channel(channel)}",
                f"two or more base stations share this channel ({names})",
            )
            failures += 1
        else:
            _result(
                "OK", f"Channel {display_channel(channel)}", f"base station {names}"
            )
    if survey.clashing_channels():
        print(
            "     Every base station must use a different channel: press the "
            "channel button on the back of one station until its display "
            "shows an unused number, then rerun this check.",
            flush=True,
        )
    trackers = ", ".join(sorted(survey.trackers)) or "none"
    if len(survey.trackers) >= _EXPECTED_TRACKERS:
        _result("OK", "Trackers", f"{len(survey.trackers)} reporting ({trackers})")
    else:
        _result(
            "FAIL",
            "Trackers",
            f"{len(survey.trackers)} of {_EXPECTED_TRACKERS} reporting ({trackers})",
        )
        failures += 1
    return failures


def run_check(_args: object = None) -> None:
    """Survey base stations and trackers, persist it, and fail on a clash."""
    from ..tracker.survive import SurviveSource
    from .mantis_bridge import require_mantis_tracker_readiness

    try:
        require_mantis_tracker_readiness("lighthouse")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None

    print("Lighthouse base-station check", flush=True)
    source = SurviveSource()
    try:
        source.start()
        print(
            f"Listening to libsurvive for {_SURVEY_S:.0f} s (leave the trackers "
            "powered and in view)...",
            flush=True,
        )
        survey = survey_lighthouses(source)
    finally:
        source.stop()

    save_lighthouse_survey(survey)
    failures = report_survey(survey)
    print(f"Saved to {LIGHTHOUSE_SURVEY_FILE}", flush=True)
    if failures:
        raise SystemExit(1)
    print("Lighthouse setup passed.", flush=True)
