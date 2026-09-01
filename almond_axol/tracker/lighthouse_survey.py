"""What libsurvive saw of the Lighthouse base stations during a run.

SteamVR Base Station 2.0 units each broadcast on one of 16 channels and every
station in view must use a different one; otherwise their sweeps collide and
tracking silently degrades or drops. libsurvive detects that clash but only
reports it in its log, so this module collects the channel/serial facts from
the recording stream and persists the most recent survey for the control
panel's readiness badges.

libsurvive numbers channels 0–15 while the base station itself displays 1–16;
:func:`display_channel` converts for operator-facing text.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.paths import almond_path
from ..utils.state_files import secure_atomic_write_json, secure_read_text

LIGHTHOUSE_SURVEY_FILE = almond_path("tracker", "lighthouse_survey.json")


def display_channel(channel: int) -> int:
    """The channel number printed on / shown by the base station."""
    return channel + 1


@dataclass
class LighthouseSurvey:
    """Base stations per libsurvive channel plus the clashes it reported."""

    channels: dict[int, set[str]] = field(default_factory=dict)
    conflicts: set[int] = field(default_factory=set)
    trackers: set[str] = field(default_factory=set)
    checked_at: float = field(default_factory=time.time)

    def note_channel(self, channel: int, serial: str | None = None) -> None:
        serials = self.channels.setdefault(channel, set())
        if serial:
            serials.add(serial.lower())

    def note_conflict(self, channel: int) -> None:
        self.conflicts.add(channel)
        self.channels.setdefault(channel, set())

    @property
    def base_station_count(self) -> int:
        """Lower bound on stations seen: a flagged clash implies at least two."""
        return sum(
            max(2 if ch in self.conflicts else 1, len(serials))
            for ch, serials in self.channels.items()
        )

    def clashing_channels(self) -> list[int]:
        """Channels libsurvive flagged or that carried two different serials."""
        clashing = set(self.conflicts)
        clashing.update(ch for ch, serials in self.channels.items() if len(serials) > 1)
        return sorted(clashing)

    def problems(self) -> list[str]:
        """Operator-facing setup problems, empty when the survey is healthy."""
        problems = self.clash_problems()
        if not self.channels:
            problems.append(
                "no base station was detected; power the base stations and give "
                "the trackers a clear view of them"
            )
        return problems

    def clash_problems(self) -> list[str]:
        """One message per channel that two or more base stations share."""
        problems: list[str] = []
        for channel in self.clashing_channels():
            serials = sorted(self.channels.get(channel, set()))
            who = (
                "base stations " + " and ".join(s.upper() for s in serials)
                if len(serials) > 1
                else "two or more base stations"
            )
            problems.append(
                f"{who} share channel {display_channel(channel)}; press the "
                "channel button on the back of one station until it shows a "
                "different number"
            )
        return problems

    def to_dict(self) -> dict[str, object]:
        return {
            "checkedAt": self.checked_at,
            "channels": {
                str(display_channel(ch)): sorted(s.upper() for s in serials)
                for ch, serials in sorted(self.channels.items())
            },
            "clashingChannels": [
                display_channel(ch) for ch in self.clashing_channels()
            ],
            "baseStationCount": self.base_station_count,
            "trackers": sorted(self.trackers),
            "problems": self.problems(),
        }


def save_lighthouse_survey(survey: LighthouseSurvey, path: Path | None = None) -> None:
    secure_atomic_write_json(
        path or LIGHTHOUSE_SURVEY_FILE, survey.to_dict(), sort_keys=True
    )


def load_lighthouse_survey(path: Path | None = None) -> dict[str, object] | None:
    """The last persisted survey as published to the control panel, if any."""
    try:
        data = json.loads(secure_read_text(path or LIGHTHOUSE_SURVEY_FILE))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("channels"), dict):
        return None
    return data
