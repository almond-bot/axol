"""What libsurvive saw of the Lighthouse base stations during a run.

SteamVR Base Station 2.0 units each broadcast on one of 16 channels and every
station in view must use a different one; otherwise their sweeps collide and
tracking silently degrades or drops. libsurvive detects that clash but only
reports it in its log, so this module collects the channel/serial facts from
the recording stream and persists the most recent survey for the control
panel's readiness badges.

A station only counts as *seen* when a tracker actually received its sweeps
during the run. libsurvive also replays the stations from its saved
calibration at startup; those are kept apart in :attr:`LighthouseSurvey.saved`
so a station that was re-channelled, unplugged or swapped since the last
session is not mistaken for a live one, and so a different station showing up
on a saved channel can be pointed out.

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

# The Mantis rig ships with two Base Station 2.0 units; one alone tracks, but
# the second is what keeps a rig visible when the operator's body blocks the
# first. Every station the operator powered must therefore be accounted for.
EXPECTED_BASE_STATIONS = 2

_CHANNEL_FIX = (
    "press the channel button on the back of one station until its display "
    "shows a number no other station uses, wait for it to settle, then check again"
)


def display_channel(channel: int) -> int:
    """The channel number printed on / shown by the base station."""
    return channel + 1


def _names(serials: set[str]) -> str:
    return " and ".join(s.upper() for s in sorted(serials))


@dataclass
class LighthouseSurvey:
    """Base stations per libsurvive channel plus the clashes it reported."""

    channels: dict[int, set[str]] = field(default_factory=dict)
    conflicts: set[int] = field(default_factory=set)
    trackers: set[str] = field(default_factory=set)
    saved: dict[int, set[str]] = field(default_factory=dict)
    expected: int = EXPECTED_BASE_STATIONS
    checked_at: float = field(default_factory=time.time)

    def note_channel(self, channel: int, serial: str | None = None) -> None:
        """Record a station whose sweeps a tracker received on ``channel``."""
        serials = self.channels.setdefault(channel, set())
        if serial:
            serials.add(serial.lower())

    def note_saved(self, channel: int, serial: str | None = None) -> None:
        """Record a station libsurvive loaded from its saved calibration."""
        serials = self.saved.setdefault(channel, set())
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

    def replaced_channels(self) -> list[int]:
        """Channels where the live station is not the one saved there last time.

        With two stations on one channel, whichever one a tracker locks onto
        wins, so the identity flipping between sessions is the clearest sign of
        a shared channel that libsurvive itself failed to flag.
        """
        replaced = []
        for channel, live in sorted(self.channels.items()):
            saved = self.saved.get(channel, set())
            if live and saved and not (live & saved):
                replaced.append(channel)
        return replaced

    def problems(self) -> list[str]:
        """Operator-facing setup problems with their fix; empty when healthy."""
        problems = self.clash_problems()
        if not self.channels:
            problems.append(
                "no base station was seen: check that every station shows a "
                "steady green light and a channel number, and that the trackers "
                "have a clear view of them"
            )
        elif not problems and self.base_station_count < self.expected:
            problems.append(self._missing_station_problem())
        return problems

    def clash_problems(self) -> list[str]:
        """One message per channel that two or more base stations share."""
        problems: list[str] = []
        for channel in self.clashing_channels():
            serials = self.channels.get(channel, set())
            who = (
                f"base stations {_names(serials)}"
                if len(serials) > 1
                else "two or more base stations"
            )
            problems.append(
                f"{who} are both set to channel {display_channel(channel)}, so "
                f"their sweeps collide and tracking will stall or jump; {_CHANNEL_FIX}"
            )
        return problems

    def _missing_station_problem(self) -> str:
        seen = ", ".join(
            f"channel {display_channel(ch)}"
            + (f" ({_names(serials)})" if serials else "")
            for ch, serials in sorted(self.channels.items())
        )
        missing = self.expected - self.base_station_count
        message = (
            f"only {self.base_station_count} of {self.expected} base stations "
            f"was seen ({seen})"
            if self.base_station_count == 1
            else f"only {self.base_station_count} of {self.expected} base "
            f"stations were seen ({seen})"
        )
        replaced = self.replaced_channels()
        if replaced:
            channel = replaced[0]
            message += (
                f"; the station now on channel {display_channel(channel)} "
                f"({_names(self.channels[channel])}) is not the one saved there "
                f"last time ({_names(self.saved[channel])}), which means both "
                f"are set to channel {display_channel(channel)}"
            )
        else:
            message += (
                f". If the other {'station is' if missing == 1 else 'stations are'} "
                "powered (steady green) it is almost certainly set to the same "
                "channel as the one that was seen — libsurvive can only receive "
                "one station per channel"
            )
        return f"{message}; {_CHANNEL_FIX}"

    def to_dict(self) -> dict[str, object]:
        return {
            "checkedAt": self.checked_at,
            "channels": {
                str(display_channel(ch)): sorted(s.upper() for s in serials)
                for ch, serials in sorted(self.channels.items())
            },
            "savedChannels": {
                str(display_channel(ch)): sorted(s.upper() for s in serials)
                for ch, serials in sorted(self.saved.items())
            },
            "clashingChannels": [
                display_channel(ch) for ch in self.clashing_channels()
            ],
            "baseStationCount": self.base_station_count,
            "expectedBaseStations": self.expected,
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
