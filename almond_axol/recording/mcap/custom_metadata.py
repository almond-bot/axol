"""Episode metadata the operator adds: ``key=value`` pairs.

Collect data's **Episode metadata** field (``episode_metadata``) takes pairs
separated by commas or new lines, such as::

    task_id=pick-12, site=lab-2, difficulty=3

Each MCAP episode stores them as given in a ``custom`` metadata record.
"""

from __future__ import annotations

import re

_KEY = re.compile(r"^[A-Za-z0-9_.\-]+$")
_MAX_VALUE = 1000


def parse_episode_metadata(text: str | None) -> dict[str, str]:
    """The pairs in ``text`` as a mapping.

    Raises ``ValueError`` with a message for the operator when a pair has no
    ``=``, a key uses other characters than letters, digits, ``_ . -``, or a
    key appears twice.
    """
    out: dict[str, str] = {}
    if not text or not text.strip():
        return out
    for raw in re.split(r"[,\n]", text):
        pair = raw.strip()
        if not pair:
            continue
        key, sep, value = (part.strip() for part in pair.partition("="))
        if not sep or not key:
            raise ValueError(
                f"Episode metadata: {pair!r} is not a key=value pair, such as "
                "site=lab-2."
            )
        if not _KEY.match(key):
            raise ValueError(
                f"Episode metadata: {key!r} is not a key. Use letters, digits, "
                "_ . or -."
            )
        if key in out:
            raise ValueError(f"Episode metadata: {key} is given twice.")
        if len(value) > _MAX_VALUE:
            raise ValueError(f"Episode metadata: the value of {key} is too long.")
        out[key] = value
    return out
