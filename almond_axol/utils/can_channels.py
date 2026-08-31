"""Validation shared by CLI and serve CAN channel flows."""

from __future__ import annotations

from collections.abc import Iterable


def require_mantis_channels(
    channels: Iterable[str | None],
) -> tuple[str, str]:
    """Return stripped left/right names, rejecting unsafe dual-rig maps."""
    values = tuple(channels)
    if len(values) != 2:
        raise ValueError("Mantis requires exactly two CAN channels")
    normalized: list[str] = []
    for side, value in zip(("left", "right"), values, strict=True):
        text = "" if value is None else str(value).strip()
        if not text or text.lower() in ("null", "none"):
            raise ValueError(
                f"Mantis {side} CAN channel is empty; configure both sides in "
                "Settings → Mantis"
            )
        normalized.append(text)
    left, right = normalized
    if left == right:
        raise ValueError(
            f"Mantis left and right CAN channels are both {left!r}; choose two "
            "distinct interfaces in Settings → Mantis"
        )
    return left, right
