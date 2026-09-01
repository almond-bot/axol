"""Validation shared by CLI and serve CAN channel flows."""

from __future__ import annotations

from collections.abc import Iterable


def require_distinct_axol_channels(
    channels: Iterable[str | None],
) -> tuple[str | None, str | None]:
    """Normalize an Axol arm map and reject one interface reused by both arms.

    Axol's two arms reuse the same motor IDs, so constructing two logical buses
    on one SocketCAN interface makes commands ambiguous.  Either side may be
    omitted for a one-arm bench setup; only two simultaneously active names
    must be distinct.
    """
    values = tuple(channels)
    if len(values) != 2:
        raise ValueError("Axol requires exactly two CAN channel slots")

    def normalize(value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return None if not text or text.lower() in ("null", "none") else text

    left, right = (normalize(value) for value in values)
    if left is not None and right is not None and left == right:
        raise ValueError(
            f"Axol left and right CAN channels are both {left!r}; choose two "
            "distinct interfaces or disable one side"
        )
    return left, right


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
