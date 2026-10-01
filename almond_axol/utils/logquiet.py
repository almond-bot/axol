"""Cap chatty third-party loggers so ``--log_level DEBUG`` stays readable.

Every CLI with a ``log_level`` field applies it to the root logger with
``logging.basicConfig(level=..., force=True)``, and the video relay and the
serve panel's log capture do the same for their processes. At ``DEBUG`` that
also unmutes libraries that log per packet or per event on a hot path —
aiortc prints one line for every RTP packet it sends (a headset stream is
several hundred a second per track), aioice every STUN check — which buries
the diagnostic lines the level was raised for (``loop sections``, IK
timings, relay queue reports). :func:`quiet_noisy_loggers` pins those
libraries at INFO regardless of the root level, so their warnings and errors
still surface.
"""

from __future__ import annotations

import logging

# Loggers capped at INFO whatever the root level is. aiortc's per-packet
# lines come from ``aiortc.rtcrtpsender`` / ``aiortc.rtcrtpreceiver``; the
# whole package is capped since its other modules log per frame or per RTCP
# report at DEBUG too. aioice logs its connectivity checks at INFO already
# (those stay) and per-STUN-message at DEBUG. websockets logs every frame at
# DEBUG. asyncio's DEBUG is the selector loop's own chatter.
NOISY_LOGGERS: tuple[str, ...] = (
    "aiortc",
    "aioice",
    "websockets",
    "asyncio",
    "av",
    "PIL",
    "urllib3",
    "httpx",
    "httpcore",
    "hpack",
    "matplotlib",
    "numba",
    "jax",
    "jaxlib",
    "filelock",
)

NOISY_LOGGER_LEVEL = logging.INFO


def quiet_noisy_loggers(level: int = NOISY_LOGGER_LEVEL) -> None:
    """Cap every logger in :data:`NOISY_LOGGERS` at ``level`` (default INFO).

    A *cap*, never a floor: the level set is ``max(level, root level)``, so
    under ``--log_level WARNING`` these packages stay at WARNING (an explicit
    INFO on the package logger would otherwise pass INFO records *through* a
    stricter root — a logger's own level overrides the inherited one). The
    serve panel changes the root level per op and calls this each time, so
    the cap follows the root in both directions.

    Idempotent and cheap; call it right after the ``basicConfig`` that sets
    the root level. Child loggers (``aiortc.rtcrtpsender``) inherit the cap
    through their parent's effective level, so only the package roots are
    listed.
    """
    cap = max(level, logging.getLogger().getEffectiveLevel())
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(cap)
