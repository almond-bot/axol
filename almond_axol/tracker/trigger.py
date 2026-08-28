"""Host-side reader for the Mantis handheld trigger's CAN messages.

The Mantis rig's onboard MCU (RP2350 running can2040 PIO CAN — the
``axol_mantis`` board in the circuits-tsx repo) sits on the same per-side
gripper CAN bus (``can_mantis_l`` / ``can_mantis_r``, SocketCAN,
1 Mbit/s) as the Damiao gripper motor and continuously
publishes the trigger state. SocketCAN delivers every frame to every
open socket, so :class:`TriggerReader` opens its own filtered bus and
coexists with the gripper driver's socket on the same interface without
stealing its frames.

The trigger is a **continuous analog squeeze**, not a switch: the rig's
trigger drives a potentiometer, and the node publishes a normalised
position so the gripper tracks the operator's hand proportionally. The
node self-calibrates (auto-zeroed rest at power-on, closing direction
latched from the first real pull), so there is nothing to calibrate
host-side.

Firmware contract (node → host only; the host never transmits to it —
source of truth: ``designs/mantis/firmware/firmware.c`` in the
circuits-py repo, protocol table in the adjacent README):

  - Classic CAN 2.0 data frame, **standard 11-bit arbitration ID
    0x009** (0x008 is reserved for the gripper motor itself, whose
    feedback answers on motor ID + 0x10), DLC 6, published at 100 Hz
    whenever the device is powered.
  - Payload layout, **little-endian** (``struct.unpack("<fH", data)``):

    =======  ===========================================================
    byte(s)  content
    =======  ===========================================================
    0-3      float32 trigger position, continuous over [0.0, 1.0] —
             0.0 fully released (gripper open), 1.0 fully squeezed
             (gripper closed). The node already clamps to this range
    4-5      uint16 raw 12-bit ADC reading, for calibration/debug
             only — ignore in production
    =======  ===========================================================

  - The node oversamples the pot 16x per published sample, so no
    host-side filtering or debounce is needed.

The position maps to the VRFrame grip convention (0.0 = fully closed …
1.0 = fully open) by :func:`position_to_grip`, which simply inverts it:
released grip 1.0 (open), fully squeezed grip 0.0 (closed).

Staleness policy: no frame for :data:`STALE_AFTER_S` (~25 missed frames
at 100 Hz — device unplugged or powered off) marks the node stale, and
the consumer **holds the last grip command** rather than jumping to a
default, so a dropout mid-demonstration never opens the jaws and drops
whatever they hold (see :meth:`~almond_axol.tracker.bridge.TrackerBridge._side_grip`).

python-can is imported lazily so the pure frame/mapping helpers work
without it; constructing a :class:`TriggerReader` without python-can
installed raises a clear error.
"""

from __future__ import annotations

import logging
import math
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)

# Standard 11-bit arbitration ID owned by the trigger node (0x008 is the
# gripper motor on the same bus).
TRIGGER_CAN_ID = 0x009

# Nominal firmware transmit rate.
TRIGGER_RATE_HZ = 100.0

# No frame for this long (~25 missed frames at 100 Hz) means the node is
# stale: hold the last grip command rather than jumping to a default.
STALE_AFTER_S = 0.25

# Payload size (float32 trigger state + uint16 raw switch level).
_FRAME_LEN = 6
_FRAME_FMT = "<fH"


@dataclass(frozen=True)
class TriggerFrame:
    """One decoded trigger-node frame.

    Attributes:
        position: Normalised trigger position, clamped to [0.0, 1.0] —
            0.0 fully released (gripper open), 1.0 fully squeezed
            (gripper closed).
        raw:      Raw 12-bit ADC reading, for calibration/debug only —
            ignore it in production code.
    """

    position: float
    raw: int


def parse_trigger_frame(data: bytes) -> TriggerFrame | None:
    """Decode one 6-byte trigger-node payload, or ``None`` if malformed.

    Pure function of the payload bytes (no CAN dependency) so it is unit
    testable and doubles as the reference decoder for the firmware
    contract in the module docstring. The node already clamps to
    [0.0, 1.0]; clamping again here (after rejecting non-finite floats)
    keeps a corrupted frame from ever commanding the gripper past its
    travel.
    """
    if len(data) != _FRAME_LEN:
        return None
    trigger, raw = struct.unpack(_FRAME_FMT, data)
    if not math.isfinite(trigger):
        return None
    return TriggerFrame(position=min(max(trigger, 0.0), 1.0), raw=raw)


def encode_trigger_frame(position: float, raw: int | None = None) -> bytes:
    """Encode one trigger-node payload (reference encoder, mirrors firmware).

    The inverse of :func:`parse_trigger_frame`; used by tests and as
    executable documentation for the firmware author. ``raw`` defaults to
    a mid-scale-ish ADC count consistent with ``position``, which only
    matters for debug readouts.
    """
    if raw is None:
        raw = int(round(position * 4095))
    return struct.pack(_FRAME_FMT, position, raw)


def position_to_grip(position: float) -> float:
    """Map the trigger position to the VRFrame grip command.

    VRFrame grip is 0.0 = fully closed … 1.0 = fully open, the inverse of
    the node's 0.0 = released … 1.0 = squeezed, so this is just
    ``1 - position``. Proportional throughout: a half-squeezed trigger
    commands a half-closed gripper.
    """
    return 1.0 - min(max(position, 0.0), 1.0)


def is_stale(last_rx: float | None, now: float, timeout: float = STALE_AFTER_S) -> bool:
    """True if a node whose last frame arrived at ``last_rx`` is stale.

    ``last_rx`` is ``None`` when no frame was ever received (always
    stale). Pure helper shared by :meth:`TriggerReader.is_stale` and the
    tests.
    """
    return last_rx is None or now - last_rx > timeout


class TriggerReader:
    """Background reader for one side's trigger node on a SocketCAN bus.

    Opens its own python-can SocketCAN socket filtered to
    :data:`TRIGGER_CAN_ID` — SocketCAN duplicates traffic to every open
    socket, so this never interferes with the gripper driver sharing the
    interface. A daemon thread decodes frames as they arrive; the
    accessors are thread-safe and dropout-safe (``None``/stale before the
    first frame).

    Args:
        channel: SocketCAN interface name, e.g. ``"can_mantis_l"``.
        bus:     Pre-opened bus-like object (``recv``/``shutdown``) — used
            by the tests to feed synthetic frames without CAN hardware.
    """

    def __init__(self, channel: str, *, bus: Any | None = None) -> None:
        if bus is None:
            try:
                import can
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise RuntimeError(
                    "python-can is required to read the Mantis trigger "
                    "(install the project dependencies with `uv sync`)"
                ) from exc
            bus = can.Bus(
                channel=channel,
                interface="socketcan",
                can_filters=[
                    {"can_id": TRIGGER_CAN_ID, "can_mask": 0x7FF, "extended": False}
                ],
            )
        self._bus = bus

        self._lock = threading.Lock()
        self._grip: float | None = None
        self._last_rx: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True, name=f"trigger-{channel}"
        )
        self._thread.start()

    # -- Thread-safe accessors -------------------------------------------

    def grip(self) -> float | None:
        """Latest grip command (0.0 = closed, 1.0 = open), or ``None`` if no
        frame has ever been received. Check :meth:`is_stale` for freshness."""
        with self._lock:
            return self._grip

    def is_stale(self, now: float | None = None) -> bool:
        """True if no frame arrived within :data:`STALE_AFTER_S` (or ever).

        ``now`` is a ``time.monotonic()`` timestamp; defaults to the
        current time.
        """
        if now is None:
            now = time.monotonic()
        with self._lock:
            return is_stale(self._last_rx, now)

    def close(self) -> None:
        """Stop the reader thread and shut the CAN socket down."""
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self._bus.shutdown()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass

    # -- Reader thread ------------------------------------------------------

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._bus.recv(timeout=0.1)
            except Exception as exc:  # noqa: BLE001 - bus torn down or errored
                if self._stop.is_set():
                    return
                _logger.warning("trigger CAN recv failed: %s", exc)
                time.sleep(0.1)
                continue
            if msg is None:
                continue
            # The kernel filter already selects the trigger ID; re-check here
            # so a test double (or a filterless bus) behaves identically.
            if msg.arbitration_id != TRIGGER_CAN_ID or msg.is_extended_id:
                continue
            frame = parse_trigger_frame(bytes(msg.data))
            if frame is None:
                continue
            grip = position_to_grip(frame.position)
            now = time.monotonic()
            with self._lock:
                self._grip = grip
                self._last_rx = now
