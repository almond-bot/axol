"""
Interface to the Jiecang JCB35N2 lift control box (two standing-desk legs).

Protocol
────────
The JCB35N2's "HS" port is an RJ45 jack, but it is NOT Ethernet — it is the
wired handset interface: 5 V active-low button lines plus a one-directional
9600 8N1 UART on which the box broadcasts its height. Plugging it into a
network port does nothing; break the pins out to GPIOs instead.

RJ45 pinout (control box side):

    pin 1   HS3 (M button)
    pin 2   DTX — height broadcast UART, 5 V TTL, controller → handset
    pin 3   GND
    pin 4   HTX — handset → controller UART. Unused here: the JCB35N2
            ignores the packet-based ``F1 F1 ...`` command protocol on
            this port (github.com/phord/Jarvis discussion #4), so
            movement must be commanded via the button lines.
    pin 5   +5 V (from the control box)
    pin 6   HS2
    pin 7   HS1 — "up" button line
    pin 8   HS0 — "down" button line

Movement is commanded exactly like the physical handset: pull HS1 (up) or
HS0 (down) to ground to press, leave the line floating to release. The box
ramps, stops on release, and has its own anti-collision stop, so hold-to-
move is the whole protocol. The control box provides 5 V pull-ups on every
line, so each GPIO is requested open-drain: it only ever sinks the line to
ground and never sources voltage.

WIRING NOTE: a released line sits at 5 V. Raspberry Pi GPIOs are not
5 V-tolerant — put a level shifter, an NPN/MOSFET stage, or at least a
series schottky diode (cathode toward the GPIO) between the control box
and the header. The same goes for the height UART on pin 2 (use a divider
or shifter into the Pi's RX).

Height feedback (optional): while moving (and for ~10 s after), the box
streams on pin 2 — start marker ``0x81 0xF5``, then 4-byte frames
``[0x01, 0x01, hi, lo]`` with the height in mm big-endian, end marker
``0x01 0x05``.

Sources: github.com/phord/Jarvis (pinout, HS button lines),
github.com/IPSW1/desk-control and github.com/auchter/esphome-jcb35n2
(JCB35N2 behavior and height-stream framing).
"""

from __future__ import annotations

UP = 1
STOP = 0
DOWN = -1

# Plausibility window for decoded heights (mm) — desk legs run roughly
# 0.6-1.3 m; anything outside means we are misaligned in the byte stream.
_HEIGHT_MIN_MM = 100
_HEIGHT_MAX_MM = 3000


class JiecangLift:
    """Drive a JCB35N2 by emulating handset button presses on open-drain GPIOs.

    Args:
        chip:        gpiochip device path, e.g. ``"/dev/gpiochip0"``.
        up_offset:   GPIO line offset wired to RJ45 pin 7 (HS1, up).
        down_offset: GPIO line offset wired to RJ45 pin 8 (HS0, down).
    """

    def __init__(self, chip: str, up_offset: int, down_offset: int) -> None:
        try:
            import gpiod
            from gpiod.line import Direction, Drive, Value
        except ImportError:
            raise SystemExit(
                "gpiod is not installed — run with the gamepad extra:\n"
                "  uv run --extra gamepad -m almond_axol.diagnostics.base.drive"
            )
        self._value = Value
        self._up = up_offset
        self._down = down_offset
        # active_low: "pressed" (ACTIVE) sinks the line to ground; open-drain
        # keeps a released line floating for the box's 5 V pull-up.
        settings = gpiod.LineSettings(
            direction=Direction.OUTPUT,
            drive=Drive.OPEN_DRAIN,
            active_low=True,
            output_value=Value.INACTIVE,
        )
        self._request = gpiod.request_lines(
            chip,
            consumer="axol-base-lift",
            config={(up_offset, down_offset): settings},
        )
        self._direction = STOP

    def command(self, direction: int) -> None:
        """Press or release the button lines. +1 = up, 0 = stop, -1 = down."""
        if direction == self._direction:
            return
        v = self._value
        self._request.set_values(
            {
                self._up: v.ACTIVE if direction == UP else v.INACTIVE,
                self._down: v.ACTIVE if direction == DOWN else v.INACTIVE,
            }
        )
        self._direction = direction

    def close(self) -> None:
        """Release both buttons and free the GPIO lines."""
        try:
            self.command(STOP)
        finally:
            self._request.release()


class HeightReader:
    """Parse the JCB35N2 height broadcast from a serial port (9600 8N1).

    ``poll()`` is non-blocking: it drains whatever bytes have arrived,
    updates :attr:`height_mm`, and returns it (``None`` until the first
    valid frame — the box only transmits while moving).
    """

    def __init__(self, port: str) -> None:
        try:
            import serial
        except ImportError:
            raise SystemExit(
                "pyserial is not installed — run with the gamepad extra:\n"
                "  uv run --extra gamepad -m almond_axol.diagnostics.base.drive"
            )
        self._serial = serial.Serial(port, 9600, timeout=0)
        self._buf = bytearray()
        self.height_mm: int | None = None

    def poll(self) -> int | None:
        data = self._serial.read(256)
        if data:
            self._feed(data)
        return self.height_mm

    def _feed(self, data: bytes) -> None:
        """Scan for [0x01, 0x01, hi, lo] frames, resyncing byte-by-byte.

        The start/end markers (0x81 0xF5 / 0x01 0x05) and any garbage fail
        either the header match or the plausibility window and are skipped.
        """
        self._buf.extend(data)
        while len(self._buf) >= 4:
            if self._buf[0] == 0x01 and self._buf[1] == 0x01:
                height = (self._buf[2] << 8) | self._buf[3]
                if _HEIGHT_MIN_MM <= height <= _HEIGHT_MAX_MM:
                    self.height_mm = height
                    del self._buf[:4]
                    continue
            del self._buf[:1]

    def close(self) -> None:
        self._serial.close()
