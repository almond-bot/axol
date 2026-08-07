"""
Telescoping lift on the powered Axol Cart — placeholder driver.

The lift used to be a Jiecang JCB35N2 control box driven by emulating its
wired handset: two active-low button lines pulled to ground to move the
legs, plus a one-directional 9600 8N1 UART carrying the box's height
broadcast. That box is being replaced by our own PCB driving the legs
directly, so nothing here touches hardware in the meantime —
:class:`Lift` just latches the commanded direction and :class:`HeightReader`
never reports a height. Every call site (:class:`almond_axol.robot.cart.Cart`
and ``almond_axol.diagnostics.base.drive``) stays wired to this interface,
so bringing the PCB up is a change to this module alone.

The JCB35N2 handset protocol, RJ45 pinout, and 5 V level-shifting notes
live in this file's git history if the old box is ever needed again.
"""

from __future__ import annotations

UP = 1
STOP = 0
DOWN = -1


class Lift:
    """Accepts hold-to-move lift commands and does nothing with them.

    The real driver moves the legs while a direction is held and stops on
    release, so :meth:`command` takes the same +1/0/-1 and acts only on
    changes.
    """

    def __init__(self) -> None:
        self._direction = STOP

    def command(self, direction: int) -> None:
        """Latch the commanded direction. +1 = up, 0 = stop, -1 = down."""
        if direction == self._direction:
            return
        self._direction = direction

    def close(self) -> None:
        """Release the lift (stops it, as dropping the buttons used to)."""
        self._direction = STOP


class HeightReader:
    """Reports the lift's height in mm; ``None`` until a driver supplies one.

    ``poll()`` is non-blocking and safe to call from a display loop at any
    rate.
    """

    def __init__(self) -> None:
        self.height_mm: int | None = None

    def poll(self) -> int | None:
        return self.height_mm

    def close(self) -> None:
        pass
