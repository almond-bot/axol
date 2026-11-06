"""Jogging: hold-to-move control of the arms, lift and Jelly base.

The control panel's Jog page drives a :class:`~almond_axol.jog.session.JogSession`
running as the ``jog`` operation. The session talks to either a simulated
robot (:mod:`almond_axol.jog.sim`) or the real one through the same backend
interface, so the page behaves the same in both modes.
"""
