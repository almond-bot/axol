"""Run your own model on Axol — no LeRobot checkpoint required.

Write a :class:`Policy` (or a plain ``obs -> chunk`` function), start it with
:func:`serve`, and run ``axol run-policy --policy_type custom`` (control
panel: **Run Policy**, policy type ``custom``). The robot sends each
:class:`Observation` — joint state, RGB camera frames, task — and executes the
action chunks you return. See :mod:`almond_axol.policy.protocol` for the wire
format if your model lives outside Python.

This package only needs Axol's base install (numpy + websockets).
"""

from .client import PolicyClient, policy_url
from .protocol import (
    PROTOCOL_VERSION,
    CameraSpec,
    Observation,
    PolicyProtocolError,
    PolicyRemoteError,
    PolicySpec,
    ReadyInfo,
)
from .server import Policy, PolicyServer, serve

__all__ = [
    "PROTOCOL_VERSION",
    "CameraSpec",
    "Observation",
    "Policy",
    "PolicyClient",
    "PolicyProtocolError",
    "PolicyRemoteError",
    "PolicyServer",
    "PolicySpec",
    "ReadyInfo",
    "policy_url",
    "serve",
]
