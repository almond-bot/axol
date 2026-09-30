"""Run your own model on Axol — no LeRobot checkpoint required.

Write a :class:`Policy` (or a plain ``obs -> chunk`` function), start it with
:func:`serve`, and run ``axol run-policy --policy_type custom`` (control
panel: **Run Policy**, policy type ``custom``). The robot sends each
:class:`Observation` — joint state, RGB camera frames, task — and executes the
action chunks you return. See :mod:`almond_axol.policy.protocol` for the wire
format if your model lives outside Python.

The custom policy interface (v2) adds lossless image transport and references
to accepted plans.
Use :class:`PlanPolicyClient` with a compatible external endpoint. The robot
owns scheduling and execution; architecture-specific processing stays remote.

Version 1 only needs numpy and websockets. Version 2 also uses OpenCV for its
PNG codec; the optional robot-side image resize uses Pillow.
"""

from .client import PolicyClient, policy_url
from .plan_client import PlanPolicyClient
from .plan_protocol import (
    PLAN_PROTOCOL_VERSION,
    Continuation,
    LastDispatched,
    PlanActions,
    PlanObservation,
    PlanSpec,
)
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
    "PLAN_PROTOCOL_VERSION",
    "PROTOCOL_VERSION",
    "CameraSpec",
    "Continuation",
    "LastDispatched",
    "Observation",
    "PlanActions",
    "PlanObservation",
    "PlanPolicyClient",
    "PlanSpec",
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
