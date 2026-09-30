"""The custom policy interface: timestamped observations and plan continuation.

``run-policy --policy_type custom`` and ``collect-dagger --policy_type custom``
always use :class:`PlanPolicyClient` and the wire contract defined in
:mod:`almond_axol.policy.plan_protocol`. The robot owns scheduling and
execution; model processing and published-plan caching stay at the endpoint.
The handshake's version 2 identifies that required contract, not an opt-in mode.

The original :class:`PolicyClient`, :class:`Policy`, :class:`PolicyServer`,
and :func:`serve` exports remain for standalone legacy SDK consumers. They
speak wire version 1 and are incompatible with the current robot commands;
there is no robot-side downgrade or legacy execution path.

The current interface uses OpenCV for PNG and Pillow for optional image resize.
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
