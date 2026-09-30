"""Run your own model on Axol — no LeRobot checkpoint required.

Write a :class:`Policy` (or a plain ``obs -> chunk`` function), start it with
:func:`serve`, and run ``axol run-policy --policy_type custom`` or
``axol collect-dagger --policy_type custom`` (control panel: policy type
``custom``). The robot sends each :class:`Observation` — joint state, RGB
camera frames, and the rows of its current plan that haven't executed yet —
and executes the action chunks you return. Test an endpoint without a robot
with :func:`check_policy` / ``axol policy.check``.

Underneath is the custom policy interface
(:mod:`almond_axol.policy.plan_protocol`, wire version 2): implement it
directly if your model lives outside Python. :class:`PlanPolicyClient` is the
robot's side of it.
"""

from .check import CheckReport, check_policy, default_spec
from .plan_client import PlanPolicyClient, policy_url
from .plan_protocol import (
    PLAN_PROTOCOL_VERSION,
    Continuation,
    LastDispatched,
    PlanActions,
    PlanObservation,
    PlanSpec,
)
from .protocol import CameraSpec, PolicyProtocolError, PolicyRemoteError
from .server import Observation, Policy, PolicyServer, PolicySpec, serve

__all__ = [
    "PLAN_PROTOCOL_VERSION",
    "CameraSpec",
    "CheckReport",
    "Continuation",
    "LastDispatched",
    "Observation",
    "PlanActions",
    "PlanObservation",
    "PlanPolicyClient",
    "PlanSpec",
    "Policy",
    "PolicyProtocolError",
    "PolicyRemoteError",
    "PolicyServer",
    "PolicySpec",
    "check_policy",
    "default_spec",
    "policy_url",
    "serve",
]
