"""Serve your own model as an Axol policy.

Subclass :class:`Policy`, implement :meth:`Policy.infer`, and hand it to
:func:`serve`. ``axol run-policy --policy_type custom`` (or **Run Policy** in
the control panel with policy type ``custom``) connects, sends joint state +
camera frames, and executes the action chunks :meth:`~Policy.infer` returns::

    import numpy as np
    from almond_axol.policy import Observation, Policy, PolicySpec, serve

    class MyPolicy(Policy):
        def setup(self, spec: PolicySpec) -> None:
            self.model = load_my_model()  # spec lists state/action/camera names

        def infer(self, obs: Observation) -> np.ndarray:
            # obs.state: float32 joints; obs.images["overhead"]: HxWx3 uint8 RGB
            return self.model(obs.state, obs.images)  # (T, D) chunk

    serve(MyPolicy(), port=8765)

A plain function works too: ``serve(lambda obs: chunk)``.

:class:`PolicyServer` implements the endpoint side of the custom policy
interface (:mod:`almond_axol.policy.plan_protocol`) so a policy never touches
it: it echoes the session contract, caches every plan it publishes, resolves
the robot's ``continuation`` reference back into the rows still executing
(:attr:`Observation.plan`), clears that history on every ``reset``, and
optionally temporally ensembles overlapping chunks before publishing them.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .plan_protocol import (
    PlanActions,
    PlanObservation,
    PlanSpec,
    decode_hello,
    decode_observation,
    decode_reset,
    encode_actions,
    encode_ready,
    encode_reset,
)
from .protocol import (
    MAX_MESSAGE_BYTES,
    PolicyProtocolError,
    as_action_chunk,
    decode_message,
    encode_error,
)

_logger = logging.getLogger(__name__)

PolicySpec = PlanSpec
"""The robot's session contract, handed to :meth:`Policy.setup`."""


@dataclass
class Observation:
    """One robot observation, as handed to :meth:`Policy.infer`.

    Attributes:
        state: Joint/EE state, float32, in :attr:`PolicySpec.state_names` order.
        state_names: Names for each entry of ``state``.
        images: Camera name → ``(height, width, 3)`` uint8 RGB frame.
        plan: The rows of the robot's current plan that have not executed yet,
            aligned with your reply: ``plan[k]`` is what the robot would
            command at the tick your row ``k`` targets. ``None`` when nothing
            is executing (episode start, after a reset or recovery). Use it to
            make a new chunk continue smoothly from the motion in progress.
        delay_steps: The robot's estimate of how many rows of your reply will
            already have passed when it arrives (``None`` unless the robot
            runs with ``plan_config.advertise_delay``).
        state_time_ns: Robot-monotonic time the state was sampled.
        image_time_ns: Robot-monotonic capture time per camera.
        request_id: This request's id (also the id of the plan you publish).
    """

    state: np.ndarray
    state_names: tuple[str, ...]
    images: Mapping[str, np.ndarray]
    plan: np.ndarray | None
    delay_steps: int | None
    state_time_ns: int
    image_time_ns: Mapping[str, int]
    request_id: str
    _state_dict: dict[str, float] | None = field(default=None, repr=False)

    @property
    def joints(self) -> dict[str, float]:
        """``state`` keyed by name, e.g. ``obs.joints["left_gripper.pos"]``."""
        if self._state_dict is None:
            self._state_dict = {
                name: float(value)
                for name, value in zip(self.state_names, self.state, strict=True)
            }
        return self._state_dict


class Policy:
    """Base class for a custom policy. Override :meth:`infer`; the rest is optional.

    Attributes:
        action_names: The action layout your model emits, in column order.
            When set, the session is refused unless the robot's layout matches
            exactly — so a joint-space model can't drive a robot configured
            for Cartesian actions (or vice versa). ``None`` accepts the
            robot's :attr:`PolicySpec.action_names`.
        fps: The control rate your model was trained at. When set, the session
            is refused unless the robot runs at the same ``--fps``.
        name: Shown in this server's log.
    """

    action_names: Sequence[str] | None = None
    fps: int | None = None
    name: str | None = None

    def setup(self, spec: PolicySpec) -> None:
        """Called once per robot connection, before any observation.

        ``spec`` is the robot's contract for the session (state/action names,
        cameras, fps, horizon). Raise to refuse the session; the message is
        shown to the operator.
        """

    def reset(self) -> None:
        """Called whenever the robot discards its plan — clear recurrent state.

        That is the start of every episode, and also any time execution
        restarts from scratch mid-episode (a stop, an operator takeover, or
        recovery from a late reply).
        """

    def infer(self, obs: Observation) -> Any:
        """Return an action chunk predicted from ``obs``.

        Returns:
            A ``(T, D)`` array-like (numpy, torch, nested lists) whose columns
            follow :attr:`PolicySpec.action_names` and whose row ``k`` is the
            target for ``k / fps`` seconds after ``obs`` was taken; a single
            ``(D,)`` action; or a list of ``{action_name: value}`` dicts.
            Joint targets are radians, the gripper is ``0`` (closed) to ``1``
            (open), as in a recorded dataset's ``action``. Rows beyond
            ``spec.actions_per_chunk`` are dropped.
        """
        raise NotImplementedError


class _FunctionPolicy(Policy):
    def __init__(self, fn: Callable[[Observation], Any]) -> None:
        self._fn = fn
        name = getattr(fn, "__name__", None)
        self.name = None if name == "<lambda>" else name

    def infer(self, obs: Observation) -> Any:
        return self._fn(obs)


def _snap_indices(action_names: Sequence[str]) -> tuple[int, ...]:
    """Dims ensembling must not average: grippers and rotation vectors.

    Averaging a bang-bang gripper command smears a grasp into a slow squeeze,
    and rotation vectors double-cover SO(3) (two near-identical orientations
    can arrive as opposite vectors), so both follow the newest prediction.
    """
    return tuple(
        i
        for i, name in enumerate(action_names)
        if name.endswith("gripper.pos") or name.endswith(("_ee.rx", "_ee.ry", "_ee.rz"))
    )


class _Session:
    """Published-plan history for one robot session, cleared on reset.

    Plans live on a local timeline: a bootstrap request (no continuation)
    starts at 0, and a request continuing plan ``P`` at ``from_row`` starts
    at ``origin(P) + from_row`` — the robot's definition of row zero. So the
    timeline follows the plan the robot actually accepted, even when some
    published plans were never adopted.
    """

    MAX_PLANS = 64

    def __init__(self, spec: PlanSpec, ensemble: float | None) -> None:
        self.spec = spec
        self.ensemble = ensemble
        self.snap = _snap_indices(spec.action_names)
        # request_id -> (origin, published rows, raw prediction rows)
        self.plans: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}

    def clear(self) -> None:
        self.plans.clear()

    def resolve(self, obs: PlanObservation) -> tuple[int, np.ndarray | None]:
        """Return this request's origin and the accepted rows still executing."""
        if obs.continuation is None:
            # Nothing is executing: whatever came before no longer conditions.
            self.clear()
            return 0, None
        entry = self.plans.get(obs.continuation.prediction_id)
        if entry is None:
            raise PolicyProtocolError(
                f"The robot is executing plan {obs.continuation.prediction_id!r}, "
                "which this server did not publish in the current session."
            )
        origin, published, _ = entry
        start = obs.continuation.from_row
        if start >= len(published):
            raise PolicyProtocolError("Continuation row is past the end of its plan.")
        remaining = published[start:].copy()
        remaining.flags.writeable = False
        return origin + start, remaining

    def publish(self, request_id: str, origin: int, raw: np.ndarray) -> np.ndarray:
        published = raw if self.ensemble is None else self._ensemble(origin, raw)
        self.plans[request_id] = (origin, published, raw)
        # Drop plans that ended before this one began, then cap the history.
        for key in [k for k, (o, p, _) in self.plans.items() if o + len(p) <= origin]:
            del self.plans[key]
        while len(self.plans) > self.MAX_PLANS:
            del self.plans[next(iter(self.plans))]
        return published

    def _ensemble(self, origin: int, raw: np.ndarray) -> np.ndarray:
        """ACT temporal ensembling over every raw prediction covering each row.

        Weight ``exp(-k·i)`` with ``i = 0`` the oldest contributor (the ACT
        paper's convention: ``k > 0`` favours older predictions, i.e. smoother).
        Snap dims (grippers, rotation vectors) take the newest prediction.
        """
        contributors = sorted(
            ((o, r) for o, _, r in self.plans.values()), key=lambda item: item[0]
        )
        contributors.append((origin, raw))
        out = raw.astype(np.float64)
        for row in range(len(raw)):
            tick = origin + row
            values = [r[tick - o] for o, r in contributors if 0 <= tick - o < len(r)]
            if len(values) < 2:
                continue
            stacked = np.stack(values).astype(np.float64)
            weights = np.exp(-self.ensemble * np.arange(len(values)))
            out[row] = weights @ stacked / weights.sum()
        out[:, list(self.snap)] = raw[:, list(self.snap)]
        return out.astype(np.float32)


class PolicyServer:
    """WebSocket endpoint that exposes one :class:`Policy` to a robot.

    One robot session at a time: a second connection is refused while the
    first is open. Use :func:`serve` for the blocking one-liner, or construct
    this directly to run it on a background thread (``serve_forever`` in a
    thread, ``shutdown`` to stop; ``port`` reports the bound port, handy with
    ``port=0``).

    Args:
        policy: A :class:`Policy`, or a callable ``obs -> chunk``.
        host: Interface to bind.
        port: TCP port (``0`` picks a free one).
        ensemble: ``None`` (default) publishes each chunk as returned. A
            number ``k`` temporally ensembles overlapping predictions before
            publishing (ACT's ``exp(-k·i)`` weights; ``0.01`` is ACT's
            default) — useful for chunked policies trained with ensembling.
            Leave it off for policies that already produce smooth plans, e.g.
            ones conditioned on :attr:`Observation.plan`.
    """

    def __init__(
        self,
        policy: Policy | Callable[[Observation], Any],
        host: str = "0.0.0.0",
        port: int = 8765,
        *,
        ensemble: float | None = None,
    ) -> None:
        from websockets.sync.server import serve as ws_serve

        if not isinstance(policy, Policy):
            if not callable(policy):
                raise TypeError("policy must be a Policy or a callable(obs) -> chunk")
            policy = _FunctionPolicy(policy)
        if ensemble is not None and not (
            isinstance(ensemble, (int, float)) and np.isfinite(ensemble)
        ):
            raise ValueError("ensemble must be a finite number or None")
        self.policy = policy
        self.ensemble = None if ensemble is None else float(ensemble)
        self._session_lock = threading.Lock()
        self._server = ws_serve(
            self._handle,
            host,
            port,
            max_size=MAX_MESSAGE_BYTES,
            compression=None,
        )

    @property
    def port(self) -> int:
        return self._server.socket.getsockname()[1]

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def shutdown(self) -> None:
        self._server.shutdown()

    def _handle(self, ws: Any) -> None:
        peer = getattr(ws, "remote_address", None)
        if not self._session_lock.acquire(blocking=False):
            _logger.warning("Refusing %s: a robot session is already open.", peer)
            ws.send(encode_error("Policy server already has a robot connected."))
            return
        try:
            _logger.info("Robot connected from %s.", peer)
            self._session_loop(ws)
        finally:
            self._session_lock.release()
            _logger.info("Robot %s disconnected.", peer)

    def _check_spec(self, spec: PlanSpec) -> None:
        declared = self.policy.action_names
        if declared is not None and tuple(declared) != spec.action_names:
            raise PolicyProtocolError(
                f"This policy emits {list(declared)}, but the robot is configured "
                f"for {list(spec.action_names)}. Match the robot's action layout "
                "(joint vs. Cartesian, gripper) to the one the model was trained on."
            )
        if self.policy.fps is not None and self.policy.fps != spec.fps:
            raise PolicyProtocolError(
                f"This policy was trained at {self.policy.fps} fps but the robot "
                f"runs at {spec.fps}; pass --fps {self.policy.fps}."
            )

    def _session_loop(self, ws: Any) -> None:
        from websockets.exceptions import ConnectionClosed

        session: _Session | None = None
        try:
            for message in ws:
                try:
                    header, payload = decode_message(message)
                    kind = header.get("type")
                    if session is None:
                        spec = decode_hello(header, payload)
                        self._check_spec(spec)
                        self.policy.setup(spec)
                        session = _Session(spec, self.ensemble)
                        ws.send(encode_ready(spec))
                        _logger.info(
                            "Session%s: %d-dim state, %d-dim actions, cameras %s, "
                            "%d fps, horizon %d, request every %d ticks.",
                            f" ({self.policy.name})" if self.policy.name else "",
                            len(spec.state_names),
                            len(spec.action_names),
                            list(spec.camera_names),
                            spec.fps,
                            spec.actions_per_chunk,
                            spec.request_interval,
                        )
                    elif kind == "reset":
                        episode = decode_reset(header, payload)
                        session.clear()
                        self.policy.reset()
                        ws.send(encode_reset(episode, reply=True))
                    else:
                        ws.send(self._infer(session, header, payload))
                except ConnectionClosed:
                    raise
                except Exception as exc:  # noqa: BLE001 - relayed to the robot
                    _logger.exception("Policy request failed")
                    ws.send(encode_error(f"{type(exc).__name__}: {exc}"))
        except ConnectionClosed:
            pass

    def _infer(self, session: _Session, header: Any, payload: memoryview) -> bytes:
        spec = session.spec
        request = decode_observation(header, payload, spec)
        origin, plan = session.resolve(request)
        obs = Observation(
            state=request.state,
            state_names=spec.state_names,
            images=request.images,
            plan=plan,
            delay_steps=request.delay_steps,
            state_time_ns=request.state_sample_time_ns,
            image_time_ns=request.image_capture_time_ns,
            request_id=request.request_id,
        )
        chunk = as_action_chunk(self.policy.infer(obs), spec.action_names)
        chunk = chunk[: spec.actions_per_chunk]
        published = session.publish(request.request_id, origin, chunk)
        return encode_actions(PlanActions(request.request_id, published), spec)


def serve(
    policy: Policy | Callable[[Observation], Any],
    host: str = "0.0.0.0",
    port: int = 8765,
    *,
    ensemble: float | None = None,
) -> None:
    """Serve ``policy`` on ``host:port`` until Ctrl+C.

    Point the robot at it with ``--server_host``/``--server_port`` (control
    panel: Settings → Inference). See :class:`PolicyServer` for ``ensemble``.
    The interface is unauthenticated plaintext: keep it on an isolated,
    trusted network, or bind ``127.0.0.1`` when the model runs on the robot's
    own machine.
    """
    server = PolicyServer(policy, host=host, port=port, ensemble=ensemble)
    _logger.info("Serving custom policy on %s:%d (Ctrl+C to stop).", host, server.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _logger.info("Policy server stopped.")
    finally:
        server.shutdown()
