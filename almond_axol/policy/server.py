"""Serve your own model as an Axol policy.

Subclass :class:`Policy`, implement :meth:`Policy.infer`, and hand it to
:func:`serve`. ``axol run-policy --policy_type custom`` (or **Run Policy** in
the control panel with policy type ``custom``) then connects, streams joint
state + camera frames to :meth:`~Policy.infer`, and executes the action
chunks it returns with the same smoothing, contact stops, episode control and
dataset recording as a LeRobot checkpoint::

    import numpy as np
    from almond_axol.policy import Observation, Policy, PolicySpec, serve

    class MyPolicy(Policy):
        def setup(self, spec: PolicySpec) -> None:
            self.model = load_my_model()  # spec lists state/action/camera names

        def infer(self, obs: Observation) -> np.ndarray:
            # obs.state: float32 joints; obs.images["overhead"]: HxWx3 uint8 RGB
            return self.model(obs.state, obs.images, obs.task)  # (T, D) chunk

    serve(MyPolicy(), port=8765)

A plain function works too: ``serve(lambda obs: chunk)``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any

from .protocol import (
    MAX_MESSAGE_BYTES,
    RESET_OK,
    Observation,
    PolicyProtocolError,
    PolicySpec,
    ReadyInfo,
    as_action_chunk,
    decode_hello,
    decode_message,
    decode_observation,
    decode_reset,
    encode_actions,
    encode_error,
    encode_message,
    encode_ready,
)

_logger = logging.getLogger(__name__)


class Policy:
    """Base class for a custom policy. Override :meth:`infer`; the rest is optional.

    Attributes:
        action_names: The action layout your model emits, in column order.
            When set, the robot refuses to run unless its own layout matches
            exactly — use it so a joint-space model can't drive a robot
            configured for Cartesian actions (or vice versa). ``None`` accepts
            the robot's :attr:`PolicySpec.action_names`.
        name: Shown in the robot's logs when it connects.
        fps: The control rate your model was trained at. When set, the robot
            refuses to run at a different ``--fps`` (unless
            ``--allow_fps_mismatch``).
    """

    action_names: Sequence[str] | None = None
    name: str | None = None
    fps: int | None = None

    def setup(self, spec: PolicySpec) -> None:
        """Called once per robot connection, before any observation.

        ``spec`` is the robot's contract for the session (state/action names,
        cameras, fps, task). Raise to refuse the session; the message is shown
        to the operator.
        """

    def reset(self) -> None:
        """Called at the start of every episode — clear any recurrent state."""

    def infer(self, obs: Observation) -> Any:
        """Return an action chunk predicted from ``obs``.

        Returns:
            A ``(T, D)`` array-like (numpy, torch, nested lists) whose columns
            follow :attr:`PolicySpec.action_names` and whose row ``k`` is the
            target for ``k / fps`` seconds after ``obs`` was taken; a single
            ``(D,)`` action; or a list of ``{action_name: value}`` dicts.
            Joint targets are radians and gripper values use the robot's
            gripper position units, as in a recorded dataset's ``action``.
            The robot uses at most ``spec.actions_per_chunk`` rows.
        """
        raise NotImplementedError


class _FunctionPolicy(Policy):
    def __init__(self, fn: Callable[[Observation], Any]) -> None:
        self._fn = fn
        self.name = getattr(fn, "__name__", None)
        if self.name == "<lambda>":
            self.name = None

    def infer(self, obs: Observation) -> Any:
        return self._fn(obs)


class PolicyServer:
    """WebSocket server that exposes one :class:`Policy` to a robot.

    One robot session at a time: a second connection is refused while the
    first is open. Use :func:`serve` for the blocking one-liner, or construct
    this directly to run it on a background thread (``serve_forever`` in a
    thread, ``shutdown`` to stop; ``port`` reports the bound port, handy with
    ``port=0``).
    """

    def __init__(
        self,
        policy: Policy | Callable[[Observation], Any],
        host: str = "0.0.0.0",
        port: int = 8765,
    ) -> None:
        from websockets.sync.server import serve as ws_serve

        if not isinstance(policy, Policy):
            if not callable(policy):
                raise TypeError("policy must be a Policy or a callable(obs) -> chunk")
            policy = _FunctionPolicy(policy)
        self.policy = policy
        self._session = threading.Lock()
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
        if not self._session.acquire(blocking=False):
            _logger.warning("Refusing %s: a robot session is already open.", peer)
            ws.send(encode_error("Policy server already has a robot connected."))
            return
        try:
            _logger.info("Robot connected from %s.", peer)
            self._session_loop(ws)
        finally:
            self._session.release()
            _logger.info("Robot %s disconnected.", peer)

    def _session_loop(self, ws: Any) -> None:
        from websockets.exceptions import ConnectionClosed

        spec: PolicySpec | None = None
        action_names: tuple[str, ...] = ()
        try:
            for message in ws:
                try:
                    header, payload = decode_message(message)
                    kind = header["type"]
                    if kind == "hello":
                        spec = decode_hello(header)
                        self.policy.setup(spec)
                        declared = self.policy.action_names
                        action_names = (
                            tuple(declared) if declared else spec.action_names
                        )
                        ws.send(
                            encode_ready(
                                ReadyInfo(
                                    action_names=action_names,
                                    name=self.policy.name,
                                    fps=self.policy.fps,
                                )
                            )
                        )
                        _logger.info(
                            "Session: %d-dim state, %d-dim actions, cameras %s, "
                            "%d fps, task %r.",
                            len(spec.state_names),
                            len(action_names),
                            list(spec.camera_names),
                            spec.fps,
                            spec.task,
                        )
                    elif spec is None:
                        raise PolicyProtocolError("Expected 'hello' first.")
                    elif kind == "reset":
                        episode = decode_reset(header)
                        self.policy.reset()
                        ws.send(encode_message(RESET_OK))
                        _logger.info("Episode %d.", episode)
                    elif kind == "observation":
                        obs = decode_observation(header, payload, spec)
                        chunk = as_action_chunk(self.policy.infer(obs), action_names)
                        ws.send(encode_actions(chunk, obs.timestep))
                    else:
                        raise PolicyProtocolError(f"Unknown message type {kind!r}.")
                except ConnectionClosed:
                    raise
                except Exception as exc:  # noqa: BLE001 - relayed to the robot
                    _logger.exception("Policy request failed")
                    ws.send(encode_error(f"{type(exc).__name__}: {exc}"))
        except ConnectionClosed:
            pass


def serve(
    policy: Policy | Callable[[Observation], Any],
    host: str = "0.0.0.0",
    port: int = 8765,
) -> None:
    """Serve ``policy`` on ``host:port`` until Ctrl+C.

    Point the robot at it with ``--server_host``/``--server_port`` (control
    panel: Settings → Inference). The protocol is unauthenticated plaintext:
    keep it on an isolated, trusted network, or bind ``127.0.0.1`` when the
    model runs on the robot's own machine.
    """
    server = PolicyServer(policy, host=host, port=port)
    _logger.info("Serving custom policy on %s:%d (Ctrl+C to stop).", host, server.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _logger.info("Policy server stopped.")
    finally:
        server.shutdown()
