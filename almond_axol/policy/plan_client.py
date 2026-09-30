"""Single-in-flight client for the custom policy interface."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from .plan_protocol import (
    PlanActions,
    PlanObservation,
    PlanSpec,
    decode_actions,
    decode_ready,
    decode_reset,
    encode_hello,
    encode_observation,
    encode_reset,
)
from .protocol import MAX_MESSAGE_BYTES, PolicyProtocolError, decode_message


def policy_url(host: str, port: int) -> str:
    """``ws://host:port`` for a bare host; a ``ws://``/``wss://`` URL passes through."""
    host = host.strip()
    if host.startswith(("ws://", "wss://")):
        return host
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # bare IPv6 literal
    return f"ws://{host}:{port}"


class PlanPolicyClient:
    """Drive all request/reply calls from one inference thread.

    Only ``close`` is safe from another thread (to cancel a blocked receive).
    Reset must run after any outstanding inference has returned. The robot
    scheduler must independently invalidate pre-reset results before waiting.
    """

    def __init__(
        self, url: str, *, open_timeout: float = 10.0, reply_timeout: float = 60.0
    ) -> None:
        self.url = url
        self.open_timeout = open_timeout
        self.reply_timeout = reply_timeout
        self.spec: PlanSpec | None = None
        self.ready: PlanSpec | None = None
        self._ws: Any = None
        self._episode: int | None = None

    def connect(self, spec: PlanSpec) -> PlanSpec:
        from websockets.sync.client import connect

        self.close()
        message = encode_hello(spec)
        self._ws = connect(
            self.url,
            open_timeout=self.open_timeout,
            max_size=MAX_MESSAGE_BYTES,
            compression=None,
        )
        try:
            ready = decode_ready(*self._request(message))
            # Compare canonical values: callers may supply lists to the tuple
            # dataclass fields; wire validation always freezes those as tuples.
            if encode_hello(ready) != message:
                raise PolicyProtocolError("Policy ready contract differs from hello.")
            self.spec = self.ready = ready
            return ready
        except Exception:
            self.close()
            raise

    def reset(self, episode: int) -> None:
        if self.spec is None:
            raise PolicyProtocolError("PlanPolicyClient.connect() must run first.")
        self._episode = None
        try:
            ack = decode_reset(*self._request(encode_reset(episode)), reply=True)
            if ack != episode:
                raise PolicyProtocolError(
                    "Policy reset acknowledgement has the wrong episode."
                )
            self._episode = episode
        except Exception:
            self.close()
            raise

    def infer(self, obs: PlanObservation) -> PlanActions:
        spec = self.spec
        if spec is None or self._episode is None:
            raise PolicyProtocolError(
                "Connect and reset the custom policy interface session before inference."
            )
        message = encode_observation(obs, spec)
        try:
            reply = decode_actions(*self._request(message), spec)
            if reply.request_id != obs.request_id:
                raise PolicyProtocolError("Policy replied with the wrong request_id.")
            return reply
        except Exception:
            # Do not consume a late reply as the answer to the next request.
            self.close()
            raise

    def _request(self, message: bytes) -> tuple[dict[str, Any], memoryview]:
        from websockets.exceptions import ConnectionClosed

        ws = self._ws
        if ws is None:
            raise PolicyProtocolError("Policy connection is closed.")
        try:
            ws.send(message)
        except ConnectionClosed as closed:
            # Refusal can arrive before hello is sent. Preserve the queued
            # remote error while never accepting success for an unsent call.
            try:
                reply = ws.recv(timeout=self.reply_timeout)
            except (ConnectionClosed, TimeoutError):
                raise closed from None
            header, payload = decode_message(reply)
            if header.get("type") != "error":
                raise closed
            return header, payload
        try:
            reply = ws.recv(timeout=self.reply_timeout)
        except TimeoutError:
            raise TimeoutError(
                f"Policy server at {self.url} did not reply within {self.reply_timeout:g}s."
            ) from None
        return decode_message(reply)

    def close(self) -> None:
        ws, self._ws = self._ws, None
        self.spec = self.ready = None
        self._episode = None
        if ws is not None:
            with suppress(Exception):
                ws.close()

    def __enter__(self) -> PlanPolicyClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
