"""Robot side of the custom policy protocol.

``axol run-policy --policy_type custom`` uses :class:`PolicyClient` to talk to
a :func:`~almond_axol.policy.serve` process; it is equally usable on its own to
exercise a policy server without a robot (see ``axol policy.check``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .protocol import (
    MAX_MESSAGE_BYTES,
    PolicyProtocolError,
    PolicySpec,
    ReadyInfo,
    decode_actions,
    decode_message,
    decode_ready,
    encode_hello,
    encode_observation,
    encode_reset,
    expect_type,
)


def policy_url(host: str, port: int) -> str:
    """``ws://host:port`` for a bare host; a ``ws://``/``wss://`` URL passes through."""
    host = host.strip()
    if host.startswith(("ws://", "wss://")):
        return host
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # bare IPv6 literal
    return f"ws://{host}:{port}"


class PolicyClient:
    """One robot session against a custom policy server.

    Not thread-safe: every call is a request/reply round trip, so drive it
    from a single thread.

    Args:
        url: ``ws://host:port`` (see :func:`policy_url`).
        open_timeout: Seconds to wait for the connection to open.
        reply_timeout: Seconds to wait for any single reply — covers the
            policy's ``setup`` (model load) and each ``infer`` call.
    """

    def __init__(
        self,
        url: str,
        *,
        open_timeout: float = 10.0,
        reply_timeout: float = 60.0,
    ) -> None:
        self.url = url
        self.open_timeout = open_timeout
        self.reply_timeout = reply_timeout
        self.spec: PolicySpec | None = None
        self.ready: ReadyInfo | None = None
        self._ws: Any = None

    def connect(self, spec: PolicySpec) -> ReadyInfo:
        """Open the session: send ``hello`` and return the policy's ``ready``."""
        from websockets.sync.client import connect

        self.close()
        self._ws = connect(
            self.url,
            open_timeout=self.open_timeout,
            max_size=MAX_MESSAGE_BYTES,
            compression=None,
        )
        self.spec = spec
        header, _ = self._request(encode_hello(spec))
        self.ready = decode_ready(header)
        return self.ready

    def reset(self, episode: int) -> None:
        """Tell the policy a new episode starts."""
        header, _ = self._request(encode_reset(episode))
        expect_type(header, "reset_ok")

    def infer(
        self,
        *,
        state: Sequence[float],
        images: Mapping[str, np.ndarray],
        task: str,
        timestep: int,
        timestamp: float,
    ) -> np.ndarray:
        """Send one observation; return the ``(T, D)`` float32 chunk predicted from it."""
        if self.spec is None or self.ready is None:
            raise PolicyProtocolError("PolicyClient.connect() must run first.")
        header, payload = self._request(
            encode_observation(
                spec=self.spec,
                state=state,
                images=images,
                task=task,
                timestep=timestep,
                timestamp=timestamp,
            )
        )
        reply_step, chunk = decode_actions(header, payload, self.ready.action_names)
        if reply_step != timestep:
            raise PolicyProtocolError(
                f"Policy answered observation {timestep} with a chunk for "
                f"{reply_step}; the session is out of sync."
            )
        return chunk

    def close(self) -> None:
        """Close the connection; safe to call from another thread to unblock a request."""
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass

    def _request(self, message: bytes) -> tuple[dict[str, Any], memoryview]:
        ws = self._ws
        if ws is None:
            raise PolicyProtocolError("Policy connection is closed.")
        ws.send(message)
        try:
            reply = ws.recv(timeout=self.reply_timeout)
        except TimeoutError:
            raise TimeoutError(
                f"Policy server at {self.url} did not reply within "
                f"{self.reply_timeout:.0f}s."
            ) from None
        return decode_message(reply)

    def __enter__(self) -> PolicyClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
