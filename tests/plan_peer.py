"""Loopback wire fixture for testing robot clients without a policy server.

This intentionally implements no prediction history, model adapter, or runtime
policy API. Test fixtures choose the exact replies the robot client receives.
"""

from __future__ import annotations

import threading

from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

from almond_axol.policy.plan_protocol import (
    PlanActions,
    decode_hello,
    decode_observation,
    decode_reset,
    encode_actions,
    encode_ready,
    encode_reset,
)
from almond_axol.policy.protocol import (
    MAX_MESSAGE_BYTES,
    decode_message,
    encode_error,
)


class PlanPeer:
    """Send scripted setup/reset/action responses over a local WebSocket."""

    def __init__(self, fixture, host="127.0.0.1", port=0, *, reply_hook=None):
        self.fixture = fixture
        self.reply_hook = reply_hook
        self._server = serve(
            self._handle, host, port, max_size=MAX_MESSAGE_BYTES, compression=None
        )
        self._thread = None

    @property
    def port(self):
        return self._server.socket.getsockname()[1]

    @property
    def url(self):
        return f"ws://127.0.0.1:{self.port}"

    def serve_forever(self):
        self._server.serve_forever()

    def shutdown(self):
        self._server.shutdown()

    def __enter__(self):
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.shutdown()
        self._thread.join(timeout=3)
        assert not self._thread.is_alive()

    def _handle(self, ws):
        spec = None
        try:
            for message in ws:
                header, payload = decode_message(message)
                if spec is None:
                    spec = decode_hello(header, payload)
                    self.fixture.setup(spec)
                    ws.send(encode_ready(spec))
                elif header["type"] == "reset":
                    episode = decode_reset(header, payload)
                    self.fixture.reset()
                    ws.send(encode_reset(episode, reply=True))
                else:
                    observation = decode_observation(header, payload, spec)
                    result = self.fixture.infer(observation)
                    reply = (
                        result
                        if isinstance(result, PlanActions)
                        else PlanActions(observation.request_id, result)
                    )
                    message = encode_actions(reply, spec)
                    if self.reply_hook is not None:
                        self.reply_hook(reply)
                    ws.send(message)
        except ConnectionClosed:
            pass
        except Exception as exc:
            try:
                ws.send(encode_error(str(exc)))
            except ConnectionClosed:
                pass
