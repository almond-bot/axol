"""Live protocol checks against the built axol-rt binary — no CAN needed.

Verifies the Unix-socket handshake and the two failure semantics that must
never regress: a clean client disconnect exits 0, and a version-skewed
target (wrong tuple size) kills the core loudly instead of misparsing or
leaving an energized orphan.

    cargo build --release && uv run python tools/rt_proto_check.py
"""

import asyncio
import os
import struct
import subprocess

BIN = os.environ.get(
    "AXOL_RT_BIN",
    os.path.join(os.path.dirname(__file__), "..", "target", "release", "axol-rt"),
)


async def session(name, actions):
    sock = f"/tmp/axol-rt-test-{os.getpid()}-{name}.sock"
    proc = subprocess.Popen(
        [BIN, "serve", "--socket", sock],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    r = w = None
    for _ in range(100):
        try:
            r, w = await asyncio.open_unix_connection(sock)
            break
        except (ConnectionRefusedError, FileNotFoundError):
            await asyncio.sleep(0.02)
    assert w is not None, "could not connect"

    def send(payload):
        w.write(struct.pack("<I", len(payload)) + payload)

    async def recv():
        (size,) = struct.unpack("<I", await r.readexactly(4))
        p = await r.readexactly(size)
        return p[:1], p[1:].decode()

    result = await actions(send, recv, w)
    w.close()
    try:
        await w.wait_closed()
    except (BrokenPipeError, ConnectionResetError):
        pass
    out, _ = await asyncio.to_thread(proc.communicate, None, 5)
    return proc.returncode, out.decode(), result


cfg = b"C" + b"loop_hz 240\njoint 0 can_alm_axol_l shoulder_1 1 250 3.5\n"


async def clean(send, recv, w):
    send(cfg)
    tag, body = await recv()
    assert (tag, body) == (b"S", "config-ok"), (tag, body)
    return "config-ok received"


async def skewed(send, recv, w):
    send(cfg)
    await recv()
    send(struct.pack("<cBI", b"T", 0, 1) + struct.pack("<5d", *([0.0] * 5)) * 8)
    await asyncio.sleep(0.5)
    return "sent skewed target"


rc, out, msg = asyncio.run(session("clean", clean))
print(f"clean disconnect: rc={rc} ({msg})")
assert rc == 0, out

rc, out, msg = asyncio.run(session("skewed", skewed))
print(f"skewed target:    rc={rc} ({msg})")
print("core output:", out.strip().splitlines()[-1] if out.strip() else "(none)")
assert rc != 0, "core must exit nonzero on protocol error"
print("protocol-level checks OK")
