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


cfg = (
    b"C"
    + b"loop_hz 240\n"
    + b"joint 0 can_alm_axol_l shoulder_1 1 250 3.5 9.4 33.0 0.6 250 0.15 0.02\n"
)


async def clean(send, recv, w):
    send(cfg)
    tag, body = await recv()
    assert (tag, body) == (b"S", "config-ok"), (tag, body)
    return "config-ok received"


async def skewed(send, recv, w):
    send(cfg)
    await recv()
    # A previous-generation 8-field target against the 9-field core.
    send(struct.pack("<cBI", b"T", 0, 1) + struct.pack("<8d", *([0.0] * 8)) * 8)
    await asyncio.sleep(0.5)
    return "sent skewed target"


def check_feedback_parse():
    """Cross-language layout check for `F` telemetry packets.

    Builds the exact byte stream `build_feedback` in serve.rs produces (see
    its `feedback_packet_layout` unit test) and asserts `RtLink._parse_feedback`
    recovers the values, including the age -> timestamp reconstruction.
    """
    import sys
    import time

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from almond_axol.rt.link import RtLink

    slots_in = {0: (1.5, -0.25, 3.0, 1200), 7: (0.5, 0.0, 0.1, 0)}
    payload = b"F" + bytes([1, 0b1000_0001])
    for i in range(8):
        pos, vel, tau, age = slots_in.get(i, (0.0, 0.0, 0.0, 0))
        payload += struct.pack("<3dI", pos, vel, tau, age)

    before = time.time()
    side, slots = RtLink._parse_feedback(payload)
    after = time.time()
    assert side == 1, side
    assert set(slots) == {0, 7}, slots
    assert slots[0][:3] == (1.5, -0.25, 3.0), slots[0]
    assert slots[7][:3] == (0.5, 0.0, 0.1), slots[7]
    assert before - 1200e-6 <= slots[0][3] <= after - 1200e-6, slots[0]
    assert before <= slots[7][3] <= after, slots[7]
    print("feedback parse:   layout + timestamp reconstruction OK")


check_feedback_parse()

rc, out, msg = asyncio.run(session("clean", clean))
print(f"clean disconnect: rc={rc} ({msg})")
assert rc == 0, out

rc, out, msg = asyncio.run(session("skewed", skewed))
print(f"skewed target:    rc={rc} ({msg})")
print("core output:", out.strip().splitlines()[-1] if out.strip() else "(none)")
assert rc != 0, "core must exit nonzero on protocol error"
print("protocol-level checks OK")
