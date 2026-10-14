"""Read or set a MyActuator joint's firmware loop gains, in RAM by default.

The X8-P20 shoulders ship with a position loop of ~0.3 Hz bandwidth
(position_kp 0.008) and a speed loop (speed_kp 0.03, speed_ki 0.0001) that
cannot hold creep speed against the gearbox's velocity-weakening friction —
an 0xA4 replay of `slow_osc` tracked with a 4-7° lag and the same 2 Hz
velocity cycle the MIT frame shows. This is the sweep tool for that: write a
few gains, replay, read the trace, repeat.

Writes go to RAM (0x31) unless ``--persist`` is given (0x32, ROM), so an
experiment is undone by a power cycle and a good set is committed
deliberately. Every write is read back (0x30) and printed.

Usage:
    uv run python scripts/fw_gains.py --r --id 1                       # read
    uv run python scripts/fw_gains.py --r --id 1 speed_kp=0.1 speed_ki=0.001
    uv run python scripts/fw_gains.py --r --id 1 position_kp=0.05 --persist

Gains: current_kp current_ki speed_kp speed_ki position_kp position_ki position_kd
"""

from __future__ import annotations

import argparse
import asyncio
import struct

from almond_axol.cli.motor import add_side_and_channel_arguments, resolve_channel
from almond_axol.motor.bus import CanBus
from almond_axol.motor.motor import make_driver
from almond_axol.motor.myactuator import _MA_PID_IDX, MyActuatorMotor

_READ = 0x30
_WRITE_RAM = 0x31
_WRITE_ROM = 0x32
_READ_ACCEL = 0x42
_WRITE_ACCEL = 0x43  # RAM + ROM


async def _read_accel(driver: MyActuatorMotor) -> tuple[int, int]:
    out = []
    for kind in (0x00, 0x01):
        resp = await driver._request(bytes([_READ_ACCEL, kind, 0, 0, 0, 0, 0, 0]))
        out.append(int(struct.unpack_from("<i", resp, 4)[0]))
    return out[0], out[1]


async def _write_accel(driver: MyActuatorMotor, dps_s2: int) -> tuple[int, int]:
    """Position-planner accel and decel, written raw (0 = direct tracking of a
    streamed 0xA4 target, which wire_mode a4 needs). Persists in ROM."""
    for kind in (0x00, 0x01):
        await driver._request(
            bytes([_WRITE_ACCEL, kind, 0, 0]) + struct.pack("<I", int(dps_s2))
        )
        await asyncio.sleep(0.3)
    return await _read_accel(driver)


async def _read(driver: MyActuatorMotor, name: str) -> float:
    resp = await driver._request(bytes([_READ, _MA_PID_IDX[name], 0, 0, 0, 0, 0, 0]))
    return float(struct.unpack_from("<f", resp, 4)[0])


async def _write(
    driver: MyActuatorMotor, name: str, value: float, persist: bool
) -> None:
    cmd = _WRITE_ROM if persist else _WRITE_RAM
    await driver._request(
        bytes([cmd, _MA_PID_IDX[name], 0, 0]) + struct.pack("<f", float(value))
    )
    await asyncio.sleep(0.3 if persist else 0.05)


async def _run(args: argparse.Namespace) -> None:
    writes: dict[str, float] = {}
    for spec in args.gains:
        name, _, raw = spec.partition("=")
        if name not in _MA_PID_IDX or not raw:
            raise SystemExit(
                f"bad gain {spec!r}; want NAME=VALUE with NAME in {', '.join(_MA_PID_IDX)}"
            )
        writes[name] = float(raw)
    async with CanBus(resolve_channel(args)) as bus:
        driver = make_driver(bus, args.id, kt=1.0)
        if not isinstance(driver, MyActuatorMotor):
            raise SystemExit(f"motor {args.id:#04x} is not a MyActuator")
        before = {n: await _read(driver, n) for n in _MA_PID_IDX}
        acc = await _read_accel(driver)
        print(f"motor {args.id:#04x} gains now:")
        for n, v in before.items():
            print(f"  {n:12s} {v:.6g}")
        print(
            f"  planner accel/decel {acc[0]}/{acc[1]} dps/s"
            + (
                "  (direct tracking)"
                if acc[0] == 0
                else "  (profiled — a streamed 0xA4 will not follow)"
            )
        )
        if args.accel is not None and acc != (args.accel, args.accel):
            got = await _write_accel(driver, args.accel)
            print(
                f"  planner accel/decel {acc[0]}/{acc[1]} -> {got[0]}/{got[1]} dps/s (ROM)"
            )
            if got[0] == 0:
                print(
                    "  ! this joint now executes a stored 0xA4 target at its speed cap the moment it wakes — set it back (--accel 5000) when done with wire_mode a4"
                )
        if not writes:
            return
        for n, v in writes.items():
            await _write(driver, n, v, args.persist)
        after = {n: await _read(driver, n) for n in _MA_PID_IDX}
        print(
            f"\nafter writing to {'ROM (persistent)' if args.persist else 'RAM (until power cycle)'}:"
        )
        for n in writes:
            flag = (
                ""
                if abs(after[n] - writes[n]) < 1e-6 * max(1.0, abs(writes[n]))
                else "   ! not accepted"
            )
            print(f"  {n:12s} {before[n]:.6g} -> {after[n]:.6g}{flag}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_side_and_channel_arguments(p)
    p.add_argument("--id", type=lambda x: int(x, 0), required=True, help="Motor CAN ID")
    p.add_argument("gains", nargs="*", metavar="NAME=VALUE", help="Gains to write")
    p.add_argument(
        "--accel",
        type=int,
        default=None,
        help="Also set the position-planner accel/decel (dps/s, ROM): 0 for the direct "
        "tracking wire_mode a4 needs, 5000 to put a joint back",
    )
    p.add_argument(
        "--persist",
        action="store_true",
        help="Write to ROM (0x32) instead of RAM (0x31); survives a power cycle",
    )
    asyncio.run(_run(p.parse_args()))


if __name__ == "__main__":
    main()
