# axol-rt

Realtime CAN control core for the Almond Axol arms, in Rust. This is the
"fast half" of the hybrid teleop architecture: Python keeps VR ingest, IK
(JAX), MuJoCo gravity/inertia, and the web/serve stack; Rust owns the CAN
buses and runs the per-tick control loop with hard, GIL-free timing.

## Status

Working today (verified on the robot, both arm buses):

- Raw SocketCAN via `libc` — no wrapper crates, direct control over
  timeouts/filters for the realtime loop.
- Full wire protocol for both vendors, ported bit-for-bit from
  `almond_axol/motor/{myactuator,damiao}.py`:
  - MyActuator RMD (IDs 0x01-0x05, request `0x140+id` / reply `0x240+id`):
    version, model, multi-turn angle, status1/2 reads; MIT frame encode.
  - Damiao (IDs 0x06-0x08, register access on `0x7FF`, feedback on
    `0x10+id`): register read, feedback request (`0xCC`), feedback decode,
    MIT frame encode.
  - `mit_encode` is validated against a Python-driver reference vector in
    `cargo test`.
- `axol-rt scan` — read-only identity/state sweep of all 16 motors.
- `axol-rt bench` — paced full-bus telemetry loop (read-only), both buses
  in parallel threads.

Measured on the robot (2026-08-27, arms at rest, serve stopped):

| rate | tick lateness p99 | full 8-motor cycle p99 | replies lost |
|------|-------------------|------------------------|--------------|
| 120 Hz | 0.000 ms | 2.70 ms | 0 / 4800 |
| 240 Hz | 0.014 ms | 2.64 ms | 0 / 9600 |
| 500 Hz | (saturated) | 2.84 ms | 18% |

For comparison, the Python control loop under teleop measured 30-57% of
ticks late. 500 Hz full-bus telemetry is a *wire* limit, not a host limit:
8 request/reply pairs = 16 frames ≈ 2.1 ms at 1 Mbps, so the bus caps a
full round-robin at ~430 Hz. 240 Hz leaves ~45% bus headroom.

## Safety

`scan` and `bench` are strictly read-only: they send version/status/angle
queries and Damiao feedback requests only — never enable, mode, or motion
commands. Safe to run against a powered robot at rest. Motion commands do
not exist in this binary yet and will only be added together with the
enable/watchdog plumbing.

## Design: the hybrid split

Per-tick work in `AxolArm.motion_control` divides cleanly by how fast its
inputs change:

**Stays in Python** (slow, heavy deps):
- VR ingest, filters, IK (JAX, ~90 Hz), serve/UI.
- MuJoCo gravity + reflected inertia `J(q)` — pose changes at human speed,
  so evaluating at IK rate and interpolating is lossless.

**Moves to Rust** (per-tick, latency-critical):
- Target interpolation between IK results (the "interpolate between
  jumps" the trapezoid filter approximates today).
- Commanded/measured differentiators (first-order LP, 20 / 80 rad/s poles).
- Host-damping band-pass (Chamberlin SVF) with pose-tracked centres.
- Friction feedforward (tanh model) and inertia feedforward.
- MIT encode + bus I/O, feedback decode, joint offsets/limits, the
  max-step safety gate, and the e-stop/ENOBUFS stall handling that
  `motor/bus.py` implements today.

**IPC** (Python -> Rust, ~90-120 Hz; Rust -> Python telemetry): a
seqlock'd shared-memory segment (or SOCK_SEQPACKET Unix socket as the
simple first cut) carrying per arm: 8 joint targets, 7 gravity torques,
7 host-damping scales, 7 inertia-FF scales, gains, and a monotonic
sequence number + timestamp. Rust extrapolates/interpolates between
updates and holds position (then ramps to rest) if updates stop.

## Build / run

```sh
cargo build --release           # needs no cross-compile: built on the Jetson
./target/release/axol-rt scan   # identity + state of every motor
./target/release/axol-rt bench --hz 240 --secs 5
./target/release/axol-rt bench --hz 240 --serial   # per-motor round-trips
```

Default interfaces are `can_alm_axol_l` and `can_alm_axol_r`; pass others
as positional args.

## Roadmap

1. **Command path** (next): enable/disable + MIT impedance command loop
   behind an explicit arm/disarm handshake; watchdog that ramps to rest on
   IPC silence; port the TX-stall (e-stop) purge logic.
2. **IPC**: shared-memory target mailbox + telemetry ring read by Python.
3. **Control math**: differentiators, band-pass, friction/inertia FF —
   each unit-tested against the Python implementation on recorded streams.
4. **Integration**: `axol teleop --rt` flag that launches `axol-rt` as the
   bus owner and switches `motion_control` to publish targets instead of
   talking CAN.
