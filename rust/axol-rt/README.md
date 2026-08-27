# axol-rt

Realtime CAN control core for the Almond Axol arms, in Rust. This is the
"fast half" of the hybrid teleop architecture: Python keeps VR ingest, IK
(JAX), MuJoCo gravity/inertia, and the web/serve stack; Rust owns the CAN
buses and runs the per-tick control loop with hard, GIL-free timing.

Run it via `axol teleop --rt` — see "The hybrid split" below.

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
- `axol-rt hold` — enable + MIT-hold the current pose + disable, gains and
  gravity feedforward from `tools/gen_hold_params.py`.
- `axol-rt serve` — the realtime core: owns both buses, paces a 240 Hz MIT
  stream, and plays impedance targets streamed from Python over a Unix
  socket. This is what `axol teleop --rt` runs.

Measured on the robot (2026-08-27):

| test | result |
|------|--------|
| bench 240 Hz telemetry | tick lateness p99 0.014 ms, 0/9600 replies lost |
| serve, teleop headless | 30000 ticks @ 240 Hz, 0.03% late, watchdog + disarm clean |
| serve, wrist_3 ±8° sinusoid + gripper cycle | worst tracking error 0.42° (moving), ≤0.11° (holding); gripper swept 1.00→0.66→1.00 as commanded |
| serve, Python killed while armed | core held 10 s, disabled everything, exited clean |

For comparison, the Python control loop under teleop measured 30-57% of
ticks late. 500 Hz full-bus telemetry is a *wire* limit, not a host limit:
8 request/reply pairs = 16 frames ≈ 2.1 ms at 1 Mbps, so the bus caps a
full round-robin at ~430 Hz. 240 Hz leaves ~45% bus headroom.

## The hybrid split (as built)

Python keeps **all** of the control math: `AxolArm.motion_control` runs
unchanged — gravity + reflected inertia (MuJoCo), pose-scheduled host
damping, friction feedforward, joint limits, the max-step gate. In rt mode
a command sink hands its fully computed per-joint MIT tuples
`(p_des, v_des, kp, kd, t_ff)` to `almond_axol.rt.RtAxol`, which ships
them to this core (~120 Hz) instead of sending CAN from Python.

The core owns the wire:

- Hard 240 Hz pacing (spin-assisted `sleep_until`, zero-late in practice).
- Linear interpolation of `p_des` / `t_ff` between successive targets over
  one estimated sender period — no steps on the bus when Python's rate
  wobbles (this costs one sender period, ~8 ms, of added latency).
- Watchdog: targets stop arriving → the in-flight segment completes and
  the arms hold there (matching what the firmware itself does if a host
  dies mid-command). Client disconnect while armed → hold 10 s, then
  disable and exit, so an orphaned core never stays energized.
- Deviation abort: any joint more than `abort_deg` (default 25°) from its
  played target disables both buses.
- Max-step gate on incoming targets (corruption defense; Python's gate is
  the real per-command limit).

The gripper rides the same target packets in slot 7 as a POSITION_FORCE
command (motor-frame target, speed limit, torque limit). It is exempt
from the max-step gate and the deviation abort (stalling against an
object is its job), is never commanded until the first target arrives,
and its bring-up — enable, open-stop calibration or attach/restore of a
holding jaw — stays in Python, run on the quiet bus before the core arms.

Measured feedback flows back to Python for free: SocketCAN broadcasts
every frame to every open socket, so Python's passive `Motor` caches keep
filling from the core's own MIT and POSITION_FORCE replies —
`motion_control`'s measured-velocity damping path works unchanged, with
real CAN timestamps.

Bring-up is split so Python's calibration stays authoritative: the core
resets the arm motors first (`prep`, gripper untouched), then Python
resolves joint offsets and MyActuator decode ranges against the
post-reset wrap state and brings up the gripper, then the core enables
and holds (`arm`). The socket protocol lives in `src/serve.rs` (Rust) and
`almond_axol/rt/link.py` (Python) — length-prefixed frames, text control
messages, packed-binary targets.

Not core-driven yet: the guarded return-to-rest paths (they play through
plain position streaming, as in sim).

## Safety

`scan` and `bench` are strictly read-only — safe against a powered robot
at rest. `hold` requires `--yes` to actuate. `serve` only actuates after
the explicit config/prep/arm handshake, and every exit path (disarm,
fault, signal, client loss) runs the disable sequence.

## Build / run

```sh
cargo build --release           # needs no cross-compile: built on the Jetson
./target/release/axol-rt scan   # identity + state of every motor
./target/release/axol-rt bench --hz 240 --secs 5
uv run python tools/gen_hold_params.py /tmp/hold.txt
./target/release/axol-rt hold --params /tmp/hold.txt --secs 5 --yes
uv run python tools/rt_smoke.py --secs 8   # end-to-end serve smoke test
uv run axol teleop --rt                    # the real thing
```

Default interfaces are `can_alm_axol_l` and `can_alm_axol_r`; pass others
as positional args (`scan` / `bench`). The teleop path finds the binary
via `AXOL_RT_BIN`, `PATH`, or this crate's `target/release/`.

## Roadmap

1. Port the TX-stall (e-stop) purge logic from `motor/bus.py`.
2. Telemetry packets (tick stats already stream as log lines; positions
   currently ride the passive CAN broadcast).
3. Optionally move the per-tick control math (differentiators, band-pass
   damping) into the core so damping acts on 240 Hz-fresh measurements.
