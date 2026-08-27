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
| TX-stall detection (unpowered bus, e-stop condition) | 348 frames queued, ENOBUFS drops for 1 s, stall declared at 1.75 s (`cargo test stall_detection_live -- --ignored`) |

For comparison, the Python control loop under teleop measured 30-57% of
ticks late. 500 Hz full-bus telemetry is a *wire* limit, not a host limit:
8 request/reply pairs = 16 frames ≈ 2.1 ms at 1 Mbps, so the bus caps a
full round-robin at ~430 Hz. 240 Hz leaves ~45% bus headroom.

## The hybrid split (as built)

Python keeps the **slow model math**: `AxolArm.motion_control` still runs
joint limits, the max-step gate, MuJoCo gravity, and the pose *scheduling*
of the fast terms (pose-scaled damping gain and inertia gain, pose-tracked
band-pass centre). The teleop pipeline's target shaping — the pose
low-pass, the IK-output EMA, and the Python trapezoid with its engage
velocity ramp and output guard — also stays: those filters condition the
*target stream* and live with IK. In rt mode a command sink hands
per-joint 9-float tuples `(p_des, mode, kp, kd, t_ff, kd_host, damp_w0,
damp_q, j_eff)` to `almond_axol.rt.RtAxol`, which ships them to this core
(~120 Hz) instead of sending CAN from Python. `t_ff` is gravity only in
tracked mode; `mode 0` (gravity comp) is a tracker-bypassing passthrough
with `v_des = 0`.

The core owns the wire and the **fast physics**, all per tick from its
own trajectory and feedback states:

- Hard 240 Hz pacing (spin-assisted `sleep_until`, zero-late in practice).
- **In-core target tracker**: the golden-ported `TrapezoidalFilter`
  (`filter::Trapezoid`) chases the latest streamed target under the
  config velocity/acceleration limits (teleop caps × 1.5 headroom),
  replacing linear segment interpolation. Its `(pos, vel, accel)` states
  are the wire command — velocity feedforward is continuous instead of
  frozen between targets, and target-rate wobble is absorbed by the
  tracker's own dynamics.
- **In-core friction + inertia feedforwards**: the tanh friction model
  (per-joint params ride the config) on the tracker velocity, and the
  streamed pose-scaled `j_eff` on the tracker acceleration — coherent
  with the executed trajectory, not with Python's 120 Hz pre-tracker view
  of it (differentiating the raw target for the inertia term is exactly
  the noise the trapezoid exists to remove).
- **In-core host damping**: band-passed `(v_des − v_meas)` scaled by the
  streamed pose-scheduled gain, computed every tick from same-tick
  feedback, with `v_des` the tracker velocity. The filter chain
  (`src/filter.rs`) is ported from `almond_axol.robot.control` and
  golden-tested against it. Damping is a phase race — computing the
  torque in Python put it ~14 ms behind the velocity it acts on (120 Hz
  sample + socket + interpolation), which pushed the shoulder burst band
  (4-9 Hz) past 90° of loop phase, where a damper *pumps* the mode: that
  was the violent rt-teleop shaking of 2026-08-27. In-core, the torque
  lands within one tick, and damping stays live through every core-owned
  hold (watchdog, orphaned client) — frozen-`t_ff` holds used to leave
  the shoulders ringing on firmware kd alone. `cargo test` includes a
  dissipated-power comparison of the two chains.
- Watchdog: targets stop arriving → the tracker converges on the last
  target and the arms hold there, damping active (matching what the
  firmware itself does if a host dies mid-command, plus the damper).
  Client disconnect while armed → hold 10 s, then disable and exit, so an
  orphaned core never stays energized.
- Deviation abort: any joint more than `abort_deg` (default 25°) from its
  *commanded* (tracker-output) position disables both buses.
- Max-step gate on incoming targets (corruption defense; Python's gate is
  the real per-command limit — and whatever gets through, the tracker's
  limits bound what the wire can see).
- Any protocol error (e.g. a version-skewed target size) stops the bus
  threads and disables the motors before the process exits — never an
  energized orphan.
- TX-stall (e-stop) handling, ported from `motor/bus.py`: `ENOBUFS`
  persisting >1 s across sends means no node is ACKing — the e-stop cut
  motor power. The core stops commanding, purges the poisoned TX queue
  (bring-up script or `ip link` flap; direct when root, `sudo -n`
  otherwise) so up to `txqueuelen` stale MIT commands can't replay and
  snap the arm when power returns, and takes the session down as a clear
  fault — re-powered motors come back disabled and need a fresh bring-up
  anyway. Transient single-frame `ENOBUFS` (host-side congestion) just
  drops that frame, like the Python path.

The gripper rides the same target packets in slot 7 as a POSITION_FORCE
command (motor-frame target, speed limit, torque limit). It is exempt
from the max-step gate and the deviation abort (stalling against an
object is its job), is never commanded until the first target arrives,
and its bring-up — enable, open-stop calibration or attach/restore of a
holding jaw — stays in Python, run on the quiet bus before the core arms.

Measured feedback flows back to Python as telemetry: once per bus per
tick the core ships an `F` packet (per-slot position, velocity, torque,
and frame age) over the socket, and Python fills its `Motor` caches from
it — positions and torques stay fresh for `get_positions`, recording, and
the contact watchdog, with receive timestamps reconstructed to within
socket transit. Python's own CAN receive path is muted at the kernel
(zero-length CAN_RAW_FILTER) while the core is armed, so the ~7,700
frames/s SocketCAN would otherwise broadcast into python-can cost Python
nothing; ~480 tiny packet decodes/s replace them.

Bring-up is split so Python's calibration stays authoritative: the core
resets the arm motors first (`prep`, gripper untouched), then Python
resolves joint offsets and MyActuator decode ranges against the
post-reset wrap state and brings up the gripper, then the core enables
and holds (`arm`). The socket protocol lives in `src/serve.rs` (Rust) and
`almond_axol/rt/link.py` (Python) — length-prefixed frames, text control
messages, packed-binary targets.

Guarded return works exactly as in classic mode: `torque_residuals` and
`reset_command_state` are cache/state-only, and `gravity_compensate`
streams its tuples through the same command sink — the contact watchdog,
the limp contact hold, and the replanned reset all run against the core.

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

Nothing pending — the split described above is fully built: the core owns
the wire in both directions (commands out, telemetry packets back), and
Python is CAN-silent from arm to disarm.
