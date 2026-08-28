# axol-rt

Realtime CAN control core for the Almond Axol arms, in Rust. This is the
"fast half" of the hybrid teleop architecture: Python keeps VR ingest, IK
(JAX), MuJoCo gravity/inertia, and the web/serve stack; Rust owns the CAN
buses and runs the per-tick control loop with hard, GIL-free timing.

`axol teleop` and every other production arm-motion flow use it
unconditionally: `gravity-comp`, `waypoints`, `tune.motion`,
`tune.repeatability`, and the LeRobot-based `collect-data`, `run-policy`, and
`replay-dataset` commands. Once armed, the core is the only CAN consumer.
Bench/calibration flows that need direct register access (`tune.pid`,
`tune.friction`, `motor.*`) remain maintenance tools outside the production
control loop. Their timed PID/friction experiments nevertheless execute in
Rust through the proxy's experiment engine; Python only plans gravity/reference
samples and fits the returned measurements.

## Status

Working today:

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
  socket. This is what `axol teleop` runs.
- `axol-rt proxy` — maintenance CAN plus a precisely paced tuning experiment
  engine and Rust-side rolling timing aggregation. Passive dashboard clients
  receive 30 Hz state frames and 10 Hz timing summaries while Rust observes
  every on-wire frame.
- `axol-rt jelly` — owns Jelly's four wheel motors: enable/disable, 50 Hz
  vector slew and x-drive mix, command watchdog, gyro heading hold, and the
  velocity/impedance park state machine.

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
*target stream* and live with IK. A command sink hands
per-joint 9-float tuples `(p_des, mode, kp, kd, t_ff, kd_host, damp_w0,
damp_q, j_eff)` to `almond_axol.rt.RtAxol`, which ships them to this core
(~120 Hz) instead of sending CAN from Python. `t_ff` is gravity only in
tracked mode; `mode 0` (gravity comp) is a tracker-bypassing passthrough
with `v_des = 0`.

The core owns the wire and the **fast physics**, all per tick from its
own trajectory and feedback states:

- Hard 240 Hz pacing (spin-assisted `sleep_until`). On 8+ core robot hosts,
  the left and right CAN threads are pinned to dedicated CPUs, separate from
  Python control, IK, camera relay, and dataset recording. They additionally
  request SCHED_FIFO priority 20 when permitted by the launcher; CPU isolation
  remains active with an explicit warning for unprivileged development runs.
- **In-core target tracker**: the golden-ported `TrapezoidalFilter`
  (`filter::Trapezoid`) chases the latest streamed target under the
  config velocity/acceleration limits (teleop caps × 1.5 headroom),
  replacing linear segment interpolation. Its position is the wire
  trajectory (the low-pass derivative below supplies wire velocity), and
  target-rate wobble is absorbed by the tracker's own dynamics.
- **In-core friction + inertia feedforwards**: the tanh friction model
  (per-joint params ride the config) and streamed pose-scaled `j_eff` use
  the classic Python 20 rad/s command-derivative chain, now driven by the
  executed tracker position. The low-pass velocity plus second low-pass
  acceleration derivative are important: the raw 240 Hz tracker
  acceleration reacts to each new 120 Hz target differently from the
  repeated-target tick, which previously produced an alternating inertia
  torque and felt vibration during motion.
- **In-core host damping**: band-passed `(v_des − v_meas)` scaled by the
  streamed pose-scheduled gain, computed every tick from the latest
  feedback, with `v_des` the fast low-pass derivative of tracker position.
  The counter-torque reaches the wire within one 240 Hz tick. The filter chain
  (`src/filter.rs`) is ported from `almond_axol.robot.control` and
  golden-tested against it. Damping is a phase race — computing the
  torque in Python put it ~14 ms behind the velocity it acts on (120 Hz
  sample + socket + interpolation), which pushed the shoulder burst band
  (4-9 Hz) past 90° of loop phase, where a damper *pumps* the mode: that
  was the violent rt-teleop shaking of 2026-08-27. In-core, the torque
  lands within one tick, and damping stays live through every core-owned
  hold (watchdog, orphaned client) — frozen-`t_ff` holds used to leave
  the shoulders ringing on firmware kd alone. `cargo test` includes a
  dissipated-power comparison of the two chains. The shared robot config
  gives shoulder-1 on both arms a Q=3 band: it keeps unity gain at the
  intended ~3.2 Hz mode while rejecting the measured 12.5-13.6 Hz
  mast/forearm structural mode. Every production flow consumes the same
  value, and explicit calibration or CLI Q values remain authoritative.
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
socket transit. The Rust maintenance proxy exits before the realtime core
arms, so Python has no CAN socket and the core is the only command/feedback
owner during control. Roughly 480 tiny telemetry packet decodes/s replace the
~7,700 frame/s dispatch load of the removed Python control path.

Bring-up is split so Python's calibration logic stays authoritative while
Rust owns every CAN syscall: the core resets the arm motors first (`prep`,
gripper untouched), then a Rust maintenance proxy carries offset/range reads
and gripper calibration frames, the proxy exits, and the realtime core enables
and holds (`arm`). The socket protocols live in `src/serve.rs` / `src/proxy.rs`
(Rust) and the Python link classes — length-prefixed messages with packed
targets or CAN frames.

Guarded return stays on the same core: `torque_residuals` and
`reset_command_state` are cache/state-only, and `gravity_compensate`
streams its tuples through the same command sink — the contact watchdog,
the limp contact hold, and the replanned reset all run against the core.

### Control-term tracing

`axol teleop --teleop.record NAME` automatically gates this trace to the
latest engaged segment and compacts both arms into `NAME_rt.npz` on teardown.
`axol collect-data` always assigns a unique prefix when none was supplied and
records tracking plus guarded-reset motion after PyRoKi is ready, so a
collection-only timing or damping fault is preserved automatically.
For low-level runs outside teleop, set `AXOL_RT_TRACE` to a path prefix to
capture one raw CSV per arm without doing file I/O on the realtime threads:

```bash
AXOL_RT_TRACE=/tmp/axol-run axol teleop
# writes /tmp/axol-run-left.csv and /tmp/axol-run-right.csv
```

Each 240 Hz joint row includes the streamed target, wire position/velocity,
measured position/velocity/torque, filter states, and separate gravity,
friction, inertia, and host-damping torque contributions. The bus threads
enqueue fixed-size rows into bounded channels; background threads format and
write them, and the regular five-second status line reports any trace drops.

## Safety

`scan` and `bench` are strictly read-only — safe against a powered robot
at rest. `hold` requires `--yes` to actuate. `serve` only actuates after
the explicit config/prep/arm handshake, and every exit path (disarm,
fault, signal, client loss) runs the disable sequence. `proxy` is the sole
frame transport for maintenance, tuning, firmware, and arm diagnostics;
`jelly` owns Jelly's wheel bus, while the proxy carries its lift bus. Both use
the realtime core's persistent-TX-stall detection and queue purge, aborting
instead of allowing stale motion frames to replay after an e-stop.

## Build / run

`axol provision` (or `axol rt.install` standalone) builds and installs the
binary automatically — rustup toolchain included, and on uv-tool installs
(no repo checkout) it fetches these sources at the installed package's
exact ref. Manual path:

```sh
cargo build --release           # needs no cross-compile: built on the Jetson
./target/release/axol-rt scan   # identity + state of every motor
./target/release/axol-rt bench --hz 240 --secs 5
uv run python tools/gen_hold_params.py /tmp/hold.txt
./target/release/axol-rt hold --params /tmp/hold.txt --secs 5 --yes
uv run python tools/rt_smoke.py --secs 8   # end-to-end serve smoke test
uv run axol teleop                         # the real thing
```

Default interfaces are `can_alm_axol_l` and `can_alm_axol_r`; pass others
as positional args (`scan` / `bench`). The teleop path finds the binary
via `AXOL_RT_BIN`, `PATH`, or this crate's `target/release/`.

## Roadmap

Nothing pending — the split described above is fully built: Rust owns the
wire in both directions for production control and every maintenance/tuning
utility. Python is orchestration, model math, fitting, and UI only.
