# Axol

<img src="assets/axol.png" width="400" alt="Axol dual-arm robot" />

Command-line interface and Python SDK for the Almond Axol dual-arm robot. CLI invoked as `axol <command> [flags]`.

**New here?** See the [Quickstart](QUICKSTART.md) to go from installation to a live teleoperation session.

## Requirements

- **Linux**
- **Python 3.13+**
- **(Optional) NVIDIA Jetson** If ZED cameras are used.

## Table of Contents

- [Installation](#installation)
- [CAN Bus Setup](#can-bus-setup)
- [Motor Commands](#motor-commands)
- [Teleoperation](#teleoperation)
- [Data Collection](#data-collection)
- [Policy Execution](#policy-execution)
- [ZED Camera](#zed-camera)
- [Tuning](#tuning)
- [Python SDK](#python-sdk)
  - [Modules](#modules)
  - [Core Concepts](#core-concepts)
  - [almond_axol.robot](#almond_axolrobot)
  - [almond_axol.kinematics](#almond_axolkinematics)
  - [almond_axol.teleop](#almond_axolteleop)
  - [almond_axol.vr](#almond_axolvr)
  - [almond_axol.zed](#almond_axolzed)
  - [almond_axol.motor](#almond_axolmotor)
  - [almond_axol.lerobot](#almond_axollerobot)

---

## Installation

Install the package using `uv`. `pyroki` and `lerobot` are sourced from Git and are resolved automatically:

```bash
uv sync
```

Install optional dependency groups as needed:

| Extra | Contents | When to use |
|---|---|---|
| `lerobot` | LeRobot (from GitHub) | `collect-data`, `run-policy` |
| `sim` | viser | `teleop --robot sim` |
| `cuda` | JAX with CUDA 13 support | GPU-accelerated JAX (IK solver used by `teleop`); note that CPU is usually faster for the JAX IK solver |
| `dev` | OpenCV (headless) | Development / debugging |

```bash
uv sync --extra lerobot --extra sim        # teleoperation + data collection
uv sync --extra lerobot --extra cuda       # policy execution on GPU
uv sync --extra lerobot --extra sim --extra cuda   # everything
```

The ZED Python bindings (`pyzed`) are not on PyPI and must be installed separately after the ZED SDK is installed:

```bash
axol zed.install
```

Before using any motor or robot commands, initialize the CAN hardware:

```bash
axol can.setup
```

---

## CAN Bus Setup

### `can.setup`

One-time setup for the Almond Axol Hub (dual-channel USB CAN adapter). Writes persistent udev rules, assigns fixed interface names, registers a startup script in the root crontab, and brings up the interfaces immediately.

- Left arm → `can_alm_axol_l`
- Right arm → `can_alm_axol_r`

```bash
axol can.setup
```

> `sudo` will be invoked automatically where required.

### `can.enable`

Re-runs the CAN startup script to bring interfaces up after plugging in the Axol without a system restart. (`can.setup` registers a `@reboot` cron hook, so this is only needed when the Axol Hub is re-plugged mid-session.) Requires `can.setup` to have been run first.

```bash
axol can.enable
```

---

## Motor Commands

All motor commands accept a mutually exclusive `--l` / `--r` flag to select the left or right arm, and `--id` for the motor's CAN address (hex or decimal). `--type` can be `myactuator` or `damiao` and is inferred from the ID if omitted.

### `motor.info`

Reads and prints a full status snapshot from a motor: status/error code, control mode (Damiao only), position, velocity, torque, temperature, and voltage.

```bash
axol motor.info --l --id 0x01
axol motor.info --r --id 6 --type damiao
```

### `motor.set-can-id`

Changes a motor's CAN ID and persists it to flash. The motor must be the only device on the bus or its current ID must be known. `--type` is required here.

| Flag | Description |
|---|---|
| `--current-id ID` | Current CAN ID |
| `--new-id ID` | New CAN ID to assign |
| `--type {myactuator,damiao}` | Motor type (required) |

```bash
axol motor.set-can-id --l --current-id 0x01 --new-id 0x03 --type myactuator
```

### `motor.set-zero-pos`

Sets the motor's zero-position reference to its current mechanical position (persisted to flash). Damiao motors require a power cycle afterward.

```bash
axol motor.set-zero-pos --l --id 0x01
```

---

## Teleoperation

### `teleop`

Launches a VR teleoperation session. When started, the hostname (`.local`) and local IP address are printed — enter either of these in the VR app at [axol.almond.bot](https://axol.almond.bot) to connect.

> **Before opening the VR app**, accept the self-signed HTTPS certificate in the VR browser by navigating to `https://<hostname>.local:8000` or `https://<local-ip>:8000` and proceeding past the security warning.

> **Network tip:** If VR tracking feels jittery or packets arrive in bursts, configure the following on your router/access point:
> - **DTIM interval** → `1`
> - **Beacon interval** → `100` ms
> - **WMM APSD (U-APSD)** → disabled
>
> These settings prevent the AP from buffering packets between beacon intervals, which causes intermittent delivery delays that are especially noticeable for latency-sensitive VR traffic.

| Flag | Description |
|---|---|
| `--robot {axol,sim}` | `axol` uses real hardware; `sim` uses the software visualizer (required) |
| `--no-left` | Disable the left arm |
| `--no-right` | Disable the right arm |
| `--gripper-torque-limit FLOAT` | Max gripper torque in POSITION_FORCE mode in Nm (default: 1.0) |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Default: `INFO` |

```bash
axol teleop --robot axol
axol teleop --robot sim --no-right
```

---

## Data Collection

### `collect-data`

Records teleoperation episodes using VR controller inputs and three ZED cameras. Saves to a [LeRobot](https://github.com/huggingface/lerobot)-format dataset. Loops until `Ctrl+C`.

| Flag | Description |
|---|---|
| `--repo-id <user>/<dataset>` | HuggingFace dataset repo ID (required) |
| `--task TEXT` | Natural language task description (required) |
| `--fps INT` | Dataset recording frame rate — camera frames captured at this rate (default: 60) |
| `--teleop-hz INT` | Motor command rate in Hz; decoupled from `--fps` for smooth control (default: 120) |
| `--root PATH` | Local dataset root (default: `$HF_LEROBOT_HOME`) |
| `--push-to-hub` | Push to HuggingFace Hub when done |
| `--zed-host IP` | IP address of the ZED camera streamer (default: `192.168.10.1`) |
| `--zed-iface IFACE` | Network interface to configure for the ZED link (e.g. `eth0`); assigns `192.168.10.2/24`, requires `sudo` |
| `--gripper-torque-limit FLOAT` | Max gripper torque in POSITION_FORCE mode in Nm (default: 1.0) |
| `--rerun-ip IP` | IP of a Rerun viewer on your local machine for live visualization |
| `--rerun-port INT` | Rerun viewer port (default: 9876); only used when `--rerun-ip` is set |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Default: `INFO` |

```bash
axol collect-data --repo-id myorg/pick-place --task "Pick the red cube and place it in the bin"
axol collect-data --repo-id myorg/pick-place --task "Pick the red cube" --fps 30 --zed-iface eth0
```

**VR controller events:**

| Event | Action |
|---|---|
| `START_RECORDING` | Begin capturing frames |
| `TERMINATE_EPISODE` | Save the episode; headset enters `Saving` state until write completes |
| `RERECORD_EPISODE` | Discard and retry |

After each episode the robot automatically returns to its rest pose before the next take begins. If an existing dataset is found at `--root`, collection resumes from where it left off.

---

## Policy Execution

### `run-policy`

Runs a trained policy autonomously on the robot using three ZED cameras. Between episodes, prompts the operator via stdin to save (`Enter`), re-record (`r`), or quit (`q`).

| Flag | Description |
|---|---|
| `--policy PATH_OR_REPO` | Local checkpoint path or HuggingFace repo ID (required) |
| `--task TEXT` | Natural language task description (required) |
| `--episode-time-s INT` | Max duration per episode in seconds (default: 30) |
| `--fps INT` | Control loop frame rate (default: 60) |
| `--repo-id <user>/<dataset>` | Optional dataset repo ID to save rollouts |
| `--root PATH` | Local dataset root (default: `$HF_LEROBOT_HOME`) |
| `--push-to-hub` | Push rollout dataset to HuggingFace Hub when done |
| `--zed-host IP` | IP address of the ZED camera streamer (default: `192.168.10.1`) |
| `--zed-iface IFACE` | Network interface to configure for the ZED link (e.g. `eth0`); assigns `192.168.10.2/24`, requires `sudo` |
| `--gripper-torque-limit FLOAT` | Max gripper torque in POSITION_FORCE mode in Nm (default: 1.0) |
| `--rerun-ip IP` | IP of a Rerun viewer on your local machine for live visualization |
| `--rerun-port INT` | Rerun viewer port (default: 9876); only used when `--rerun-ip` is set |
| `--device STR` | PyTorch device for inference (default: `cuda`) |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Default: `INFO` |

```bash
axol run-policy --policy myorg/pick-place-policy --task "Pick the red cube"
axol run-policy --policy ./checkpoints/epoch_100 --task "Stack blocks" --episode-time-s 20 --device cpu
```

---

## ZED Camera

### `zed.stream`

Streams ZED-X One cameras over the local network using HEVC encoding. At least one camera must be specified. Streams until `Ctrl+C`. Sender IP is `192.168.10.1/24`.

| Flag | Description |
|---|---|
| `--overhead SERIAL` | Serial number of the overhead camera |
| `--left-arm SERIAL` | Serial number of the left-arm camera |
| `--right-arm SERIAL` | Serial number of the right-arm camera |
| `--resolution {HD1080,HD1200,SVGA}` | Default: `HD1080` |
| `--fps FPS` | Default: 60 |
| `--bitrate KBPS` | HEVC bitrate in kbit/s (default: 8000) |
| `--setup-ip IFACE` | Assign sender IP to a network interface before streaming (e.g. `eth0`); requires `sudo` |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Default: `INFO` |

```bash
axol zed.stream --overhead 12345678 --left-arm 23456789 --right-arm 34567890
axol zed.stream --overhead 12345678 --resolution SVGA --fps 30 --bitrate 4000
```

### `zed.install`

Downloads and installs the `pyzed` Python wheel matching the installed ZED SDK version. Caches the wheel in `~/.almond/wheels/`.

```bash
axol zed.install
```

---

## Tuning

### `tune.pid`

Tunes `Kp`/`Kd` gains for a single joint at ~100 Hz using sinusoidal tracking or step-response, and prints error statistics.

| Flag | Description |
|---|---|
| `--l` / `--r` | Arm side (required) |
| `--joint JOINT` | `shoulder_1`, `shoulder_2`, `shoulder_3`, `elbow`, `wrist_1`, `wrist_2`, `wrist_3` (required) |
| `--kp FLOAT` | Proportional gain (required) |
| `--kd FLOAT` | Derivative gain (required) |
| `--tff` | Apply full feedforward (gravity + friction) |
| `--mode {sine,step}` | `sine` = sinusoidal tracking (default); `step` = step response |
| `--amp FLOAT` | Motion amplitude in rad (default: auto safe value) |
| `--freq FLOAT` | [sine] Frequency in Hz (default: 1.0) |
| `--duration FLOAT` | [sine] Duration in seconds (default: 5.0) |
| `--hold FLOAT` | [step] Hold time per phase in seconds (default: 2.0) |
| `--rate FLOAT` | Command rate in Hz (default: 100.0) |

```bash
axol tune.pid --l --joint elbow --kp 25 --kd 0.6
axol tune.pid --r --joint shoulder_1 --kp 35 --kd 1.2 --mode step
```

### `tune.friction`

Identifies the four friction-model parameters for one joint via a bidirectional velocity sweep. Gravity is computed centrally from the URDF (see [Gravity compensation](#gravity-compensation)), so this command only fits the friction half-difference and a constant offset.

**Friction model:** `τ = Fc·tanh(0.1·k·v) + Fv·v + Fo`

| Flag | Description |
|---|---|
| `--l` / `--r` | Arm side (required) |
| `--joint JOINT` | `shoulder_1`, `shoulder_2`, `shoulder_3`, `elbow`, `wrist_1`, `wrist_2`, `wrist_3` (required) |
| `--kp FLOAT` | Proportional gain (default: from `AxolConfig`) |
| `--kd FLOAT` | Derivative gain (default: from `AxolConfig`) |
| `--velocities V [V ...]` | Velocity setpoints in rad/s (default: ~0.1, 0.3, 0.6, 0.9, 1.3) |
| `--lo RAD` | Override lower joint limit for the sweep |
| `--hi RAD` | Override upper joint limit for the sweep |
| `--dump-csv [PATH]` | Write per-bin `(v, q, tau_fwd, tau_bwd, tau_avg, tau_halfdiff)` rows to a CSV for offline plotting / arm-vs-arm comparison. Pass without a value to auto-name as `logs/friction_<side>_<joint>_<timestamp>.csv` |

```bash
axol tune.friction --l --joint shoulder_1 --kp 30 --kd 0.8
axol tune.friction --r --joint elbow --kp 20 --kd 0.6
axol tune.friction --l --joint wrist_1 --velocities 0.2 0.6 1.0
axol tune.friction --l --joint shoulder_2 --dump-csv
```

### `gravity_comp`

Holds both arms in gravity-compensation mode so you can move them by hand. Each *free* arm joint is sent `set_impedance` with `kp=0`, `kd=KD`, and a feedforward torque equal to the URDF-modelled gravity. Joints not in the free set are held rigidly at their current position with their configured `ArmConfig` `kp`/`kd` (still gravity-compensated). The grippers are softly held at their current positions.

| Flag | Description |
|---|---|
| `--no-left` / `--no-right` | Disable an arm |
| `--joints J1,J2,...` | Comma-separated joints to gravity-compensate (e.g. `WRIST_3` or `SHOULDER_1,ELBOW`). Other arm joints are held in place. Default: all 7 joints free. |
| `--kd FLOAT` | Velocity damping coefficient on *free* joints (Nm·s/rad). Higher = less floppy. Default 0.5 |
| `--rate FLOAT` | Control loop rate in Hz (default: 100) |
| `--telemetry-rate FLOAT` | Joint telemetry poll rate in Hz (default: 500) |

```bash
axol gravity_comp                                 # all 7 joints free, both arms
axol gravity_comp --no-right                      # left arm only, all joints free
axol gravity_comp --kd 1.0                        # heavier damping
axol gravity_comp --joints WRIST_3                # only WRIST_3 free; everything else held rigid
axol gravity_comp --no-right --joints SHOULDER_1,WRIST_3   # one-arm, two-joint isolation
```

Use `--joints` to test gravity comp on a single joint at a time: only the named joints float freely, while the rest of the arm holds its pose so you can isolate the effect.

If the arm sags or pushes back, tune the per-joint `mass` and `com` fields on `AxolConfig` (each `JointConfig` carries the inertial of the URDF body it drives) — see [Gravity compensation](#gravity-compensation).

---

## Python SDK

Install the package with `uv sync` (see [Installation](#installation)), then import directly from `almond_axol`.

- [Modules](#modules)
- [Core Concepts](#core-concepts)
- [almond_axol.robot](#almond_axolrobot)
- [almond_axol.kinematics](#almond_axolkinematics)
- [almond_axol.teleop](#almond_axolteleop)
- [almond_axol.vr](#almond_axolvr)
- [almond_axol.zed](#almond_axolzed)
- [almond_axol.motor](#almond_axolmotor)
- [almond_axol.lerobot](#almond_axollerobot)

---

### Modules

```
almond_axol/
├── robot/        Axol (hardware) and Sim (visualizer) — start here
├── kinematics/   Bimanual IK solver (JAX + pyroki)
├── teleop/       VR headset → IK → robot control loop
├── vr/           WebSocket server that receives VR frames
├── zed/          ZED-X One camera streaming
├── motor/        Low-level async CAN motor interface
└── lerobot/      LeRobot Robot / Teleoperator / Camera wrappers (requires lerobot extra)
    ├── robot/    AxolRobot — LeRobot Robot wrapping the async hardware driver
    ├── teleop/   AxolVRTeleop — LeRobot Teleoperator wrapping VRTeleop
    └── camera/   ZedCamera — LeRobot Camera wrapping a ZED stream receiver
```

End-to-end data flow for teleoperation:

```
VR headset → VRServer (WSS) → VRTeleop → KinematicsSolver → Axol → motors
```

End-to-end data flow for LeRobot data collection:

```
VR headset → AxolVRTeleop.get_action() → AxolRobot.send_action() → motors
                                                 ↑
                                  AxolRobot.get_observation() → dataset
                                  (joints from telemetry, cameras from ZedCamera)
```

---

### Core Concepts

**Async context managers.** `Axol`, `Sim`, `VRTeleop`, `VRServer`, and `ZedStreamer` all open and close hardware resources in `__aenter__` / `__aexit__`. Always use them with `async with`.

**Joint arrays.** Every method that reads or writes joint state uses `np.ndarray` of shape `(8,)` in `Joint` enum order: `SHOULDER_1`, `SHOULDER_2`, `SHOULDER_3`, `ELBOW`, `WRIST_1`, `WRIST_2`, `WRIST_3`, `GRIPPER`. Arm joints are in radians; the gripper is normalized to `[0.0 = closed, 1.0 = fully open]`.

**Telemetry vs. one-shot reads.** `start_telemetry(hz)` launches a background polling loop and populates `.positions` and `.torques` as non-blocking cached properties — read them in a tight control loop without `await`. Direct calls like `await get_positions()` issue individual CAN reads and are suitable for diagnostics but too slow for real-time control. `start_telemetry` is not required if you are already running a `motion_control` loop — every impedance command sent to the arm motors returns a position and torque reading, which is used to populate the same cache automatically.

---

### almond_axol.robot

Hardware controller for both arms. `Axol` opens one SocketCAN bus per arm on entry, enables all 16 motors, and calibrates the gripper open-stop. `Sim` is a drop-in replacement that renders the robot in a browser using viser (requires the `sim` extra).

```python
from almond_axol.robot import Axol, AxolConfig, ArmConfig, JointConfig, FrictionParams, Sim
```

#### `Axol`

```python
Axol(
    config: AxolConfig = AxolConfig(),
    left_channel: str | None = "can_alm_axol_l",
    right_channel: str | None = "can_alm_axol_r",
)
```

Pass `left_channel=None` or `right_channel=None` to operate a single arm. Both arms are brought up concurrently on `__aenter__`.

```python
import asyncio
import numpy as np
from almond_axol.robot import Axol

async def main():
    async with Axol() as axol:
        await axol.start_telemetry(500)   # 500 Hz background polling

        # non-blocking cached reads (after telemetry warms up)
        print("left positions (rad):", axol.left.positions)
        print("left torques (Nm):", axol.left.torques)

        # primary control: impedance for arm joints, position-force for gripper
        q = np.zeros(8, dtype=np.float32)
        q[7] = 1.0  # open gripper
        await axol.motion_control(left=q, right=q)

asyncio.run(main())
```

**Lifecycle methods**

| Method | Description |
|---|---|
| `enable()` | Start CAN buses, enable all motors, calibrate grippers |
| `disable()` | Disable all motors and close CAN buses |
| `start_telemetry(hz, torque=False)` | Begin background polling loop on all motors |
| `stop_telemetry()` | Stop background polling |
| `clear_errors()` | Clear latched error flags on all motors |
| `set_control_mode(mode)` | Set `ControlMode` on all motors |

**State reads** — each returns `(left_array, right_array)` where the absent arm is `None`

| Method | Units |
|---|---|
| `get_positions()` | rad (gripper: `[0, 1]`) |
| `get_velocities()` | rad/s |
| `get_torques()` | Nm (Damiao) / A (MyActuator) |
| `get_temperatures()` | °C |
| `get_voltages()` | V |
| `get_error_codes()` | `list[MotorStatus]` |
| `get_gains()` | `list[MotorGains]` |

**State writes**

| Method | Description |
|---|---|
| `motion_control(left, right)` | Impedance (arm) + position-force (gripper); primary control method |
| `set_positions_velocity(left, right, max_speed)` | Motor built-in position controller |
| `set_velocity(left, right)` | Motor built-in speed controller |
| `set_gains(left, right)` | Write PID gains; persisted to flash |
| `set_zero_position(left, right)` | Save current shaft position as encoder zero |
| `set_acceleration(left, right)` | Set per-joint acceleration ramp (rad/s²) |

Individual arms are accessible via `axol.left` and `axol.right` (`AxolArm`), which expose the same methods operating on a single arm.

#### `Sim`

`Sim` implements the same interface as `Axol`. Use it to visualise motion without hardware. Requires the `sim` extra.

```python
import asyncio
import numpy as np
from almond_axol.robot import Sim

async def main():
    async with Sim(port=8080) as sim:
        q = np.zeros(8, dtype=np.float32)
        await sim.motion_control(left=q)
        await asyncio.sleep(float("inf"))  # keep the viser server alive

asyncio.run(main())
```

Open `http://localhost:8080` in a browser to view the robot.

#### Configuration — `AxolConfig`, `ArmConfig`, `JointConfig`

Each arm joint is configured with a single `JointConfig` carrying its impedance gains, friction-comp model, and the inertial of the body it drives:

```python
from almond_axol.robot import Axol, AxolConfig, FrictionParams

config = AxolConfig()
config.left.elbow.kp = 200
config.left.elbow.mass = 0.6
config.left.elbow.com = (-0.025, 0.0, -0.07)
config.left.elbow.friction = FrictionParams(fc=0.4, k=10.0, fv=0.05, fo=0.0)
async with Axol(config=config) as axol: ...
```

Or build a fully custom arm with `dataclasses.replace` (start from the
`AxolConfig` defaults so you keep the per-side friction values that get
injected at construction):

```python
from dataclasses import replace
from almond_axol.robot import AxolConfig, JointConfig, FrictionParams

left = replace(
    AxolConfig().left,
    shoulder_1=JointConfig(
        kp=35.0, kd=1.2,
        friction=FrictionParams(fc=0.0, k=0.0, fv=0.0, fo=0.0),
        mass=2.0, com=(0.065, 0.0, 0.0),
    ),
)
```

**`JointConfig` fields**

| Field | Type | Description |
|---|---|---|
| `kp` | `float` (`[0, 500]`) | Impedance position stiffness |
| `kd` | `float` (`[0, 5]`) | Impedance velocity damping |
| `friction` | `FrictionParams` | `fc`, `k`, `fv`, `fo` — friction-comp model `fc·tanh(k·v) + fv·v + fo` |
| `mass` | `float` | Mass of the URDF body this joint drives (kg). For `wrist_3` this includes the gripper. |
| `com` | `tuple[float, float, float]` | Centre-of-mass of the same body in its URDF link frame (m). Used by gravity comp. |

Gravity feedforward is computed centrally from the URDF — see [Gravity compensation](#gravity-compensation) — and uses the per-joint `mass` and `com` directly.

`ArmConfig.gripper` is a `PositionForceConfig` with `torque_limit` (Nm) and `max_speed` (rad/s); the gripper's mass is already lumped into `wrist_3.mass` (the gripper joint is fixed).

`AxolConfig` also exposes a top-level safety parameter:

| Field | Default | Description |
|---|---|---|
| `max_step_rad` | `0.5` | Maximum allowed change in any arm joint (rad) between consecutive `motion_control` calls. Commands that exceed this are dropped and a warning is logged. Set to `float("inf")` to disable. At 30 Hz, 0.5 rad/step ≈ 15 rad/s — roughly 2.5× the teleop velocity ceiling. |

Both arms share the same `ArmConfig` defaults for gains and masses; the right arm gets CoMs mirrored across X via `ArmConfig.mirror_to_right()`. Per-motor friction values are identified separately for each arm (left/right motors measurably differ) — see `_LEFT_FRICTION` / `_RIGHT_FRICTION` in `almond_axol/robot/config.py`. Pass an explicit `left=` / `right=` to override either side.

### Gravity compensation

`almond_axol.robot.gravity.GravityCompensator` builds a MuJoCo model from the bundled URDF and computes per-joint gravity torques as `qfrc_bias` with `qvel=0` (Coriolis terms vanish). Because the URDF is the full kinematic chain, each parent joint's gravity load includes the contribution of every child link — this is the main improvement over the previous per-joint `ga·cos(q) + gb·sin(q)` model, which silently ignored child-link mass.

Per-link masses are not taken from the bundled URDF — the Onshape exporter leaves placeholder sub-gram values that produce essentially zero gravity. Real per-link mass and CoM live on each `JointConfig.mass` / `JointConfig.com` in `almond_axol/robot/config.py` (CoMs come from the CAD inertial origins; masses are tuned in place against measured joint torques and are typically lower than the CAD values, since Onshape often over-assigns aluminum-class densities to parts that are hollow / 3D-printed).

If the arms sag or push back in gravity-comp mode, tune the relevant joint's `mass` and `com` on `AxolConfig` and pass it to `Axol`:

```python
from almond_axol.robot import AxolConfig, Axol

config = AxolConfig()
config.left.elbow.mass = 0.6
config.left.elbow.com = (-0.025, 0.0, -0.07)
async with Axol(config=config) as axol: ...
```

---

### almond_axol.kinematics

Bimanual inverse kinematics using pyroki and JAX/jaxls. Loads the bundled URDF, builds a collision model, and JIT-compiles the solver during `__init__` — the first call takes a few seconds; subsequent calls are fast.

```python
from almond_axol.kinematics import KinematicsSolver, KinematicsConfig
```

```python
import numpy as np
from almond_axol.kinematics import KinematicsSolver, KinematicsConfig

solver = KinematicsSolver(KinematicsConfig(pos_weight=100.0))
q = np.zeros(solver.num_joints, dtype=np.float32)

# Forward kinematics → (left_SE3, right_SE3) as jaxlie.SE3
left_se3, right_se3 = solver.fk(q)

# Inverse kinematics → new joint array
pos = np.array([0.3, 0.2, 0.4], dtype=np.float32)
rot = np.eye(3, dtype=np.float32)
q = solver.ik(q, left_pose=(pos, rot))
```

**`KinematicsSolver` interface**

| Member | Description |
|---|---|
| `num_joints` | Total actuated joints across both arms |
| `joint_names` | Joint name strings (left arm then right) |
| `left_indices` / `right_indices` | Indices into the full `q` array for each arm |
| `fk(q)` | Forward kinematics → `(left_SE3, right_SE3)` |
| `ik(q, left_pose, right_pose, left_elbow_pos, right_elbow_pos)` | IK; `pose` is `(pos_3, rot_3x3)`; elbow hints are optional `(3,)` arrays |
| `set_posture_pose(q)` | Set the null-space attractor (home pose for joint drift prevention) |

**`KinematicsConfig` fields**

| Field | Default | Description |
|---|---|---|
| `pos_weight` | `50.0` | End-effector position tracking weight |
| `ori_weight` | `10.0` | End-effector orientation tracking weight |
| `elbow_weight` | `5.0` | Elbow hint tracking weight |
| `rest_weight` | `7.5` | Per-step damping; penalises deviation from `q_current` |
| `posture_weight` | `5.0` | Persistent attractor to home pose (prevents null-space drift) |
| `manipulability_weight` | `0.05` | Reward for configurations with high manipulability |
| `limit_weight` | `75.0` | Joint limit penalty weight |
| `self_collision_margin` | `0.1` m | Minimum clearance between collision bodies |
| `self_collision_weight` | `75.0` | Self-collision penalty weight |
| `max_iterations` | `8` | Solver iterations per `ik()` call |
| `cost_tolerance` | `1e-2` | Convergence tolerance |
| `max_joint_delta` | `~0.035` rad | Maximum joint change per `ik()` call |
| `max_reach` | `0.8` m | EE target clamped to this distance from shoulder |

---

### almond_axol.teleop

Connects a VR headset to the robot (or simulator) for teleoperation. IK runs in a dedicated subprocess to keep JAX off the asyncio event loop. The pipeline is: VR frames → One Euro filtering → IK solve → EMA smoothing → trapezoidal velocity profiling → `motion_control()`.

```python
from almond_axol.teleop import VRTeleop, VRTeleopConfig
```

```python
import asyncio
from almond_axol.robot import Axol
from almond_axol.teleop import VRTeleop

async def main():
    async with VRTeleop(Axol()) as teleop:
        await teleop.run()  # blocking; Ctrl+C to exit

asyncio.run(main())
```

Use `step()` instead of `run()` to integrate teleoperation into your own control loop:

```python
async with VRTeleop(axol) as teleop:
    while True:
        left_q, right_q = teleop.step()  # returns latest smoothed (8,) arrays
        # ... custom logic ...
        await asyncio.sleep(1 / 120)
```

See `teleop --robot axol` in [Teleoperation](#teleoperation) for the equivalent CLI command.

**`VRTeleopConfig` fields**

| Field | Default | Description |
|---|---|---|
| `frequency` | `120` Hz | Control loop rate used by `run()` and reset trajectory density |
| `teleop_max_vel` | `1.0 rev/s` | Trapezoidal filter velocity cap during normal teleoperation |
| `teleop_max_accel` | `3.5 rev/s²` | Trapezoidal filter acceleration cap |
| `engage_max_vel` | `0.1 rev/s` | Slower velocity limit when the deadman switch is first pressed after a reset |
| `engage_duration` | `1.0` s | How long `engage_max_vel` is held before restoring `teleop_max_vel` |
| `startup_max_accel` | `0.3 rev/s²` | Gentler accel during the initial startup move to rest pose |
| `ik_alpha` | `0.5` | EMA blend factor on IK output; `1.0` disables smoothing |
| `pose_min_cutoff` | `1.5` Hz | One Euro Filter tremor cutoff for raw VR poses |
| `pose_beta` | `5.0` | One Euro Filter speed coefficient (raises cutoff during fast moves) |
| `reset_speed` | `0.1 rev/s` | Speed of the collision-aware return-to-rest trajectory |
| `rest_pose_left` / `rest_pose_right` | near-zero | Reset target for each arm, shape `(7,)` in `ARM_JOINTS` order |

**Deadman switch behaviour.** Grip is a toggle, not a hold. Press both grip buttons together to enable arm movement; press either grip alone to freeze the arms. A rising edge on the reset button triggers a collision-aware trajectory back to the rest pose.

---

### almond_axol.vr

Secure WebSocket server (WSS) that receives `VRFrame` JSON messages from the VR app. A self-signed TLS certificate is auto-generated in `~/.almond/vr/certs/` on first use. This module can be used standalone to read raw VR data without full teleoperation — useful for custom control loops or data collection.

```python
from almond_axol.vr import VRServer, VRServerConfig
```

```python
import asyncio
from almond_axol.vr import VRServer, VRServerConfig

async def main():
    async with VRServer(VRServerConfig(port=8000)) as vr:
        while True:
            frame = vr.get_frame()
            if frame is not None:
                print(frame.l_ee, frame.r_ee)
            await asyncio.sleep(0.01)

asyncio.run(main())
```

Or use a callback instead of polling:

```python
def on_frame(frame):
    print(frame.l_grip, frame.r_grip)

async with VRServer() as vr:
    vr.set_on_frame(on_frame)
    await asyncio.sleep(float("inf"))
```

**`VRFrame` fields**

| Field | Type | Description |
|---|---|---|
| `l_ee` / `r_ee` | `VRPose` | 6-DOF end-effector pose (position + quaternion) |
| `l_elbow` / `r_elbow` | `VRPosition` | 3D elbow positions |
| `l_grip` / `r_grip` | `float [0, 1]` | Gripper commands |
| `l_lock` / `r_lock` | `bool` | Deadman switches |
| `reset` | `bool` | Rising edge triggers a reset move |
| `state` | `VRState` | `TELEOP`, `DATA_COLLECTION`, or `RECORDING` (headset-driven); `SAVING` and `ERROR` are server-pushed only |

**Server → headset feedback**

The server can push a state override to all connected headsets at any time using `VRServer.broadcast_text()`. The headset interprets messages of the form `{"type": "state", "value": "saving"}` as a state override that blocks recording controls. `{"type": "state", "value": "error"}` shows an error indicator in the headset UI. `{"type": "state", "value": "data_collection"}` re-enables controls after saving. The `AxolVRTeleop.send_feedback_state(state)` helper wraps this for all `VRState` values.

**`VRServerConfig` fields**

| Field | Default | Description |
|---|---|---|
| `port` | `8000` | WSS listen port |
| `certfile` | auto | Path to TLS certificate; `None` uses the auto-generated cert |
| `keyfile` | auto | Path to TLS private key |

> Before opening the VR app, accept the self-signed certificate by navigating to `https://<hostname>.local:8000` in the VR browser and proceeding past the security warning.

---

### almond_axol.zed

Streams up to three ZED-X One cameras over the local network using HEVC (H.265) encoding via the ZED SDK. Requires `pyzed` — install it with `axol zed.install` (see [ZED Camera](#zed-camera)).

```python
from almond_axol.zed import ZedStreamer, ZedConfig
```

```python
import asyncio
from almond_axol.zed import ZedStreamer, ZedConfig

async def main():
    config = ZedConfig(
        overhead_serial=12345678,
        left_arm_serial=23456789,
        right_arm_serial=34567890,
    )
    async with ZedStreamer(config):
        await asyncio.sleep(float("inf"))

asyncio.run(main())
```

**`ZedConfig` fields**

| Field | Default | Description |
|---|---|---|
| `overhead_serial` | `None` | Serial number of the overhead camera |
| `left_arm_serial` | `None` | Serial number of the left-arm camera |
| `right_arm_serial` | `None` | Serial number of the right-arm camera |
| `overhead_port` | `30000` | Streaming port for the overhead camera |
| `left_arm_port` | `30002` | Streaming port for the left-arm camera |
| `right_arm_port` | `30004` | Streaming port for the right-arm camera |
| `resolution` | `HD1080` | `sl.RESOLUTION`: `HD1200`, `HD1080`, or `SVGA` |
| `fps` | `60` | Capture frame rate for all cameras |
| `bitrate` | `8000` kbps | HEVC encoding bitrate |

At least one serial number must be provided. The sender IP is `192.168.10.1/24`; set it automatically with `--setup-ip` via the CLI or by calling `setup_link_ip(iface, "192.168.10.1/24")` from `almond_axol.shared` before streaming.

---

### almond_axol.motor

Low-level async SocketCAN interface for individual motors. Most users work through `Axol` — this layer is exposed for diagnostics, custom control modes, and bench testing individual motors.

```python
from almond_axol.motor import CanBus, Motor, ControlMode, MotorStatus, MotorGains, Joint
```

```python
import asyncio
from almond_axol.motor import CanBus, Motor, ControlMode, Joint

async def main():
    async with CanBus("can_alm_axol_l") as bus:
        elbow = Motor(bus, Joint.ELBOW)
        await elbow.enable()
        await elbow.set_control_mode(ControlMode.IMPEDANCE)

        pos = await elbow.get_position()  # rad
        print("elbow position:", pos)

        await elbow.set_impedance(p_des=pos, v_des=0.0, kp=100.0, kd=2.0, t_ff=0.0)
        await elbow.disable()

asyncio.run(main())
```

**`Motor` methods**

| Method | Description |
|---|---|
| `enable()` / `disable()` | Enable motor / engage brake |
| `clear_errors()` | Clear latched error flags |
| `set_zero_position()` | Save current position as encoder zero (persisted to flash) |
| `set_control_mode(mode)` | Set `ControlMode`; required before mode-specific commands |
| `get_control_mode()` | Read active mode from hardware (`None` for MyActuator) |
| `get_position()` | Shaft position (rad); raises if telemetry is active |
| `get_velocity()` | Shaft velocity (rad/s) |
| `get_torque()` | Torque estimate (Nm); raises if telemetry is active |
| `get_temperature()` | Motor temperature (°C) |
| `get_voltage()` | Bus voltage (V) |
| `get_error_code()` | `MotorStatus` |
| `get_gains()` / `set_gains(gains)` | Read/write PID gains (persisted to flash) |
| `set_impedance(p_des, v_des, kp, kd, t_ff)` | MIT impedance command; requires `IMPEDANCE` mode |
| `set_position_velocity(position, max_speed)` | Built-in position controller; requires `POSITION_VELOCITY` mode |
| `set_velocity(velocity)` | Built-in speed controller; requires `VELOCITY` mode |
| `set_position_force(position, max_speed, max_torque)` | Damiao only; requires `POSITION_FORCE` mode |
| `set_acceleration(acceleration, deceleration)` | Acceleration ramp (rad/s²) |
| `set_can_id(can_id)` | Change CAN ID (persisted to flash) |
| `start_telemetry(hz, torque=False)` / `stop_telemetry()` | Background polling loop |
| `motor.position` | Cached position (rad); populated by telemetry or `set_impedance` responses |
| `motor.torque` | Cached torque (Nm); populated by telemetry with `torque=True` |

**`ControlMode` values**

| Value | Description |
|---|---|
| `IMPEDANCE` | MIT impedance control (arm joints) |
| `POSITION_VELOCITY` | Motor built-in position controller |
| `VELOCITY` | Motor built-in speed controller |
| `POSITION_FORCE` | Position with hard torque cap; Damiao only (gripper) |

**`MotorStatus` values**

`OK`, `DISABLED`, `OVER_VOLTAGE`, `UNDER_VOLTAGE`, `OVER_CURRENT`, `OVER_TEMPERATURE`, `MOS_OVER_TEMP`, `ROTOR_OVER_TEMP`, `LOST_COMM`, `OVERLOAD`, `MOTOR_STALL`\*, `ENCODER_ERROR`\*, `POWER_OVERRUN`\*, `SPEEDING`\*, `UNKNOWN`

\* MyActuator only

**Joint → driver mapping**

| Joint | Driver | CAN ID |
|---|---|---|
| `SHOULDER_1` – `WRIST_1` | MyActuator | `0x01` – `0x05` |
| `WRIST_2`, `WRIST_3`, `GRIPPER` | Damiao | `0x06` – `0x08` |

---

### almond_axol.lerobot

LeRobot-compatible wrappers for the Axol hardware. Requires the `lerobot` extra. These classes implement the LeRobot `Robot`, `Teleoperator`, and `Camera` interfaces so the Axol works with any LeRobot training or data-collection pipeline without modification. The `collect-data` and `run-policy` CLI commands are built on top of this layer.

```python
from almond_axol.lerobot.robot import AxolRobot, AxolRobotConfig
from almond_axol.lerobot.teleop import AxolVRTeleop, AxolVRTeleopConfig
from almond_axol.lerobot.camera import ZedCamera, ZedCameraConfig
```

#### `AxolRobot`

LeRobot `Robot` wrapping the async `Axol` hardware driver. A background thread runs a dedicated asyncio event loop so motor telemetry keeps streaming while the synchronous `get_observation()` and `send_action()` calls block on the calling thread.

```python
from almond_axol.lerobot.robot import AxolRobot, AxolRobotConfig
from almond_axol.lerobot.camera import ZedCameraConfig

config = AxolRobotConfig(
    cameras={
        "overhead":  ZedCameraConfig(host="192.168.10.1", port=30000),
        "left_arm":  ZedCameraConfig(host="192.168.10.1", port=30002),
        "right_arm": ZedCameraConfig(host="192.168.10.1", port=30004),
    },
)
with AxolRobot(config) as robot:
    obs = robot.get_observation()           # joints + camera frames
    joint_obs = robot.get_joint_observation()  # joints only — use in tight control loops
    robot.send_action(obs)                  # hold current position
```

**`AxolRobotConfig` fields**

| Field | Default | Description |
|---|---|---|
| `cameras` | `{}` | `ZedCameraConfig` instances keyed by name |
| `axol_config` | `AxolConfig()` | Per-joint gains and safety parameters forwarded to the hardware driver |
| `telemetry_hz` | `120.0` | Background joint telemetry polling rate in Hz |
| `observe_torques` | `False` | Include joint torques in `observation.state` |
| `left_channel` | `"can_alm_axol_l"` | SocketCAN interface for the left arm |
| `right_channel` | `"can_alm_axol_r"` | SocketCAN interface for the right arm |

**Key methods**

| Method | Description |
|---|---|
| `get_observation()` | Returns joint positions (+ torques if enabled) and latest camera frames |
| `get_joint_observation()` | Returns joint positions only — no camera reads; use in the high-frequency teleop path |
| `send_action(action)` | Sends joint position targets via impedance control (arm) and position-force control (gripper) |
| `positions` | `(left, right)` cached arm positions from telemetry, each shape `(8,)` |

---

#### `AxolVRTeleop`

LeRobot `Teleoperator` wrapping `VRTeleop`. Runs the VR WebSocket server and IK subprocess on a background thread so `get_action()` is non-blocking and safe to call from any thread.

```python
from almond_axol.lerobot.teleop import AxolVRTeleop, AxolVRTeleopConfig

teleop = AxolVRTeleop(AxolVRTeleopConfig())
pos_l, pos_r = robot.positions
teleop.connect(q_start_left=pos_l, q_start_right=pos_r)

while True:
    action = teleop.get_action()
    events = teleop.get_teleop_events()
```

**`AxolVRTeleopConfig` fields**

| Field | Default | Description |
|---|---|---|
| `vr_teleop_config` | `VRTeleopConfig()` | Rest poses, IK frequency, filter parameters — see [`almond_axol.teleop`](#almond_axolteleop) |
| `kinematics_config` | `KinematicsConfig()` | IK solver weights — see [`almond_axol.kinematics`](#almond_axolkinematics) |
| `vr_server_config` | `VRServerConfig()` | WSS port and TLS certificate paths — see [`almond_axol.vr`](#almond_axolvr) |

**Key methods**

| Method | Description |
|---|---|
| `get_action()` | Returns the latest smoothed joint positions as a LeRobot `RobotAction` dict |
| `get_teleop_events()` | Returns and clears latched episode-control events (`start_recording`, `TERMINATE_EPISODE`, `RERECORD_EPISODE`) |
| `request_reset()` | Triggers a collision-aware trajectory back to the rest pose |
| `is_resetting` | `True` while the reset move is pending or in progress |
| `send_feedback_state(state)` | Broadcasts a `VRState` override (e.g. `SAVING`) to all connected VR headsets |

---

#### `ZedCamera`

LeRobot `Camera` wrapping a ZED stream receiver. Connects to a single port on the ZED streamer and decodes HEVC frames in a background thread. Resolution and FPS are **always overridden from the live stream** on `connect()` — the config defaults just need to match the sender so `RobotConfig` validation passes before the robot connects.

```python
from almond_axol.lerobot.camera import ZedCamera, ZedCameraConfig

cam = ZedCamera(ZedCameraConfig(host="192.168.10.1", port=30000))
cam.connect()
frame = cam.read_latest()   # shape (H, W, 3), non-blocking, returns most recent frame
cam.disconnect()
```

**`ZedCameraConfig` fields**

| Field | Default | Description |
|---|---|---|
| `host` | `"192.168.10.1"` | IP address of the `zed.stream` sender |
| `port` | `30000` | Streaming port; overhead=30000, left_arm=30002, right_arm=30004 |
| `fps` | `60` | Expected stream FPS; validated against the live stream on connect |
| `width` | `960` | Expected frame width (SVGA); validated on connect |
| `height` | `600` | Expected frame height (SVGA); validated on connect |
| `warmup_s` | `1` | Seconds to read frames during `connect()` before returning |

If the live stream parameters differ from the config, `connect()` raises a `RuntimeError` with the mismatch details. Update the config to match the `--resolution` and `--fps` passed to `zed.stream`.
