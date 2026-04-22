# Axol CLI

Command-line interface for the Almond Axol dual-arm robot. Invoked as `axol <command> [flags]`.

## Requirements

- **Linux**
- **Python 3.13+**
- **(Optional) NVIDIA Jetson** If Zed cameras are used.

## Table of Contents

- [Installation](#installation)
- [CAN Bus Setup](#can-bus-setup)
- [Motor Commands](#motor-commands)
- [Teleoperation](#teleoperation)
- [Data Collection](#data-collection)
- [Policy Execution](#policy-execution)
- [ZED Camera](#zed-camera)
- [Tuning](#tuning)

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
| `--fps INT` | Frame rate (default: 30) |
| `--root PATH` | Local dataset root (default: `$HF_LEROBOT_HOME`) |
| `--push-to-hub` | Push to HuggingFace Hub when done |
| `--gripper-torque-limit FLOAT` | Max gripper torque in POSITION_FORCE mode in Nm (default: 1.0) |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Default: `INFO` |

```bash
axol collect-data --repo-id myorg/pick-place --task "Pick the red cube and place it in the bin"
```

**VR controller events:**

| Event | Action |
|---|---|
| `START_RECORDING` | Begin capturing frames |
| `TERMINATE_EPISODE` | Save the episode |
| `RERECORD_EPISODE` | Discard and retry |

If an existing dataset is found at `--root`, collection resumes from where it left off.

---

## Policy Execution

### `run-policy`

Runs a trained policy autonomously on the robot using three ZED cameras. Between episodes, prompts the operator via stdin to save (`Enter`), re-record (`r`), or quit (`q`).

| Flag | Description |
|---|---|
| `--policy PATH_OR_REPO` | Local checkpoint path or HuggingFace repo ID (required) |
| `--task TEXT` | Natural language task description (required) |
| `--episode-time-s INT` | Max duration per episode in seconds (default: 30) |
| `--fps INT` | Control loop frame rate (default: 30) |
| `--repo-id <user>/<dataset>` | Optional dataset repo ID to save rollouts |
| `--root PATH` | Local dataset root (default: `$HF_LEROBOT_HOME`) |
| `--push-to-hub` | Push rollout dataset to HuggingFace Hub when done |
| `--gripper-torque-limit FLOAT` | Max gripper torque in POSITION_FORCE mode in Nm (default: 1.0) |
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

### `tune.feedforward`

Identifies all six feedforward parameters for one joint via a bidirectional velocity sweep. Fits gravity and friction models and prints a ready-to-paste config snippet.

**Gravity model:** `τ = ga·cos(q) + gb·sin(q) + Fo`

**Friction model:** `τ = Fc·tanh(0.1·k·v) + Fv·v`

| Flag | Description |
|---|---|
| `--l` / `--r` | Arm side (required) |
| `--joint JOINT` | Joint to identify (required) |
| `--kp FLOAT` | Proportional gain (default: from `AxolConfig`) |
| `--kd FLOAT` | Derivative gain (default: from `AxolConfig`) |
| `--velocities V [V ...]` | Velocity setpoints in rad/s (default: ~0.1, 0.3, 0.6, 0.9, 1.3) |

```bash
axol tune.feedforward --l --joint shoulder_1 --kp 30 --kd 0.8
axol tune.feedforward --r --joint elbow --kp 20 --kd 0.6
axol tune.feedforward --l --joint wrist_1 --velocities 0.2 0.6 1.0
```
