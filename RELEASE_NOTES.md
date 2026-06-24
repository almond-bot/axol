# Almond Axol v0.1.0 — First Release

We're excited to announce the **first release** of the Almond Axol SDK: the command-line interface and Python SDK for the Almond Axol dual-arm robot.

This release establishes the full operating stack — from low-level CAN/motor control to VR teleoperation, data collection, and policy execution — plus a browser-based control panel for driving the robot without touching a terminal.

## Highlights

- **VR teleoperation** — Drive Axol live from a WebXR headset (or in `--sim` mode without hardware) through the hosted interface at [axol.almond.bot](https://axol.almond.bot). Includes a JAX/PyRoki inverse-kinematics solver and motion filtering for smooth, responsive control.
- **Web control panel** — Operate the robot entirely from the browser via `axol serve`, with a self-updating server that stays in sync with `main` and restarts onto new versions once idle.
- **One-command install** — `curl https://axol.almond.bot/install -fsS | bash` sets up `uv`, the `axol` CLI, and a systemd service that keeps the robot server running at boot.
- **Data collection** — Record teleop episodes straight to a LeRobot dataset for training.
- **Policy execution** — Run trained policies on the robot with local or remote inference via the bundled inference server.
- **Quest-over-USB transport** — Low-latency wired controller transport that tunnels headset poses over a USB `adb` connection while keeping camera video on the LAN.
- **ZED camera streaming** — Hardware-accelerated NVENC encoding on the Jetson/ZED Box, streamed to the headset over WebRTC.
- **Gravity compensation** — Hold the arms weightless for hand-guiding and demonstration.
- **Tuning tools** — Built-in routines for PID, friction, and repeatability characterization.

## CLI Commands

- `teleop` — VR teleoperation session (supports `--sim`)
- `serve` — web control panel + API server
- `gravity-comp` — gravity-compensation / hand-guiding mode
- `collect-data` — record teleoperation episodes
- `run-policy` / `inference-server` — policy execution and inference
- `can.setup` / `can.enable` / `can.driver` — CAN bus initialization
- `motor.info` / `motor.set-can-id` / `motor.set-zero-pos` — motor configuration
- `zed.install` / `gst.install` / `gst.build-zed` / `jetson.setup` — camera & encode setup
- `tune.pid` / `tune.friction` / `tune.repeatability` — tuning utilities
- `provision` — machine provisioning

## Python SDK

Public modules for building on top of Axol:

- `almond_axol.robot` — `Axol`, `Sim`, configuration, gravity compensation
- `almond_axol.kinematics` — IK solver
- `almond_axol.teleop` — teleoperation control loop
- `almond_axol.vr` — WebXR/VR transport and video
- `almond_axol.zed` — ZED camera integration
- `almond_axol.motor` — low-level motor bus and drivers
- `almond_axol.lerobot` — LeRobot dataset and policy integration

## Requirements

- Linux
- Python 3.13+
- (Optional) NVIDIA Jetson / ZED Box for GMSL-attached ZED cameras

## Getting Started

```bash
curl https://axol.almond.bot/install -fsS | bash
```

Then open [axol.almond.bot](https://axol.almond.bot) and connect to your machine. Full documentation is available at [docs.almond.bot](https://docs.almond.bot).
