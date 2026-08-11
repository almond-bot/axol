# lerobot_robot_axol

LeRobot plugin for the [Almond Axol](https://docs.almond.bot) dual-arm robot.

Installing this package makes the Axol devices available to the stock
[LeRobot](https://github.com/huggingface/lerobot) CLI tools (`lerobot-record`,
`lerobot-teleoperate`, `lerobot-replay`, ...) through LeRobot's plugin
auto-discovery — no changes to LeRobot itself.

| Kind | `type` | Class |
|---|---|---|
| Robot | `axol` | `almond_axol.lerobot.robot.AxolRobot` |
| Teleoperator | `axol_vr` | `almond_axol.lerobot.teleop.AxolVRTeleop` |
| Camera | `zed` | `almond_axol.lerobot.camera.ZedCamera` |

The device classes live in the [`almond-axol` SDK](https://github.com/almond-bot/axol);
this package is a thin shim that registers them with LeRobot at CLI startup.

## Requirements

- The robot machine (typically the ZED Box / NVIDIA Jetson wired to the Axol) with
  CAN set up — see the [installation guide](https://docs.almond.bot/installation).
- Python 3.13, Linux.
- LeRobot >= 0.6.1.
- For the `zed` camera: the ZED SDK and pyzed (`axol zed.install`).

## Install

One command — the plugin pulls in `almond-axol` (and its pinned dependencies)
automatically; both are distributed from GitHub, not PyPI:

```bash
pip install "lerobot_robot_axol @ git+https://github.com/almond-bot/axol#subdirectory=plugins/lerobot_robot_axol"
```

If you already installed the `axol` CLI via the
[one-line installer](https://docs.almond.bot/installation), add the plugin to
that environment instead:

```bash
uv tool install --with "lerobot_robot_axol @ git+https://github.com/almond-bot/axol#subdirectory=plugins/lerobot_robot_axol" almond-axol
```

## Usage

```bash
lerobot-teleoperate \
    --robot.type=axol \
    --teleop.type=axol_vr

lerobot-record \
    --robot.type=axol \
    --robot.cameras "{overhead: {type: zed, serial: 41234567, stereo: true}}" \
    --teleop.type=axol_vr \
    --dataset.repo_id=${HF_USER}/my-dataset \
    --dataset.num_episodes=5
```

Teleoperation uses a VR headset: open [axol.almond.bot](https://axol.almond.bot)
in the headset browser and connect to the robot machine — see the
[teleop guide](https://docs.almond.bot/operations/teleop).

For the highest-rate data collection (out-of-process ZED capture, Jetson NVENC
dataset encoding, headset video streaming), prefer the SDK's own
[`axol collect-data`](https://docs.almond.bot/cli/collect-data), which is built
on the same LeRobot classes.
