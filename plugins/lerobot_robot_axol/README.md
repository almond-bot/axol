# lerobot_robot_axol

LeRobot plugin for the [Almond Axol](https://docs.almond.bot) dual-arm robot.

Installing this package makes the Axol devices available to the stock
[LeRobot](https://github.com/huggingface/lerobot) CLI tools (`lerobot-record`,
`lerobot-teleoperate`, `lerobot-replay`, ...) through LeRobot's plugin
auto-discovery — no changes to LeRobot itself.

| Kind | `type` | Class |
|---|---|---|
| Robot | `axol` | `almond_axol.lerobot.robot.AxolRobot` |
| Robot | `axol_mantis` | `almond_axol.lerobot.robot.MantisRobot` |
| Teleoperator | `axol_vr` | `almond_axol.lerobot.teleop.AxolVRTeleop` |
| Camera | `zed` | `almond_axol.lerobot.camera.ZedCamera` |

The device classes live in the [`almond-axol` SDK](https://github.com/almond-bot/axol);
this package is a thin shim that registers them with LeRobot at CLI startup.

## Requirements

- The robot machine (typically the ZED Box / NVIDIA Jetson wired to the Axol) with
  CAN set up — see the [installation guide](https://docs.almond.bot/installation).
- Python 3.12+ (the hosted Axol installer bundles 3.13), Linux.
- LeRobot 0.6.1 (pinned by `almond-axol[lerobot]`).
- Almond Axol SDK 0.1.x (the plugin imports SDK module paths and intentionally
  stays below the next pre-1.0 API line).
- For the `zed` camera: the ZED SDK and pyzed (`axol zed.install`).

## Install

```bash
pip install lerobot_robot_axol
```

This pulls in `almond-axol[lerobot]` automatically. To install from source
instead:

```bash
pip install "lerobot_robot_axol @ git+https://github.com/almond-bot/axol#subdirectory=plugins/lerobot_robot_axol"
```

If you already installed the `axol` CLI via the
[one-line installer](https://docs.almond.bot/installation), add the plugin to
that shared environment in place, then put its LeRobot entry points on your
interactive `PATH`:

```bash
sudo /usr/local/bin/uv pip install \
    --python /opt/axol/uv/tools/almond-axol/bin/python \
    --no-deps \
    "lerobot_robot_axol==0.1.1"
export PATH="/opt/axol/uv/tools/almond-axol/bin:$PATH"
```

Use the in-place command above rather than rebuilding the Axol tool with
`uv tool install --with`: a tool rebuild exactly reconciles dependencies and
would prune separately managed packages such as the VIVE Ultimate runtime or a
JetPack-compatible CUDA Torch build. The hosted environment already contains
the compatible Axol SDK/LeRobot dependencies, so `--no-deps` also prevents a
plugin install from changing that running environment. Later hosted-installer
or control-panel Axol updates detect a published plugin installed this way and
preserve its exact version in the same transaction. A direct/VCS/custom plugin
source blocks those force updates before mutation because its source cannot be
reconstructed safely; update that deployment manually with the same source.

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
