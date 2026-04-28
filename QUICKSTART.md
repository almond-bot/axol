# Quickstart

From a fresh clone to a live teleoperation session in five steps.

---

## 1. Install

Install the package using `uv`. The `sim` extra is optional — add it if you want to test with the software visualizer before touching hardware.

```bash
uv sync              # hardware only
uv sync --extra sim  # + software visualizer
```

Then activate the virtual environment so the `axol` CLI is on your path:

```bash
source .venv/bin/activate
```

> You will need to do this in every new shell session. Alternatively, prefix every command with `uv run` (e.g. `uv run axol can.setup`) to skip activation.

---

## 2. Set Up CAN (one-time)

Plug in the Axol Hub, then run the one-time CAN setup. This writes udev rules, assigns fixed interface names, and registers a startup script so the interfaces come up automatically on every reboot.

```bash
axol can.setup
```

> `sudo` is invoked automatically. You will not need to run this again unless you reinstall the OS or swap the Axol Hub.

After any subsequent replug mid-session (without a reboot), bring the interfaces back up with:

```bash
axol can.enable
```

---

## 3. Verify Motors

With the robot powered on and CAN up, read a status snapshot from one motor to confirm communication is working:

```bash
axol motor.info --l --id 0x01   # left shoulder_1
axol motor.info --r --id 0x01   # right shoulder_1
```

You should see position, velocity, torque, temperature, and voltage. Any `DISABLED` or error status means power or CAN isn't reaching that motor.

---

## 4. Accept the TLS Certificate

The VR app connects over a self-signed HTTPS/WSS connection. Before opening the app, you must accept the certificate in the VR headset's browser — otherwise the WebSocket handshake will be silently rejected.

1. Start `axol teleop` briefly (see step 5) — it will print your hostname and IP.
2. In the VR headset browser, navigate to:
   ```
   https://<hostname>.local:8000
   ```
   or
   ```
   https://<local-ip>:8000
   ```
3. Proceed past the security warning.
4. You only need to do this once per machine (the cert is cached in `~/.almond/vr/certs/`).

---

## 5. Teleop

```bash
axol teleop --robot axol
```

The terminal will print the hostname and IP address. Open the VR app at [axol.almond.bot](https://axol.almond.bot) on the headset and enter either address to connect.

### Controller Layout

![Quest controller diagram](assets/quest.png)

| # | Button | Action |
|---|---|---|
| 1 | Left grip | Press both 1 + 2 to **enable** arm movement; press either alone to **disable** |
| 2 | Right grip | Same as above |
| 3 | Left trigger | Actuate left gripper |
| 4 | Right trigger | Actuate right gripper |
| 5 | X | **Reset** — returns both arms to the rest pose |
| 7 | Y | **Exit AR** — closes the VR session |

> Grip is a toggle, not a hold. Press both grips together to start moving the arms; press either grip on its own to freeze them.

### Flags

| Flag | Description |
|---|---|
| `--no-left` | Disable the left arm |
| `--no-right` | Disable the right arm |
| `--gripper-torque-limit FLOAT` | Max gripper torque in Nm (default: `1.0`) |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Default: `INFO` |

### Test without hardware

```bash
axol teleop --robot sim
```

Opens a browser visualizer at `http://localhost:8080` instead of commanding real motors. Useful for verifying network connectivity and VR tracking before powering the robot.

---

## Network Tip

If VR tracking feels jittery or arrives in bursts, configure the following on your router or access point:

| Setting | Value |
|---|---|
| DTIM interval | `1` |
| Beacon interval | `100` ms |
| WMM APSD (U-APSD) | disabled |

These prevent the AP from batching packets between beacon intervals, which causes intermittent latency spikes that are especially noticeable in VR.

---

## Next Steps

- [Full CLI reference](README.md) — all commands, flags, and tuning tools
- [Python SDK](README.md#python-sdk) — use `Axol`, `KinematicsSolver`, and `VRTeleop` directly in Python
