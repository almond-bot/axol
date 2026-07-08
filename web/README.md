# Almond Axol Web

The browser front-ends for the Almond Axol robot. This directory lives inside the main `axol` repo (it was previously the standalone `axol-vr` repo) and builds two surfaces from one app:

- **VR interface** (`/vr`) — WebXR teleoperation. Streams hand/elbow pose from a Meta Quest headset to the Almond Axol SDK over WebSocket. Deployed to Vercel at [axol.almond.bot](https://axol.almond.bot).
- **Control panel** (`/control`) — browser UI for driving the robot (connect, teleop, gravity comp, collect data, run policy). Served by `axol serve`.

The base path `/` redirects by device: headset browsers go to `/vr`, everything else to `/control`.

> Docs: [Web Control Panel](https://docs.almond.bot/guides/control-panel) · [VR Interface](https://docs.almond.bot/guides/vr-interface). The `serve` backend (FastAPI) that the control panel talks to lives in `almond_axol/serve/`.

## Structure

```
web/
├── app/                        # Vite + React app — both /vr and /control routes
│   ├── src/routes/VrApp.tsx        # WebXR teleop interface
│   ├── src/routes/ControlPanel.tsx # control panel UI
│   └── dist/                       # build output — served by `axol serve` and Vercel
└── packages/
    └── axol-vr-client/         # Reusable R3F components and hooks
```

## Packages

### `@almond/axol-vr-client`

React components and hooks for connecting to the Almond Axol SDK WebSocket server from inside an XR session.

**Exports**

| Export | Description |
|---|---|
| `AxolVRClient` | R3F component — reads XR input sources each frame and streams pose data over the main WebSocket, mirroring each frame onto a dedicated USB pose socket when one is supplied so the server can prefer the wired link and fall back to WiFi |
| `useAxolVRClient` | Hook — manages WebSocket lifecycle (connect, disconnect, auto-retry) |
| `useAxolPoseSocket` | Hook — maintains a dedicated pose WebSocket to `wss://localhost:<port>` (the Quest-over-USB `adb reverse` tunnel) so controller poses avoid WiFi latency; returns `{ poseWsRef, status }` |
| `useAxolVideo` | Hook — negotiates a WebRTC connection over the same WebSocket and returns the camera video tracks streamed by the server (overhead / wrist cams), labelled by camera name |
| `useAxolTracking` | Hook — returns a frame-readable `ref` reflecting whether the robot is currently engaged (mirroring the operator), driven by the server's `{"type":"tracking"}` pushes with a local grip-toggle fallback. Used to gate camera-screen repositioning to when the robot isn't being controlled |
| `useAxolUrdfState` | Hook — returns a frame-readable `ref` with the latest `{"type":"urdf_state"}` push (robot base transform in the XR reference space + IK joint solution), used to render the virtual robot overlay in absolute (UMI) mode |
| `AxolState` | Enum — `Teleop`, `DataCollection`, `Recording`, `Saving`, `Error` |
| `AxolConnectionStatus` | Enum — `Idle`, `Connecting`, `Open`, `Error`, `Failed` |
| `AxolPoseData` | Type — shape of each frame sent over the WebSocket |
| `AxolMode` | Type — `"teleop" \| "data_collection"`, the server-announced operating mode that locks the HUD |
| `ConfirmAction` | Type — `"save" \| "discard"`, which episode action a stop-recording confirmation popup is gating |
| `CameraStreams` | Type — `Record<string, MediaStream>`, the camera-name → stream map returned by `useAxolVideo` |

**`AxolVRClient` props**

| Prop | Type | Description |
|---|---|---|
| `wsRef` | `RefObject<WebSocket \| null>` | WebSocket ref from `useAxolVRClient` |
| `poseWsRef` | `RefObject<WebSocket \| null>` (optional) | Dedicated pose WebSocket from `useAxolPoseSocket` (Quest-over-USB). When supplied and open, each frame is sent over **both** this and `wsRef`; the server prefers the low-latency USB stream and uses the network frames only while USB is quiet, so a USB drop fails over to WiFi with no reconnect |
| `onStateChange` | `(state: AxolState) => void` | Fires when the controller state machine transitions |
| `onPendingRecording` | `(pendingAt: number \| null) => void` | Fires with a timestamp when a 3-second recording countdown begins; `null` when cancelled or resolved |
| `onPendingConfirm` | `(action: ConfirmAction \| null) => void` | Fires with `"save"` / `"discard"` when the stop-recording confirmation popup is armed, and `null` when it's confirmed or cancelled |
| `onMode` | `(mode: AxolMode) => void` | Fires once per connection with the server-announced operating mode (`"teleop"` / `"data_collection"`) that locks the HUD |
| `onEpisode` | `(episode: number \| null) => void` | Fires with the current 1-based episode number during data collection (and `null` when the server clears it, e.g. on a connection change); drives the `Episode N` HUD readout |
| `onExit` | `() => void` | Fires when the Y button exits the XR session |

**`useAxolVRClient` params**

```ts
useAxolVRClient(hostname: string, port = 8000, maxRetries = 3, retryMs = 1000)
// returns: { status, connect, disconnect, wsRef }
```

**`useAxolPoseSocket` params**

```ts
useAxolPoseSocket(enabled: boolean, port = 8000)
// returns: { poseWsRef, status }
```

When `enabled`, maintains `wss://localhost:<port>` — the Quest-over-USB
`adb reverse` tunnel — with auto-retry, and closes when disabled. Pass
`poseWsRef` to `AxolVRClient` and each frame is mirrored over both the USB cable
and the network socket; the server prefers the cable (avoiding WiFi power-save
buffering) and falls back to the network frames whenever USB goes quiet, so a
cable drop fails over to WiFi with no reconnect. Camera video keeps using the
LAN connection. See **Quest over USB** in the repo README for the operator flow.

**Frame data (`AxolPoseData`)**

Each frame sends a JSON message over the WebSocket:

```ts
{
  l_ee:    { position: { x, y, z }, quaternion: { x, y, z, w } }  // left controller
  r_ee:    { position: { x, y, z }, quaternion: { x, y, z, w } }  // right controller
  l_elbow: { x, y, z }
  r_elbow: { x, y, z }
  l_lock:  boolean   // left grip button state (True = pressed); rising edge of both together enables tracking, either alone disables it
  r_lock:  boolean   // right grip button state (True = pressed); see l_lock
  l_grip:  number    // left grip (0 = fully gripped, 1 = open)
  r_grip:  number    // right grip
  reset:   boolean   // true on the frame X (reset) or Y (exit) was pressed — Y piggy-backs a reset so the arms return to rest before the session ends
  state:   "teleop" | "data_collection" | "recording"  // client-driven; "saving" is server-pushed via feedback message
  seq:     number    // monotonic frame counter; the same frame is sent over both USB and WiFi with one seq, and the server processes each seq once (from whichever link delivers it first)
}
```

## Controller bindings

![Quest controller diagram](assets/quest.png)

The operating mode (teleop vs. data collection) is **announced by the server on connect and locked** for the session — there's no in-headset toggle. In plain teleop the recording controls are inert; in data collection they drive episodes.

| # | Button | Action |
|---|---|---|
| 1 | Left grip | Press both grips (1 + 2) together to **enable** arm tracking; press either alone to **disable** it (toggle, not hold) |
| 2 | Right grip | See above |
| 3 | Left trigger | Actuate left gripper; while tracking is disengaged, point at a camera screen and hold to **move** it — grab one screen with **both** triggers to **resize** it |
| 4 | Right trigger | Actuate right gripper; while tracking is disengaged, point at a camera screen and hold to **move** it — grab one screen with **both** triggers to **resize** it |
| 5 | Left **X** | Reset pose; cancels a recording countdown. While recording, arms the **Discard episode?** confirmation — press **X** again to discard and re-record, or **A** to cancel and keep recording |
| 7 | Left **Y** | Exit the XR session — sends a reset first, so the arms return to rest and disengage instead of holding the last pose |
| 6 | Right **A** | **Record**: start a take (3-second countdown). While recording, arms the **Save episode?** confirmation — press **A** again to save, or **X** to cancel and keep recording — **data collection only** (no effect during plain teleop) |
| — | Right thumbstick (click) | Re-anchor the camera screens to your current gaze and clear all moves + resizes |

## State machine

In **teleop** mode the headset stays in `Teleop` with the recording controls inert. In **data collection** mode it starts in `DataCollection` and drives episodes with **A** / **X**:

```
DataCollection ──[A]──► (countdown 3s) ──► Recording
      ▲                                          │
      │                             [A]=save · [X]=discard
      │                            (arms a confirm popup — press
      │                             the same button again to commit)
      │                                          │
      ├──────── Saving ◄──(server push)◄─── [A→A] save
      │                                          │
      └────────────────────────────────── [X→X] discard
```

During the 3-second countdown the state sent to the server remains `DataCollection`. Once the countdown completes it transitions to `Recording`.

Stopping a recording is **confirmation-gated**: while recording, the first **A** (save) or **X** (discard) press arms an in-headset **Save episode?** / **Discard episode?** popup instead of stopping immediately. Pressing the **same** button confirms; the **other** cancels and keeps recording. Nothing is committed server-side until a save is confirmed — a confirmed discard carries the reset flag so the server drops the take and rewinds to re-record.

The `Saving` state is **server-driven**: the Python SDK broadcasts `{"type": "state", "value": "saving"}` over the WebSocket immediately when recording stops, then `{"type": "state", "value": "data_collection"}` once `save_episode()` completes. While in `Saving`, all A/X button actions except Y (exit) are blocked.

The `Error` state is also **server-driven**: broadcasting `{"type": "state", "value": "error"}` displays an error indicator in the headset UI and blocks all recording controls.

## App

The `app/` package is a Vite + React app that serves both the WebXR teleop interface (`/vr`, wrapping the `axol-vr-client` library) and the control panel (`/control`). The two routes are lazy-loaded so opening the control panel doesn't pull in the heavy three.js / XR bundle.

**Dev**

```bash
npm install
npm run dev --workspace=app
```

- **VR**: open the printed localhost URL on your Quest browser, enter the hostname of the machine running the Almond Axol SDK, press **Connect**, then **Start** to enter the AR session.
- **Control panel**: open `/control` in a normal browser. It talks to the `axol serve` API (default `https://localhost:8001`).

**Build**

```bash
npm run build --workspace=packages/axol-vr-client   # client package first
npm run build --workspace=app                        # → app/dist/
```

The built `app/dist/` is served two ways: by Vercel (the hosted VR app) and by `axol serve` locally (which hosts both routes from the same bundle).

## Deployment

The app is deployed on Vercel. `vercel.json` builds the client package first so it is available as a local workspace dependency:

```json
{
  "buildCommand": "npm run build --workspace=packages/axol-vr-client && npm run build --workspace=app",
  "outputDirectory": "app/dist",
  "installCommand": "rm -f package-lock.json && npm install"
}
```

The `installCommand` removes any macOS-generated lock file to avoid missing Linux rollup binaries on the Vercel build machine.

## Python SDK

The Almond Axol SDK receives frames from the headset and can push state feedback back. The relevant models live in `almond_axol/vr/models.py`:

```python
class VRState(str, Enum):
    TELEOP = "teleop"
    DATA_COLLECTION = "data_collection"
    RECORDING = "recording"
    SAVING = "saving"          # server-pushed only; blocks recording controls
    ERROR = "error"            # server-pushed only; shows error indicator in headset UI

class VRFrame(BaseModel):     # headset → server (every XR frame)
    l_ee: VRPose
    r_ee: VRPose
    l_elbow: VRPosition
    r_elbow: VRPosition
    l_lock: bool
    r_lock: bool
    l_grip: float
    r_grip: float
    reset: bool
    state: VRState             # one of TELEOP / DATA_COLLECTION / RECORDING
```

**Server → headset feedback**

The server can push a state override to all connected headsets at any time:

```json
{ "type": "state", "value": "saving" }
```

Use `AxolVRTeleop.send_feedback_state(VRState.SAVING)` / `send_feedback_state(VRState.DATA_COLLECTION)` to block and unblock recording controls on the headset while an episode is being written to disk.

On connect the server announces its operating mode — `{ "type": "mode", "value": "teleop" | "data_collection" }` (via `VRServer.set_mode()`) — which locks the headset HUD to that mode. During data collection it also pushes the current episode number — `{ "type": "episode", "value": N }` (1-based) — via `AxolVRTeleop.send_feedback_episode(episode)`, rendered as an `Episode N` HUD readout; the latest value is stored (`VRServer.set_episode()`) and re-sent on connect so a headset joining mid-session shows the right number.

The server also pushes `{ "type": "tracking", "value": true|false }` whenever the engage toggle changes; the headset uses it to only allow repositioning the camera screens while the robot isn't being controlled.

In absolute (UMI) mode the server additionally streams `{ "type": "urdf_state", "base": { "pos": [x,y,z], "quat": [x,y,z,w] } | null, "joints": { "<urdf joint>": rad, ... }, "engaged": bool }` at ~30 Hz. `base` is the robot base transform solved at engage (null before the first engage) in the same reference space as the controller poses; the app renders the robot URDF there (`RobotModel` in the app), fetching the model from the server's `https://<host>:8000/urdf/` static mount.

**Camera video (WebRTC)**

When the server has video sources registered (`VRServer.set_video_sources`, see `almond_axol/video/video.py`), the headset negotiates a WebRTC connection over the same WebSocket: it sends `{ "type": "webrtc-request" }`, the server replies with `{ "type": "webrtc-offer", "sdp": ..., "tracks": { mid: cameraName } }`, and the client answers with `{ "type": "webrtc-answer", "sdp": ... }`. The `useAxolVideo` hook implements the client side and returns the labelled video tracks. A stereo overhead arrives as the two tracks `overhead_left` / `overhead_right`, rendered per-lens.
