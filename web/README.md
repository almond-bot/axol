# Almond Axol VR

WebXR teleoperation interface for the Almond Axol robot. Streams hand/elbow pose data from a Meta Quest headset to the Almond Axol SDK over WebSocket.

## Monorepo structure

```
axol-vr/
├── app/                        # Vite + React app (deployed to Vercel)
└── packages/
    └── axol-vr-client/         # Reusable R3F components and hooks
```

## Packages

### `@almond/axol-vr-client`

React components and hooks for connecting to the Almond Axol SDK WebSocket server from inside an XR session.

**Exports**

| Export | Description |
|---|---|
| `AxolVRClient` | R3F component — reads XR input sources each frame and streams pose data over WebSocket |
| `useAxolVRClient` | Hook — manages WebSocket lifecycle (connect, disconnect, auto-retry) |
| `AxolState` | Enum — `Teleop`, `DataCollection`, `Recording`, `Saving`, `Error` |
| `AxolConnectionStatus` | Enum — `Idle`, `Connecting`, `Open`, `Error`, `Failed` |
| `AxolPoseData` | Type — shape of each frame sent over the WebSocket |

**`AxolVRClient` props**

| Prop | Type | Description |
|---|---|---|
| `wsRef` | `RefObject<WebSocket \| null>` | WebSocket ref from `useAxolVRClient` |
| `onStateChange` | `(state: AxolState) => void` | Fires when the controller state machine transitions |
| `onPendingRecording` | `(pendingAt: number \| null) => void` | Fires with a timestamp when a 3-second recording countdown begins; `null` when cancelled or resolved |
| `onExit` | `() => void` | Fires when the Y button exits the XR session |

**`useAxolVRClient` params**

```ts
useAxolVRClient(hostname: string, port = 8000, maxRetries = 3, retryMs = 1000)
// returns: { status, connect, disconnect, wsRef }
```

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
  reset:   boolean   // true on the frame X was pressed
  state:   "teleop" | "data_collection" | "recording"  // client-driven; "saving" is server-pushed via feedback message
}
```

## Controller bindings

![Quest controller diagram](assets/quest.png)

| # | Button | Action |
|---|---|---|
| 1 | Left grip | Press both grips (1 + 2) together to **enable** arm tracking; press either alone to **disable** it (toggle, not hold) |
| 2 | Right grip | See above |
| 3 | Left trigger | Actuate left gripper |
| 4 | Right trigger | Actuate right gripper |
| 5 | Left **X** | Reset pose; cancels recording countdown; exits Recording → DataCollection |
| 7 | Left **Y** | Exit XR session |
| 6 | Right **A** | Start recording (3-second countdown); stop immediately if already recording; cancels countdown if pressed during it |
| 8 | Right **B** | Toggle between Teleop and DataCollection (disabled while recording or countdown) |

## State machine

```
Teleop ──[B]──► DataCollection ──[A]──► (countdown 3s) ──► Recording
   ▲                 ▲                                          │
   └────────[B]──────┘                                   [A or X]
                                                               │
                                                          (server push)
                                                               │
                                                             Saving
                                                               │
                                                          (save done)
                                                               │
                                                         DataCollection
```

During the 3-second countdown the state sent to the server remains `DataCollection`. Once the countdown completes it transitions to `Recording`.

The `Saving` state is **server-driven**: the Python SDK broadcasts `{"type": "state", "value": "saving"}` over the WebSocket immediately when recording stops, then `{"type": "state", "value": "data_collection"}` once `save_episode()` completes. While in `Saving`, all A/B/X button actions except Y (exit) are blocked.

The `Error` state is also **server-driven**: broadcasting `{"type": "state", "value": "error"}` displays an error indicator in the headset UI and blocks all recording controls.

## App

The `app/` package is a Vite + React app that wraps the client library into a full WebXR UI deployed to Vercel.

**Dev**

```bash
npm install
npm run dev --workspace=app
```

Open the printed localhost URL on your Quest browser, enter the hostname of the machine running the Almond Axol SDK, and press **Connect**. Once connected, press **Start** to enter the AR session.

**Build**

```bash
npm run build --workspace=packages/axol-vr-client
npm run build --workspace=app
# output: app/dist/
```

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
