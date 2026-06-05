import { useRef, useState, type ReactNode } from "react"
import { Canvas, useFrame, useThree } from "@react-three/fiber"
import { Text } from "@react-three/drei"
import { createXRStore, XR, useXR } from "@react-three/xr"
import * as THREE from "three"
import {
  AxolConnectionStatus,
  AxolVRClient,
  AxolState,
  useAxolVRClient,
} from "@almond/axol-vr-client"

const store = createXRStore({
  handTracking: false,
  bodyTracking: true,
  controller: { model: false },
})

const L_ELBOW_JOINT = "left-arm-lower" as XRBodyJoint
const R_ELBOW_JOINT = "right-arm-lower" as XRBodyJoint

const AXIS_LEN = 0.1
const SHAFT_R = 0.004
const TIP_R = 0.009
const TIP_LEN = 0.025
const DOT_RADIUS = 0.014

const AXES: { color: string; rotation: [number, number, number] }[] = [
  { color: "#FF0000", rotation: [0, 0, -Math.PI / 2] }, // X — red
  { color: "#00FF00", rotation: [0, 0, 0] }, // Y — green
  { color: "#0000FF", rotation: [Math.PI / 2, 0, 0] }, // Z — blue
]

function Arrow({ color, rotation }: { color: string; rotation: [number, number, number] }) {
  const shaftLen = AXIS_LEN - TIP_LEN
  return (
    <group rotation={rotation}>
      <mesh position={[0, shaftLen / 2, 0]}>
        <cylinderGeometry args={[SHAFT_R, SHAFT_R, shaftLen, 8]} />
        <meshBasicMaterial color={color} />
      </mesh>
      <mesh position={[0, shaftLen + TIP_LEN / 2, 0]}>
        <coneGeometry args={[TIP_R, TIP_LEN, 8]} />
        <meshBasicMaterial color={color} />
      </mesh>
    </group>
  )
}

function AxesMarker({ groupRef }: { groupRef: React.RefObject<THREE.Group | null> }) {
  return (
    <group ref={groupRef} visible={false}>
      {AXES.map((a) => (
        <Arrow key={a.color} color={a.color} rotation={a.rotation} />
      ))}
      <mesh>
        <sphereGeometry args={[DOT_RADIUS, 10, 10]} />
        <meshBasicMaterial color="#FFFF00" />
      </mesh>
    </group>
  )
}

function PoseVisualizer() {
  const { gl } = useThree()
  const leftRef = useRef<THREE.Group>(null)
  const rightRef = useRef<THREE.Group>(null)
  const lElbowRef = useRef<THREE.Group>(null)
  const rElbowRef = useRef<THREE.Group>(null)

  useFrame(() => {
    const session = gl.xr.getSession()
    if (!session) return
    const frame = gl.xr.getFrame()
    const refSpace = gl.xr.getReferenceSpace()
    if (!frame || !refSpace) return

    function applyPose(group: THREE.Group | null, space: XRSpace | null | undefined) {
      if (!group) return
      if (!space) {
        group.visible = false
        return
      }
      const pose = frame.getPose(space, refSpace!)
      if (!pose) {
        group.visible = false
        return
      }
      const { position: p, orientation: o } = pose.transform
      group.position.set(p.x, p.y, p.z)
      group.quaternion.set(o.x, o.y, o.z, o.w)
      group.visible = true
    }

    function applyPosition(group: THREE.Group | null, space: XRSpace | null | undefined) {
      if (!group) return
      if (!space) {
        group.visible = false
        return
      }
      const pose = frame.getPose(space, refSpace!)
      if (!pose) {
        group.visible = false
        return
      }
      const { position: p } = pose.transform
      group.position.set(p.x, p.y, p.z)
      group.visible = true
    }

    const leftSource = Array.from(session.inputSources).find(
      (s: XRInputSource) => s.handedness === "left"
    )
    const rightSource = Array.from(session.inputSources).find(
      (s: XRInputSource) => s.handedness === "right"
    )

    applyPose(leftRef.current, leftSource?.targetRaySpace ?? null)
    applyPose(rightRef.current, rightSource?.targetRaySpace ?? null)

    const body = (frame as XRFrame & { body?: XRBody }).body
    applyPosition(lElbowRef.current, body?.get(L_ELBOW_JOINT))
    applyPosition(rElbowRef.current, body?.get(R_ELBOW_JOINT))
  })

  return (
    <>
      <AxesMarker groupRef={leftRef} />
      <AxesMarker groupRef={rightRef} />
      <group ref={lElbowRef} visible={false}>
        <mesh>
          <sphereGeometry args={[DOT_RADIUS, 10, 10]} />
          <meshBasicMaterial color="#FFFF00" />
        </mesh>
      </group>
      <group ref={rElbowRef} visible={false}>
        <mesh>
          <sphereGeometry args={[DOT_RADIUS, 10, 10]} />
          <meshBasicMaterial color="#FFFF00" />
        </mesh>
      </group>
    </>
  )
}

const hudBg = { backgroundColor: "#000000", backgroundOpacity: 0.5, padding: 0.006 } as object

function XRHud({ children }: { children: ReactNode }) {
  const session = useXR((s) => s.session)
  const groupRef = useRef<THREE.Group>(null)

  useFrame(({ gl }) => {
    if (!groupRef.current) return
    groupRef.current.visible = gl.xr.isPresenting
    if (!gl.xr.isPresenting) return
    const activeCam = gl.xr.getCamera()
    groupRef.current.position.copy(activeCam.position)
    groupRef.current.quaternion.copy(activeCam.quaternion)
  })

  if (!session) return null

  return (
    <group ref={groupRef} visible={false}>
      {children}
    </group>
  )
}

function ExitButton() {
  const [hovered, setHovered] = useState(false)

  return (
    <Text
      position={[-0.2, 0.1, -0.5]}
      fontSize={0.02}
      fontWeight="bold"
      color={hovered ? "yellow" : "white"}
      anchorX="left"
      anchorY="top"
      renderOrder={999}
      material-depthTest={false}
      {...hudBg}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
      onClick={() => store.getState().session?.end()}
    >
      Exit
    </Text>
  )
}

const STATUS_DISPLAY: Partial<Record<AxolState | "pending", { color: string; label: string }>> = {
  pending: { color: "yellow", label: "● Starting…" },
  [AxolState.Error]: { color: "#f87171", label: "● Error" },
  [AxolState.Recording]: { color: "red", label: "● Recording" },
  [AxolState.Saving]: { color: "orange", label: "● Saving…" },
  [AxolState.DataCollection]: { color: "blue", label: "● Data Collection" },
}

function StateDisplay({
  state,
  isRecordingPending,
}: {
  state: AxolState
  isRecordingPending: boolean
}) {
  const displayState: AxolState | "pending" = isRecordingPending ? "pending" : state
  const { color, label } = STATUS_DISPLAY[displayState] ?? { color: "white", label: "● Teleop" }

  return (
    <Text
      position={[0.2, 0.1, -0.5]}
      fontSize={0.02}
      fontWeight="bold"
      color={color}
      anchorX="right"
      anchorY="top"
      renderOrder={999}
      material-depthTest={false}
      {...hudBg}
    >
      {label}
    </Text>
  )
}

function HelpPanel({ onDismiss }: { onDismiss: () => void }) {
  const W = 0.3
  const H = 0.075
  const col = 0.07

  return (
    <group position={[0, -0.038, 0]}>
      {/* Large dismiss plane behind everything */}
      <mesh position={[0, 0, -0.002]} renderOrder={996} onClick={onDismiss}>
        <planeGeometry args={[2, 2]} />
        <meshBasicMaterial transparent opacity={0} depthTest={false} side={THREE.DoubleSide} />
      </mesh>
      {/* Panel background */}
      <mesh position={[0, -H / 2, -0.001]} renderOrder={998} onClick={(e) => e.stopPropagation()}>
        <planeGeometry args={[W, H]} />
        <meshBasicMaterial
          color="black"
          transparent
          opacity={0.97}
          depthTest={false}
          side={THREE.DoubleSide}
        />
      </mesh>
      {/* Vertical divider */}
      <mesh position={[0, -H / 2, 0]} renderOrder={999}>
        <planeGeometry args={[0.002, H]} />
        <meshBasicMaterial color="white" depthTest={false} side={THREE.DoubleSide} />
      </mesh>
      {/* LEFT header */}
      <Text
        position={[-col, -0.004, 0]}
        fontSize={0.013}
        color="white"
        fontWeight="bold"
        anchorX="center"
        anchorY="top"
        renderOrder={1000}
        material-depthTest={false}
      >
        LEFT
      </Text>
      {/* RIGHT header */}
      <Text
        position={[col, -0.004, 0]}
        fontSize={0.013}
        color="white"
        fontWeight="bold"
        anchorX="center"
        anchorY="top"
        renderOrder={1000}
        material-depthTest={false}
      >
        RIGHT
      </Text>
      {/* Left buttons */}
      <Text
        position={[-col, -0.022, 0]}
        fontSize={0.013}
        color="white"
        anchorX="center"
        anchorY="top"
        renderOrder={1000}
        material-depthTest={false}
        lineHeight={1.6}
      >
        {`[Y]  Exit VR\n[X]  Reset Pose`}
      </Text>
      {/* Right buttons */}
      <Text
        position={[col, -0.022, 0]}
        fontSize={0.013}
        color="white"
        anchorX="center"
        anchorY="top"
        renderOrder={1000}
        material-depthTest={false}
        lineHeight={1.6}
      >
        {`[B]  Toggle Mode\n[A]  Start / Stop Rec`}
      </Text>
    </group>
  )
}

function HelpIcon() {
  const [open, setOpen] = useState(false)

  return (
    <group position={[0, 0.1, -0.5]}>
      <Text
        fontSize={0.02}
        fontWeight="bold"
        color={open ? "yellow" : "white"}
        anchorX="center"
        anchorY="top"
        renderOrder={999}
        material-depthTest={false}
        {...hudBg}
        onClick={() => setOpen((v) => !v)}
      >
        ?
      </Text>
      {open && <HelpPanel onDismiss={() => setOpen(false)} />}
    </group>
  )
}

function CountdownDisplay({ recordingPendingAt }: { recordingPendingAt: number | null }) {
  const [count, setCount] = useState(3)
  const prevCountRef = useRef(3)

  useFrame(() => {
    if (recordingPendingAt === null) return
    const remaining = Math.ceil((3000 - (Date.now() - recordingPendingAt)) / 1000)
    const clamped = Math.max(1, Math.min(3, remaining))
    if (clamped !== prevCountRef.current) {
      prevCountRef.current = clamped
      setCount(clamped)
    }
  })

  if (recordingPendingAt === null) return null

  return (
    <Text
      position={[0, 0, -0.5]}
      fontSize={0.1}
      fontWeight="bold"
      color="white"
      anchorX="center"
      anchorY="middle"
      renderOrder={999}
      material-depthTest={false}
    >
      {String(count)}
    </Text>
  )
}

export default function App() {
  const [hostname, setHostname] = useState(() => localStorage.getItem("wsHostname") ?? "")
  const [vrState, setVrState] = useState<AxolState>(AxolState.Teleop)
  const [recordingPendingAt, setRecordingPendingAt] = useState<number | null>(null)
  const { status, connect, disconnect, wsRef } = useAxolVRClient(hostname)

  const handleConnect = () => {
    localStorage.setItem("wsHostname", hostname)
    connect()
  }

  const isActive =
    status === AxolConnectionStatus.Connecting || status === AxolConnectionStatus.Open

  return (
    <>
      <div
        style={{
          position: "fixed",
          inset: 0,
          zIndex: 10,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 12,
          pointerEvents: "none",
        }}
      >
        <div
          style={{
            pointerEvents: "auto",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 10,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
            <img src="/almond.svg" alt="Almond" style={{ width: 48, height: 48 }} />
            <span style={{ fontSize: 32, fontWeight: 800 }}>Almond Axol VR</span>
          </div>
          {status === AxolConnectionStatus.Open && (
            <button type="button" onClick={() => store.enterAR()}>
              Start
            </button>
          )}
          {isActive && (
            <span
              style={{
                fontSize: 12,
                opacity: 0.9,
                color: status === AxolConnectionStatus.Open ? "#4ade80" : "#eab308",
              }}
            >
              {status === AxolConnectionStatus.Open ? "● Connected" : "○ Connecting…"}
            </span>
          )}
          {!isActive && (
            <form
              onSubmit={(e) => {
                e.preventDefault()
                handleConnect()
              }}
              style={{ display: "flex", alignItems: "center", gap: 6 }}
            >
              <input
                type="text"
                value={hostname}
                onChange={(e) => setHostname(e.target.value)}
                placeholder="workstation.local"
                style={{
                  fontSize: 12,
                  padding: "4px 8px",
                  borderRadius: 4,
                  border: "1px solid #333",
                  background: "#1e1e1e",
                  color: "white",
                  width: 160,
                }}
              />
              <button
                type="submit"
                style={{ padding: "4px 10px", fontSize: 12, cursor: "pointer" }}
              >
                Connect
              </button>
            </form>
          )}
          {isActive && (
            <button
              type="button"
              onClick={disconnect}
              style={{ padding: "4px 10px", fontSize: 12, cursor: "pointer" }}
            >
              Disconnect
            </button>
          )}
          {status === AxolConnectionStatus.Open && (
            <div style={{ display: "flex", gap: 16, fontSize: 14, color: "#9ca3af", marginTop: 8 }}>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>LEFT</div>
                <div>[Y] Exit VR</div>
                <div>[X] Reset Pose</div>
              </div>
              <div style={{ width: 1, background: "#9ca3af", alignSelf: "stretch" }} />
              <div style={{ textAlign: "center" }}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>RIGHT</div>
                <div>[B] Toggle Mode</div>
                <div>[A] Start / Stop Rec</div>
              </div>
            </div>
          )}
          {status === AxolConnectionStatus.Failed && (
            <div
              style={{
                fontSize: 12,
                color: "#f87171",
                padding: "6px 10px",
                background: "rgba(248, 113, 113, 0.15)",
                borderRadius: 6,
                maxWidth: 280,
                textAlign: "center",
              }}
            >
              Could not connect to {hostname}. Check that the server is running.
            </div>
          )}
        </div>
      </div>

      <Canvas>
        <XR store={store}>
          <AxolVRClient
            wsRef={wsRef}
            onStateChange={setVrState}
            onPendingRecording={setRecordingPendingAt}
            onExit={() => store.getState().session?.end()}
          />
          <XRHud>
            <ExitButton />
            <HelpIcon />
            <StateDisplay state={vrState} isRecordingPending={recordingPendingAt !== null} />
            <CountdownDisplay recordingPendingAt={recordingPendingAt} />
          </XRHud>
          <PoseVisualizer />
        </XR>
      </Canvas>
    </>
  )
}
