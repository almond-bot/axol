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
import { Headset, Loader2, ShieldCheck } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { SiteNav } from "@/components/site-nav"
import { authorizeCert } from "@/lib/cert-accept"
import { cn } from "@/lib/utils"

// The VR teleop WebSocket server runs on this port (see useAxolVRClient default).
const VR_WS_PORT = 8000

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

function ControlHints({ title, rows }: { title: string; rows: [string, string][] }) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.02] p-3">
      <div className="mb-1.5 font-mono text-[0.65rem] tracking-widest text-white/40 uppercase">
        {title}
      </div>
      <div className="flex flex-col gap-1">
        {rows.map(([key, label]) => (
          <div key={key} className="flex items-center gap-2">
            <kbd className="flex size-5 items-center justify-center rounded border border-white/15 bg-white/[0.06] font-mono text-[0.65rem] text-white/70">
              {key}
            </kbd>
            <span className="text-white/60">{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function ConnectionStatus({ status }: { status: AxolConnectionStatus }) {
  const meta =
    status === AxolConnectionStatus.Open
      ? { dot: "bg-emerald-400", ring: "bg-emerald-400/40", label: "Connected" }
      : status === AxolConnectionStatus.Connecting
        ? { dot: "bg-amber-400", ring: "bg-amber-400/40", label: "Connecting…" }
        : status === AxolConnectionStatus.Failed
          ? { dot: "bg-red-400", ring: "bg-red-400/40", label: "Connection failed" }
          : { dot: "bg-white/40", ring: "bg-white/10", label: "Not connected" }

  return (
    <div className="flex items-center justify-center gap-2 text-sm text-white/60">
      <span className="relative flex size-2.5">
        {status === AxolConnectionStatus.Connecting && (
          <span
            className={cn(
              "absolute inline-flex h-full w-full animate-ping rounded-full",
              meta.ring
            )}
          />
        )}
        <span className={cn("relative inline-flex size-2.5 rounded-full", meta.dot)} />
      </span>
      {meta.label}
    </div>
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

  return (
    <>
      <div className="pointer-events-none fixed inset-0 z-10 flex flex-col bg-[#121212]/70 backdrop-blur-sm">
        <div className="pointer-events-auto">
          <SiteNav current="vr" />
        </div>
        <div className="flex flex-1 items-center justify-center p-6">
          <Card className="pointer-events-auto w-full max-w-sm gap-6">
            <div className="flex flex-col items-center gap-3 text-center">
              <img src="/almond.svg" alt="Almond" className="h-12 w-12" />
              <div>
                <h1 className="font-heading text-2xl font-bold tracking-tight">Almond Axol</h1>
                <p className="text-sm text-white/40">VR Teleoperation</p>
              </div>
            </div>

            <ConnectionStatus status={status} />

            {status === AxolConnectionStatus.Open ? (
              <div className="flex flex-col gap-2">
                <Button size="lg" className="w-full" onClick={() => store.enterAR()}>
                  <Headset />
                  Enter VR
                </Button>
                <Button variant="ghost" className="w-full" onClick={disconnect}>
                  Disconnect
                </Button>
              </div>
            ) : status === AxolConnectionStatus.Connecting ? (
              <Button variant="secondary" className="w-full" onClick={disconnect}>
                <Loader2 className="animate-spin" />
                Cancel
              </Button>
            ) : (
              <form
                onSubmit={(e) => {
                  e.preventDefault()
                  handleConnect()
                }}
                className="flex flex-col gap-2"
              >
                <label
                  htmlFor="vr-host"
                  className="text-xs font-medium tracking-widest text-white/40 uppercase"
                >
                  Axol Host Address
                </label>
                <Input
                  id="vr-host"
                  type="text"
                  value={hostname}
                  onChange={(e) => setHostname(e.target.value)}
                  placeholder="axol-host.local"
                />
                <Button type="submit" className="w-full" disabled={!hostname.trim()}>
                  Connect
                </Button>
              </form>
            )}

            {status === AxolConnectionStatus.Open && (
              <div className="grid grid-cols-2 gap-3 text-left text-xs">
                <ControlHints
                  title="Left"
                  rows={[
                    ["Y", "Exit VR"],
                    ["X", "Reset pose"],
                  ]}
                />
                <ControlHints
                  title="Right"
                  rows={[
                    ["B", "Toggle mode"],
                    ["A", "Start / stop rec"],
                  ]}
                />
              </div>
            )}

            {status === AxolConnectionStatus.Failed && (
              <div className="flex flex-col gap-2">
                <p className="rounded-lg border border-red-400/25 bg-red-400/10 p-3 text-xs text-red-300">
                  Could not connect to <span className="font-mono">{hostname || "the server"}</span>
                  . Check that <span className="font-mono">axol serve</span> is running, then
                  authorize its self-signed certificate below.
                </p>
                {hostname.trim() && (
                  <Button
                    variant="outline"
                    className="w-full"
                    onClick={() =>
                      authorizeCert(`https://${hostname.trim()}:${VR_WS_PORT}`).then(handleConnect)
                    }
                  >
                    <ShieldCheck />
                    Authorize certificate
                  </Button>
                )}
              </div>
            )}
          </Card>
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
