/**
 * The live lidar view for a running operation: the scan around the robot,
 * drawn with the robot model at its measured pose when Axol is on Jelly.
 *
 * The scene's origin is the floor under the centre of Jelly. Points come in
 * that frame from `/api/lidar/ws`; the robot model is shifted so its `jelly`
 * link sits there too.
 */
import { useEffect, useMemo, useRef, useState, type RefObject } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import { Grid, OrbitControls } from "@react-three/drei"
import { type OrbitControls as OrbitControlsImpl } from "three-stdlib"
import { Vector3 } from "three"
import { AlertTriangle, Loader2, Maximize2, Minimize2, RotateCcw } from "lucide-react"
import { useUrdfRobot } from "@/components/settings/urdf-robot"
import { LidarMarker, LidarPoints } from "@/components/lidar/lidar-points"
import { LidarControls } from "@/components/lidar/lidar-controls"
import { useLidarStream } from "@/lib/lidar-stream"
import {
  lidarStatusLine,
  loadLidarSettings,
  saveLidarSettings,
  type LidarSettings,
} from "@/lib/lidar"
import { liveRobotTargets, type LiveRobotSample } from "@/lib/live-robot"
import { type AxolModel, type GripperType } from "@/lib/supervisor"

const CAMERA_START = new Vector3(-3.2, 2.6, 3.2) // Y-up: behind-left of the robot, above
const TARGET_START = new Vector3(0, 0.6, 0)
const SMOOTHING_TAU = 0.06

/** The base link the lidar's mount is measured from, by robot model. */
const BASE_LINKS = ["jelly", "cart"]

/** Latest `joints` push from the teleop server on `socket`. */
function useJointPush(socket: WebSocket | null): RefObject<LiveRobotSample | null> {
  const ref = useRef<LiveRobotSample | null>(null)
  useEffect(() => {
    ref.current = null
    if (!socket) return
    const onMessage = (event: MessageEvent) => {
      if (typeof event.data !== "string" || !event.data.includes('"joints"')) return
      try {
        const msg = JSON.parse(event.data) as { type?: string; value?: Partial<LiveRobotSample> }
        const v = msg.type === "joints" ? msg.value : null
        if (!v || typeof v.q !== "object" || v.q === null) return
        ref.current = {
          q: v.q,
          l_grip: typeof v.l_grip === "number" ? v.l_grip : 1,
          r_grip: typeof v.r_grip === "number" ? v.r_grip : 1,
          lift: typeof v.lift === "number" ? v.lift : null,
        }
      } catch {
        // not ours
      }
    }
    socket.addEventListener("message", onMessage)
    return () => socket.removeEventListener("message", onMessage)
  }, [socket])
  return ref
}

function RobotAtPose({
  robotModel,
  gripper,
  joints,
}: {
  robotModel: AxolModel
  gripper: GripperType
  joints: RefObject<LiveRobotSample | null>
}) {
  const { model } = useUrdfRobot(robotModel, gripper)
  // Shift the model so its base link sits on the scene's origin.
  const offset = useMemo(() => {
    if (!model) return null
    const link = BASE_LINKS.map((name) => model.robot.links[name]).find(Boolean)
    if (!link) return null
    model.scene.updateWorldMatrix(true, true)
    return link.getWorldPosition(new Vector3()).negate()
  }, [model])
  const smooth = useRef<Record<string, number> | null>(null)

  useFrame((_, delta) => {
    const sample = joints.current
    if (!model || !sample) return
    const targets = liveRobotTargets(sample, model.robot)
    if (!smooth.current) smooth.current = { ...targets }
    const alpha = 1 - Math.exp(-delta / SMOOTHING_TAU)
    for (const [name, value] of Object.entries(targets)) {
      const prev = smooth.current[name] ?? value
      smooth.current[name] = prev + (value - prev) * alpha
      model.robot.joints[name]?.setJointValue(smooth.current[name])
    }
  })

  if (!model || !offset) return null
  return (
    <group position={offset}>
      <primitive object={model.scene} />
    </group>
  )
}

/**
 * A card with the live lidar scan. `socket` is the teleop server's socket,
 * whose joint push poses the robot model; without one the model holds its
 * rest pose.
 */
export default function LidarView({
  robot,
  socket,
}: {
  robot: { model: AxolModel; gripper: GripperType } | null
  socket: WebSocket | null
}) {
  const { cloud, status } = useLidarStream(true)
  const [settings, setSettings] = useState<LidarSettings>(loadLidarSettings)
  const [large, setLarge] = useState(false)
  const controls = useRef<OrbitControlsImpl | null>(null)
  const joints = useJointPush(socket)
  const line = lidarStatusLine(status)

  const change = (next: LidarSettings) => {
    setSettings(next)
    saveLidarSettings(next)
  }

  const resetView = () => {
    const c = controls.current
    if (!c) return
    c.object.position.copy(CAMERA_START)
    c.target.copy(TARGET_START)
    c.update()
  }

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-white/10 bg-white/[0.02] p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-xs tracking-widest text-white/40 uppercase">Lidar</span>
        <div className="flex items-center gap-3">
          <span
            className={
              line.tone === "ok"
                ? "font-mono text-[0.65rem] text-white/40"
                : line.tone === "bad"
                  ? "flex items-center gap-1 text-xs text-amber-300"
                  : "flex items-center gap-1 text-xs text-white/45"
            }
          >
            {line.tone === "bad" && <AlertTriangle className="size-3.5" />}
            {line.tone === "wait" && <Loader2 className="size-3.5 animate-spin" />}
            {/* The error itself shows over the view. */}
            {line.tone === "bad" ? "No scan" : line.text}
          </span>
          <button
            type="button"
            onClick={resetView}
            title="Reset view"
            className="text-white/40 transition-colors hover:text-white/80"
          >
            <RotateCcw className="size-4" />
          </button>
          <button
            type="button"
            onClick={() => setLarge((v) => !v)}
            title={large ? "Smaller view" : "Larger view"}
            className="text-white/40 transition-colors hover:text-white/80"
          >
            {large ? <Minimize2 className="size-4" /> : <Maximize2 className="size-4" />}
          </button>
        </div>
      </div>

      <div
        className={`relative overflow-hidden rounded-md bg-[#0b0c0e] ${
          large ? "h-[clamp(460px,75vh,900px)]" : "h-[clamp(320px,48vh,560px)]"
        }`}
      >
        <Canvas
          camera={{ position: CAMERA_START.toArray(), fov: 50, near: 0.05, far: 200 }}
          dpr={[1, 2]}
        >
          <color attach="background" args={["#0b0c0e"]} />
          <ambientLight intensity={0.7} />
          <directionalLight position={[2, 4, 3]} intensity={1.6} />
          <directionalLight position={[-3, 2, -2]} intensity={0.5} />
          <Grid
            infiniteGrid
            cellSize={0.5}
            sectionSize={2}
            cellColor="#24272c"
            sectionColor="#3a3f46"
            fadeDistance={30}
            fadeStrength={1.5}
          />
          {robot && (
            <RobotAtPose robotModel={robot.model} gripper={robot.gripper} joints={joints} />
          )}
          {/* Jelly's base frame: Z up. */}
          <group rotation={[-Math.PI / 2, 0, 0]}>
            <LidarPoints cloud={cloud} settings={settings} />
            {status.mount && <LidarMarker mount={status.mount} />}
          </group>
          <OrbitControls
            ref={controls}
            makeDefault
            target={TARGET_START.toArray()}
            maxPolarAngle={Math.PI * 0.495}
          />
        </Canvas>
        {status.state === "error" && (
          <div className="pointer-events-none absolute inset-x-0 bottom-3 flex justify-center px-3">
            <span className="rounded-md border border-amber-400/25 bg-black/70 px-3 py-1.5 text-xs text-amber-200">
              {status.error ?? "The lidar isn't sending."} If it still fails, run{" "}
              <code>axol lidar.info</code> on the robot.
            </span>
          </div>
        )}
      </div>

      <LidarControls settings={settings} onChange={change} />
    </div>
  )
}
