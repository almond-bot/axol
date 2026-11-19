/**
 * The Sensors page's 3D view: the live map, where the robot has been, and
 * the robot at its pose on the map with the live lidar scan around it.
 *
 * Everything inside the scene's root group is in the map frame (Z up, floor
 * at z = 0). Without a map the robot stays at the origin.
 */
import {
  useEffect,
  useMemo,
  useRef,
  type MutableRefObject,
  type ReactNode,
  type RefObject,
} from "react"
import { Canvas, useFrame, useThree } from "@react-three/fiber"
import { Grid, OrbitControls } from "@react-three/drei"
import { type OrbitControls as OrbitControlsImpl } from "three-stdlib"
import {
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  Group,
  Line,
  LineBasicMaterial,
  Matrix4,
  Plane,
  Quaternion,
  Vector3,
  type Texture,
} from "three"
import { useUrdfRobot } from "@/components/settings/urdf-robot"
import { LidarMarker, LidarPoints } from "@/components/lidar/lidar-points"
import { useJointPose } from "@/components/lidar/joint-push"
import { type LidarCloud, type LidarMount, type LidarSettings } from "@/lib/lidar"
import { type LidarMap } from "@/lib/lidar-map"
import { type LiveRobotSample } from "@/lib/live-robot"
import { type AxolModel, type GripperType } from "@/lib/supervisor"

/** Camera moves the page's buttons ask for. */
export interface SceneView {
  reset: () => void
  top: () => void
}

/** The base link the lidar's mount and the map pose are measured from. */
const BASE_LINKS = ["jelly", "cart"]
const MAP_POINT_SIZE = 0.05
const POSE_TAU = 0.08

let disc: Texture | null = null
function discTexture(): Texture {
  if (disc) return disc
  const size = 64
  const canvas = document.createElement("canvas")
  canvas.width = canvas.height = size
  const ctx = canvas.getContext("2d")!
  ctx.beginPath()
  ctx.arc(size / 2, size / 2, size / 2 - 1, 0, Math.PI * 2)
  ctx.fillStyle = "#ffffff"
  ctx.fill()
  disc = new CanvasTexture(canvas)
  return disc
}

/** T_map_base from a pose message, or identity. */
function poseMatrix(m: number[] | null | undefined, out: Matrix4): Matrix4 {
  if (!m) return out.identity()
  return out.set(
    m[0],
    m[1],
    m[2],
    m[3],
    m[4],
    m[5],
    m[6],
    m[7],
    m[8],
    m[9],
    m[10],
    m[11],
    0,
    0,
    0,
    1
  )
}

function MapPoints({ map, clip }: { map: LidarMap; clip: Plane[] }) {
  const invalidate = useThree((s) => s.invalidate)
  const geometry = useMemo(() => new BufferGeometry(), [])
  const texture = useMemo(() => discTexture(), [])
  const drawn = useRef({ generation: -1, count: 0 })
  useEffect(() => map.subscribe(() => invalidate()), [map, invalidate])
  useEffect(() => () => geometry.dispose(), [geometry])

  useFrame(() => {
    const last = drawn.current
    if (last.generation === map.generation && last.count === map.count) return
    if (last.generation !== map.generation) {
      geometry.setAttribute("position", new BufferAttribute(map.positions, 3))
      geometry.setAttribute("color", new BufferAttribute(map.colors, 3))
      last.generation = map.generation
    } else if (map.count > last.count) {
      // Upload only the new points.
      for (const name of ["position", "color"]) {
        const attr = geometry.getAttribute(name) as BufferAttribute
        attr.addUpdateRange(last.count * 3, (map.count - last.count) * 3)
        attr.needsUpdate = true
      }
    }
    last.count = map.count
    geometry.setDrawRange(0, map.count)
  })

  return (
    <points geometry={geometry} frustumCulled={false}>
      {/* Dimmer than the live scan, so the scan stands out. */}
      <pointsMaterial
        vertexColors
        color="#8a939e"
        sizeAttenuation
        map={texture}
        alphaTest={0.5}
        size={MAP_POINT_SIZE}
        clippingPlanes={clip}
      />
    </points>
  )
}

function Trail({ map }: { map: LidarMap }) {
  const line = useMemo(() => {
    const geometry = new BufferGeometry()
    const material = new LineBasicMaterial({ color: "#f59e0b", transparent: true, opacity: 0.8 })
    return new Line(geometry, material)
  }, [])
  const drawn = useRef({ length: -1, revision: -1 })
  useEffect(
    () => () => {
      line.geometry.dispose()
      ;(line.material as LineBasicMaterial).dispose()
    },
    [line]
  )
  useFrame(() => {
    const t = map.trail
    if (drawn.current.length === t.length && drawn.current.revision === map.generation) return
    drawn.current = { length: t.length, revision: map.generation }
    // Lift it off the floor a little so the grid doesn't hide it.
    const pts = new Float32Array(t.length)
    for (let i = 0; i < t.length; i += 3) {
      pts[i] = t[i]
      pts[i + 1] = t[i + 1]
      pts[i + 2] = 0.02
    }
    line.geometry.setAttribute("position", new BufferAttribute(pts, 3))
    line.geometry.setDrawRange(0, t.length / 3)
  })
  return <primitive object={line} />
}

/** The robot model, in base-frame coordinates (its base link at the origin). */
function RobotModel({
  robotModel,
  gripper,
  joints,
}: {
  robotModel: AxolModel
  gripper: GripperType
  joints: RefObject<LiveRobotSample | null>
}) {
  const { model } = useUrdfRobot(robotModel, gripper)
  const robot = model?.robot ?? null
  const offset = useMemo(() => {
    if (!robot) return null
    robot.updateMatrixWorld(true)
    const link = BASE_LINKS.map((name) => robot.links[name]).find(Boolean)
    if (!link) return new Matrix4()
    const rel = new Matrix4().copy(robot.matrixWorld).invert().multiply(link.matrixWorld)
    return rel.invert()
  }, [robot])
  useJointPose(robot, joints)
  const group = useRef<Group>(null)
  useEffect(() => {
    if (group.current && offset) {
      group.current.matrix.copy(offset)
      group.current.matrixWorldNeedsUpdate = true
    }
  }, [offset])
  if (!robot || !offset) return null
  return (
    <group ref={group} matrixAutoUpdate={false}>
      <primitive object={robot} />
    </group>
  )
}

/**
 * Moves the robot's group to its pose on the map, smoothed, and keeps the
 * camera following it when asked.
 */
function RobotOnMap({
  map,
  useMap,
  follow,
  baseRef,
  children,
}: {
  map: LidarMap
  useMap: boolean
  follow: boolean
  baseRef: MutableRefObject<Vector3>
  children: ReactNode
}) {
  const group = useRef<Group>(null)
  const controls = useThree((s) => s.controls) as unknown as OrbitControlsImpl | null
  const camera = useThree((s) => s.camera)
  const stateRef = useRef({
    target: new Matrix4(),
    pos: new Vector3(),
    quat: new Quaternion(),
    tPos: new Vector3(),
    tQuat: new Quaternion(),
    scale: new Vector3(),
    started: false,
    world: new Vector3(),
    lastWorld: new Vector3(),
  })
  useFrame((_, delta) => {
    const g = group.current
    const state = stateRef.current
    if (!g) return
    poseMatrix(useMap ? map.pose?.m : null, state.target)
    state.target.decompose(state.tPos, state.tQuat, state.scale)
    if (!state.started) {
      state.pos.copy(state.tPos)
      state.quat.copy(state.tQuat)
      state.started = true
    } else {
      const a = 1 - Math.exp(-delta / POSE_TAU)
      state.pos.lerp(state.tPos, a)
      state.quat.slerp(state.tQuat, a)
    }
    g.position.copy(state.pos)
    g.quaternion.copy(state.quat)
    g.updateMatrixWorld()
    // The robot's floor point in the (Y-up) world.
    g.getWorldPosition(state.world)
    baseRef.current.copy(state.world)
    if (follow && controls) {
      const d = state.world.clone().sub(state.lastWorld)
      if (d.lengthSq() > 0 && d.lengthSq() < 25) {
        camera.position.add(d)
        controls.target.add(d)
        controls.update()
      }
    }
    state.lastWorld.copy(state.world)
  })
  return <group ref={group}>{children}</group>
}

function ViewRig({
  viewRef,
  baseRef,
  map,
}: {
  viewRef: MutableRefObject<SceneView | null>
  baseRef: MutableRefObject<Vector3>
  map: LidarMap
}) {
  const controls = useThree((s) => s.controls) as unknown as OrbitControlsImpl | null
  const invalidate = useThree((s) => s.invalidate)
  useEffect(() => {
    if (!controls) return
    const heading = () => {
      const m = map.pose?.m
      return m ? Math.atan2(m[4], m[0]) : 0
    }
    const place = (offset: Vector3, lookUp: number) => {
      const p = baseRef.current
      controls.object.position.copy(p).add(offset)
      controls.target.set(p.x, p.y + lookUp, p.z)
      controls.update()
      invalidate()
    }
    viewRef.current = {
      // Behind the robot and above it. Map +x is world +x, map +y is world -z.
      reset: () => {
        const yaw = heading()
        place(new Vector3(-Math.cos(yaw) * 4, 3, Math.sin(yaw) * 4), 0.6)
      },
      top: () => {
        const yaw = heading()
        place(new Vector3(-Math.cos(yaw) * 0.01, 14, Math.sin(yaw) * 0.01), 0)
      },
    }
    return () => {
      viewRef.current = null
    }
  }, [controls, viewRef, baseRef, map, invalidate])
  return null
}

export function SensorScene({
  cloud,
  map,
  useMap,
  showMap,
  mapEpoch,
  settings,
  mount,
  robot,
  joints,
  follow,
  viewRef,
}: {
  cloud: LidarCloud
  map: LidarMap
  /** Place the robot by the map's pose. */
  useMap: boolean
  /** Draw the map's points. Off shows only the live scan. */
  showMap: boolean
  /** The live map whose frame the scans are drawn in, or null for around the robot. */
  mapEpoch: number | null
  settings: LidarSettings
  mount: LidarMount | null
  robot: { model: AxolModel; gripper: GripperType } | null
  joints: RefObject<LiveRobotSample | null>
  follow: boolean
  viewRef: MutableRefObject<SceneView | null>
}) {
  const baseRef = useRef(new Vector3())
  // World is Y-up; keep map points at or below the cut.
  const clip = useMemo(
    () => [new Plane(new Vector3(0, -1, 0), settings.hideAbove)],
    [settings.hideAbove]
  )
  return (
    <Canvas
      camera={{ position: [-4, 3, 0], fov: 50, near: 0.05, far: 400 }}
      dpr={[1, 2]}
      onCreated={({ gl }) => {
        gl.localClippingEnabled = true
      }}
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
        fadeDistance={40}
        fadeStrength={1.5}
      />
      <group rotation={[-Math.PI / 2, 0, 0]}>
        {useMap && showMap && <MapPoints map={map} clip={clip} />}
        {useMap && <Trail map={map} />}
        {/* On a map, each scan is drawn where it was taken. */}
        {mapEpoch !== null && <LidarPoints cloud={cloud} settings={settings} mapEpoch={mapEpoch} />}
        <RobotOnMap map={map} useMap={useMap} follow={follow} baseRef={baseRef}>
          {robot && <RobotModel robotModel={robot.model} gripper={robot.gripper} joints={joints} />}
          {mapEpoch === null && <LidarPoints cloud={cloud} settings={settings} />}
          {mount && <LidarMarker mount={mount} />}
        </RobotOnMap>
      </group>
      <OrbitControls makeDefault target={[0, 0.6, 0]} maxPolarAngle={Math.PI * 0.495} />
      <ViewRig viewRef={viewRef} baseRef={baseRef} map={map} />
    </Canvas>
  )
}
