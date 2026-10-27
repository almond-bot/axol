/**
 * 3D view for the Jog page: the robot's URDF at its live pose, a combined
 * move-and-turn gizmo on the target gripper pose, and an orange ghost of the arm: the
 * planned move's solution with a dashed path from the gripper, or, before a
 * plan, the live IK preview of the target being dragged.
 *
 * Every pose the page exchanges with the jog session is in Axol's torso frame
 * (the URDF's `root` link). The overlay group below follows that link, so
 * target, path and markers sit in the same frame IK uses wherever the lift
 * and base have put the torso.
 */
import {
  memo,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react"
import { Canvas, invalidate, useFrame, type ThreeEvent } from "@react-three/fiber"
import { Line, OrbitControls, PivotControls } from "@react-three/drei"
import {
  Box3,
  Color,
  EdgesGeometry,
  Group,
  LineBasicMaterial,
  LineSegments,
  LoadingManager,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Object3D,
  Quaternion,
  Sphere,
  Vector3,
  type BufferGeometry,
  type Material,
} from "three"
import URDFLoader, { type URDFRobot } from "urdf-loader"
import { Loader2 } from "lucide-react"
import { apiUrl, urdfUrl, type AxolModel, type GripperType } from "@/lib/supervisor"
import {
  jointValuesFor,
  type JogArm,
  type JogPlan,
  type JogPose,
  type JogPreview,
  type JogState,
} from "@/lib/jog"

const PLAN_COLOR = "#ff9a3c"
const PATH_COLOR = "#eff483"
const EDGE_COLOR = "#0d0f12"

/** What the orange ghost shows: one arm at a solution, and where its tip lands. */
type Ghost = Pick<JogPreview, "arm" | "joints" | "reached">

interface LoadedRobot {
  /** Z-up URDF wrapped in a Y-up group. */
  scene: Group
  robot: URDFRobot
}

function loadRobot(model: AxolModel, gripper: GripperType): Promise<URDFRobot> {
  return new Promise((resolve, reject) => {
    const manager = new LoadingManager()
    const loader = new URDFLoader(manager)
    loader.packages = { assembly: apiUrl("/api/urdf") }
    let robot: URDFRobot | null = null
    let failed = false
    manager.onError = (url) => {
      failed = true
      reject(new Error(`Couldn't load ${url} from the host.`))
    }
    // The manager reports the URDF loaded before the meshes are attached to
    // their links, so wait until every visual has its mesh.
    manager.onLoad = () => {
      const deadline = Date.now() + 15_000
      const check = () => {
        if (failed || !robot) return
        const visuals: Object3D[] = []
        robot.traverse((o) => {
          if ((o as { isURDFVisual?: boolean }).isURDFVisual) visuals.push(o as unknown as Object3D)
        })
        if (visuals.every((v) => v.children.length > 0)) resolve(robot)
        else if (Date.now() > deadline) reject(new Error("The robot model's meshes did not load."))
        else setTimeout(check, 50)
      }
      check()
    }
    loader.load(
      urdfUrl(model, gripper),
      (value) => {
        robot = value
      },
      undefined,
      () => {
        failed = true
        reject(new Error("Couldn't load the robot model from the host."))
      }
    )
  })
}

function disposeMaterial(material: Material | Material[] | undefined) {
  if (Array.isArray(material)) material.forEach((m) => m.dispose())
  else material?.dispose()
}

/** Shaded solids with crease edges, like MoveIt's RViz robot display. */
function styleSolid(root: Group) {
  const meshes: Mesh[] = []
  root.traverse((object) => {
    if ((object as Mesh).isMesh) meshes.push(object as Mesh)
  })
  const edgeMaterial = new LineBasicMaterial({
    color: EDGE_COLOR,
    transparent: true,
    opacity: 0.55,
  })
  for (const mesh of meshes) {
    const source = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material
    const color = new Color((source as MeshStandardMaterial | undefined)?.color ?? "#a9b0b8")
    disposeMaterial(mesh.material)
    mesh.material = new MeshStandardMaterial({
      color,
      roughness: 0.55,
      metalness: 0.15,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    })
    mesh.castShadow = false
    const edges = new LineSegments(
      new EdgesGeometry(mesh.geometry as BufferGeometry, 28),
      edgeMaterial
    )
    edges.raycast = () => undefined
    mesh.add(edges)
  }
}

/** Translucent orange solids for the planned pose. */
function styleGhost(root: Group) {
  const material = new MeshBasicMaterial({
    color: PLAN_COLOR,
    transparent: true,
    opacity: 0.38,
    depthWrite: false,
  })
  root.traverse((object) => {
    const mesh = object as Mesh
    if (!mesh.isMesh) return
    disposeMaterial(mesh.material)
    mesh.material = material
    mesh.raycast = () => undefined
    mesh.renderOrder = 2
  })
}

function wrap(robot: URDFRobot): Group {
  const scene = new Group()
  // URDF is Z-up; three.js is Y-up.
  scene.rotation.x = -Math.PI / 2
  scene.add(robot as unknown as Parameters<Group["add"]>[0])
  return scene
}

/** Show only one arm's links of the planned-pose ghost; true if that changed anything. */
function showArmOnly(robot: URDFRobot, arm: JogArm): boolean {
  let changed = false
  for (const [name, link] of Object.entries(robot.links)) {
    const visible = name.startsWith(`${arm}_`)
    for (const child of link.children)
      if ((child as { isURDFVisual?: boolean }).isURDFVisual && child.visible !== visible) {
        child.visible = visible
        changed = true
      }
  }
  return changed
}

function applyJoints(robot: URDFRobot, values: Record<string, number>) {
  let changed = false
  for (const [name, value] of Object.entries(values))
    if (robot.joints[name]?.setJointValue(value)) changed = true
  if (!changed) return
  robot.updateMatrixWorld(true)
  // The canvas renders on demand, and only a pose that moved is a reason to
  // draw: state arrives at 20 Hz even when the robot is still, and every
  // needless frame delays the page's hold heartbeat.
  invalidate()
}

/** Frame the whole robot: fit its bounding sphere in the vertical field of view. */
function framing(scene: Group, fov: number) {
  scene.updateMatrixWorld(true)
  const box = new Box3().setFromObject(scene)
  const sphere = box.getBoundingSphere(new Sphere())
  const distance = (sphere.radius / Math.sin(((fov / 2) * Math.PI) / 180)) * 1.08
  const direction = new Vector3(1.0, 0.5, 0.9).normalize()
  return {
    target: sphere.center.clone(),
    camera: sphere.center.clone().addScaledVector(direction, distance),
    radius: sphere.radius,
    floor: box.min.y,
  }
}

function poseFrom(matrix: Matrix4): JogPose {
  const p = new Vector3()
  const q = new Quaternion()
  matrix.decompose(p, q, new Vector3())
  return { position: [p.x, p.y, p.z], quaternion: [q.x, q.y, q.z, q.w] }
}

/** Group that tracks the URDF's torso link every frame. */
function TorsoFrame({ robot, children }: { robot: URDFRobot; children: React.ReactNode }) {
  const group = useRef<Group>(null)
  useFrame(() => {
    const root = robot.links.root
    const g = group.current
    if (!root || !g) return
    g.matrix.copy(root.matrixWorld)
    g.matrixWorldNeedsUpdate = true
  })
  return (
    <group ref={group} matrixAutoUpdate={false}>
      {children}
    </group>
  )
}

function Triad({ size = 0.06, opacity = 1 }: { size?: number; opacity?: number }) {
  // Stable points, so a re-render doesn't rebuild the lines or force a frame.
  const axes = useMemo(
    (): [Vector3[], string][] => [
      [[new Vector3(), new Vector3(size, 0, 0)], "#ef4444"],
      [[new Vector3(), new Vector3(0, size, 0)], "#22c55e"],
      [[new Vector3(), new Vector3(0, 0, size)], "#3b82f6"],
    ],
    [size]
  )
  return (
    <>
      {axes.map(([points, color]) => (
        <Line
          key={color}
          points={points}
          color={color}
          lineWidth={2.5}
          transparent
          opacity={opacity}
          depthTest={false}
          renderOrder={5}
        />
      ))}
    </>
  )
}

/**
 * The combined move-and-turn gizmo. Memoised: drei's handles pass fresh
 * inline props on every render, and R3F draws a frame for each, so it must
 * not re-render with the overlay's 20 Hz state.
 */
const TargetGizmo = memo(function TargetGizmo({
  matrix,
  handlers,
}: {
  matrix: Matrix4
  handlers: {
    dragStart: () => void
    drag: (local: Matrix4) => void
    dragEnd: () => void
  }
}) {
  return (
    <PivotControls
      matrix={matrix}
      autoTransform={false}
      disableScaling
      depthTest={false}
      fixed
      scale={120}
      lineWidth={5}
      onDragStart={handlers.dragStart}
      onDrag={handlers.drag}
      onDragEnd={handlers.dragEnd}
    />
  )
})

function Overlay({
  robot,
  tips,
  plan,
  ghost,
  activeArm,
  target,
  onTargetChange,
  onSelectArm,
  onDragging,
}: {
  robot: URDFRobot
  tips: JogState["tips"]
  plan: JogPlan | null
  ghost: Ghost | null
  activeArm: JogArm
  target: JogPose | null
  onTargetChange: (pose: JogPose) => void
  onSelectArm: (arm: JogArm) => void
  onDragging: (dragging: boolean) => void
}) {
  // The target pose in the torso frame; the gizmo moves it in place.
  const targetMatrix = useMemo(() => new Matrix4(), [])
  const dragging = useRef(false)

  // Follow the target prop unless the operator is dragging the gizmo.
  useEffect(() => {
    if (!target || dragging.current) return
    const next = new Matrix4().compose(
      new Vector3(...target.position),
      new Quaternion(...target.quaternion),
      new Vector3(1, 1, 1)
    )
    // The page hands over a fresh target with every state message.
    if (next.equals(targetMatrix)) return
    targetMatrix.copy(next)
    invalidate()
  }, [target, targetMatrix])

  // The canvas draws on demand, and R3F counts a new handler or object prop
  // as a change worth a frame. State arrives at 20 Hz, so everything handed
  // to the scene below stays stable across renders: handlers reach the
  // latest callbacks through a ref.
  const callbacks = useRef({ onTargetChange, onSelectArm, onDragging })
  useLayoutEffect(() => {
    callbacks.current = { onTargetChange, onSelectArm, onDragging }
  })
  const handlers = useMemo(
    () => ({
      dragStart: () => {
        dragging.current = true
        callbacks.current.onDragging(true)
      },
      drag: (local: Matrix4) => {
        targetMatrix.copy(local)
        callbacks.current.onTargetChange(poseFrom(targetMatrix))
      },
      dragEnd: () => {
        dragging.current = false
        callbacks.current.onDragging(false)
        callbacks.current.onTargetChange(poseFrom(targetMatrix))
      },
      select: {
        left: (event: ThreeEvent<PointerEvent>) => {
          event.stopPropagation()
          callbacks.current.onSelectArm("left")
        },
        right: (event: ThreeEvent<PointerEvent>) => {
          event.stopPropagation()
          callbacks.current.onSelectArm("right")
        },
      },
    }),
    [targetMatrix]
  )
  const pathPoints = useMemo(() => plan?.path.map((p) => new Vector3(...p)) ?? [], [plan])
  const ghostTurn = useMemo(
    () => (ghost ? new Quaternion(...ghost.reached.quaternion) : undefined),
    [ghost]
  )
  return (
    <>
      <TorsoFrame robot={robot}>
        {/* One gizmo moves and turns the target: arrows and planes slide it,
          rings turn it, all along the gripper's own axes. Dragging a handle
          pauses the orbit controls. */}
        {target && <TargetGizmo matrix={targetMatrix} handlers={handlers} />}
        {tips &&
          (["left", "right"] as const).map((arm) => (
            <mesh
              key={arm}
              position={tips[arm].position}
              // Only the other arm's marker is clickable, so the active one
              // never sits over the gizmo's inner handles.
              onPointerDown={arm === activeArm ? undefined : handlers.select[arm]}
              renderOrder={4}
            >
              <sphereGeometry args={[arm === activeArm ? 0.022 : 0.03, 20, 14]} />
              <meshBasicMaterial
                color={arm === activeArm ? "#eff483" : "#9ca3af"}
                transparent
                opacity={arm === activeArm ? 0.9 : 0.55}
                depthTest={false}
              />
            </mesh>
          ))}
        {pathPoints.length > 1 && (
          <Line
            points={pathPoints}
            color={PATH_COLOR}
            lineWidth={3}
            dashed
            dashSize={0.02}
            gapSize={0.012}
            depthTest={false}
            renderOrder={6}
          />
        )}
        {ghost && (
          <group position={ghost.reached.position} quaternion={ghostTurn}>
            <Triad size={0.045} opacity={0.9} />
            <mesh renderOrder={6}>
              <sphereGeometry args={[0.012, 16, 12]} />
              <meshBasicMaterial color={PLAN_COLOR} depthTest={false} />
            </mesh>
          </group>
        )}
      </TorsoFrame>
    </>
  )
}

/** Keep the camera on the robot as the base drives: shift it by the torso's
 *  move across the floor, leaving height and orbit angle alone. */
function FollowBase({ robot }: { robot: URDFRobot }) {
  const last = useRef<Vector3 | null>(null)
  useFrame((three) => {
    const root = robot.links.root
    if (!root) return
    const now = root.getWorldPosition(new Vector3())
    const previous = last.current
    last.current = now
    if (!previous) return
    const shift = new Vector3(now.x - previous.x, 0, now.z - previous.z)
    if (shift.lengthSq() === 0) return
    three.camera.position.add(shift)
    const controls = three.controls as unknown as { target?: Vector3; update?: () => void } | null
    controls?.target?.add(shift)
    controls?.update?.()
  })
  return null
}

function CameraRig({ target, radius }: { target: Vector3; radius: number }) {
  return (
    <OrbitControls
      makeDefault
      target={target}
      enableDamping
      dampingFactor={0.12}
      minDistance={radius * 0.2}
      maxDistance={radius * 8}
    />
  )
}

const FOV = 38

/** Everything the scene shows or reports that changes while it runs. */
interface SceneInput {
  state: JogState | null
  plan: JogPlan | null
  preview: Ghost | null
  activeArm: JogArm
  target: JogPose | null
  onTargetChange: (pose: JogPose) => void
  onSelectArm: (arm: JogArm) => void
  onDragging: (dragging: boolean) => void
}

/**
 * Hands the scene its input without re-rendering the canvas. R3F's Canvas
 * draws a frame every time it re-renders, so the page's 20 Hz state must not
 * reach it as props; the scene inside subscribes here instead and draws only
 * when something it shows has moved.
 */
function createSceneStore(initial: SceneInput) {
  let value = initial
  const listeners = new Set<() => void>()
  return {
    get: () => value,
    set(next: SceneInput) {
      value = next
      for (const listener of listeners) listener()
    },
    subscribe(listener: () => void) {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
  }
}
type SceneStore = ReturnType<typeof createSceneStore>

function LiveScene({
  live,
  planned,
  store,
}: {
  live: LoadedRobot
  planned: LoadedRobot
  store: SceneStore
}) {
  const { state, plan, preview, activeArm, target, onTargetChange, onSelectArm, onDragging } =
    useSyncExternalStore(store.subscribe, store.get)
  const liftTravel = Number(live.robot.joints.lift?.limit?.upper ?? 0)
  const values = useMemo(() => jointValuesFor(state, liftTravel), [state, liftTravel])
  useEffect(() => {
    applyJoints(live.robot, values)
  }, [live, values])
  const shown: Ghost | null = plan ?? preview
  useEffect(() => {
    if (!shown) return
    if (showArmOnly(planned.robot, shown.arm)) invalidate()
    applyJoints(planned.robot, jointValuesFor(state, liftTravel, shown))
  }, [planned, shown, state, liftTravel])
  return (
    <>
      <primitive object={live.scene} />
      <primitive object={planned.scene} visible={Boolean(shown)} />
      <Overlay
        robot={live.robot}
        tips={state?.tips}
        plan={plan}
        ghost={shown}
        activeArm={activeArm}
        target={target}
        onTargetChange={onTargetChange}
        onSelectArm={onSelectArm}
        onDragging={onDragging}
      />
    </>
  )
}

const SceneCanvas = memo(function SceneCanvas({
  live,
  planned,
  store,
}: {
  live: LoadedRobot
  planned: LoadedRobot
  store: SceneStore
}) {
  // Frame the whole robot once per model.
  const view = useMemo(() => framing(live.scene, FOV), [live])
  return (
    <Canvas
      data-testid="jog-canvas"
      // Draw only when something changes, so the page's 50 ms hold
      // heartbeat never waits behind a busy render loop.
      frameloop="demand"
      camera={{
        fov: FOV,
        position: view.camera.toArray(),
        near: view.radius / 200,
        far: view.radius * 60,
      }}
      gl={{ antialias: true, preserveDrawingBuffer: true }}
      onCreated={({ gl }) => gl.setClearColor("#0d0d0f", 1)}
    >
      <hemisphereLight args={["#f4f6f8", "#2a2d33", 1.15]} />
      <directionalLight position={[3, 5, 2]} intensity={1.5} />
      <directionalLight position={[-3, 2.5, -2]} intensity={0.45} />
      <LiveScene live={live} planned={planned} store={store} />
      <gridHelper
        args={[12, 48, "#3a3d44", "#24262b"]}
        position={[view.target.x, view.floor, view.target.z]}
      />
      <CameraRig target={view.target} radius={view.radius} />
      <FollowBase robot={live.robot} />
    </Canvas>
  )
})

export default function JogViewer({
  robotModel,
  gripper,
  state,
  plan,
  preview = null,
  activeArm,
  target,
  onTargetChange,
  onSelectArm,
  onDragging = () => undefined,
}: {
  robotModel: AxolModel
  gripper: GripperType
  state: JogState | null
  plan: JogPlan | null
  /** The live IK preview of the target, shown as the ghost until there is a plan. */
  preview?: Ghost | null
  activeArm: JogArm
  target: JogPose | null
  onTargetChange: (pose: JogPose) => void
  onSelectArm: (arm: JogArm) => void
  onDragging?: (dragging: boolean) => void
}) {
  const [live, setLive] = useState<LoadedRobot | null>(null)
  const [planned, setPlanned] = useState<LoadedRobot | null>(null)
  const [error, setError] = useState<string | null>(null)
  const input: SceneInput = {
    state,
    plan,
    preview,
    activeArm,
    target,
    onTargetChange,
    onSelectArm,
    onDragging,
  }
  const [store] = useState(() => createSceneStore(input))
  useLayoutEffect(() => store.set(input))

  useEffect(() => {
    // The page remounts the viewer (key) for another model, so loading
    // starts from empty state.
    let cancelled = false
    Promise.all([loadRobot(robotModel, gripper), loadRobot(robotModel, gripper)])
      .then(([robot, ghostRobot]) => {
        if (cancelled) return
        const scene = wrap(robot)
        styleSolid(scene)
        const ghostScene = wrap(ghostRobot)
        styleGhost(ghostScene)
        setLive({ scene, robot })
        setPlanned({ scene: ghostScene, robot: ghostRobot })
      })
      .catch((reason: Error) => {
        if (!cancelled) setError(reason.message)
      })
    return () => {
      cancelled = true
    }
  }, [robotModel, gripper])

  if (error)
    return (
      <div className="flex h-full items-center justify-center p-6 text-center text-sm text-white/50">
        {error}
      </div>
    )
  if (!live || !planned)
    return (
      <div className="flex h-full items-center justify-center gap-2 text-sm text-white/45">
        <Loader2 className="size-4 animate-spin" /> Loading the robot model…
      </div>
    )
  return <SceneCanvas live={live} planned={planned} store={store} />
}
