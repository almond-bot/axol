/**
 * 3D view for the Jog page: the robot's URDF at its live pose, a transform
 * gizmo on the target gripper pose, and an orange ghost of the arm: the
 * planned move's solution with a dashed path from the gripper, or, before a
 * plan, the live IK preview of the target being dragged.
 *
 * Every pose the page exchanges with the jog session is in Axol's torso frame
 * (the URDF's `root` link). The overlay group below follows that link, so
 * target, path and markers sit in the same frame IK uses wherever the lift
 * and base have put the torso.
 */
import { useEffect, useMemo, useRef, useState } from "react"
import { Canvas, invalidate, useFrame, type ThreeEvent } from "@react-three/fiber"
import { Line, OrbitControls, TransformControls } from "@react-three/drei"
import {
  Box3,
  Color,
  EdgesGeometry,
  Group,
  LineBasicMaterial,
  LineSegments,
  LoadingManager,
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

export type GizmoMode = "translate" | "rotate"
export type GizmoSpace = "world" | "local"
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

/** Show only one arm's links of the planned-pose ghost. */
function showArmOnly(robot: URDFRobot, arm: JogArm) {
  for (const [name, link] of Object.entries(robot.links)) {
    const visible = name.startsWith(`${arm}_`)
    for (const child of link.children)
      if ((child as { isURDFVisual?: boolean }).isURDFVisual) child.visible = visible
  }
}

function applyJoints(robot: URDFRobot, values: Record<string, number>) {
  for (const [name, value] of Object.entries(values)) robot.joints[name]?.setJointValue(value)
  robot.updateMatrixWorld(true)
  // The canvas renders on demand; a new pose is a reason to draw.
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

function poseFrom(object: Object3D): JogPose {
  const p = object.position
  const q = object.quaternion
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
  const axes: [Vector3, string][] = [
    [new Vector3(size, 0, 0), "#ef4444"],
    [new Vector3(0, size, 0), "#22c55e"],
    [new Vector3(0, 0, size), "#3b82f6"],
  ]
  return (
    <>
      {axes.map(([end, color]) => (
        <Line
          key={color}
          points={[new Vector3(), end]}
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

function Overlay({
  robot,
  state,
  plan,
  ghost,
  activeArm,
  target,
  gizmoMode,
  gizmoSpace,
  onTargetChange,
  onSelectArm,
  onDragging,
}: {
  robot: URDFRobot
  state: JogState | null
  plan: JogPlan | null
  ghost: Ghost | null
  activeArm: JogArm
  target: JogPose | null
  gizmoMode: GizmoMode
  gizmoSpace: GizmoSpace
  onTargetChange: (pose: JogPose) => void
  onSelectArm: (arm: JogArm) => void
  onDragging: (dragging: boolean) => void
}) {
  const targetObject = useMemo(() => new Object3D(), [])
  const dragging = useRef(false)

  // Follow the target prop unless the operator is dragging the gizmo.
  useEffect(() => {
    if (!target || dragging.current) return
    targetObject.position.set(...target.position)
    targetObject.quaternion.set(...target.quaternion)
    targetObject.updateMatrixWorld(true)
  }, [target, targetObject])

  const tips = state?.tips
  const select = (arm: JogArm) => (event: ThreeEvent<PointerEvent>) => {
    event.stopPropagation()
    onSelectArm(arm)
  }
  return (
    <>
      <TorsoFrame robot={robot}>
        <primitive object={targetObject}>{target && <Triad size={0.07} />}</primitive>
        {tips &&
          (["left", "right"] as const).map((arm) => (
            <mesh
              key={arm}
              position={tips[arm].position}
              onPointerDown={select(arm)}
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
        {plan && plan.path.length > 1 && (
          <Line
            points={plan.path.map((p) => new Vector3(...p))}
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
          <group
            position={ghost.reached.position}
            quaternion={new Quaternion(...ghost.reached.quaternion)}
          >
            <Triad size={0.045} opacity={0.9} />
            <mesh renderOrder={6}>
              <sphereGeometry args={[0.012, 16, 12]} />
              <meshBasicMaterial color={PLAN_COLOR} depthTest={false} />
            </mesh>
          </group>
        )}
      </TorsoFrame>
      {/* The gizmo places itself in world space from the target's world
        matrix, so it lives at the scene root, not inside the torso frame. */}
      {target && (
        <TransformControls
          object={targetObject}
          mode={gizmoMode}
          space={gizmoSpace}
          size={0.75}
          onMouseDown={() => {
            dragging.current = true
            onDragging(true)
          }}
          onMouseUp={() => {
            dragging.current = false
            onDragging(false)
            onTargetChange(poseFrom(targetObject))
          }}
          onObjectChange={() => onTargetChange(poseFrom(targetObject))}
        />
      )}
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

export default function JogViewer({
  robotModel,
  gripper,
  state,
  plan,
  preview = null,
  activeArm,
  target,
  gizmoMode,
  gizmoSpace,
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
  gizmoMode: GizmoMode
  gizmoSpace: GizmoSpace
  onTargetChange: (pose: JogPose) => void
  onSelectArm: (arm: JogArm) => void
  onDragging?: (dragging: boolean) => void
}) {
  const [live, setLive] = useState<LoadedRobot | null>(null)
  const [ghost, setGhost] = useState<LoadedRobot | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // The page remounts the viewer (key) for another model, so loading
    // starts from empty state.
    let cancelled = false
    Promise.all([loadRobot(robotModel, gripper), loadRobot(robotModel, gripper)])
      .then(([robot, planned]) => {
        if (cancelled) return
        const scene = wrap(robot)
        styleSolid(scene)
        const ghostScene = wrap(planned)
        styleGhost(ghostScene)
        setLive({ scene, robot })
        setGhost({ scene: ghostScene, robot: planned })
      })
      .catch((reason: Error) => {
        if (!cancelled) setError(reason.message)
      })
    return () => {
      cancelled = true
    }
  }, [robotModel, gripper])

  const liftTravel = Number(live?.robot.joints.lift?.limit?.upper ?? 0)
  const values = useMemo(() => jointValuesFor(state, liftTravel), [state, liftTravel])
  useEffect(() => {
    if (live) applyJoints(live.robot, values)
  }, [live, values])
  const shown: Ghost | null = plan ?? preview
  useEffect(() => {
    if (!ghost || !shown) return
    showArmOnly(ghost.robot, shown.arm)
    applyJoints(ghost.robot, jointValuesFor(state, liftTravel, shown))
  }, [ghost, shown, state, liftTravel])

  // Frame the whole robot once per model.
  const view = useMemo(() => (live ? framing(live.scene, FOV) : null), [live])

  if (error)
    return (
      <div className="flex h-full items-center justify-center p-6 text-center text-sm text-white/50">
        {error}
      </div>
    )
  if (!live || !ghost || !view)
    return (
      <div className="flex h-full items-center justify-center gap-2 text-sm text-white/45">
        <Loader2 className="size-4 animate-spin" /> Loading the robot model…
      </div>
    )
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
      <primitive object={live.scene} />
      <primitive object={ghost.scene} visible={Boolean(shown)} />
      <Overlay
        robot={live.robot}
        state={state}
        plan={plan}
        ghost={shown}
        activeArm={activeArm}
        target={target}
        gizmoMode={gizmoMode}
        gizmoSpace={gizmoSpace}
        onTargetChange={onTargetChange}
        onSelectArm={onSelectArm}
        onDragging={onDragging}
      />
      <gridHelper
        args={[12, 48, "#3a3d44", "#24262b"]}
        position={[view.target.x, view.floor, view.target.z]}
      />
      <CameraRig target={view.target} radius={view.radius} />
      <FollowBase robot={live.robot} />
    </Canvas>
  )
}
