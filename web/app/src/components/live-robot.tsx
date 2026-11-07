import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from "react"
import { useFrame } from "@react-three/fiber"
import * as THREE from "three"
import { type URDFRobot } from "urdf-loader"
import { type AxolJointSample } from "@almond/axol-vr-client"
import { disposeMaterial, useHeadsetUrdf } from "@/components/headset-urdf"
import { LIVE_ROBOT_HEIGHT, liveRobotTargets } from "@/lib/live-robot"

// Time constant (s) of the per-frame smoothing toward the latest joint push.
// The server pushes at ~20 Hz; smoothing at the headset's render rate hides
// the steps without adding noticeable lag.
const SMOOTHING_TAU = 0.06

// Shaded copies of the URDF's own colors. A few lights inside the model's
// group light it; everything else in the headset scene is unlit.
function styleLive(robot: URDFRobot) {
  const cache = new Map<number, THREE.MeshStandardMaterial>()
  const replaced = new Set<THREE.Material>()
  robot.traverse((obj) => {
    const mesh = obj as THREE.Mesh
    if (!mesh.isMesh) return
    const sources = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
    const color =
      (sources[0] as THREE.MeshStandardMaterial | undefined)?.color?.getHex() ?? 0xa9b0b8
    let mat = cache.get(color)
    if (!mat) {
      mat = new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.1 })
      cache.set(color, mat)
    }
    for (const old of sources) if (old) replaced.add(old)
    mesh.material = mat
    mesh.raycast = () => undefined
  })
  for (const material of replaced) disposeMaterial(material)
}

interface Fitted {
  /** The robot, turned to face away from the viewer and sized to fit. */
  root: THREE.Group
  /** Size and centre of the fitted model, for the grab target. */
  size: THREE.Vector3
  center: THREE.Vector3
}

/** Wrap the robot so its floor sits at the origin, facing -Z, LIVE_ROBOT_HEIGHT tall. */
function fit(robot: URDFRobot): Fitted {
  for (const joint of Object.values(robot.joints)) joint.setJointValue(0)
  robot.removeFromParent()
  robot.position.set(0, 0, 0)
  robot.updateMatrixWorld(true)
  const box = new THREE.Box3().setFromObject(
    robot as unknown as Parameters<THREE.Box3["setFromObject"]>[0]
  )
  const center = box.getCenter(new THREE.Vector3())
  // Floor at the origin, centred over it (URDF frame, Z up).
  robot.position.set(-center.x, -center.y, -box.min.z)
  // URDF is Z-up with +X forward; turn it Y-up, then about Y so its front
  // faces away from the viewer (-Z) and its left arm is on their left.
  const zUp = new THREE.Group()
  zUp.rotation.x = -Math.PI / 2
  zUp.add(robot as unknown as Parameters<THREE.Group["add"]>[0])
  const turn = new THREE.Group()
  turn.rotation.y = Math.PI / 2
  turn.add(zUp)
  const root = new THREE.Group()
  const height = box.max.z - box.min.z
  root.scale.setScalar(height > 1e-3 ? LIVE_ROBOT_HEIGHT / height : 1)
  root.add(turn)
  root.updateMatrixWorld(true)
  const fitted = new THREE.Box3().setFromObject(root)
  return {
    root,
    size: fitted.getSize(new THREE.Vector3()),
    center: fitted.getCenter(new THREE.Vector3()),
  }
}

/**
 * A small live model of the robot for the headset, showing its measured
 * pose: both arms, the grippers and Jelly's lift, from the teleop server's
 * ~20 Hz `joints` push (`useAxolJoints`). The URDF comes from the server too,
 * named by the push. Hidden until the model has loaded and while no push is
 * arriving.
 *
 * The caller places it: `groupRef` is the model's group (floor at its origin,
 * front facing -Z) and `hitRef` an invisible box around the model to point
 * the controllers at.
 */
export function LiveRobot({
  hostname,
  jointsRef,
  groupRef,
  hitRef,
}: {
  hostname: string
  jointsRef: RefObject<AxolJointSample | null>
  groupRef: RefObject<THREE.Group | null>
  hitRef: RefObject<THREE.Mesh | null>
}) {
  const [urdf, setUrdf] = useState<string | null>(null)
  const prepare = useCallback((robot: URDFRobot) => styleLive(robot), [])
  const robot = useHeadsetUrdf(hostname, urdf, prepare)
  const fitted = useMemo(() => (robot ? fit(robot) : null), [robot])
  // The key light aims at the model's floor, wherever the group is placed.
  const lightTarget = useMemo(() => new THREE.Group(), [])
  const lightRef = useRef<THREE.DirectionalLight>(null)
  useEffect(() => {
    if (lightRef.current) lightRef.current.target = lightTarget
  }, [fitted, lightTarget])
  const smoothRef = useRef<Record<string, number> | null>(null)

  useFrame((_, delta) => {
    const sample = jointsRef.current
    if (sample?.urdf && sample.urdf !== urdf) setUrdf(sample.urdf)
    const group = groupRef.current
    if (!group) return
    if (!robot || !fitted || !sample || sample.urdf !== urdf) {
      group.visible = false
      smoothRef.current = null
      return
    }
    group.visible = true
    const targets = liveRobotTargets(sample, robot)
    const smooth = smoothRef.current
    if (!smooth) {
      // First push after (re)appearing: snap instead of gliding in.
      smoothRef.current = { ...targets }
    } else {
      const alpha = 1 - Math.exp(-delta / SMOOTHING_TAU)
      for (const [name, value] of Object.entries(targets)) {
        const prev = smooth[name] ?? value
        smooth[name] = prev + (value - prev) * alpha
      }
    }
    for (const [name, value] of Object.entries(smoothRef.current!)) {
      robot.joints[name]?.setJointValue(value)
    }
  })

  return (
    <group ref={groupRef} visible={false}>
      {fitted ? (
        <>
          <primitive object={fitted.root} />
          <mesh ref={hitRef} position={fitted.center}>
            <boxGeometry args={[fitted.size.x, fitted.size.y, fitted.size.z]} />
            <meshBasicMaterial visible={false} />
          </mesh>
          <primitive object={lightTarget} />
          <hemisphereLight args={["#ffffff", "#444444", 1.5]} />
          <directionalLight ref={lightRef} position={[0.3, 1, 1]} intensity={1.5} />
        </>
      ) : null}
    </group>
  )
}
