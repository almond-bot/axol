import { useRef, useState, type RefObject } from "react"
import { useFrame } from "@react-three/fiber"
import * as THREE from "three"
import { type URDFRobot } from "urdf-loader"
import { useAxolUrdfState } from "@almond/axol-vr-client"
import { disposeMaterial, useHeadsetUrdf } from "@/components/headset-urdf"

// Overlay translucency: the operator must see the physical device *through*
// the virtual robot to judge alignment.
const OVERLAY_OPACITY = 0.55

// Time constant (s) of the per-frame exponential smoothing toward the latest
// server target. The server streams ~60 Hz; smoothing at the headset's render
// rate hides the discrete updates without adding perceptible lag.
const SMOOTHING_TAU = 0.04

// Convert every mesh material to an unlit translucent copy that keeps the
// URDF's own colors (the passthrough scene has no lights, so anything lit
// renders black).
function applyOverlayMaterials(root: THREE.Object3D) {
  const cache = new Map<number, THREE.MeshBasicMaterial>()
  const replaced = new Set<THREE.Material>()
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh
    if (!mesh.isMesh) return
    const sources = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
    const source = sources[0]
    const color = (source as THREE.MeshStandardMaterial | undefined)?.color?.getHex() ?? 0x888888
    let mat = cache.get(color)
    if (!mat) {
      mat = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: OVERLAY_OPACITY,
        depthWrite: false,
      })
      cache.set(color, mat)
    }
    for (const oldMaterial of sources) {
      if (oldMaterial) replaced.add(oldMaterial)
    }
    mesh.material = mat
  })
  for (const material of replaced) disposeMaterial(material)
}

const _targetPos = new THREE.Vector3()
const _targetQuat = new THREE.Quaternion()

/**
 * Virtual Axol overlay for absolute (Mantis) mode.
 *
 * Fetches the robot URDF + meshes from the connected teleop server
 * (`https://host:8000/urdf/`) and renders it in the passthrough scene at the
 * base transform the server calibrated at engage, with arm joints driven by
 * the live `urdf_state` stream (`useAxolUrdfState`). Hidden until the first
 * engage (the server sends `base: null` before calibration), and while an
 * external tracker world has not been registered to the viewer.
 *
 * This is the hardware↔URDF alignment check: at engage the virtual grippers
 * should coincide with the physical devices, and stay on them as you move —
 * any drift while translating/rotating exposes mount-transform or frame
 * errors.
 */
export function RobotModel({
  hostname,
  wsRef,
}: {
  hostname: string
  wsRef: RefObject<WebSocket | null>
}) {
  const urdfStateRef = useAxolUrdfState(wsRef)
  // An ordinary Axol session never publishes urdf_state, so wait for the
  // server to prove this is an overlay-capable Mantis session before fetching
  // the robot model — the first state also names this Axol version's URDF.
  const [overlayUrdf, setOverlayUrdf] = useState<string | null>(null)
  const robot: URDFRobot | null = useHeadsetUrdf(hostname, overlayUrdf, applyOverlayMaterials)
  const overlayRequestedRef = useRef(false)
  const groupRef = useRef<THREE.Group>(null)

  // Smoothed render state, advanced toward the latest server target each
  // frame. Snaps on the first target after (re)appearing.
  const smoothPos = useRef(new THREE.Vector3())
  const smoothQuat = useRef(new THREE.Quaternion())
  const smoothJoints = useRef<Record<string, number>>({})
  const trackingRef = useRef(false)

  useFrame((_, delta) => {
    const group = groupRef.current
    if (!group) return
    const state = urdfStateRef.current
    if (state && !overlayRequestedRef.current) {
      overlayRequestedRef.current = true
      setOverlayUrdf(state.urdf)
    }
    if (!robot || !state?.base || !state.viewerWorldAligned) {
      group.visible = false
      trackingRef.current = false
      return
    }
    group.visible = true

    _targetPos.set(...state.base.pos)
    _targetQuat.set(...state.base.quat)

    if (!trackingRef.current) {
      // First target after (re)appearing: snap instead of gliding in.
      trackingRef.current = true
      smoothPos.current.copy(_targetPos)
      smoothQuat.current.copy(_targetQuat)
      smoothJoints.current = { ...state.joints }
    } else {
      const alpha = 1 - Math.exp(-delta / SMOOTHING_TAU)
      smoothPos.current.lerp(_targetPos, alpha)
      smoothQuat.current.slerp(_targetQuat, alpha)
      for (const [name, value] of Object.entries(state.joints)) {
        const prev = smoothJoints.current[name] ?? value
        smoothJoints.current[name] = prev + (value - prev) * alpha
      }
    }

    group.position.copy(smoothPos.current)
    group.quaternion.copy(smoothQuat.current)
    for (const [name, value] of Object.entries(smoothJoints.current)) {
      robot.joints[name]?.setJointValue(value)
    }
  })

  return (
    <group ref={groupRef} visible={false}>
      {robot ? <primitive object={robot} /> : null}
    </group>
  )
}
