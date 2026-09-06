import { useEffect, useRef, useState, type RefObject } from "react"
import { useFrame } from "@react-three/fiber"
import * as THREE from "three"
import URDFLoader, { type URDFRobot } from "urdf-loader"
import { axolHttpsOrigin, useAxolUrdfState } from "@almond/axol-vr-client"

const VR_WS_PORT = 8000

const INITIAL_RETRY_MS = 3000
const MAX_RETRY_MS = 30_000

// Overlay translucency: the operator must see the physical device *through*
// the virtual robot to judge alignment.
const OVERLAY_OPACITY = 0.55

// Time constant (s) of the per-frame exponential smoothing toward the latest
// server target. The server streams ~60 Hz; smoothing at the headset's render
// rate hides the discrete updates without adding perceptible lag.
const SMOOTHING_TAU = 0.04

// Convert every mesh material to an unlit translucent copy that keeps the
// URDF's own colors (the passthrough scene has no lights, so anything lit
// renders black). Must run after the meshes have actually loaded — urdf-loader
// attaches them asynchronously, well after the URDF parse callback.
function disposeMaterial(material: THREE.Material) {
  // Collada assets may carry textures even though the overlay replaces them
  // with an unlit solid colour. Dispose those GPU resources along with the
  // material instead of retaining an unreachable source material.
  for (const value of Object.values(material)) {
    if (value instanceof THREE.Texture) value.dispose()
  }
  material.dispose()
}

function disposeRobot(root: THREE.Object3D) {
  const geometries = new Set<THREE.BufferGeometry>()
  const materials = new Set<THREE.Material>()
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh
    if (!mesh.isMesh) return
    if (mesh.geometry) geometries.add(mesh.geometry)
    const meshMaterials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
    for (const material of meshMaterials) {
      if (material) materials.add(material)
    }
  })
  for (const geometry of geometries) geometry.dispose()
  for (const material of materials) disposeMaterial(material)
}

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
  const [robot, setRobot] = useState<URDFRobot | null>(null)
  // An ordinary Axol session never publishes urdf_state, so wait for the
  // server to prove this is an overlay-capable Mantis session before fetching
  // the robot model.
  const [overlayRequested, setOverlayRequested] = useState(false)
  const overlayRequestedRef = useRef(false)
  const liveRobotRef = useRef<URDFRobot | null>(null)
  const groupRef = useRef<THREE.Group>(null)

  // Smoothed render state, advanced toward the latest server target each
  // frame. Snaps on the first target after (re)appearing.
  const smoothPos = useRef(new THREE.Vector3())
  const smoothQuat = useRef(new THREE.Quaternion())
  const smoothJoints = useRef<Record<string, number>>({})
  const trackingRef = useRef(false)

  useEffect(() => {
    if (!hostname || !overlayRequested) return
    const origin = axolHttpsOrigin(hostname, VR_WS_PORT)
    let cancelled = false
    let retryTimer: ReturnType<typeof setTimeout> | null = null
    let activeManager: THREE.LoadingManager | null = null
    let activeUrdfRequest: AbortController | null = null
    let retryMs = INITIAL_RETRY_MS

    const scheduleRetry = () => {
      if (cancelled || retryTimer !== null) return
      const delay = retryMs
      retryMs = Math.min(retryMs * 2, MAX_RETRY_MS)
      retryTimer = setTimeout(() => {
        retryTimer = null
        attempt()
      }, delay)
    }

    // A one-shot load can race the teleop server's startup or the self-signed
    // cert authorization and then never recover ("URDF doesn't appear until I
    // restart the app") — retry with bounded backoff until it loads or the
    // component unmounts.
    function attempt() {
      if (cancelled) return
      // Meshes load asynchronously after the URDF parse callback; the
      // manager's onLoad fires once every referenced mesh has arrived, which
      // is the first moment the overlay materials can actually be applied.
      let loaded: URDFRobot | null = null
      let failed = false
      const manager = new THREE.LoadingManager(() => {
        if (activeManager === manager) activeManager = null
        activeUrdfRequest = null
        if (cancelled) {
          if (loaded) disposeRobot(loaded)
          return
        }
        if (!loaded || failed) {
          if (loaded) disposeRobot(loaded)
          scheduleRetry()
          return
        }
        applyOverlayMaterials(loaded)
        liveRobotRef.current = loaded
        setRobot(loaded)
      })
      manager.onError = () => {
        failed = true
      }
      activeManager = manager
      activeUrdfRequest = new AbortController()
      const loader = new URDFLoader(manager)
      loader.fetchOptions = { signal: activeUrdfRequest.signal }
      loader.parseCollision = false
      // The longstanding URDF uses package://assembly/meshes/<name>.stl; the
      // server exposes the whole URDF directory at /urdf.
      loader.packages = { assembly: `${origin}/urdf` }
      loader.load(
        `${origin}/urdf/axol.urdf`,
        (r) => {
          loaded = r
        },
        undefined,
        // Retry scheduling happens in the manager's onLoad (which also fires
        // after errored items end), so this only logs — scheduling here too
        // would fork parallel retry loops.
        (err) => {
          failed = true
          if (!cancelled) {
            console.warn("failed to load robot URDF from", origin, "- retrying", err)
          }
        }
      )
    }
    attempt()

    return () => {
      cancelled = true
      if (retryTimer !== null) clearTimeout(retryTimer)
      activeUrdfRequest?.abort()
      activeManager?.abort()
      if (liveRobotRef.current) {
        disposeRobot(liveRobotRef.current)
        liveRobotRef.current = null
      }
    }
  }, [hostname, overlayRequested])

  useFrame((_, delta) => {
    const group = groupRef.current
    if (!group) return
    const state = urdfStateRef.current
    if (state && !overlayRequestedRef.current) {
      overlayRequestedRef.current = true
      setOverlayRequested(true)
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
