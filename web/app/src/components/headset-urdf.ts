import { useEffect, useRef, useState } from "react"
import * as THREE from "three"
import URDFLoader, { type URDFRobot } from "urdf-loader"
import { axolHttpsOrigin } from "@almond/axol-vr-client"

const VR_WS_PORT = 8000

const INITIAL_RETRY_MS = 3000
const MAX_RETRY_MS = 30_000

// Collada assets may carry textures even though the headset views replace
// them with their own materials. Dispose those GPU resources along with the
// material instead of retaining an unreachable source material.
export function disposeMaterial(material: THREE.Material) {
  for (const value of Object.values(material)) {
    if (value instanceof THREE.Texture) value.dispose()
  }
  material.dispose()
}

export function disposeRobot(root: THREE.Object3D) {
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

/**
 * Load a robot URDF + meshes from the connected teleop server
 * (`https://host:8000/urdf/<file>`) for a headset view.
 *
 * `prepare` restyles the robot once every mesh is in (urdf-loader attaches
 * meshes asynchronously, well after the URDF parse callback). A one-shot load
 * can race the teleop server's startup or the self-signed cert authorization
 * and then never recover, so a failed load retries with bounded backoff until
 * it succeeds or the inputs change. `null` until loaded, and while `urdfFile`
 * is null. The robot is disposed on unmount or when the inputs change.
 */
export function useHeadsetUrdf(
  hostname: string,
  urdfFile: string | null,
  prepare: (robot: URDFRobot) => void
): URDFRobot | null {
  const [robot, setRobot] = useState<URDFRobot | null>(null)
  const prepareRef = useRef(prepare)
  useEffect(() => {
    prepareRef.current = prepare
  }, [prepare])

  useEffect(() => {
    if (!hostname || !urdfFile) return
    const origin = axolHttpsOrigin(hostname, VR_WS_PORT)
    let cancelled = false
    let retryTimer: ReturnType<typeof setTimeout> | null = null
    let activeManager: THREE.LoadingManager | null = null
    let activeUrdfRequest: AbortController | null = null
    let live: URDFRobot | null = null
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

    function attempt() {
      if (cancelled) return
      // The manager's onLoad fires once every referenced mesh has arrived,
      // which is the first moment the materials can be restyled.
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
        prepareRef.current(loaded)
        live = loaded
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
      // The classic URDF uses package://assembly/meshes/..., the generated
      // ones paths relative to themselves; the server exposes the whole URDF
      // directory at /urdf, so both resolve.
      loader.packages = { assembly: `${origin}/urdf` }
      loader.load(
        `${origin}/urdf/${encodeURIComponent(urdfFile!)}`,
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
      if (live) disposeRobot(live)
      setRobot(null)
    }
  }, [hostname, urdfFile])

  return robot
}
