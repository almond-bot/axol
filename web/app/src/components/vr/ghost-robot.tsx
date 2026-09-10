/**
 * In-headset "ghost" of the robot at its live joint state.
 *
 * A translucent copy of the URDF, posed from the teleop server's ~20 Hz joint
 * pushes (see `useAxolJoints`), anchored so the robot's shoulders sit roughly
 * where the operator's own are: stand still, look ahead, and the ghost's arms
 * hang where the real arms are relative to *you*. That makes re-engaging after
 * someone hand-guided the arms a matter of putting your hands where the ghost
 * grippers are (or, in box mode, of seeing the pair you're about to lead)
 * instead of guessing from the camera feeds.
 *
 * Anchored once when shown (yaw-only, from the current gaze); toggle it off and
 * on to re-anchor. The URDF and meshes are fetched from the teleop server's
 * own origin (`/urdf/...`), whose certificate the app has already accepted.
 */

import { useEffect, useRef, useState, type RefObject } from "react"
import { useFrame, useThree } from "@react-three/fiber"
import * as THREE from "three"
import URDFLoader, { type URDFRobot } from "urdf-loader"
import type { AxolJointSample } from "@almond/axol-vr-client"

// Robot base frame placement relative to the head (metres). Shoulders in the
// URDF sit 0.86 m above the base origin; a standing operator's shoulders are
// ~0.22 m below the eyes; the base is pushed slightly forward so the torso
// mesh doesn't intersect the head.
const SHOULDER_Z = 0.86
const EYE_TO_SHOULDER = 0.22
const FORWARD_OFFSET = 0.15
// Joint pushes older than this (ms) mean the stream died: hide the ghost
// rather than show a stale pose as if it were live.
const STALE_MS = 1500

const _fwd = new THREE.Vector3()
const _yAxis = new THREE.Vector3(0, 1, 0)

// Two-tone so an engaged (tracking) robot reads differently from a parked one.
const COLOR_IDLE = new THREE.Color("#7dd3fc")
const COLOR_ENGAGED = new THREE.Color("#4ade80")

function ghostMaterial(): THREE.MeshBasicMaterial {
  return new THREE.MeshBasicMaterial({
    color: COLOR_IDLE,
    transparent: true,
    opacity: 0.32,
    depthWrite: false,
    side: THREE.DoubleSide,
  })
}

export function GhostRobot({
  enabled,
  urdfBase,
  jointsRef,
}: {
  enabled: boolean
  /** Origin + path of the URDF directory, e.g. `https://host:8000/urdf`. */
  urdfBase: string
  jointsRef: RefObject<AxolJointSample | null>
}) {
  const { gl } = useThree()
  const anchorRef = useRef<THREE.Group>(null)
  const anchoredRef = useRef(false)
  const [robot, setRobot] = useState<URDFRobot | null>(null)
  const materialRef = useRef<THREE.MeshBasicMaterial | null>(null)
  // Only the first enable triggers the (mesh-heavy) load; later toggles just
  // show/hide and re-anchor.
  // Base the current (or in-flight) load was started from; a load is never
  // cancelled by a disable, only replaced when the base changes.
  const loadedFromRef = useRef<string | null>(null)
  const unmountedRef = useRef(false)

  useEffect(() => {
    if (!enabled || !urdfBase || loadedFromRef.current === urdfBase) return
    loadedFromRef.current = urdfBase
    const manager = new THREE.LoadingManager()
    const loader = new URDFLoader(manager)
    // The URDF references its meshes as package://assembly/meshes/*.stl.
    loader.packages = { assembly: urdfBase }
    let loaded: URDFRobot | null = null
    loader.load(`${urdfBase}/axol.urdf`, (r) => {
      loaded = r
    })
    manager.onError = (url) => {
      console.warn(`ghost robot: failed to load ${url}`)
    }
    manager.onLoad = () => {
      // Stale if the base changed (or we unmounted) while the load was in flight.
      if (unmountedRef.current || loadedFromRef.current !== urdfBase || !loaded) return
      const material = ghostMaterial()
      loaded.traverse((obj) => {
        const mesh = obj as THREE.Mesh
        if (mesh.isMesh) {
          mesh.material = material
          mesh.renderOrder = 5
        }
      })
      materialRef.current?.dispose()
      materialRef.current = material
      setRobot(loaded)
    }
  }, [enabled, urdfBase])

  useEffect(() => {
    unmountedRef.current = false
    return () => {
      unmountedRef.current = true
      materialRef.current?.dispose()
      materialRef.current = null
    }
  }, [])

  useEffect(() => {
    // Re-anchor on every show.
    if (enabled) anchoredRef.current = false
  }, [enabled])

  useFrame(() => {
    const anchor = anchorRef.current
    if (!anchor) return
    const sample = jointsRef.current
    const presenting = gl.xr.isPresenting
    const fresh = sample !== null && performance.now() - sample.receivedAt < STALE_MS
    anchor.visible = enabled && presenting && robot !== null && fresh
    if (!presenting) anchoredRef.current = false
    if (!anchor.visible || !robot || !sample) return

    if (!anchoredRef.current) {
      anchoredRef.current = true
      const cam = gl.xr.getCamera()
      _fwd.set(0, 0, -1).applyQuaternion(cam.quaternion)
      _fwd.y = 0
      if (_fwd.lengthSq() < 1e-6) _fwd.set(0, 0, -1)
      _fwd.normalize()
      anchor.quaternion.setFromAxisAngle(_yAxis, Math.atan2(-_fwd.x, -_fwd.z))
      anchor.position.copy(cam.position).addScaledVector(_fwd, FORWARD_OFFSET)
      anchor.position.y = cam.position.y - EYE_TO_SHOULDER - SHOULDER_Z
      anchor.updateMatrixWorld(true)
    }

    for (const [name, value] of Object.entries(sample.q)) {
      robot.joints[name]?.setJointValue(value)
    }
    const material = materialRef.current
    if (material) material.color.copy(sample.engaged ? COLOR_ENGAGED : COLOR_IDLE)
  })

  return (
    <group ref={anchorRef} visible={false}>
      {/* Robot FLU (+x forward, +y left, +z up) → three.js (-z forward, -x
          left, +y up): Z-up→Y-up about x, then a quarter turn about y. */}
      <group rotation={[0, Math.PI / 2, 0]}>
        <group rotation={[-Math.PI / 2, 0, 0]}>
          {robot && <primitive object={robot as unknown as THREE.Object3D} />}
        </group>
      </group>
    </group>
  )
}
