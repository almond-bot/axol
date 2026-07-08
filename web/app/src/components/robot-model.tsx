import { useEffect, useRef, useState, type RefObject } from "react"
import { useFrame } from "@react-three/fiber"
import * as THREE from "three"
import URDFLoader, { type URDFRobot } from "urdf-loader"
import { axolHttpsOrigin, useAxolUrdfState } from "@almond/axol-vr-client"

const VR_WS_PORT = 8000

// Translucent unlit materials: the operator has to see the physical device
// *through* the virtual robot to judge alignment (and the passthrough scene
// has no lights, so anything lit would render black). The grippers are the
// alignment reference, so they get a distinct highlight color.
function makeBodyMaterial() {
  return new THREE.MeshBasicMaterial({
    color: 0x4fc3f7,
    transparent: true,
    opacity: 0.4,
    depthWrite: false,
  })
}

function makeGripperMaterial() {
  return new THREE.MeshBasicMaterial({
    color: 0xffb74d,
    transparent: true,
    opacity: 0.65,
    depthWrite: false,
  })
}

/**
 * Virtual Axol overlay for absolute (UMI) mode.
 *
 * Fetches the robot URDF + meshes from the connected teleop server
 * (`https://host:8000/urdf/`) and renders it in the passthrough scene at the
 * base transform the server calibrated at engage, with joints driven by the
 * live IK solution (`useAxolUrdfState`). Hidden until the first engage (the
 * server sends `base: null` before calibration).
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
  const groupRef = useRef<THREE.Group>(null)

  useEffect(() => {
    if (!hostname) return
    const origin = axolHttpsOrigin(hostname, VR_WS_PORT)
    const loader = new URDFLoader()
    // The URDF references meshes as package://assembly/meshes/<name>.stl;
    // the server exposes the whole urdf directory at /urdf.
    loader.packages = { assembly: `${origin}/urdf` }
    let cancelled = false
    loader.load(
      `${origin}/urdf/axol.urdf`,
      (loaded) => {
        if (cancelled) return
        const body = makeBodyMaterial()
        const gripper = makeGripperMaterial()
        loaded.traverse((obj) => {
          const mesh = obj as THREE.Mesh
          if (mesh.isMesh) mesh.material = body
        })
        for (const linkName of ["left_gripper", "right_gripper"]) {
          loaded.links[linkName]?.traverse((obj) => {
            const mesh = obj as THREE.Mesh
            if (mesh.isMesh) mesh.material = gripper
          })
        }
        setRobot(loaded)
      },
      undefined,
      (err) => console.warn("failed to load robot URDF from", origin, err)
    )
    return () => {
      cancelled = true
      setRobot(null)
    }
  }, [hostname])

  useFrame(() => {
    const group = groupRef.current
    if (!group) return
    const state = urdfStateRef.current
    if (!robot || !state?.base) {
      group.visible = false
      return
    }
    group.visible = true
    group.position.set(...state.base.pos)
    group.quaternion.set(...state.base.quat)
    for (const [name, value] of Object.entries(state.joints)) {
      robot.joints[name]?.setJointValue(value)
    }
  })

  return (
    <group ref={groupRef} visible={false}>
      {robot ? <primitive object={robot} /> : null}
    </group>
  )
}
