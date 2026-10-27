/**
 * Interactive 3D preview of the robot at a given joint configuration.
 *
 * Loads the robot's URDF (+ STL meshes) from the serve host's /api/urdf mount
 * and renders it with react-three-fiber. This module pulls in three.js, so it
 * is only ever imported lazily (React.lazy in pose-panel.tsx) — keep it out of
 * any eagerly-loaded control panel code.
 */

import { useEffect } from "react"
import { Canvas } from "@react-three/fiber"
import { OrbitControls } from "@react-three/drei"
import { Loader2 } from "lucide-react"
import type { AxolModel, GripperType } from "@/lib/supervisor"
import { useUrdfRobot, type JointLimits } from "./urdf-robot"

export type { JointLimits }

export default function PoseViewer({
  jointValues,
  onLoaded,
  robotModel = "classic",
  gripper = "parallel",
  autoRotate = false,
  distance = 1,
}: {
  /** URDF joint name -> angle (rad). */
  jointValues: Record<string, number>
  /** Reports the movable joints' limits once the URDF is in. */
  onLoaded?: (limits: JointLimits) => void
  /** Which Axol version's URDF to show. Fixed per mount: key the viewer by it. */
  robotModel?: AxolModel
  /** The fitted gripper (the mobile Axol has a URDF per gripper). Fixed per mount too. */
  gripper?: GripperType
  /** Slowly turn the model (a showcase turntable); dragging still orbits. */
  autoRotate?: boolean
  /** Camera distance, as a multiple of the default framing. */
  distance?: number
}) {
  const { model, error } = useUrdfRobot(robotModel, gripper, onLoaded)

  useEffect(() => {
    if (!model) return
    for (const [name, value] of Object.entries(jointValues)) {
      model.robot.joints[name]?.setJointValue(value)
    }
  }, [model, jointValues])

  if (error) {
    return (
      <div className="flex h-full items-center justify-center p-4 text-center text-xs text-white/40">
        Couldn&apos;t load the robot model from the Host Computer: {error}
      </div>
    )
  }
  if (!model) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="size-5 animate-spin text-white/30" />
      </div>
    )
  }

  const { center, size } = model
  const camPos: [number, number, number] = [
    center.x + size * 0.9 * distance,
    center.y + size * 0.45 * distance,
    center.z + size * 0.9 * distance,
  ]
  return (
    // Keyed by the framing: a later re-frame remounts with the new camera.
    <Canvas
      key={`${size.toFixed(4)}:${center.toArray().map((v) => v.toFixed(4))}`}
      camera={{ position: camPos, fov: 40, near: size / 100, far: size * 20 }}
    >
      <ambientLight intensity={0.7} />
      <directionalLight position={[2, 4, 3]} intensity={1.6} />
      <directionalLight position={[-3, 2, -2]} intensity={0.5} />
      <primitive object={model.scene} />
      <gridHelper args={[size * 2, 20, "#3a3a3a", "#242424"]} position={model.floor.toArray()} />
      <OrbitControls
        target={center.toArray()}
        enableDamping
        makeDefault
        autoRotate={autoRotate}
        autoRotateSpeed={1.2}
      />
    </Canvas>
  )
}
