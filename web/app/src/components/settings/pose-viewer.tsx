/**
 * Interactive 3D preview of the robot at a given joint configuration.
 *
 * Loads the robot's URDF (+ STL meshes) from the serve host's /api/urdf mount
 * and renders it with react-three-fiber. This module pulls in three.js, so it
 * is only ever imported lazily (React.lazy in pose-panel.tsx) — keep it out of
 * any eagerly-loaded control panel code.
 */

import { useEffect, useRef, useState } from "react"
import { Canvas } from "@react-three/fiber"
import { OrbitControls } from "@react-three/drei"
import { Box3, Group, LoadingManager, Vector3 } from "three"
import URDFLoader, { type URDFRobot } from "urdf-loader"
import { Loader2 } from "lucide-react"
import { apiUrl, urdfUrl } from "@/lib/supervisor"

export interface JointLimits {
  [jointName: string]: { lower: number; upper: number }
}

interface LoadedModel {
  /** The robot wrapped in a Z-up→Y-up group, ready to render. */
  scene: Group
  robot: URDFRobot
  center: Vector3
  /** Rough model diameter, for camera placement. */
  size: number
}

export default function PoseViewer({
  jointValues,
  onLoaded,
}: {
  /** URDF joint name -> angle (rad). */
  jointValues: Record<string, number>
  /** Reports the movable joints' limits once the URDF is in. */
  onLoaded?: (limits: JointLimits) => void
}) {
  const [model, setModel] = useState<LoadedModel | null>(null)
  const [error, setError] = useState<string | null>(null)
  const onLoadedRef = useRef(onLoaded)
  useEffect(() => {
    onLoadedRef.current = onLoaded
  }, [onLoaded])

  useEffect(() => {
    let cancelled = false
    const manager = new LoadingManager()
    const loader = new URDFLoader(manager)
    // The URDF references its meshes as package://assembly/meshes/*.stl; the
    // serve host exposes that directory at /api/urdf.
    loader.packages = { assembly: apiUrl("/api/urdf") }
    let robot: URDFRobot | null = null
    loader.load(urdfUrl(), (r) => {
      robot = r
    })
    manager.onError = (url) => {
      if (!cancelled) setError(`failed to load ${url}`)
    }
    // Meshes stream in through the manager after the URDF callback; only when
    // everything is in can we size the camera to the real bounding box.
    manager.onLoad = () => {
      if (cancelled || !robot) return
      const scene = new Group()
      scene.rotation.x = -Math.PI / 2 // URDF is Z-up; three.js is Y-up.
      // urdf-loader resolves a second @types/three; structurally identical.
      scene.add(robot as unknown as Parameters<Group["add"]>[0])
      scene.updateWorldMatrix(true, true)
      const box = new Box3().setFromObject(scene)
      const center = box.getCenter(new Vector3())
      const size = Math.max(box.getSize(new Vector3()).length(), 0.1)
      const limits: JointLimits = {}
      for (const [name, joint] of Object.entries(robot.joints)) {
        if (joint.jointType === "fixed") continue
        limits[name] = { lower: Number(joint.limit.lower), upper: Number(joint.limit.upper) }
      }
      setModel({ scene, robot, center, size })
      onLoadedRef.current?.(limits)
    }
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!model) return
    for (const [name, value] of Object.entries(jointValues)) {
      model.robot.joints[name]?.setJointValue(value)
    }
  }, [model, jointValues])

  if (error) {
    return (
      <div className="flex h-full items-center justify-center p-4 text-center text-xs text-white/40">
        Couldn&apos;t load the robot model from the serve host: {error}
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
    center.x + size * 0.9,
    center.y + size * 0.45,
    center.z + size * 0.9,
  ]
  return (
    <Canvas camera={{ position: camPos, fov: 40, near: size / 100, far: size * 20 }}>
      <ambientLight intensity={0.7} />
      <directionalLight position={[2, 4, 3]} intensity={1.6} />
      <directionalLight position={[-3, 2, -2]} intensity={0.5} />
      <primitive object={model.scene} />
      <gridHelper args={[size * 2, 20, "#3a3a3a", "#242424"]} />
      <OrbitControls target={center.toArray()} enableDamping makeDefault />
    </Canvas>
  )
}
