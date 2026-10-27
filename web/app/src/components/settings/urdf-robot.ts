/**
 * Load an Axol version's robot model (URDF + STL meshes) from the Host
 * Computer's /api/urdf mount for a three.js scene. This module pulls in
 * three.js, so only lazily-loaded viewers import it.
 */

import { useEffect, useRef, useState } from "react"
import { Box3, Group, LoadingManager, Vector3 } from "three"
import URDFLoader, { type URDFRobot } from "urdf-loader"
import { apiUrl, urdfUrl, type AxolModel, type GripperType } from "@/lib/supervisor"

export interface JointLimits {
  [jointName: string]: { lower: number; upper: number }
}

export interface LoadedModel {
  /** The robot wrapped in a Z-up→Y-up group, ready to render. */
  scene: Group
  robot: URDFRobot
  center: Vector3
  /** Rough model diameter, for camera placement. */
  size: number
  /** Where the ground plane goes (three.js coords): the URDF's `floor` frame
   *  when it has one (Axol Mobile), else the world origin. */
  floor: Vector3
}

/**
 * Load an Axol version's URDF and meshes from the Host Computer, framed for a
 * camera once every mesh is in. Shared by the pose viewer and the dataset
 * replay; each call loads its own copy, so callers may add to its scene.
 */
export function useUrdfRobot(
  robotModel: AxolModel,
  gripper: GripperType,
  onLoaded?: (limits: JointLimits) => void
): { model: LoadedModel | null; error: string | null } {
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
    // The classic URDF references its meshes as package://assembly/meshes/*.stl,
    // the generated ones relative to themselves; the serve host exposes that
    // directory at /api/urdf, so both resolve.
    loader.packages = { assembly: apiUrl("/api/urdf") }
    let robot: URDFRobot | null = null
    loader.load(urdfUrl(robotModel, gripper), (r) => {
      robot = r
    })
    manager.onError = (url) => {
      if (!cancelled) setError(url)
    }
    // Meshes stream in through the manager after the URDF callback; only when
    // everything is in can we size the camera to the real bounding box.
    // onLoad fires whenever this manager's queue drains — which is not when
    // the robot is complete: three's FileLoader shares a request already in
    // flight for the same URL (another viewer on the page loading the same
    // meshes) without telling this manager, so the queue can drain with the
    // URDF alone. Framing an empty robot puts the camera inside the model
    // once its meshes arrive (the Canvas only reads its camera on mount).
    // So frame once every visual holds its mesh, polling for the shared
    // ones; after a timeout, frame whatever arrived.
    let timer: ReturnType<typeof setTimeout> | undefined
    const deadline = Date.now() + 10_000
    let framed = false
    const frame = () => {
      if (cancelled || framed || !robot) return
      const visuals: { children: unknown[] }[] = []
      robot.traverse((o) => {
        if ((o as { isURDFVisual?: boolean }).isURDFVisual) visuals.push(o)
      })
      const complete = visuals.every((v) => v.children.length > 0)
      if (!complete && Date.now() < deadline) {
        timer = setTimeout(frame, 100)
        return
      }
      const scene = new Group()
      scene.rotation.x = -Math.PI / 2 // URDF is Z-up; three.js is Y-up.
      // urdf-loader resolves a second @types/three; structurally identical.
      scene.add(robot as unknown as Parameters<Group["add"]>[0])
      scene.updateWorldMatrix(true, true)
      const box = new Box3().setFromObject(scene)
      if (box.isEmpty()) {
        setError("the robot model has no meshes")
        return
      }
      const center = box.getCenter(new Vector3())
      const size = Math.max(box.getSize(new Vector3()).length(), 0.1)
      const floor = new Vector3()
      robot.links.floor?.getWorldPosition(floor)
      const limits: JointLimits = {}
      for (const [name, joint] of Object.entries(robot.joints)) {
        if (joint.jointType === "fixed") continue
        limits[name] = { lower: Number(joint.limit.lower), upper: Number(joint.limit.upper) }
      }
      framed = true
      setModel({ scene, robot, center, size, floor })
      onLoadedRef.current?.(limits)
    }
    manager.onLoad = () => {
      clearTimeout(timer)
      frame()
    }
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [robotModel, gripper])

  return { model, error }
}
