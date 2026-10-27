/**
 * A saved episode replayed on the robot model: the arms follow the recorded
 * joint angles, and each gripper's path is drawn with a marker at the
 * current frame (its axes show the rotation). Commanded paths are drawn
 * see-through when asked for. Cartesian episodes (Mantis) record no joint
 * angles, so their arms stay still and the paths and markers carry the data.
 *
 * Pulls in three.js: import it lazily.
 */

import { useEffect, useMemo } from "react"
import { Canvas } from "@react-three/fiber"
import { Line, OrbitControls } from "@react-three/drei"
import { Quaternion, Vector3 } from "three"
import { Loader2 } from "lucide-react"
import { useUrdfRobot } from "@/components/settings/urdf-robot"
import { SIDE_COLORS, SIDES, replayJoints, type Side } from "@/lib/episode-data"
import type { AxolModel, EpisodeData, EpisodePose, GripperType } from "@/lib/supervisor"

function pathPoints(pose: EpisodePose | undefined): [number, number, number][] {
  if (!pose) return []
  const points: [number, number, number][] = []
  for (let i = 0; i < pose.x.length; i++) {
    const x = pose.x[i]
    const y = pose.y[i]
    const z = pose.z[i]
    if (x != null && y != null && z != null) points.push([x, y, z])
  }
  return points
}

function markerAt(pose: EpisodePose | undefined, frame: number) {
  if (!pose) return null
  const [x, y, z, rx, ry, rz] = (["x", "y", "z", "rx", "ry", "rz"] as const).map(
    (axis) => pose[axis][frame]
  )
  if (x == null || y == null || z == null) return null
  const axis = new Vector3(rx ?? 0, ry ?? 0, rz ?? 0)
  const angle = axis.length()
  const rotation =
    angle > 1e-9 ? new Quaternion().setFromAxisAngle(axis.normalize(), angle) : new Quaternion()
  return { position: [x, y, z] as [number, number, number], rotation }
}

function GripperMarker({ side, pose, frame }: { side: Side; pose?: EpisodePose; frame: number }) {
  const marker = markerAt(pose, frame)
  if (!marker) return null
  return (
    <group position={marker.position} quaternion={marker.rotation}>
      <mesh>
        <sphereGeometry args={[0.012, 16, 16]} />
        <meshBasicMaterial color={SIDE_COLORS[side]} />
      </mesh>
      <axesHelper args={[0.07]} />
    </group>
  )
}

export default function EpisodeReplay({
  data,
  frame,
  withCommanded,
  robotModel = "classic",
  gripper = "parallel",
}: {
  data: EpisodeData
  frame: number
  withCommanded: boolean
  robotModel?: AxolModel
  gripper?: GripperType
}) {
  const { model, error } = useUrdfRobot(robotModel, gripper)
  const joints = useMemo(() => replayJoints(data, frame), [data, frame])

  useEffect(() => {
    if (!model) return
    for (const [name, value] of Object.entries(joints)) {
      model.robot.joints[name]?.setJointValue(value)
    }
  }, [model, joints])

  const paths = useMemo(
    () =>
      SIDES.map((side) => ({
        side,
        measured: pathPoints(data.pose[side]),
        commanded: pathPoints(data.actionPose[side]),
      })),
    [data]
  )

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
  return (
    <Canvas
      key={`${size.toFixed(4)}:${center.toArray().map((v) => v.toFixed(4))}`}
      camera={{
        position: [center.x + size * 0.9, center.y + size * 0.5, center.z + size * 0.9],
        fov: 40,
        near: size / 100,
        far: size * 20,
      }}
    >
      <ambientLight intensity={0.7} />
      <directionalLight position={[2, 4, 3]} intensity={1.6} />
      <directionalLight position={[-3, 2, -2]} intensity={0.5} />
      <primitive object={model.scene} />
      {/* Paths and markers in the URDF root frame, like the robot (Z-up). */}
      <group rotation={[-Math.PI / 2, 0, 0]}>
        {paths.map(({ side, measured, commanded }) => (
          <group key={side}>
            {measured.length > 1 && (
              <Line points={measured} color={SIDE_COLORS[side]} lineWidth={2} />
            )}
            {withCommanded && commanded.length > 1 && (
              <Line
                points={commanded}
                color={SIDE_COLORS[side]}
                lineWidth={1.5}
                transparent
                opacity={0.35}
                dashed
                dashSize={0.01}
                gapSize={0.008}
              />
            )}
            <GripperMarker side={side} pose={data.pose[side]} frame={frame} />
          </group>
        ))}
      </group>
      <gridHelper args={[size * 2, 20, "#3a3a3a", "#242424"]} position={model.floor.toArray()} />
      <OrbitControls target={center.toArray()} enableDamping makeDefault />
    </Canvas>
  )
}
