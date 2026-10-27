/**
 * Gripper-mount poses from recorded joint angles, worked out in the browser.
 *
 * A joint-space episode records joint angles only; its gripper-pose charts,
 * paths and jump check need where each gripper mount was. Computing that on
 * the Host Computer would run in the process that also runs a recording's
 * control loop, so the panel does it here: the robot model's URDF (no
 * meshes) is posed frame by frame and each `{side}_gripper` link read back in
 * the URDF root frame, the frame Mantis records in (metres, axis-angle
 * radians). Pulls in three.js and urdf-loader: import it lazily.
 */

import { Matrix4, Quaternion, Vector3 } from "three"
import URDFLoader, { type URDFRobot } from "urdf-loader"
import { SIDES, type Side } from "@/lib/episode-data"
import { urdfUrl, type AxolModel, type EpisodePose, type GripperType } from "@/lib/supervisor"

/** Dataset joint name suffix -> URDF joint suffix (urdf_arm_joint_names). */
const URDF_JOINTS = [
  ["shoulder_1", "s1_0"],
  ["shoulder_2", "s2_0"],
  ["shoulder_3", "s3_0"],
  ["elbow", "e1_0"],
  ["wrist_1", "e2_0"],
  ["wrist_2", "w1_0"],
  ["wrist_3", "w2_0"],
] as const

/** Parse a URDF's kinematic tree only: no visuals, no meshes to fetch.
 *  Joints take recorded values as they are, past their limits too (as the
 *  Host Computer's kinematics do), so a bad take shows as recorded. */
export function parseKinematics(urdf: string): URDFRobot {
  const loader = new URDFLoader()
  loader.parseVisual = false
  loader.parseCollision = false
  const robot = loader.parse(urdf)
  for (const joint of Object.values(robot.joints)) joint.ignoreLimits = true
  return robot
}

const kinematics = new Map<string, Promise<URDFRobot>>()

/** The kinematic model of an Axol version, fetched from the Host Computer
 *  once per page. */
export function loadKinematics(robotModel: AxolModel, gripper: GripperType): Promise<URDFRobot> {
  const url = urdfUrl(robotModel, gripper)
  let robot = kinematics.get(url)
  if (!robot) {
    robot = fetch(url).then(async (response) => {
      if (!response.ok) throw new Error(`the robot model answered HTTP ${response.status}`)
      return parseKinematics(await response.text())
    })
    robot.catch(() => kinematics.delete(url))
    kinematics.set(url, robot)
  }
  return robot
}

/** Axis-angle vector of a rotation, with the angle in [0, π]. */
function rotationVector(q: Quaternion): [number, number, number] {
  const w = q.w < 0 ? -q.w : q.w
  const sign = q.w < 0 ? -1 : 1
  const s = Math.hypot(q.x, q.y, q.z)
  const scale = s > 1e-12 ? (2 * Math.atan2(s, w)) / s : 2
  return [q.x * sign * scale, q.y * sign * scale, q.z * sign * scale]
}

/**
 * Each side's gripper-mount pose per frame, from `joints` (dataset joint name
 * -> angle per frame, rad). A side whose arm joints aren't all recorded is
 * left out; a frame with a missing angle gets nulls.
 */
export function posesFromJoints(
  robot: URDFRobot,
  joints: Record<string, (number | null)[]>,
  frames: number
): Partial<Record<Side, EpisodePose>> {
  const sides = SIDES.filter(
    (side) =>
      URDF_JOINTS.every(([name]) => joints[`${side}_${name}.pos`]) &&
      robot.links[`${side}_gripper`] &&
      URDF_JOINTS.every(([, urdf]) => robot.joints[`${side}_${urdf}`])
  )
  const out: Partial<Record<Side, EpisodePose>> = {}
  for (const side of sides) {
    out[side] = { x: [], y: [], z: [], rx: [], ry: [], rz: [] }
  }
  const rootInverse = new Matrix4()
  const relative = new Matrix4()
  const position = new Vector3()
  const rotation = new Quaternion()
  const scale = new Vector3()
  for (let frame = 0; frame < frames; frame++) {
    const usable = new Set<Side>()
    for (const side of sides) {
      let complete = true
      for (const [name, urdf] of URDF_JOINTS) {
        const value = joints[`${side}_${name}.pos`][frame]
        if (value == null) complete = false
        else robot.joints[`${side}_${urdf}`].setJointValue(value)
      }
      if (complete) usable.add(side)
    }
    robot.updateMatrixWorld(true)
    rootInverse.copy(robot.matrixWorld).invert()
    for (const side of sides) {
      const pose = out[side]!
      if (!usable.has(side)) {
        for (const axis of ["x", "y", "z", "rx", "ry", "rz"] as const) pose[axis].push(null)
        continue
      }
      relative.multiplyMatrices(rootInverse, robot.links[`${side}_gripper`].matrixWorld)
      relative.decompose(position, rotation, scale)
      const [rx, ry, rz] = rotationVector(rotation)
      pose.x.push(position.x)
      pose.y.push(position.y)
      pose.z.push(position.z)
      pose.rx.push(rx)
      pose.ry.push(ry)
      pose.rz.push(rz)
    }
  }
  return out
}
