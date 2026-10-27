import type { ChartSeries } from "@/components/diagnostics/telemetry-chart"
import type { EpisodeData, EpisodePose } from "@/lib/supervisor"
import { JOINT_COLORS, type TelemetryFrame } from "@/lib/telemetry"

/**
 * A saved episode's robot data shaped for the dataset view: chart series and
 * the joint angles that drive the 3D replay. The host sends columns per
 * feature name (see recording/episode_data.py); this turns them into the
 * frames TelemetryChart draws and the URDF joint values the viewer takes.
 */

export type Side = "left" | "right"
export const SIDES: readonly Side[] = ["left", "right"]

/** Dataset joint name suffix -> URDF joint suffix (urdf_arm_joint_names). */
const ARM_JOINTS = [
  ["shoulder_1", "s1_0", "SHOULDER_1"],
  ["shoulder_2", "s2_0", "SHOULDER_2"],
  ["shoulder_3", "s3_0", "SHOULDER_3"],
  ["elbow", "e1_0", "ELBOW"],
  ["wrist_1", "e2_0", "WRIST_1"],
  ["wrist_2", "w1_0", "WRIST_2"],
  ["wrist_3", "w2_0", "WRIST_3"],
] as const

/** Colors for the gripper mount's axes and for each side. */
export const AXIS_COLORS = { x: "#e66767", y: "#199e70", z: "#3987e5" } as const
export const SIDE_COLORS: Record<Side, string> = { left: "#3987e5", right: "#d95926" }

const RAD_TO_DEG = 180 / Math.PI

/** The frame shown at `time` (s): the last one at or before it. */
export function frameAt(t: number[], time: number): number {
  let lo = 0
  let hi = t.length - 1
  if (hi < 0) return 0
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (t[mid] <= time) lo = mid
    else hi = mid - 1
  }
  return lo
}

/** Does the episode record joint angles the replay can drive the arms with? */
export function hasJointState(data: EpisodeData): boolean {
  return SIDES.some((side) => `${side}_${ARM_JOINTS[0][0]}.pos` in data.state)
}

/** URDF joint name -> angle (rad) for one frame of the measured joints. */
export function replayJoints(data: EpisodeData, frame: number): Record<string, number> {
  const values: Record<string, number> = {}
  for (const side of SIDES) {
    for (const [name, urdf] of ARM_JOINTS) {
      const v = data.state[`${side}_${name}.pos`]?.[frame]
      if (v != null) values[`${side}_${urdf}`] = v
    }
  }
  return values
}

/** The largest move of a gripper mount between two frames (mirrors the
 *  Host Computer's check for recorded poses). */
export function largestPoseJump(
  poses: Partial<Record<Side, EpisodePose>>,
  t: number[]
): EpisodeData["checks"]["poseJump"] {
  let best: EpisodeData["checks"]["poseJump"] = null
  for (const side of SIDES) {
    const pose = poses[side]
    if (!pose) continue
    for (let i = 1; i < pose.x.length; i++) {
      const [x0, y0, z0, x1, y1, z1] = [
        pose.x[i - 1],
        pose.y[i - 1],
        pose.z[i - 1],
        pose.x[i],
        pose.y[i],
        pose.z[i],
      ]
      if ([x0, y0, z0, x1, y1, z1].some((v) => v == null)) continue
      const m = Math.hypot(x1! - x0!, y1! - y0!, z1! - z0!)
      if (!best || m > best.m) best = { side, m, t: t[i] }
    }
  }
  return best
}

/** The episode with the gripper poses the panel worked out from its joints. */
export function withJointPoses(
  data: EpisodeData,
  pose: Partial<Record<Side, EpisodePose>>,
  actionPose: Partial<Record<Side, EpisodePose>>
): EpisodeData {
  return {
    ...data,
    pose,
    actionPose,
    checks: { ...data.checks, poseJump: largestPoseJump(pose, data.t) },
  }
}

/** One chart: what it shows, its unit and its series over the episode. */
export interface ChartSpec {
  key: string
  title: string
  unit: string
  series: ChartSeries[]
  frames: TelemetryFrame[]
}

interface Column {
  key: string
  label: string
  color: string
  values: (number | null)[]
  scale?: number
}

function toFrames(t: number[], columns: Column[]): TelemetryFrame[] {
  return t.map((time, i) => {
    const m: TelemetryFrame["m"] = {}
    for (const c of columns) {
      const v = c.values[i]
      m[c.key] = [v == null ? null : v * (c.scale ?? 1)]
    }
    return { t: time, m }
  })
}

function chart(
  key: string,
  title: string,
  unit: string,
  t: number[],
  columns: Column[]
): ChartSpec | null {
  const present = columns.filter((c) => c.values.length > 0)
  if (present.length === 0) return null
  return {
    key,
    title,
    unit,
    series: present.map(({ key, label, color }) => ({ key, label, color })),
    frames: toFrames(t, present),
  }
}

/** A commanded series: the measured one's color, see-through. */
const commanded = (color: string) => `${color}80`

export type ChartGroup = "joints" | "pose" | "gripper"

/**
 * The charts for one group. With `withCommanded`, each measured series gets
 * its commanded twin, drawn see-through, so tracking error shows as a gap.
 */
export function episodeCharts(
  data: EpisodeData,
  group: ChartGroup,
  withCommanded: boolean
): ChartSpec[] {
  const { t } = data
  const charts: (ChartSpec | null)[] = []
  const sideLabel = (side: Side) => (side === "left" ? "Left" : "Right")

  if (group === "joints") {
    for (const side of SIDES) {
      const columns: Column[] = []
      for (const [name, , joint] of ARM_JOINTS) {
        const key = `${side}_${name}.pos`
        const label = name.replace("_", " ")
        const color = JOINT_COLORS[joint]
        if (data.state[key]) {
          columns.push({ key, label, color, values: data.state[key], scale: RAD_TO_DEG })
        }
        if (withCommanded && data.action[key]) {
          columns.push({
            key: `cmd:${key}`,
            label: `${label} commanded`,
            color: commanded(color),
            values: data.action[key],
            scale: RAD_TO_DEG,
          })
        }
      }
      charts.push(chart(`joints-${side}`, `${sideLabel(side)} arm joints`, "°", t, columns))
    }
  }

  if (group === "pose") {
    for (const side of SIDES) {
      const pose = data.pose[side]
      const target = withCommanded ? data.actionPose[side] : undefined
      const columns = (axes: (keyof EpisodePose)[], scale: number): Column[] =>
        axes.flatMap((axis, i) => {
          const color = Object.values(AXIS_COLORS)[i]
          const out: Column[] = []
          if (pose) out.push({ key: axis, label: axis, color, values: pose[axis], scale })
          if (target) {
            out.push({
              key: `cmd:${axis}`,
              label: `${axis} commanded`,
              color: commanded(color),
              values: target[axis],
              scale,
            })
          }
          return out
        })
      charts.push(
        chart(
          `position-${side}`,
          `${sideLabel(side)} gripper position`,
          "cm",
          t,
          columns(["x", "y", "z"], 100)
        ),
        chart(
          `rotation-${side}`,
          `${sideLabel(side)} gripper rotation`,
          "°",
          t,
          columns(["rx", "ry", "rz"], RAD_TO_DEG)
        )
      )
    }
    if (data.poseLagMs) {
      charts.push(
        chart("pose-lag", "Camera to tracker lag", "ms", t, [
          { key: "lag", label: "lag", color: "#9085e9", values: data.poseLagMs },
        ])
      )
    }
  }

  if (group === "gripper") {
    const columns: Column[] = []
    for (const side of SIDES) {
      const key = `${side}_gripper.pos`
      const color = SIDE_COLORS[side]
      if (data.state[key]) {
        columns.push({ key, label: side, color, values: data.state[key], scale: 100 })
      }
      if (withCommanded && data.action[key]) {
        columns.push({
          key: `cmd:${key}`,
          label: `${side} commanded`,
          color: commanded(color),
          values: data.action[key],
          scale: 100,
        })
      }
    }
    charts.push(chart("gripper", "Gripper opening", "% open", t, columns))
  }

  return charts.filter((c): c is ChartSpec => c !== null)
}

/** "0:04.2": time into the episode, for chart axes. */
export function formatEpisodeTime(t: number): string {
  const m = Math.floor(t / 60)
  const s = t - m * 60
  return `${m}:${s.toFixed(1).padStart(4, "0")}`
}
