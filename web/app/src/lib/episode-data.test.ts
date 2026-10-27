import { describe, expect, it } from "vitest"

import {
  episodeCharts,
  formatEpisodeTime,
  frameAt,
  hasJointState,
  largestPoseJump,
  replayJoints,
  withJointPoses,
} from "./episode-data"
import type { EpisodeData } from "./supervisor"

const pose = (x: number) => ({
  x: [x, x],
  y: [0, 0],
  z: [1, 1],
  rx: [0, 0],
  ry: [0, 0],
  rz: [0, 0],
})

const joint: EpisodeData = {
  repoId: "local/teleop",
  episode: 0,
  fps: 10,
  frames: 2,
  space: "joint",
  t: [0, 0.1],
  state: {
    "left_shoulder_1.pos": [0, 0.5],
    "left_wrist_1.pos": [0.2, 0.3],
    "left_gripper.pos": [1, 0.5],
  },
  action: { "left_shoulder_1.pos": [0.1, 0.6], "left_gripper.pos": [1, 0.4] },
  pose: { left: pose(0.3) },
  actionPose: { left: pose(0.31) },
  poseLagMs: null,
  intervention: null,
  checks: { droppedFrames: 0, longestGapS: 0.1, jointJump: null, poseJump: null },
}

const cartesian: EpisodeData = {
  ...joint,
  space: "cartesian",
  state: { "right_ee.x": [0.3, 0.31], "right_gripper.pos": [0.2, 0.3] },
  action: {},
  pose: { right: pose(0.3) },
  actionPose: {},
  poseLagMs: [12, 14],
}

describe("frameAt", () => {
  it("shows the last frame at or before the time", () => {
    const t = [0, 0.1, 0.2, 0.3]
    expect(frameAt(t, 0)).toBe(0)
    expect(frameAt(t, 0.15)).toBe(1)
    expect(frameAt(t, 0.3)).toBe(3)
    expect(frameAt(t, 9)).toBe(3)
    expect(frameAt([], 1)).toBe(0)
  })
})

describe("replayJoints", () => {
  it("maps recorded joints to the robot model's joints", () => {
    expect(replayJoints(joint, 1)).toEqual({ left_s1_0: 0.5, left_e2_0: 0.3 })
    expect(hasJointState(joint)).toBe(true)
    expect(hasJointState(cartesian)).toBe(false)
  })
})

describe("episodeCharts", () => {
  it("charts each arm's joints in degrees, with commanded twins on request", () => {
    const [left] = episodeCharts(joint, "joints", false)
    expect(left.title).toBe("Left arm joints")
    expect(left.series.map((s) => s.key)).toEqual(["left_shoulder_1.pos", "left_wrist_1.pos"])
    expect(left.frames[1].m["left_shoulder_1.pos"][0]).toBeCloseTo(28.648, 2)
    const withCommanded = episodeCharts(joint, "joints", true)[0]
    expect(withCommanded.series.map((s) => s.key)).toContain("cmd:left_shoulder_1.pos")
  })

  it("charts each gripper's position in cm and its rotation, and the Mantis lag", () => {
    const titles = episodeCharts(cartesian, "pose", false).map((c) => c.title)
    expect(titles).toEqual([
      "Right gripper position",
      "Right gripper rotation",
      "Camera to tracker lag",
    ])
    const [position] = episodeCharts(cartesian, "pose", false)
    expect(position.frames[0].m.x[0]).toBeCloseTo(30)
  })

  it("charts the opening of each gripper that has one", () => {
    const [opening] = episodeCharts(joint, "gripper", true)
    expect(opening.series.map((s) => s.key)).toEqual(["left_gripper.pos", "cmd:left_gripper.pos"])
    expect(opening.frames[1].m["left_gripper.pos"][0]).toBe(50)
  })
})

describe("largestPoseJump", () => {
  it("finds the largest move of either gripper between two frames", () => {
    const moving = { ...pose(0), x: [0, 0.01, 0.05], y: [0, 0, 0], z: [1, 1, 1] }
    const still = { ...pose(0.2), x: [0.2, 0.2, 0.2], y: [0, 0, 0], z: [1, 1, 1] }
    const jump = largestPoseJump({ left: still, right: moving }, [0, 0.1, 0.2])
    expect(jump?.side).toBe("right")
    expect(jump?.m).toBeCloseTo(0.04)
    expect(jump?.t).toBe(0.2)
    expect(largestPoseJump({}, [0])).toBeNull()
  })

  it("fills in a joint episode's poses and its gripper check", () => {
    const filled = withJointPoses({ ...joint, pose: {}, actionPose: {} }, { left: pose(0.3) }, {})
    expect(filled.pose.left?.x).toEqual([0.3, 0.3])
    expect(filled.checks.poseJump).toEqual({ side: "left", m: 0, t: 0.1 })
  })
})

describe("formatEpisodeTime", () => {
  it("reads as minutes and seconds", () => {
    expect(formatEpisodeTime(4.25)).toBe("0:04.3")
    expect(formatEpisodeTime(65)).toBe("1:05.0")
  })
})
