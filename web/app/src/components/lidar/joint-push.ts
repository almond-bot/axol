/**
 * Pose a robot model from a running operation's joint push: the
 * `{"type": "joints"}` messages teleop and data collection send on their VR
 * server's socket.
 */
import { useEffect, useRef, type RefObject } from "react"
import { useFrame } from "@react-three/fiber"
import { type URDFRobot } from "urdf-loader"
import { liveRobotTargets, type LiveRobotSample } from "@/lib/live-robot"

const SMOOTHING_TAU = 0.06

/** Latest `joints` push on `socket`, or null before one arrives. */
export function useJointPush(socket: WebSocket | null): RefObject<LiveRobotSample | null> {
  const ref = useRef<LiveRobotSample | null>(null)
  useEffect(() => {
    ref.current = null
    if (!socket) return
    const onMessage = (event: MessageEvent) => {
      if (typeof event.data !== "string" || !event.data.includes('"joints"')) return
      try {
        const msg = JSON.parse(event.data) as { type?: string; value?: Partial<LiveRobotSample> }
        const v = msg.type === "joints" ? msg.value : null
        if (!v || typeof v.q !== "object" || v.q === null) return
        ref.current = {
          q: v.q,
          l_grip: typeof v.l_grip === "number" ? v.l_grip : 1,
          r_grip: typeof v.r_grip === "number" ? v.r_grip : 1,
          lift: typeof v.lift === "number" ? v.lift : null,
        }
      } catch {
        // not ours
      }
    }
    socket.addEventListener("message", onMessage)
    return () => socket.removeEventListener("message", onMessage)
  }, [socket])
  return ref
}

/** Move `robot`'s joints smoothly toward the latest push, every frame. */
export function useJointPose(robot: URDFRobot | null, joints: RefObject<LiveRobotSample | null>) {
  const smooth = useRef<Record<string, number> | null>(null)
  useFrame((_, delta) => {
    const sample = joints.current
    if (!robot || !sample) return
    const targets = liveRobotTargets(sample, robot)
    if (!smooth.current) smooth.current = { ...targets }
    const alpha = 1 - Math.exp(-delta / SMOOTHING_TAU)
    for (const [name, value] of Object.entries(targets)) {
      const prev = smooth.current[name] ?? value
      smooth.current[name] = prev + (value - prev) * alpha
      robot.joints[name]?.setJointValue(smooth.current[name])
    }
  })
}
