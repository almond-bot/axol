import { wheelName } from "./jelly-view"
import {
  motorFaultLabel,
  wheelFaults,
  type JellyStatus,
  type RobotStatus,
  type SafetyLinkStatus,
} from "./supervisor"

/**
 * Faulted motors on Axol's power supply, which a power cycle resets: the arm
 * motors and Jelly's wheels. Mantis grippers are not on that supply.
 */
export function motorFaultLabels(
  robot: RobotStatus | null | undefined,
  jelly: JellyStatus | null | undefined
): string[] {
  const arms =
    robot && robot.profile !== "mantis" && robot.state === "connected"
      ? (robot.faults ?? []).map(motorFaultLabel)
      : []
  const wheels = wheelFaults(jelly?.wheels).map((m) => {
    const problem = m.reachable
      ? (m.status ?? "error").replace(/_/g, " ").toLowerCase()
      : "unreachable"
    return `${wheelName(m)} wheel: ${problem}`
  })
  return [...arms, ...wheels]
}

/** How the header's safety control looks. */
export type SafetyTone = "on" | "off" | "estop" | "fault" | "unknown"

export interface SafetyView {
  /** False on a Jelly without a safety board (or no host): show nothing. */
  visible: boolean
  tone: SafetyTone
  /** One to three words for the header. */
  label: string
  /** One or two sentences for the panel. */
  detail: string
  canStop: boolean
  canStart: boolean
  /** Why power can't be restored from here, when it can't. */
  startHint: string | null
  /** Extra lines under the detail (faults worth knowing). */
  notes: string[]
}

const HIDDEN: SafetyView = {
  visible: false,
  tone: "unknown",
  label: "",
  detail: "",
  canStop: false,
  canStart: false,
  startHint: null,
  notes: [],
}

const STOP_CAUSE_DETAIL: Record<string, string> = {
  can: "Power was cut by a software stop.",
  heartbeat: "Power was cut because the robot computer stopped responding.",
  console: "Power was cut from the safety board's service console.",
  update: "Power was cut for a safety board firmware update.",
}

/** The safety board's state, in operator terms. */
export function safetyView(link: SafetyLinkStatus | null | undefined): SafetyView {
  if (!link || !link.detected) return HIDDEN
  const s = link.status
  if (!link.reachable || !s) {
    return {
      ...HIDDEN,
      visible: true,
      tone: "unknown",
      label: link.paused ? "Safety paused" : "Safety offline",
      detail: link.paused
        ? "The software stop is paused while the robot computer sets up its CAN adapters. Use the E-Stop."
        : "Jelly's safety board is not answering, so the software stop may not work. Use the E-Stop.",
      canStop: link.connected,
      startHint: "Wait for the safety board to answer.",
    }
  }

  const notes: string[] = []
  if (s.faults.remoteStartFailed && !s.outputOn) {
    notes.push("The last restore did not switch power on.")
  }
  if (s.faults.heartbeatLost && s.stopCause !== "heartbeat") {
    notes.push("The robot computer stopped responding earlier.")
  }
  if (s.faults.supplyDip && !s.outputOn) {
    notes.push("Jelly's 24 V supply dipped, which can cut power on its own.")
  }

  if (s.faults.relay) {
    return {
      ...HIDDEN,
      visible: true,
      tone: "fault",
      label: "Safety fault",
      detail:
        "The safety board reports a relay fault. Use the E-Stop, then contact Almond support.",
      canStop: s.outputOn,
      startHint: "Contact Almond support before restoring power.",
      notes,
    }
  }
  if (s.updateLocked && !s.outputOn) {
    return {
      ...HIDDEN,
      visible: true,
      tone: "off",
      label: "Updating firmware",
      detail:
        "The safety board is installing new firmware. Power stays off until the update finishes.",
      startHint: "Wait for the update to finish.",
      notes,
    }
  }
  if (s.outputOn) {
    return {
      ...HIDDEN,
      visible: true,
      tone: "on",
      label: "Power on",
      detail: "Cut power works like the E-Stop. Axol loses power and the arms drop.",
      canStop: true,
      notes,
    }
  }
  if (s.remoteStartActive) {
    return {
      ...HIDDEN,
      visible: true,
      tone: "off",
      label: "Restoring power",
      detail: "Switching power on.",
      canStop: true,
      startHint: "Power is switching on.",
      notes,
    }
  }
  if (!s.estopOk) {
    return {
      ...HIDDEN,
      visible: true,
      tone: "estop",
      label: "E-Stop pressed",
      detail: "Power is off. Release the E-Stop, then restore power.",
      startHint: "Release the E-Stop first.",
      notes,
    }
  }
  if (!s.supplyOk) {
    return {
      ...HIDDEN,
      visible: true,
      tone: "off",
      label: "Power off",
      detail: `The battery reads ${s.supplyVolts.toFixed(1)} V, too low to switch power on.`,
      startHint: "Charge Jelly first.",
      notes,
    }
  }
  if (!s.softwareEnable) {
    return {
      ...HIDDEN,
      visible: true,
      tone: "off",
      label: "Power cut",
      detail: STOP_CAUSE_DETAIL[s.stopCause] ?? "Power was cut by a software stop.",
      canStart: true,
      notes,
    }
  }
  return {
    ...HIDDEN,
    visible: true,
    tone: "off",
    label: "Power off",
    detail: "Press the green on-switch on Jelly, or restore power here.",
    canStart: true,
    notes,
  }
}
