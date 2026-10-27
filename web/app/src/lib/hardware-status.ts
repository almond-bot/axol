import { useState } from "react"
import type { CanDiscoveryState } from "@/lib/supervisor"

/**
 * Hardware problems as an operator reads them. The host reports CAN
 * discovery results and link errors in terms of interfaces, hubs and
 * identification; the panel says what to do about them instead ("turn on its
 * power", "check its USB cable"), and keeps the host's text for a tooltip.
 */

export interface HardwareNotice {
  tone: "attention" | "error"
  text: string
}

/**
 * What the operator needs to know while plugged-in hardware isn't identified
 * yet, or null when there is nothing to say. Unidentified hardware is almost
 * always hardware whose motors are off: the panel checks it again on its own,
 * so turning the power on is all it takes.
 */
export function discoveryNotice(
  discovery: CanDiscoveryState | null | undefined
): HardwareNotice | null {
  switch (discovery?.status) {
    case "unidentified":
      return {
        tone: "attention",
        text: "Hardware is plugged in but not answering. Turn on its power. It connects on its own.",
      }
    case "partial":
      return {
        tone: "attention",
        text: "Some hardware is plugged in but not answering. Turn on its power.",
      }
    case "error":
      return {
        tone: "error",
        text: "The Host Computer couldn't check its hardware. It tries again on its own. If this keeps happening, restart the Host Computer.",
      }
    default:
      return null
  }
}

/** Plugged-in hardware the panel should keep checking until it answers. */
export function discoveryNeedsRecheck(discovery: CanDiscoveryState | null | undefined): boolean {
  return (
    discovery?.status === "unidentified" ||
    discovery?.status === "partial" ||
    discovery?.status === "error"
  )
}

/**
 * The discovery result to show and act on. A check of silent hardware takes
 * the host many seconds and reports "running" meanwhile; the last result
 * stands until the check ends, so the notice doesn't blink off and the
 * backoff doesn't start over.
 */
export function settledDiscovery(
  previous: CanDiscoveryState | null,
  next: CanDiscoveryState | null
): CanDiscoveryState | null {
  if (next?.status === "running" && discoveryNeedsRecheck(previous)) return previous
  return next
}

/** `settledDiscovery` over a page's polled discovery state. */
export function useSettledDiscovery(discovery: CanDiscoveryState | null): CanDiscoveryState | null {
  const [state, setState] = useState({ seen: discovery, shown: discovery })
  if (state.seen === discovery) return state.shown
  const next = { seen: discovery, shown: settledDiscovery(state.shown, discovery) }
  setState(next)
  return next.shown
}

/** How long to wait before the next automatic check: 5 s, doubling to a minute. */
export function nextRecheckDelay(previous: number | null): number {
  return previous == null ? 5000 : Math.min(previous * 2, 60_000)
}

/**
 * A device's connection error in plain words. `device` is the product name
 * ("Axol", "Mantis"). Unknown errors say only that the connection failed;
 * the host's own text is for the tooltip.
 */
export function deviceErrorText(error: string | null | undefined, device: string): string {
  const raw = (error ?? "").replace(/^Error:\s*/, "")
  if (/not detected|not found|no such device|no can interfaces|not plugged/i.test(raw)) {
    return `${device} isn't plugged in. Check its USB cable.`
  }
  if (
    /time(d)? ?out|no response|not answer|silent|did not respond|unresponsive|revalidat|identif|power/i.test(
      raw
    )
  ) {
    return `${device} isn't answering. Turn on its power.`
  }
  if (/permission|denied|not permitted/i.test(raw)) {
    return "The Host Computer can't open the USB device. Restart the Host Computer."
  }
  if (/busy|in use|owns/i.test(raw)) {
    return `${device} is in use. Stop the running operation or test first.`
  }
  return `Couldn't connect to ${device}.`
}
