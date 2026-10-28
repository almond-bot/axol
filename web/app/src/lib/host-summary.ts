import { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"
import type { HostSummary } from "@/components/app-shell"
import { fetchInfo } from "@/lib/supervisor"

/**
 * The header pill for an operation screen with its own route (Jog), outside
 * the control panel's state: the Host Computer's name once it answers, and
 * Devices on click, as on the control panel's pages.
 *
 * `reachable` is whether the page's own status poll reached the host, or null
 * before its first answer.
 */
export function useHostSummary(reachable: boolean | null): HostSummary {
  const navigate = useNavigate()
  const [hostname, setHostname] = useState<string | null>(null)
  useEffect(() => {
    if (!reachable || hostname) return
    let active = true
    fetchInfo()
      .then((info) => active && setHostname(info.hostname))
      .catch(() => undefined)
    return () => {
      active = false
    }
  }, [reachable, hostname])
  return {
    dot: reachable ? "ok" : reachable === false ? "err" : "idle",
    label: reachable
      ? hostname || "Connected"
      : reachable === false
        ? "Can't reach the robot"
        : "Connecting…",
    onClick: () => navigate("/control/devices"),
  }
}
