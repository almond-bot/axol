import { useEffect, useMemo, useState } from "react"
import { serverHttpBase } from "@/lib/supervisor"

/** Bare hostname of the serve machine: the stored host may carry a scheme or
 * the control-panel port; same-origin panels have no host at all. */
export function vrHostname(host: string): string {
  const base = serverHttpBase(host)
  if (base) {
    try {
      return new URL(base).hostname
    } catch {
      // fall through to the page's own host
    }
  }
  return window.location.hostname
}

/**
 * A plain connection to the running operation's VR server
 * (`wss://host:vrPort/ws`), reconnecting every 3 s while `enabled`: the
 * server only comes up partway through the operation's startup. Returns the
 * socket once open, null otherwise.
 *
 * For panels that need the server's messages but not its video (the live
 * session settings when no camera feeds are shown, e.g. in sim); the
 * camera-feed card owns its own socket and shares it instead.
 */
export function useVrSocket(host: string, vrPort: number, enabled: boolean): WebSocket | null {
  const hostname = useMemo(() => vrHostname(host), [host])
  const [socket, setSocket] = useState<WebSocket | null>(null)

  useEffect(() => {
    if (!enabled) return
    let closed = false
    let timer: ReturnType<typeof setTimeout> | null = null
    let current: WebSocket | null = null

    function connect() {
      let ws: WebSocket
      try {
        ws = new WebSocket(`wss://${hostname}:${vrPort}/ws`)
      } catch {
        timer = setTimeout(connect, 3000)
        return
      }
      current = ws
      ws.onopen = () => setSocket(ws)
      ws.onclose = () => {
        if (current === ws) current = null
        setSocket(null)
        if (!closed) timer = setTimeout(connect, 3000)
      }
    }

    connect()
    return () => {
      closed = true
      if (timer) clearTimeout(timer)
      if (current) {
        current.onclose = null
        current.close()
      }
      setSocket(null)
    }
  }, [hostname, vrPort, enabled])

  return socket
}
