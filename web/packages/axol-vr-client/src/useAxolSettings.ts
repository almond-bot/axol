import type { RefObject } from "react"
import { useCallback, useEffect, useState } from "react"
import type { AxolSettingDef, AxolSettings } from "./types"

export type AxolSettingValue = boolean | number | string

/**
 * Mirrors the teleop server's live session settings and lets the client
 * change them.
 *
 * The server pushes `{"type":"settings","value":{schema, values}}` once on
 * connect and again after every change from any client (headset HUD, control
 * panel, SDK), so `settings` always reflects the server's state — a request
 * the server rejects (out of range, unknown key) simply never echoes back.
 * `null` until the first push on the current connection; an older server that
 * has no live settings leaves it null and the UI hides the controls.
 *
 * The connect-time push races this hook: the server sends it the moment the
 * socket is accepted, while our listener is attached in an effect a render
 * after `connected` flips — on a fast link the push lands first and is lost,
 * and `settings` is only re-sent on change. So once the listener is in place
 * the hook asks for the announces again with `{"type":"get"}` (ignored by
 * older servers).
 *
 * `setSetting(key, value)` sends `{"type":"set","key","value"}` on the socket.
 * `step(def, direction)` is a helper for the generic controls: it computes the
 * next value of a setting (toggle a boolean, cycle a select, step a number
 * within its range) and sends it.
 *
 * A `message` *listener* is added (not `ws.onmessage`) so this coexists with
 * the other consumers on the same socket.
 */
export function useAxolSettings(
  wsRef: RefObject<WebSocket | null>,
  connected: boolean
): {
  settings: AxolSettings | null
  setSetting: (key: string, value: AxolSettingValue) => void
  step: (def: AxolSettingDef, direction: 1 | -1) => void
} {
  const [settings, setSettings] = useState<AxolSettings | null>(null)

  useEffect(() => {
    const ws = wsRef.current
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reset the mirror for the new connection
    setSettings(null)
    if (!connected || !ws) return
    const onMessage = (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data as string) as { type: string; value: unknown }
        if (msg.type !== "settings") return
        const v = msg.value as Partial<AxolSettings> | null
        if (!v || !Array.isArray(v.schema) || !v.values || typeof v.values !== "object") return
        setSettings({ schema: v.schema as AxolSettingDef[], values: { ...v.values } })
      } catch {
        // ignore malformed messages
      }
    }
    ws.addEventListener("message", onMessage)
    // Re-request the connect-time announces now that we're listening (the
    // server's own copy may have arrived before this effect ran).
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "get" }))
    return () => {
      ws.removeEventListener("message", onMessage)
      setSettings(null)
    }
  }, [wsRef, connected])

  const setSetting = useCallback(
    (key: string, value: AxolSettingValue) => {
      const ws = wsRef.current
      if (!ws || ws.readyState !== WebSocket.OPEN) return
      ws.send(JSON.stringify({ type: "set", key, value }))
    },
    [wsRef]
  )

  const step = useCallback(
    (def: AxolSettingDef, direction: 1 | -1) => {
      if (!settings) return
      const next = nextSettingValue(def, settings.values[def.key], direction)
      if (next !== undefined) setSetting(def.key, next)
    },
    [settings, setSetting]
  )

  return { settings, setSetting, step }
}

/**
 * The value one "step" away from `current` for a setting: booleans toggle,
 * selects cycle through their options, numbers move by `step` and clamp to
 * `[min, max]`. `undefined` when already at the end of a numeric range.
 */
export function nextSettingValue(
  def: AxolSettingDef,
  current: AxolSettingValue | undefined,
  direction: 1 | -1
): AxolSettingValue | undefined {
  if (def.type === "boolean") return !current
  if (def.type === "select") {
    const opts = def.options
    if (opts.length === 0) return undefined
    const i = opts.indexOf(String(current))
    return opts[(i + direction + opts.length) % opts.length]
  }
  const step = def.step ?? 1
  const cur = typeof current === "number" ? current : (def.min ?? 0)
  let next = cur + direction * step
  if (def.min !== null && next < def.min - 1e-9) next = def.min
  if (def.max !== null && next > def.max + 1e-9) next = def.max
  // Snap to the step grid so repeated ±0.1 clicks don't accumulate float dust.
  const decimals = Math.max(0, Math.ceil(-Math.log10(step)) + 1)
  next = Number(next.toFixed(decimals))
  return next === cur ? undefined : next
}

/** Display string for a setting value (`"1.2 Nm"`, `"ramp"`, `"ON"`). */
export function formatSettingValue(
  def: AxolSettingDef,
  value: AxolSettingValue | undefined
): string {
  if (def.type === "boolean") return value ? "ON" : "OFF"
  if (def.type === "select") return String(value ?? "—")
  if (typeof value !== "number") return "—"
  const step = def.step ?? 1
  const decimals = Math.max(0, Math.ceil(-Math.log10(step)))
  const num = value.toFixed(decimals)
  return def.unit ? `${num} ${def.unit}` : num
}
