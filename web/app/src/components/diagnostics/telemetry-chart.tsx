import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import type { TelemetryFrame } from "@/lib/telemetry"

/** One plotted series: `key` indexes frame.m, color is the joint's fixed slot. */
export interface ChartSeries {
  key: string
  label: string
  color: string
}

interface TelemetryChartProps {
  title: string
  unit: string
  series: ChartSeries[]
  /** Shared frame buffer (mutated in place by the stream hook). */
  frames: TelemetryFrame[]
  /** Redraw trigger — ticks when the buffer changes. */
  version: number
  /** Index into the fast sample: 0 = position, 1 = velocity, 2 = torque. */
  metric: number
  /** Visible window in seconds; null fits the whole buffer (run viewer). */
  windowSec: number | null
  /** Live charts anchor the window to now; static ones to the last frame. */
  live: boolean
  /** Why the stream is quiet (shown as an in-plot notice), if it is. */
  quietReason?: string | null
  className?: string
}

// Chart chrome on the card surface (#161618) — hairline recessive grid/axes,
// text in muted ink tokens; only the data wears the series colors.
const GRID = "rgba(255,255,255,0.07)"
const BASELINE = "rgba(255,255,255,0.16)"
const AXIS_INK = "#898781"
const PAD = { top: 12, right: 12, bottom: 26, left: 48 }
// A gap between samples longer than this breaks the line (bus handed to a
// task, server restart) instead of drawing a false bridge across it.
const GAP_BREAK_S = 1.5
const MAX_POINTS_PER_SERIES = 1500

interface Extracted {
  /** Per series, [t, value] pairs within the window (strided). */
  points: [number, number][][]
  t0: number
  t1: number
  min: number
  max: number
}

function extract(
  frames: TelemetryFrame[],
  series: ChartSeries[],
  metric: number,
  windowSec: number | null,
  live: boolean
): Extracted | null {
  if (frames.length === 0) return null
  const tEnd = live ? Date.now() / 1000 : frames[frames.length - 1].t
  const t1 = tEnd
  const t0 = windowSec == null ? frames[0].t : t1 - windowSec
  let lo = 0
  while (lo < frames.length && frames[lo].t < t0) lo++
  const visible = frames.slice(lo)
  if (visible.length === 0) return null
  const stride = Math.max(1, Math.ceil(visible.length / MAX_POINTS_PER_SERIES))

  let min = Infinity
  let max = -Infinity
  const points: [number, number][][] = series.map(() => [])
  for (let i = 0; i < visible.length; i += stride) {
    const frame = visible[i]
    for (let s = 0; s < series.length; s++) {
      const sample = frame.m[series[s].key]
      const value = sample?.[metric]
      if (value == null || Number.isNaN(value)) continue
      points[s].push([frame.t, value])
      if (value < min) min = value
      if (value > max) max = value
    }
  }
  if (min === Infinity) return null
  if (min === max) {
    min -= 0.5
    max += 0.5
  }
  const pad = (max - min) * 0.08
  return { points, t0, t1, min: min - pad, max: max + pad }
}

/** ~n clean tick values across [min, max]. */
function ticks(min: number, max: number, n: number): number[] {
  const span = max - min
  if (span <= 0) return [min]
  const step = Math.pow(10, Math.floor(Math.log10(span / n)))
  const err = span / (n * step)
  const mult = err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1
  const s = step * mult
  const out: number[] = []
  for (let v = Math.ceil(min / s) * s; v <= max + s * 1e-6; v += s) out.push(v)
  return out
}

function fmtValue(v: number): string {
  const a = Math.abs(v)
  if (a >= 100) return v.toFixed(0)
  if (a >= 10) return v.toFixed(1)
  if (a >= 0.1 || a === 0) return v.toFixed(2)
  return v.toFixed(3)
}

function fmtClock(t: number): string {
  const d = new Date(t * 1000)
  return d.toLocaleTimeString(undefined, { hour12: false })
}

/**
 * Streaming multi-series line chart on canvas: 2px round-joined lines, hairline
 * solid grid, crosshair + all-series tooltip (pointer or ←/→ on keyboard
 * focus), and a persistent legend with live per-series values under the plot.
 */
export function TelemetryChart({
  title,
  unit,
  series,
  frames,
  version,
  metric,
  windowSec,
  live,
  quietReason,
  className,
}: TelemetryChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 0, h: 220 })
  // Hovered/focused time (seconds); null hides the crosshair.
  const [hoverT, setHoverT] = useState<number | null>(null)

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      setSize({ w: el.clientWidth, h: 220 })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const data = useMemo(
    () => extract(frames, series, metric, windowSec, live),
    // The live buffer is mutated in place (stable identity) — `version` is its
    // change signal; static charts pass a fresh `frames` array instead.
    // eslint-disable-next-line react-hooks/exhaustive-deps -- version is the buffer's change signal
    [frames, version, series, metric, windowSec, live]
  )

  // Values shown in the tooltip + the frame the crosshair snaps to.
  const hover = useMemo(() => {
    if (hoverT == null || !data) return null
    let best: { t: number; values: (number | null)[] } | null = null
    let bestDist = Infinity
    // Snap to the nearest sampled time across all series (they share frames).
    for (let s = 0; s < data.points.length; s++) {
      for (const [t] of data.points[s]) {
        const d = Math.abs(t - hoverT)
        if (d < bestDist) {
          bestDist = d
          best = { t, values: [] }
        }
      }
      break // frames are shared — the first non-empty series is enough
    }
    if (!best) return null
    const snapT = best.t
    best.values = data.points.map((pts) => {
      let v: number | null = null
      let dBest = 0.6 // within ~half a second counts as "at" the crosshair
      for (const [t, value] of pts) {
        const d = Math.abs(t - snapT)
        if (d < dBest) {
          dBest = d
          v = value
        }
      }
      return v
    })
    return best
  }, [hoverT, data])

  // Latest value per series, for the legend's live value labels.
  const latest = useMemo(() => {
    return series.map((s) => {
      for (let i = frames.length - 1; i >= Math.max(0, frames.length - 20); i--) {
        const v = frames[i].m[s.key]?.[metric]
        if (v != null) return v
      }
      return null
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps -- version is the buffer's change signal
  }, [frames, version, series, metric])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || size.w === 0) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = size.w * dpr
    canvas.height = size.h * dpr
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, size.w, size.h)

    const plotW = size.w - PAD.left - PAD.right
    const plotH = size.h - PAD.top - PAD.bottom
    if (plotW <= 0 || plotH <= 0) return

    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace"

    if (!data) {
      ctx.fillStyle = AXIS_INK
      ctx.textAlign = "center"
      ctx.fillText(quietReason ?? "No telemetry", size.w / 2, size.h / 2)
      return
    }

    const { t0, t1, min, max } = data
    const x = (t: number) => PAD.left + ((t - t0) / (t1 - t0 || 1)) * plotW
    const y = (v: number) => PAD.top + (1 - (v - min) / (max - min)) * plotH

    // Grid + y labels (hairline, solid, recessive).
    ctx.textAlign = "right"
    ctx.textBaseline = "middle"
    for (const tick of ticks(min, max, 4)) {
      const py = y(tick)
      ctx.strokeStyle = GRID
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(PAD.left, py)
      ctx.lineTo(size.w - PAD.right, py)
      ctx.stroke()
      ctx.fillStyle = AXIS_INK
      ctx.fillText(fmtValue(tick), PAD.left - 6, py)
    }
    // X labels (clock time) — tick count scales with width so an HH:MM:SS
    // label (~55px) never collides with its neighbor.
    ctx.textAlign = "center"
    ctx.textBaseline = "top"
    const nTicksX = Math.max(2, Math.floor(plotW / 90))
    for (const tick of ticks(t0, t1, nTicksX)) {
      ctx.fillStyle = AXIS_INK
      ctx.fillText(fmtClock(tick), x(tick), size.h - PAD.bottom + 8)
    }
    // Baseline.
    ctx.strokeStyle = BASELINE
    ctx.beginPath()
    ctx.moveTo(PAD.left, size.h - PAD.bottom)
    ctx.lineTo(size.w - PAD.right, size.h - PAD.bottom)
    ctx.stroke()

    // Series lines: 2px, round joins, broken across sampling gaps.
    ctx.lineWidth = 2
    ctx.lineJoin = "round"
    ctx.lineCap = "round"
    for (let s = 0; s < series.length; s++) {
      const pts = data.points[s]
      if (pts.length === 0) continue
      ctx.strokeStyle = series[s].color
      ctx.beginPath()
      let prevT = -Infinity
      for (const [t, v] of pts) {
        if (t - prevT > GAP_BREAK_S) ctx.moveTo(x(t), y(v))
        else ctx.lineTo(x(t), y(v))
        prevT = t
      }
      ctx.stroke()
    }

    // Crosshair + marker dots (2px surface ring so dots read over lines).
    if (hover) {
      const px = x(hover.t)
      ctx.strokeStyle = BASELINE
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(px, PAD.top)
      ctx.lineTo(px, size.h - PAD.bottom)
      ctx.stroke()
      for (let s = 0; s < series.length; s++) {
        const v = hover.values[s]
        if (v == null) continue
        ctx.beginPath()
        ctx.arc(px, y(v), 4, 0, Math.PI * 2)
        ctx.fillStyle = series[s].color
        ctx.fill()
        ctx.lineWidth = 2
        ctx.strokeStyle = "#161618"
        ctx.stroke()
      }
    }
  }, [data, size, series, hover, quietReason])

  const toTime = useCallback(
    (clientX: number) => {
      if (!data || !wrapRef.current) return null
      const rect = wrapRef.current.getBoundingClientRect()
      const frac = (clientX - rect.left - PAD.left) / (rect.width - PAD.left - PAD.right)
      return data.t0 + Math.min(1, Math.max(0, frac)) * (data.t1 - data.t0)
    },
    [data]
  )

  function onKeyDown(e: React.KeyboardEvent) {
    if (!data) return
    const span = data.t1 - data.t0
    const step = span / 60
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault()
      const dir = e.key === "ArrowLeft" ? -1 : 1
      setHoverT((t) => Math.min(data.t1, Math.max(data.t0, (t ?? data.t1) + dir * step)))
    } else if (e.key === "Escape") {
      setHoverT(null)
    }
  }

  // Tooltip placement: clamp inside the plot, flip sides near the right edge.
  const tooltipLeft =
    hover && data && size.w > 0
      ? PAD.left + ((hover.t - data.t0) / (data.t1 - data.t0 || 1)) * (size.w - PAD.left - PAD.right)
      : 0
  const flip = tooltipLeft > size.w - 190

  return (
    <Card className={cn("gap-3 p-4", className)}>
      <div className="flex items-baseline gap-2">
        <h3 className="font-heading text-sm font-semibold">{title}</h3>
        <span className="text-xs text-white/35">{unit}</span>
        {quietReason && data && (
          <span className="ml-auto text-xs text-amber-200/70">{quietReason}</span>
        )}
      </div>
      <div
        ref={wrapRef}
        role="img"
        aria-label={`${title} chart`}
        tabIndex={0}
        onKeyDown={onKeyDown}
        onPointerMove={(e) => setHoverT(toTime(e.clientX))}
        onPointerLeave={() => setHoverT(null)}
        onBlur={() => setHoverT(null)}
        className="relative outline-none focus-visible:ring-1 focus-visible:ring-[#eff483]/50"
        style={{ height: 220 }}
      >
        <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />
        {hover && (
          <div
            className="pointer-events-none absolute top-2 z-10 w-44 rounded-md border border-white/10 bg-[#1c1c1c]/95 px-2.5 py-2 text-xs shadow-xl"
            style={flip ? { right: size.w - tooltipLeft + 8 } : { left: tooltipLeft + 8 }}
          >
            <div className="mb-1 font-mono text-[0.65rem] text-white/40">
              {fmtClock(hover.t)}
            </div>
            {series.map((s, i) => (
              <div key={s.key} className="flex items-center gap-2 leading-5">
                <span
                  className="inline-block h-0.5 w-3 shrink-0 rounded"
                  style={{ background: s.color }}
                />
                <span className="font-mono font-semibold text-white/90 tabular-nums">
                  {hover.values[i] == null ? "–" : fmtValue(hover.values[i]!)}
                </span>
                <span className="truncate text-white/45 capitalize">{s.label}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      {/* Legend: the dependable identity channel + live value per series. */}
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {series.map((s, i) => (
          <span key={s.key} className="inline-flex items-center gap-1.5 text-xs">
            <span className="inline-block h-0.5 w-3 rounded" style={{ background: s.color }} />
            <span className="text-white/50 capitalize">{s.label}</span>
            <span className="font-mono text-white/80 tabular-nums">
              {latest[i] == null ? "–" : fmtValue(latest[i]!)}
            </span>
          </span>
        ))}
      </div>
    </Card>
  )
}
