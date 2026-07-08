import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Maximize2, Minimize2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import type { TelemetryFrame } from "@/lib/telemetry"

/** One plotted series: `key` indexes frame.m, color is the joint's fixed slot. */
export interface ChartSeries {
  key: string
  label: string
  color: string
}

/** Visible time range (epoch seconds). Charts sharing a view stay in sync. */
export interface ChartView {
  t0: number
  t1: number
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
  view: ChartView
  /** Called on wheel-zoom / drag-pan; the parent owns the view. */
  onViewChange?: (view: ChartView) => void
  /** Why the stream is quiet (shown as an in-plot notice), if it is. */
  quietReason?: string | null
  /** Plot height in px (fullscreen mode sizes itself). */
  height?: number
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
// Zoom limits (visible span, seconds).
const MIN_SPAN_S = 3
const MAX_SPAN_S = 3600

interface Extracted {
  /** Per series, [t, value] pairs within the window (strided). */
  points: [number, number][][]
  min: number
  max: number
}

function extract(
  frames: TelemetryFrame[],
  series: ChartSeries[],
  metric: number,
  view: ChartView
): Extracted | null {
  if (frames.length === 0 || series.length === 0) return null
  let lo = 0
  while (lo < frames.length && frames[lo].t < view.t0) lo++
  let hi = frames.length
  while (hi > lo && frames[hi - 1].t > view.t1) hi--
  const visible = frames.slice(lo, hi)
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
  return { points, min: min - pad, max: max + pad }
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
 * Interactive multi-series line chart on canvas: 2px round-joined lines,
 * hairline solid grid, crosshair + all-series tooltip (pointer or ←/→ on
 * keyboard focus), wheel-zoom and drag-pan on the time axis, and a persistent
 * legend with live per-series values under the plot.
 */
export function TelemetryChart({
  title,
  unit,
  series,
  frames,
  version,
  metric,
  view,
  onViewChange,
  quietReason,
  height = 260,
  className,
}: TelemetryChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const [expanded, setExpanded] = useState(false)
  const [size, setSize] = useState({ w: 0, h: height })
  // Hovered/focused time (seconds); null hides the crosshair.
  const [hoverT, setHoverT] = useState<number | null>(null)
  const drag = useRef<{ x: number; view: ChartView; moved: boolean } | null>(null)

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      setSize({ w: el.clientWidth, h: el.clientHeight })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [expanded])

  // Fullscreen closes on Escape.
  useEffect(() => {
    if (!expanded) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setExpanded(false)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [expanded])

  const data = useMemo(
    () => extract(frames, series, metric, view),
    // The live buffer is mutated in place (stable identity) — `version` is its
    // change signal; static charts pass a fresh `frames` array instead.
    // eslint-disable-next-line react-hooks/exhaustive-deps -- version is the buffer's change signal
    [frames, version, series, metric, view]
  )

  const plotW = size.w - PAD.left - PAD.right

  /** Clamp a candidate view to the data extent (with a little slack). */
  const clampView = useCallback(
    (t0: number, t1: number): ChartView => {
      const span = Math.min(Math.max(t1 - t0, MIN_SPAN_S), MAX_SPAN_S)
      // Bound pan/zoom to the data's own timestamps (server clock) rather than
      // the browser's Date.now(): a skewed client clock must not let the view
      // scroll past the newest sample into empty space.
      const dataMin = frames.length > 0 ? frames[0].t : t0
      const dataMax = frames.length > 0 ? frames[frames.length - 1].t : t1
      let start = t0
      if (start + span > dataMax) start = dataMax - span
      if (start < dataMin - span * 0.5) start = dataMin - span * 0.5
      return { t0: start, t1: start + span }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps -- version is the buffer's change signal
    [frames, version]
  )

  // Wheel-zoom needs preventDefault, so the listener must be non-passive.
  useEffect(() => {
    const el = wrapRef.current
    if (!el || !onViewChange) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const frac = Math.min(
        1,
        Math.max(0, (e.clientX - rect.left - PAD.left) / (rect.width - PAD.left - PAD.right))
      )
      const span = view.t1 - view.t0
      const newSpan = Math.min(Math.max(span * Math.exp(e.deltaY * 0.0015), MIN_SPAN_S), MAX_SPAN_S)
      const anchor = view.t0 + frac * span
      onViewChange(clampView(anchor - frac * newSpan, anchor + (1 - frac) * newSpan))
    }
    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  }, [onViewChange, view, clampView])

  // Values shown in the tooltip + the frame the crosshair snaps to.
  const hover = useMemo(() => {
    if (hoverT == null || !data) return null
    let snapT: number | null = null
    let bestDist = Infinity
    for (const pts of data.points) {
      for (const [t] of pts) {
        const d = Math.abs(t - hoverT)
        if (d < bestDist) {
          bestDist = d
          snapT = t
        }
      }
      if (pts.length > 0) break // frames are shared — one series is enough
    }
    if (snapT == null) return null
    const values = data.points.map((pts) => {
      let v: number | null = null
      let dBest = 0.6 // within ~half a second counts as "at" the crosshair
      for (const [t, value] of pts) {
        const d = Math.abs(t - snapT!)
        if (d < dBest) {
          dBest = d
          v = value
        }
      }
      return v
    })
    return { t: snapT, values }
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

    const plotH = size.h - PAD.top - PAD.bottom
    if (plotW <= 0 || plotH <= 0) return

    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace"

    if (!data) {
      ctx.fillStyle = AXIS_INK
      ctx.textAlign = "center"
      ctx.fillText(
        quietReason ?? (series.length === 0 ? "No joints selected" : "No telemetry in view"),
        size.w / 2,
        size.h / 2
      )
      return
    }

    const { min, max } = data
    const x = (t: number) => PAD.left + ((t - view.t0) / (view.t1 - view.t0 || 1)) * plotW
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
    for (const tick of ticks(view.t0, view.t1, nTicksX)) {
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
  }, [data, size, series, hover, quietReason, view, plotW])

  const toTime = useCallback(
    (clientX: number) => {
      if (!wrapRef.current) return null
      const rect = wrapRef.current.getBoundingClientRect()
      const frac = (clientX - rect.left - PAD.left) / (rect.width - PAD.left - PAD.right)
      return view.t0 + Math.min(1, Math.max(0, frac)) * (view.t1 - view.t0)
    },
    [view]
  )

  function onPointerDown(e: React.PointerEvent) {
    if (!onViewChange || e.button !== 0) return
    drag.current = { x: e.clientX, view, moved: false }
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
  }

  function onPointerMove(e: React.PointerEvent) {
    const d = drag.current
    if (d && onViewChange) {
      const dx = e.clientX - d.x
      if (d.moved || Math.abs(dx) > 3) {
        d.moved = true
        setHoverT(null)
        const span = d.view.t1 - d.view.t0
        const dt = (-dx / Math.max(1, plotW)) * span
        onViewChange(clampView(d.view.t0 + dt, d.view.t1 + dt))
        return
      }
    }
    setHoverT(toTime(e.clientX))
  }

  function onPointerUp() {
    drag.current = null
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (!data) return
    const span = view.t1 - view.t0
    const step = span / 60
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault()
      const dir = e.key === "ArrowLeft" ? -1 : 1
      setHoverT((t) => Math.min(view.t1, Math.max(view.t0, (t ?? view.t1) + dir * step)))
    } else if (e.key === "Escape") {
      setHoverT(null)
    }
  }

  // Tooltip placement: clamp inside the plot, flip sides near the right edge.
  const tooltipLeft =
    hover && size.w > 0
      ? PAD.left + ((hover.t - view.t0) / (view.t1 - view.t0 || 1)) * plotW
      : 0
  const flip = tooltipLeft > size.w - 190

  const card = (
    // The Card surface is normally a translucent wash; full screen needs a
    // solid one so the page behind doesn't bleed through the plot.
    <Card
      className={cn("gap-3 p-4", expanded && "h-full w-full bg-[#161618]", className)}
    >
      <div className="flex items-baseline gap-2">
        <h3 className="font-heading text-sm font-semibold">{title}</h3>
        <span className="text-xs text-white/35">{unit}</span>
        {quietReason && data && (
          <span className="ml-auto text-xs text-amber-200/70">{quietReason}</span>
        )}
        <Button
          variant="ghost"
          size="icon"
          className={cn("size-7 self-center text-white/40", !quietReason && "ml-auto")}
          onClick={() => setExpanded((v) => !v)}
          aria-label={expanded ? "Exit full screen" : "Full screen"}
          title={expanded ? "Exit full screen (Esc)" : "Full screen"}
        >
          {expanded ? <Minimize2 /> : <Maximize2 />}
        </Button>
      </div>
      <div
        ref={wrapRef}
        role="img"
        aria-label={`${title} chart`}
        tabIndex={0}
        onKeyDown={onKeyDown}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => {
          setHoverT(null)
          drag.current = null
        }}
        onBlur={() => setHoverT(null)}
        className={cn(
          "relative touch-none outline-none focus-visible:ring-1 focus-visible:ring-[#eff483]/50",
          onViewChange && "cursor-crosshair",
          expanded && "min-h-0 flex-1"
        )}
        style={expanded ? undefined : { height }}
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

  if (expanded) {
    return <div className="fixed inset-0 z-50 flex bg-black/85 p-4 sm:p-8">{card}</div>
  }
  return card
}
