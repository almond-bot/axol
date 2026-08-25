import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Maximize2, Minimize2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"

/**
 * One static line on a run chart. Each series carries its own time axis
 * (shifted to start at 0), so signals sampled at different rates overlay.
 */
export interface RunChartSeries {
  label: string
  color: string
  x: (number | null)[]
  data: (number | null)[]
}

interface RunChartProps {
  title: string
  unit?: string
  series: RunChartSeries[]
  /**
   * Optional error lane rendered under the main plot on a shared time axis
   * with its own (zero-centered) value scale. Position traces overlap when
   * tracking is decent, hiding the error at position scale — the lane makes
   * the residual visible at its own magnitude, so failures show up.
   */
  sub?: RunChartSeries[]
  height?: number
  className?: string
  id?: string
}

const GRID = "rgba(255,255,255,0.07)"
const BASELINE = "rgba(255,255,255,0.16)"
const AXIS_INK = "#898781"
const PAD = { top: 12, right: 12, bottom: 26, left: 52 }
const SUB_H = 56
const SUB_GAP = 12

/** Visible time window (relative seconds); null means the full run. */
interface View {
  t0: number
  t1: number
}

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
  return v.toFixed(4)
}

/** All of a pane's points, relative-time, precomputed once per run. */
function toPoints(series: RunChartSeries[]): { points: [number, number][][]; t1: number } {
  let t1 = 0
  const points: [number, number][][] = series.map(() => [])
  for (let s = 0; s < series.length; s++) {
    const { x, data } = series[s]
    let start: number | null = null
    for (const v of x) {
      if (v != null) {
        start = v
        break
      }
    }
    if (start == null) continue
    const n = Math.min(x.length, data.length)
    for (let i = 0; i < n; i++) {
      const t = x[i]
      const v = data[i]
      if (t == null || v == null || Number.isNaN(v)) continue
      const rel = t - start
      if (rel > t1) t1 = rel
      points[s].push([rel, v])
    }
  }
  return { points, t1 }
}

interface Pane {
  min: number
  max: number
  /** Per series, the visible slice (plus one point past each edge, so lines
   * run through the viewport instead of stopping at the first inside sample;
   * drawing is clipped to the plot rect). */
  points: [number, number][][]
}

function windowPane(
  all: [number, number][][],
  view: View,
  zeroCentered: boolean
): Pane | null {
  let min = Infinity
  let max = -Infinity
  const points: [number, number][][] = all.map((pts) => {
    let lo = 0
    while (lo < pts.length && pts[lo][0] < view.t0) lo++
    let hi = pts.length
    while (hi > lo && pts[hi - 1][0] > view.t1) hi--
    const slice = pts.slice(Math.max(0, lo - 1), Math.min(pts.length, hi + 1))
    for (let i = slice === pts ? 0 : 1; i < slice.length; i++) {
      const v = slice[i][1]
      if (v < min) min = v
      if (v > max) max = v
    }
    return slice
  })
  // Scale to strictly-visible values; fall back to edge points when the view
  // is narrower than the sample spacing.
  if (min === Infinity) {
    for (const pts of points) {
      for (const [, v] of pts) {
        if (v < min) min = v
        if (v > max) max = v
      }
    }
  }
  if (min === Infinity) return null
  if (min === max) {
    min -= 0.5
    max += 0.5
  }
  if (zeroCentered) {
    const bound = Math.max(Math.abs(min), Math.abs(max), 1e-9) * 1.1
    return { min: -bound, max: bound, points }
  }
  const pad = (max - min) * 0.08
  return { min: min - pad, max: max + pad, points }
}

/**
 * Static multi-series line chart for saved tuning runs: same visual language
 * and interactions as the live telemetry charts — canvas, hairline grid,
 * crosshair tooltip, wheel-zoom anchored at the cursor, drag-pan, and a
 * full-screen toggle (Esc exits) — over fixed arrays with a relative-seconds
 * time axis. Double-click resets the zoom. The y-scale (and the optional
 * zero-centered error lane's) refits to what's in view, so zooming into a
 * failure re-resolves it.
 */
export function RunChart({
  title,
  unit,
  series,
  sub,
  height = 240,
  className,
  id,
}: RunChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const [expanded, setExpanded] = useState(false)
  const [size, setSize] = useState({ w: 0, h: height })
  const [view, setView] = useState<View | null>(null)
  const [hoverT, setHoverT] = useState<number | null>(null)
  const drag = useRef<{ x: number; view: View; moved: boolean } | null>(null)

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }))
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

  const allMain = useMemo(() => toPoints(series), [series])
  const allSub = useMemo(() => (sub && sub.length > 0 ? toPoints(sub) : null), [sub])
  const fullT1 = Math.max(allMain.t1, allSub?.t1 ?? 0, 1e-6)
  const minSpan = fullT1 / 1000

  const v: View = view ?? { t0: 0, t1: fullT1 }
  const zoomed = view != null

  const main = useMemo(() => windowPane(allMain.points, v, false), [allMain, v.t0, v.t1]) // eslint-disable-line react-hooks/exhaustive-deps
  const subPane = useMemo(
    () => (allSub ? windowPane(allSub.points, v, true) : null),
    [allSub, v.t0, v.t1] // eslint-disable-line react-hooks/exhaustive-deps
  )
  const allSeries = useMemo(() => [...series, ...(sub ?? [])], [series, sub])
  const plotW = size.w - PAD.left - PAD.right

  const clampView = useCallback(
    (t0: number, t1: number): View => {
      const span = Math.min(Math.max(t1 - t0, minSpan), fullT1)
      let start = t0
      if (start + span > fullT1) start = fullT1 - span
      if (start < 0) start = 0
      return { t0: start, t1: start + span }
    },
    [fullT1, minSpan]
  )

  // Wheel-zoom needs preventDefault, so the listener must be non-passive.
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const frac = Math.min(
        1,
        Math.max(0, (e.clientX - rect.left - PAD.left) / (rect.width - PAD.left - PAD.right))
      )
      setView((prev) => {
        const cur = prev ?? { t0: 0, t1: fullT1 }
        const span = cur.t1 - cur.t0
        const newSpan = Math.min(Math.max(span * Math.exp(e.deltaY * 0.0015), minSpan), fullT1)
        const anchor = cur.t0 + frac * span
        const next = clampView(anchor - frac * newSpan, anchor + (1 - frac) * newSpan)
        return next.t0 <= 0 && next.t1 >= fullT1 ? null : next
      })
    }
    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  }, [clampView, fullT1, minSpan])

  const hover = useMemo(() => {
    if (hoverT == null || !main) return null
    const panes = subPane ? [main, subPane] : [main]
    let snapT: number | null = null
    let best = Infinity
    for (const pane of panes) {
      for (const pts of pane.points) {
        for (const [t] of pts) {
          const d = Math.abs(t - hoverT)
          if (d < best) {
            best = d
            snapT = t
          }
        }
      }
    }
    if (snapT == null) return null
    const window = (v.t1 - v.t0) / 100
    const values: (number | null)[] = []
    for (const pane of panes) {
      for (const pts of pane.points) {
        let value: number | null = null
        let dBest = window
        for (const [t, pv] of pts) {
          const d = Math.abs(t - snapT!)
          if (d <= dBest) {
            dBest = d
            value = pv
          }
        }
        values.push(value)
      }
    }
    return { t: snapT, values }
  }, [hoverT, main, subPane, v.t0, v.t1])

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

    const hasSub = subPane != null
    const mainH = size.h - PAD.top - PAD.bottom - (hasSub ? SUB_H + SUB_GAP : 0)
    if (plotW <= 0 || mainH <= 0) return
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace"

    if (!main) {
      ctx.fillStyle = AXIS_INK
      ctx.textAlign = "center"
      ctx.fillText("No data", size.w / 2, size.h / 2)
      return
    }

    const px = (t: number) => PAD.left + ((t - v.t0) / (v.t1 - v.t0 || 1)) * plotW

    const drawPane = (
      pane: Pane,
      top: number,
      h: number,
      lines: RunChartSeries[],
      nTicks: number,
      zeroLine: boolean
    ) => {
      const py = (val: number) => top + (1 - (val - pane.min) / (pane.max - pane.min)) * h
      ctx.textAlign = "right"
      ctx.textBaseline = "middle"
      for (const tick of ticks(pane.min, pane.max, nTicks)) {
        const y = py(tick)
        ctx.strokeStyle = GRID
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(PAD.left, y)
        ctx.lineTo(size.w - PAD.right, y)
        ctx.stroke()
        ctx.fillStyle = AXIS_INK
        ctx.fillText(fmtValue(tick), PAD.left - 6, y)
      }
      if (zeroLine && pane.min < 0 && pane.max > 0) {
        ctx.strokeStyle = BASELINE
        ctx.beginPath()
        ctx.moveTo(PAD.left, py(0))
        ctx.lineTo(size.w - PAD.right, py(0))
        ctx.stroke()
      }
      // Clip line drawing to the plot rect: the visible slice keeps one point
      // past each edge so lines run through the viewport, not into the axes.
      ctx.save()
      ctx.beginPath()
      ctx.rect(PAD.left, top - 2, plotW, h + 4)
      ctx.clip()
      ctx.lineJoin = "round"
      ctx.lineCap = "round"
      for (let s = 0; s < lines.length; s++) {
        const pts = pane.points[s]
        if (!pts || pts.length === 0) continue
        ctx.strokeStyle = lines[s].color
        ctx.lineWidth = 2
        ctx.beginPath()
        let first = true
        for (const [t, val] of pts) {
          if (first) {
            ctx.moveTo(px(t), py(val))
            first = false
          } else {
            ctx.lineTo(px(t), py(val))
          }
        }
        ctx.stroke()
      }
      ctx.restore()
      return py
    }

    const pyMain = drawPane(main, PAD.top, mainH, series, 4, false)
    let pySub: ((val: number) => number) | null = null
    if (subPane && sub) {
      pySub = drawPane(subPane, PAD.top + mainH + SUB_GAP, SUB_H, sub, 2, true)
    }

    ctx.textAlign = "center"
    ctx.textBaseline = "top"
    for (const tick of ticks(v.t0, v.t1, Math.max(2, Math.floor(plotW / 70)))) {
      ctx.fillStyle = AXIS_INK
      ctx.fillText(`${fmtValue(tick)}s`, px(tick), size.h - PAD.bottom + 8)
    }
    ctx.strokeStyle = BASELINE
    ctx.beginPath()
    ctx.moveTo(PAD.left, size.h - PAD.bottom)
    ctx.lineTo(size.w - PAD.right, size.h - PAD.bottom)
    ctx.stroke()

    if (hover && hover.t >= v.t0 && hover.t <= v.t1) {
      const hx = px(hover.t)
      ctx.strokeStyle = BASELINE
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(hx, PAD.top)
      ctx.lineTo(hx, size.h - PAD.bottom)
      ctx.stroke()
      for (let s = 0; s < allSeries.length; s++) {
        const val = hover.values[s]
        if (val == null) continue
        const py = s < series.length ? pyMain : pySub
        if (!py) continue
        ctx.beginPath()
        ctx.arc(hx, py(val), 4, 0, Math.PI * 2)
        ctx.fillStyle = allSeries[s].color
        ctx.fill()
        ctx.lineWidth = 2
        ctx.strokeStyle = "#161618"
        ctx.stroke()
      }
    }
  }, [main, subPane, size, series, sub, allSeries, hover, plotW, v.t0, v.t1])

  const toTime = useCallback(
    (clientX: number): number | null => {
      if (!wrapRef.current) return null
      const rect = wrapRef.current.getBoundingClientRect()
      const frac = (clientX - rect.left - PAD.left) / (rect.width - PAD.left - PAD.right)
      return v.t0 + Math.min(1, Math.max(0, frac)) * (v.t1 - v.t0)
    },
    [v.t0, v.t1]
  )

  function onPointerDown(e: React.PointerEvent) {
    if (e.button !== 0) return
    drag.current = { x: e.clientX, view: v, moved: false }
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
  }

  function onPointerMove(e: React.PointerEvent) {
    const d = drag.current
    if (d) {
      const dx = e.clientX - d.x
      if (d.moved || Math.abs(dx) > 3) {
        d.moved = true
        setHoverT(null)
        const span = d.view.t1 - d.view.t0
        const dt = (-dx / Math.max(1, plotW)) * span
        const next = clampView(d.view.t0 + dt, d.view.t1 + dt)
        setView(next.t0 <= 0 && next.t1 >= fullT1 ? null : next)
        return
      }
    }
    setHoverT(toTime(e.clientX))
  }

  function onPointerUp() {
    drag.current = null
  }

  const tooltipLeft =
    hover && size.w > 0 ? PAD.left + ((hover.t - v.t0) / (v.t1 - v.t0 || 1)) * plotW : 0
  const flip = tooltipLeft > size.w - 200

  const card = (
    // The Card surface is normally a translucent wash; full screen needs a
    // solid one so the page behind doesn't bleed through the plot.
    <Card id={id} className={cn("gap-2 p-4", expanded && "h-full w-full bg-[#161618]", className)}>
      <div className="flex items-baseline gap-2">
        <h3 className="font-heading text-sm font-semibold">{title}</h3>
        {unit && <span className="text-xs text-white/35">{unit}</span>}
        <span className="ml-auto flex items-center gap-1 self-center">
          {zoomed && (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-xs text-white/40"
              onClick={() => setView(null)}
              title="Reset zoom (double-click the plot)"
            >
              reset zoom
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="size-7 text-white/40"
            onClick={() => setExpanded((x) => !x)}
            aria-label={expanded ? "Exit full screen" : "Full screen"}
            title={expanded ? "Exit full screen (Esc)" : "Full screen"}
          >
            {expanded ? <Minimize2 /> : <Maximize2 />}
          </Button>
        </span>
      </div>
      <div
        ref={wrapRef}
        role="img"
        aria-label={`${title} chart`}
        className={cn("relative cursor-crosshair touch-none", expanded && "min-h-0 flex-1")}
        style={expanded ? undefined : { height }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => {
          setHoverT(null)
          drag.current = null
        }}
        onDoubleClick={() => setView(null)}
      >
        <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />
        {hover && (
          <div
            className="pointer-events-none absolute top-2 z-10 w-48 rounded-md border border-white/10 bg-[#1c1c1c]/95 px-2.5 py-2 text-xs shadow-xl"
            style={flip ? { right: size.w - tooltipLeft + 8 } : { left: tooltipLeft + 8 }}
          >
            <div className="mb-1 font-mono text-[0.65rem] text-white/40">
              t = {fmtValue(hover.t)}s
            </div>
            {allSeries.map((s, i) => (
              <div key={`${s.label}-${i}`} className="flex items-center gap-2 leading-5">
                <span
                  className="inline-block h-0.5 w-3 shrink-0 rounded"
                  style={{ background: s.color }}
                />
                <span className="font-mono font-semibold text-white/90 tabular-nums">
                  {hover.values[i] == null ? "–" : fmtValue(hover.values[i]!)}
                </span>
                <span className="truncate text-white/45">{s.label}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        {allSeries.map((s, i) => (
          <span key={`${s.label}-${i}`} className="inline-flex items-center gap-1.5 text-xs">
            <span className="inline-block h-0.5 w-3 rounded" style={{ background: s.color }} />
            <span className="text-white/50">{s.label}</span>
          </span>
        ))}
        <span className="ml-auto text-[0.65rem] text-white/25">
          scroll to zoom · drag to pan · double-click resets
        </span>
      </div>
    </Card>
  )

  if (expanded) {
    return <div className="fixed inset-0 z-50 flex bg-black/85 p-4 sm:p-8">{card}</div>
  }
  return card
}
