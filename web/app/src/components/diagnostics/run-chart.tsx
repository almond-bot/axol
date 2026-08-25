import { useEffect, useMemo, useRef, useState } from "react"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"

/**
 * One static line on a run chart. Each series carries its own time axis so
 * two runs recorded at different times/rates can overlay (both are shifted
 * to start at 0).
 */
export interface RunChartSeries {
  label: string
  color: string
  x: (number | null)[]
  data: (number | null)[]
  /** Dashed stroke — the B run in an A/B overlay. */
  dashed?: boolean
}

interface RunChartProps {
  title: string
  unit?: string
  series: RunChartSeries[]
  height?: number
  className?: string
}

const GRID = "rgba(255,255,255,0.07)"
const BASELINE = "rgba(255,255,255,0.16)"
const AXIS_INK = "#898781"
const PAD = { top: 12, right: 12, bottom: 26, left: 52 }

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

interface Prepared {
  t1: number
  min: number
  max: number
  /** Per series, [t, value] with each series' time offset removed. */
  points: [number, number][][]
}

function prepare(series: RunChartSeries[]): Prepared | null {
  let t1 = 0
  let min = Infinity
  let max = -Infinity
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
      if (v < min) min = v
      if (v > max) max = v
    }
  }
  if (min === Infinity) return null
  if (min === max) {
    min -= 0.5
    max += 0.5
  }
  const pad = (max - min) * 0.08
  return { t1: Math.max(t1, 1e-6), min: min - pad, max: max + pad, points }
}

function swatchStyle(s: RunChartSeries): React.CSSProperties {
  return {
    background: s.dashed
      ? `repeating-linear-gradient(90deg, ${s.color} 0 4px, transparent 4px 7px)`
      : s.color,
  }
}

/**
 * Static multi-series line chart for saved tuning runs: same visual language
 * as the live telemetry charts (canvas, hairline grid, crosshair tooltip),
 * but over fixed arrays with a relative-seconds time axis. Dashed strokes
 * carry the B run of an A/B overlay.
 */
export function RunChart({ title, unit, series, height = 240, className }: RunChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  const [hoverT, setHoverT] = useState<number | null>(null)

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setWidth(el.clientWidth))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const data = useMemo(() => prepare(series), [series])
  const plotW = width - PAD.left - PAD.right

  const hover = useMemo(() => {
    if (hoverT == null || !data) return null
    let snapT: number | null = null
    let best = Infinity
    for (const pts of data.points) {
      for (const [t] of pts) {
        const d = Math.abs(t - hoverT)
        if (d < best) {
          best = d
          snapT = t
        }
      }
    }
    if (snapT == null) return null
    const window = data.t1 / 100
    const values = data.points.map((pts) => {
      let v: number | null = null
      let dBest = window
      for (const [t, value] of pts) {
        const d = Math.abs(t - snapT!)
        if (d <= dBest) {
          dBest = d
          v = value
        }
      }
      return v
    })
    return { t: snapT, values }
  }, [hoverT, data])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || width === 0) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, width, height)

    const plotH = height - PAD.top - PAD.bottom
    if (plotW <= 0 || plotH <= 0) return
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace"

    if (!data) {
      ctx.fillStyle = AXIS_INK
      ctx.textAlign = "center"
      ctx.fillText("No data", width / 2, height / 2)
      return
    }

    const { t1, min, max } = data
    const px = (t: number) => PAD.left + (t / t1) * plotW
    const py = (v: number) => PAD.top + (1 - (v - min) / (max - min)) * plotH

    ctx.textAlign = "right"
    ctx.textBaseline = "middle"
    for (const tick of ticks(min, max, 4)) {
      const y = py(tick)
      ctx.strokeStyle = GRID
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(PAD.left, y)
      ctx.lineTo(width - PAD.right, y)
      ctx.stroke()
      ctx.fillStyle = AXIS_INK
      ctx.fillText(fmtValue(tick), PAD.left - 6, y)
    }
    ctx.textAlign = "center"
    ctx.textBaseline = "top"
    for (const tick of ticks(0, t1, Math.max(2, Math.floor(plotW / 70)))) {
      ctx.fillStyle = AXIS_INK
      ctx.fillText(`${fmtValue(tick)}s`, px(tick), height - PAD.bottom + 8)
    }
    ctx.strokeStyle = BASELINE
    ctx.beginPath()
    ctx.moveTo(PAD.left, height - PAD.bottom)
    ctx.lineTo(width - PAD.right, height - PAD.bottom)
    ctx.stroke()

    ctx.lineJoin = "round"
    ctx.lineCap = "round"
    for (let s = 0; s < series.length; s++) {
      const pts = data.points[s]
      if (pts.length === 0) continue
      ctx.strokeStyle = series[s].color
      ctx.lineWidth = series[s].dashed ? 1.5 : 2
      ctx.setLineDash(series[s].dashed ? [5, 4] : [])
      ctx.beginPath()
      let first = true
      for (const [t, v] of pts) {
        if (first) {
          ctx.moveTo(px(t), py(v))
          first = false
        } else {
          ctx.lineTo(px(t), py(v))
        }
      }
      ctx.stroke()
    }
    ctx.setLineDash([])

    if (hover) {
      const hx = px(hover.t)
      ctx.strokeStyle = BASELINE
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(hx, PAD.top)
      ctx.lineTo(hx, height - PAD.bottom)
      ctx.stroke()
      for (let s = 0; s < series.length; s++) {
        const v = hover.values[s]
        if (v == null) continue
        ctx.beginPath()
        ctx.arc(hx, py(v), 4, 0, Math.PI * 2)
        ctx.fillStyle = series[s].color
        ctx.fill()
        ctx.lineWidth = 2
        ctx.strokeStyle = "#161618"
        ctx.stroke()
      }
    }
  }, [data, width, height, series, hover, plotW])

  function toTime(clientX: number): number | null {
    if (!wrapRef.current || !data) return null
    const rect = wrapRef.current.getBoundingClientRect()
    const frac = (clientX - rect.left - PAD.left) / (rect.width - PAD.left - PAD.right)
    return Math.min(1, Math.max(0, frac)) * data.t1
  }

  const tooltipLeft = hover && data && width > 0 ? PAD.left + (hover.t / data.t1) * plotW : 0
  const flip = tooltipLeft > width - 200

  return (
    <Card className={cn("gap-2 p-4", className)}>
      <div className="flex items-baseline gap-2">
        <h3 className="font-heading text-sm font-semibold">{title}</h3>
        {unit && <span className="text-xs text-white/35">{unit}</span>}
      </div>
      <div
        ref={wrapRef}
        role="img"
        aria-label={`${title} chart`}
        className="relative cursor-crosshair touch-none"
        style={{ height }}
        onPointerMove={(e) => setHoverT(toTime(e.clientX))}
        onPointerLeave={() => setHoverT(null)}
      >
        <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />
        {hover && (
          <div
            className="pointer-events-none absolute top-2 z-10 w-48 rounded-md border border-white/10 bg-[#1c1c1c]/95 px-2.5 py-2 text-xs shadow-xl"
            style={flip ? { right: width - tooltipLeft + 8 } : { left: tooltipLeft + 8 }}
          >
            <div className="mb-1 font-mono text-[0.65rem] text-white/40">
              t = {fmtValue(hover.t)}s
            </div>
            {series.map((s, i) => (
              <div key={`${s.label}-${i}`} className="flex items-center gap-2 leading-5">
                <span className="inline-block h-0.5 w-3 shrink-0 rounded" style={swatchStyle(s)} />
                <span className="font-mono font-semibold text-white/90 tabular-nums">
                  {hover.values[i] == null ? "–" : fmtValue(hover.values[i]!)}
                </span>
                <span className="truncate text-white/45">{s.label}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {series.map((s, i) => (
          <span key={`${s.label}-${i}`} className="inline-flex items-center gap-1.5 text-xs">
            <span className="inline-block h-0.5 w-3 rounded" style={swatchStyle(s)} />
            <span className="text-white/50">{s.label}</span>
          </span>
        ))}
      </div>
    </Card>
  )
}
