import { useMemo, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import {
  TelemetryChart,
  type ChartSeries,
  type ChartView,
} from "@/components/diagnostics/telemetry-chart"
import { cn } from "@/lib/utils"
import type { ArmSide, ControlTiming, TelemetryFrame, TimingFrame } from "@/lib/telemetry"

const SIDES: ArmSide[] = ["left", "right"]

const METRICS = [
  { key: "rate", label: "Loop rate", title: "On-wire control rate", unit: "Hz", index: 0 },
  { key: "jitter", label: "Jitter", title: "CAN period jitter (p95)", unit: "ms", index: 1 },
  {
    key: "latency",
    label: "Latency",
    title: "Per-motor command → feedback latency (p95)",
    unit: "ms",
    index: 2,
  },
  {
    key: "cycle",
    label: "CAN cycle",
    title: "Full-arm CAN transaction time (p95)",
    unit: "ms",
    index: 3,
  },
  {
    key: "gap",
    label: "Max gap",
    title: "Worst control interval (rolling 1s)",
    unit: "ms",
    index: 4,
  },
  {
    key: "misses",
    label: "Misses",
    title: "Timing failures (rolling 1s)",
    unit: "count",
    index: 5,
  },
] as const

const COLORS = {
  leftCommand: "#3987e5",
  leftFeedback: "#76b5ff",
  rightCommand: "#d55181",
  rightFeedback: "#ff8caf",
  leftCycle: "#54d6dc",
  rightCycle: "#f2a35e",
  target: "#898781",
}

type TimingSeries = "command" | "feedback" | "roundtrip" | "send" | "reply" | "cycle"

function sample(timing: ControlTiming, kind: TimingSeries): (number | null)[] {
  if (kind === "command") {
    return [
      timing.commandHz,
      timing.commandJitterP95Ms,
      null,
      null,
      timing.commandGapMaxMs,
      timing.deadlineMisses,
    ]
  }
  if (kind === "feedback") {
    return [
      timing.feedbackHz,
      timing.feedbackJitterP95Ms,
      null,
      null,
      timing.feedbackGapMaxMs,
      timing.missedFeedback,
    ]
  }
  if (kind === "roundtrip") return [null, null, timing.roundTripP95Ms, null, null, null]
  if (kind === "send") return [null, null, null, timing.commandBatchP95Ms, null, null]
  if (kind === "reply") return [null, null, null, timing.feedbackBatchP95Ms, null, null]
  return [null, null, null, timing.canCycleP95Ms, null, null]
}

/** Adapt timing messages to the generic chart's keyed numeric-series shape. */
function chartFrames(frames: TimingFrame[]): TelemetryFrame[] {
  return frames.map((frame) => {
    const m: TelemetryFrame["m"] = {
      target: [240, null, null, null, 1000 / 240, null],
    }
    for (const side of SIDES) {
      const timing = frame.arms[side]
      if (!timing) continue
      m[`${side}:command`] = sample(timing, "command")
      m[`${side}:feedback`] = sample(timing, "feedback")
      m[`${side}:roundtrip`] = sample(timing, "roundtrip")
      m[`${side}:send`] = sample(timing, "send")
      m[`${side}:reply`] = sample(timing, "reply")
      m[`${side}:cycle`] = sample(timing, "cycle")
    }
    return { t: frame.t, m }
  })
}

function fmt(value: number | null | undefined, digits = 1): string {
  return value == null || !Number.isFinite(value) ? "–" : value.toFixed(digits)
}

function sourceLabel(value: string): string {
  return value.toLowerCase().replace("_", " ")
}

interface ControlHealthProps {
  frames: TimingFrame[]
  version: number
  /** Newest server timestamp from any live stream; keeps freshness checks pure. */
  nowT: number
  view: ChartView
  onViewChange: (view: ChartView) => void
}

/** Live proof of the cadence and response timing that reached the CAN wire. */
export function ControlHealth({ frames, version, nowT, view, onViewChange }: ControlHealthProps) {
  const [metricKey, setMetricKey] = useState<(typeof METRICS)[number]["key"]>("rate")
  const metric = METRICS.find((item) => item.key === metricKey) ?? METRICS[0]
  const converted = useMemo(
    () => chartFrames(frames),
    // The stream mutates its buffer in place; version is the change signal.
    // eslint-disable-next-line react-hooks/exhaustive-deps -- version is intentional
    [frames, version]
  )
  const latest = useMemo(() => {
    const out: Partial<Record<ArmSide, { timing: ControlTiming; t: number }>> = {}
    for (let i = frames.length - 1; i >= 0 && Object.keys(out).length < SIDES.length; i--) {
      for (const side of SIDES) {
        const timing = frames[i].arms[side]
        if (timing && !out[side]) out[side] = { timing, t: frames[i].t }
      }
    }
    return out
    // eslint-disable-next-line react-hooks/exhaustive-deps -- version is intentional
  }, [frames, version])

  const commandSeries: ChartSeries[] = SIDES.map((side) => ({
    key: `${side}:command`,
    label: `${side} command`,
    color: side === "left" ? COLORS.leftCommand : COLORS.rightCommand,
  }))
  const feedbackSeries: ChartSeries[] = SIDES.map((side) => ({
    key: `${side}:feedback`,
    label: `${side} feedback`,
    color: side === "left" ? COLORS.leftFeedback : COLORS.rightFeedback,
  }))
  const series: ChartSeries[] =
    metric.key === "rate"
      ? [
          ...commandSeries,
          ...feedbackSeries,
          { key: "target", label: "RT target", color: COLORS.target },
        ]
      : metric.key === "jitter"
        ? [...commandSeries, ...feedbackSeries]
        : metric.key === "latency"
          ? SIDES.map((side) => ({
              key: `${side}:roundtrip`,
              label: `${side} motor round-trip`,
              color: side === "left" ? COLORS.leftCommand : COLORS.rightCommand,
            }))
          : metric.key === "cycle"
            ? SIDES.flatMap((side) => [
                {
                  key: `${side}:send`,
                  label: `${side} send batch`,
                  color: side === "left" ? COLORS.leftCommand : COLORS.rightCommand,
                },
                {
                  key: `${side}:reply`,
                  label: `${side} reply batch`,
                  color: side === "left" ? COLORS.leftFeedback : COLORS.rightFeedback,
                },
                {
                  key: `${side}:cycle`,
                  label: `${side} full cycle`,
                  color: side === "left" ? COLORS.leftCycle : COLORS.rightCycle,
                },
              ])
            : metric.key === "gap"
              ? [
                  ...commandSeries,
                  ...feedbackSeries,
                  { key: "target", label: "4.17 ms target", color: COLORS.target },
                ]
              : [...commandSeries, ...feedbackSeries]

  const newest = frames[frames.length - 1]?.t ?? 0
  const stale = newest === 0 || nowT - newest > 2
  const quietReason =
    frames.length === 0
      ? "start teleop and wait for PyRoKi to finish"
      : stale
        ? "control traffic stopped"
        : null

  return (
    <section className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="mr-2">
          <h2 className="font-heading text-base font-semibold">Control loop & CAN timing</h2>
          <p className="text-xs text-white/35">
            Post-PyRoKi teleop only · passive kernel timestamps · rolling 1s · zero added bus
            traffic
          </p>
        </div>
        <div className="flex overflow-hidden rounded-md border border-white/10">
          {METRICS.map((item) => (
            <button
              key={item.key}
              type="button"
              onClick={() => setMetricKey(item.key)}
              className={cn(
                "px-2.5 py-1 text-xs transition-colors",
                metric.key === item.key
                  ? "bg-[#eff483]/15 text-[#eff483]"
                  : "text-white/50 hover:bg-white/[0.05]"
              )}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {SIDES.map((side) => {
          const reading = latest[side]
          const timing = reading?.timing
          const isFresh = reading != null && nowT - reading.t <= 2
          const clean =
            isFresh &&
            timing != null &&
            timing.commandHz != null &&
            timing.feedbackHz != null &&
            Math.abs(timing.commandHz - timing.targetHz) <= timing.targetHz * 0.02 &&
            Math.abs(timing.feedbackHz - timing.targetHz) <= timing.targetHz * 0.02 &&
            timing.commandJitterP95Ms != null &&
            timing.commandJitterP95Ms <= 0.5 &&
            timing.feedbackJitterP95Ms != null &&
            timing.feedbackJitterP95Ms <= 0.5 &&
            timing.roundTripP95Ms != null &&
            timing.roundTripP95Ms <= 3 &&
            timing.canHeadroomP05Ms != null &&
            timing.canHeadroomP05Ms >= 0.25 &&
            timing.deadlineMisses === 0 &&
            timing.missedFeedback === 0
          return (
            <Card key={side} className="gap-3 p-3.5">
              <div className="flex items-center gap-2">
                <h3 className="font-heading text-sm font-semibold capitalize">{side} arm</h3>
                <Badge variant={clean ? "success" : isFresh ? "warning" : "neutral"}>
                  {clean ? "240 Hz clean" : isFresh ? "timing issue" : "idle"}
                </Badge>
                {timing && (
                  <span className="ml-auto text-[0.65rem] text-white/30">
                    via {sourceLabel(timing.sourceJoint)}
                  </span>
                )}
              </div>
              <div className="grid grid-cols-3 gap-x-3 gap-y-2 text-xs">
                <div>
                  <p className="text-white/35">command</p>
                  <p className="font-mono text-sm text-white/85 tabular-nums">
                    {fmt(timing?.commandHz)}{" "}
                    <span className="text-[0.65rem] text-white/35">Hz</span>
                  </p>
                </div>
                <div>
                  <p className="text-white/35">feedback</p>
                  <p className="font-mono text-sm text-white/85 tabular-nums">
                    {fmt(timing?.feedbackHz)}{" "}
                    <span className="text-[0.65rem] text-white/35">Hz</span>
                  </p>
                </div>
                <div>
                  <p className="text-white/35">full cycle p95</p>
                  <p className="font-mono text-sm text-white/85 tabular-nums">
                    {fmt(timing?.canCycleP95Ms, 2)}{" "}
                    <span className="text-[0.65rem] text-white/35">ms</span>
                  </p>
                </div>
                <div>
                  <p className="text-white/35">send batch p95</p>
                  <p className="font-mono text-white/75 tabular-nums">
                    {fmt(timing?.commandBatchP95Ms, 2)} ms
                  </p>
                </div>
                <div>
                  <p className="text-white/35">reply batch p95</p>
                  <p className="font-mono text-white/75 tabular-nums">
                    {fmt(timing?.feedbackBatchP95Ms, 2)} ms
                  </p>
                </div>
                <div>
                  <p className="text-white/35">motor RTT p95</p>
                  <p className="font-mono text-white/75 tabular-nums">
                    {fmt(timing?.roundTripP95Ms, 2)} ms
                  </p>
                </div>
                <div>
                  <p className="text-white/35">command jitter</p>
                  <p className="font-mono text-white/75 tabular-nums">
                    {fmt(timing?.commandJitterP95Ms, 2)} ms
                  </p>
                </div>
                <div>
                  <p className="text-white/35">worst command gap</p>
                  <p className="font-mono text-white/75 tabular-nums">
                    {fmt(timing?.commandGapMaxMs, 2)} ms
                  </p>
                </div>
                <div>
                  <p className="text-white/35">CAN occupied p95</p>
                  <p className="font-mono text-white/75 tabular-nums">
                    {fmt(timing?.canUtilizationP95Pct, 0)}%
                  </p>
                </div>
                <div>
                  <p className="text-white/35">missed 240 Hz</p>
                  <p className="font-mono text-white/75 tabular-nums">
                    {timing?.deadlineMisses ?? "–"}/s
                  </p>
                </div>
                <div>
                  <p className="text-white/35">lost feedback</p>
                  <p className="font-mono text-white/75 tabular-nums">
                    {timing?.missedFeedback ?? "–"}/s
                  </p>
                </div>
                <div>
                  <p className="text-white/35">headroom p05</p>
                  <p className="font-mono text-white/75 tabular-nums">
                    {fmt(timing?.canHeadroomP05Ms, 2)} ms
                  </p>
                </div>
              </div>
            </Card>
          )
        })}
      </div>

      <TelemetryChart
        title={metric.title}
        unit={metric.unit}
        series={series}
        frames={converted}
        version={version}
        metric={metric.index}
        view={view}
        onViewChange={onViewChange}
        quietReason={quietReason}
        height={300}
      />
      <p className="text-xs leading-relaxed text-white/35">
        Send batch is first-to-last arm command; full cycle continues through the final feedback.
        “Missed 240 Hz” counts command gaps that lost one or more 4.17 ms deadlines. All values come
        from passive kernel-timestamped evidence on the Rust-owned CAN wire.
      </p>
    </section>
  )
}
