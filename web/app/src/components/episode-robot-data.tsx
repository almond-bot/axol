import { lazy, Suspense, useEffect, useMemo, useState } from "react"
import { CircleAlert, CircleCheck, Loader2 } from "lucide-react"
import { TelemetryChart, type ChartView } from "@/components/diagnostics/telemetry-chart"
import { Switch } from "@/components/ui/switch"
import {
  episodeCharts,
  formatEpisodeTime,
  frameAt,
  hasJointState,
  withJointPoses,
  type ChartGroup,
} from "@/lib/episode-data"
import {
  fetchEpisodeData,
  type AxolModel,
  type DatasetEpisode,
  type EpisodeData,
  type GripperType,
} from "@/lib/supervisor"
import { cn } from "@/lib/utils"

// three.js only loads once an episode's data is shown.
const EpisodeReplay = lazy(() => import("@/components/episode-replay"))

// A jump between two frames larger than this reads as a glitch: a re-engage,
// lost tracking or a bad IK solution.
const JOINT_JUMP_WARN_DEG = 5
const POSE_JUMP_WARN_M = 0.03

const GROUPS: { key: ChartGroup; label: string }[] = [
  { key: "joints", label: "Joints" },
  { key: "pose", label: "Gripper pose" },
  { key: "gripper", label: "Gripper opening" },
]

function Check({
  ok,
  children,
  onClick,
}: {
  ok: boolean
  children: React.ReactNode
  onClick?: () => void
}) {
  const Icon = ok ? CircleCheck : CircleAlert
  const body = (
    <>
      <Icon className={cn("size-3.5 shrink-0", ok ? "text-emerald-400" : "text-amber-300")} />
      <span>{children}</span>
    </>
  )
  const className = cn(
    "flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs",
    ok
      ? "border-white/10 text-white/60"
      : "border-amber-400/25 bg-amber-400/[0.05] text-amber-100/85"
  )
  return onClick ? (
    <button
      type="button"
      onClick={onClick}
      title="Go to this frame"
      className={cn(className, "transition-colors hover:border-white/25")}
    >
      {body}
    </button>
  ) : (
    <span className={className}>{body}</span>
  )
}

function Checks({ data, onSeek }: { data: EpisodeData; onSeek: (t: number) => void }) {
  const { droppedFrames, longestGapS, jointJump, poseJump } = data.checks
  const jointName = (name: string) => name.replace(".pos", "").replaceAll("_", " ")
  return (
    <div className="flex flex-wrap gap-2">
      <Check ok={droppedFrames === 0}>
        {droppedFrames === 0
          ? "No dropped frames"
          : `${droppedFrames} dropped ${droppedFrames === 1 ? "frame" : "frames"} (longest gap ${longestGapS.toFixed(2)} s)`}
      </Check>
      {jointJump && (
        <Check ok={jointJump.deg <= JOINT_JUMP_WARN_DEG} onClick={() => onSeek(jointJump.t)}>
          Largest joint jump {jointJump.deg.toFixed(1)}° ({jointName(jointJump.name)} at{" "}
          {formatEpisodeTime(jointJump.t)})
        </Check>
      )}
      {poseJump && (
        <Check ok={poseJump.m <= POSE_JUMP_WARN_M} onClick={() => onSeek(poseJump.t)}>
          Largest gripper jump {(poseJump.m * 100).toFixed(1)} cm ({poseJump.side} at{" "}
          {formatEpisodeTime(poseJump.t)})
        </Check>
      )}
    </div>
  )
}

/**
 * A joint episode with its gripper poses, worked out here from the robot
 * model: the Host Computer sends joints only, so a panel following a live
 * recording costs that recording no CPU. If the model can't load, the
 * episode shows without gripper poses.
 */
async function withPoses(
  data: EpisodeData,
  robotModel: AxolModel | undefined,
  gripper: GripperType | undefined
): Promise<EpisodeData> {
  if (!hasJointState(data) || Object.keys(data.pose).length > 0) return data
  try {
    const { loadKinematics, posesFromJoints } = await import("@/lib/episode-fk")
    const robot = await loadKinematics(robotModel ?? "classic", gripper ?? "parallel")
    return withJointPoses(
      data,
      posesFromJoints(robot, data.state, data.frames),
      posesFromJoints(robot, data.action, data.frames)
    )
  } catch {
    return data
  }
}

/**
 * An episode's recorded robot data, to check a take before training on it:
 * quick checks (dropped frames, jumps), a 3D replay and charts of the joints,
 * the gripper pose and opening. Everything follows the player's time
 * (`time`), and a click on a chart or a check moves it (`onSeek`).
 */
export function EpisodeRobotData({
  repoId,
  episode,
  time,
  onSeek,
  robotModel,
  gripper,
}: {
  repoId: string
  episode: DatasetEpisode
  time: number
  onSeek: (t: number) => void
  robotModel?: AxolModel
  gripper?: GripperType
}) {
  const requestKey = `${repoId}:${episode.index}`
  const [loaded, setLoaded] = useState<{
    key: string
    data: EpisodeData | null
    error: string | null
  } | null>(null)
  const [chosenGroup, setGroup] = useState<ChartGroup | null>(null)
  const [withCommanded, setWithCommanded] = useState(false)
  const [zoom, setZoom] = useState<{ key: string; view: ChartView } | null>(null)

  useEffect(() => {
    let cancelled = false
    fetchEpisodeData(repoId, episode.index)
      .then((data) => withPoses(data, robotModel, gripper))
      .then((data) => {
        if (!cancelled) setLoaded({ key: requestKey, data, error: null })
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setLoaded({
            key: requestKey,
            data: null,
            error: String(e instanceof Error ? e.message : e),
          })
        }
      })
    return () => {
      cancelled = true
    }
  }, [repoId, episode.index, requestKey, robotModel, gripper])

  const current = loaded?.key === requestKey ? loaded : null
  const data = current?.data ?? null
  // Cartesian (Mantis) episodes have no joints to chart: open on the pose.
  const group: ChartGroup = chosenGroup ?? (data && !hasJointState(data) ? "pose" : "joints")
  const charts = useMemo(
    () => (data ? episodeCharts(data, group, withCommanded) : []),
    [data, group, withCommanded]
  )
  const duration = data?.t.at(-1) ?? episode.durationS
  const fullView: ChartView = { t0: 0, t1: Math.max(duration, 0.1) }
  const view = zoom?.key === requestKey ? zoom.view : fullView
  const frame = data ? frameAt(data.t, time) : 0

  if (!current) {
    return (
      <div className="flex items-center gap-2 text-sm text-white/45">
        <Loader2 className="size-4 animate-spin" /> Loading the robot data…
      </div>
    )
  }
  if (!data) {
    return (
      <p className="text-sm text-white/45">
        Couldn&apos;t load this episode&apos;s robot data: {current.error}
      </p>
    )
  }

  return (
    <section className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="font-heading text-sm font-semibold text-white/80">Robot data</h3>
        <Checks data={data} onSeek={onSeek} />
      </div>
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.25fr)]">
        <div className="flex flex-col gap-2">
          <div className="h-80 overflow-hidden rounded-xl border border-white/10 bg-black/30 lg:h-[26rem]">
            <Suspense
              fallback={
                <div className="flex h-full items-center justify-center">
                  <Loader2 className="size-5 animate-spin text-white/30" />
                </div>
              }
            >
              <EpisodeReplay
                data={data}
                frame={frame}
                withCommanded={withCommanded}
                robotModel={robotModel}
                gripper={gripper}
              />
            </Suspense>
          </div>
          <p className="text-xs text-white/45">
            {hasJointState(data)
              ? "The arms follow the recorded joints. Lines show each gripper's path."
              : "This episode records gripper poses, not joints, so the arms stay still. Lines show each gripper's path."}
          </p>
        </div>
        <div className="flex min-w-0 flex-col gap-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex overflow-hidden rounded-md border border-white/10">
              {GROUPS.map((item) => (
                <button
                  key={item.key}
                  type="button"
                  onClick={() => setGroup(item.key)}
                  className={cn(
                    "px-2.5 py-1 text-xs transition-colors",
                    group === item.key
                      ? "bg-[#eff483]/15 text-[#eff483]"
                      : "text-white/50 hover:bg-white/[0.05]"
                  )}
                >
                  {item.label}
                </button>
              ))}
            </div>
            <label className="flex items-center gap-2 text-xs text-white/60">
              Show commanded
              <Switch
                checked={withCommanded}
                onChange={setWithCommanded}
                aria-label="Show commanded values"
              />
            </label>
          </div>
          {charts.length === 0 ? (
            <p className="text-sm text-white/45">This episode doesn&apos;t record these values.</p>
          ) : (
            charts.map((c) => (
              <TelemetryChart
                key={c.key}
                title={c.title}
                unit={c.unit}
                series={c.series}
                frames={c.frames}
                version={0}
                metric={0}
                view={view}
                onViewChange={(next) => setZoom({ key: requestKey, view: next })}
                gapBreakS={Math.max(0.5, 3 / Math.max(data.fps, 1))}
                formatTime={formatEpisodeTime}
                cursorT={time}
                onSeek={onSeek}
                height={200}
              />
            ))
          )}
        </div>
      </div>
    </section>
  )
}
