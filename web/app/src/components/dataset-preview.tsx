import { useEffect, useMemo, useRef, useState } from "react"
import { Loader2, Pause, Play, RefreshCw } from "lucide-react"
import { cn } from "@/lib/utils"
import {
  ApiRequestError,
  episodeVideoSource,
  fetchDatasetEpisodes,
  fetchDatasets,
  previewStep,
  setEpisodeTask,
  type DatasetEpisode,
  type DatasetEpisodes,
  type DatasetInfo,
} from "@/lib/supervisor"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Select, SelectOption } from "@/components/ui/select"
import { useToast } from "@/components/ui/toast"

const OPEN_KEY = "datasetPreviewOpen"
const PREVIEW_FPS_KEY = "datasetPreviewFps"

/** Preview rates offered (0 = the recorded file). Axol's all-intra cameras
 *  are ~21 Mbps each at 60 fps; four of them outrun most links, so the
 *  default is the light cut (every 4th frame of a 60 fps dataset). */
const PREVIEW_FPS_CHOICES = [15, 30, 0]
const DEFAULT_PREVIEW_FPS = 15

function readPreviewFps(): number {
  try {
    const stored = localStorage.getItem(PREVIEW_FPS_KEY)
    const fps = stored === null ? NaN : Number(stored)
    return PREVIEW_FPS_CHOICES.includes(fps) ? fps : DEFAULT_PREVIEW_FPS
  } catch {
    return DEFAULT_PREVIEW_FPS
  }
}

function writePreviewFps(fps: number) {
  try {
    localStorage.setItem(PREVIEW_FPS_KEY, String(fps))
  } catch {
    // Blocked storage: the default applies next time.
  }
}

function readOpen(): boolean {
  try {
    return localStorage.getItem(OPEN_KEY) === "1"
  } catch {
    return false
  }
}

function writeOpen(open: boolean) {
  try {
    localStorage.setItem(OPEN_KEY, open ? "1" : "0")
  } catch {
    // Private window / blocked storage: the card just opens closed next time.
  }
}

/** `observation.images.left_arm` → `left_arm`. */
function cameraLabel(key: string): string {
  return key.replace(/^observation\.images\./, "")
}

function formatSeconds(s: number): string {
  return `${s.toFixed(1)} s`
}

function errorText(e: unknown): string {
  return String(e).replace(/^Error:\s*/, "")
}

/**
 * Browse a dataset on the serve host: its saved episodes, each camera's video
 * and the episode's task, which can be renamed in place. While an operation
 * records, the card follows that session's dataset (its episode snapshot
 * names it) and refreshes after every save, so the previous takes can be
 * reviewed — and a mistyped task fixed — without stopping. The take in
 * progress is not listed until it is saved.
 */
export function DatasetPreview({
  connected,
  liveDataset,
  episodesRecorded,
}: {
  connected: boolean
  /** The running operation's dataset, if it records one. */
  liveDataset: { repoId: string; root: string } | null
  /** Bumps after every save of the running operation (triggers a refresh). */
  episodesRecorded: number | null
}) {
  const toast = useToast()
  const [open, setOpen] = useState(readOpen)
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  // The operator's explicit pick; null follows the live session (else newest).
  const [chosen, setChosen] = useState<string | null>(null)
  const [data, setData] = useState<DatasetEpisodes | null>(null)
  const [error, setError] = useState<string | null>(null)
  // The request the last response answered (loading = it isn't the current one).
  const [settledKey, setSettledKey] = useState<string | null>(null)
  const [chosenEpisode, setChosenEpisode] = useState<number | null>(null)
  const [reloadNonce, setReloadNonce] = useState(0)
  const [previewFps, setPreviewFps] = useState(readPreviewFps)

  // A live dataset is listed under the scanned root's repo id, which differs
  // from the session's own repo id when it records outside that root.
  const liveRepoId = liveDataset
    ? (datasets.find((d) => d.root === liveDataset.root)?.repoId ?? liveDataset.repoId)
    : null
  const selected = chosen ?? liveRepoId ?? datasets[0]?.repoId ?? null
  const following = selected !== null && selected === liveRepoId
  // Refetch after each save only while showing the session's own dataset.
  const liveSaves = following ? episodesRecorded : null
  const requestKey = `${selected}|${liveSaves}|${reloadNonce}`
  const loading = open && selected !== null && settledKey !== requestKey

  useEffect(() => {
    if (!open || !connected) return
    let cancelled = false
    fetchDatasets()
      .then((found) => {
        if (!cancelled) setDatasets(found)
      })
      .catch(() => {
        // The listing is a convenience; the episodes fetch reports errors.
      })
    return () => {
      cancelled = true
    }
  }, [open, connected, liveDataset?.root, episodesRecorded, reloadNonce])

  useEffect(() => {
    if (!open || !connected || !selected) return
    let cancelled = false
    fetchDatasetEpisodes(selected)
      .then((next) => {
        if (cancelled) return
        setData(next)
        setError(null)
      })
      .catch((e) => {
        if (cancelled) return
        setData(null)
        setError(
          // An older host answers the unknown route with the SPA's generic 404.
          e instanceof ApiRequestError && e.status === 404 && e.message === "not found"
            ? "This host does not offer dataset preview (update axol)."
            : errorText(e)
        )
      })
      .finally(() => {
        if (!cancelled) setSettledKey(requestKey)
      })
    return () => {
      cancelled = true
    }
  }, [open, connected, selected, requestKey])

  const shown = data && data.repoId === selected ? data : null
  const episodes = useMemo(() => [...(shown?.episodes ?? [])].reverse(), [shown])
  const episode =
    episodes.find((e) => e.index === chosenEpisode) ?? (episodes.length ? episodes[0] : null)

  function toggle() {
    setOpen((was) => {
      writeOpen(!was)
      return !was
    })
  }

  function onRenamed(updated: DatasetEpisode) {
    setData((prev) => {
      if (!prev) return prev
      const tasks = updated.tasks.filter((t) => !prev.tasks.includes(t))
      return {
        ...prev,
        tasks: [...prev.tasks, ...tasks],
        episodes: prev.episodes.map((e) => (e.index === updated.index ? updated : e)),
      }
    })
    toast.success(`Episode ${updated.index + 1} task saved.`)
  }

  if (!connected) return null

  return (
    <Card>
      <div className="flex items-start justify-between gap-3">
        <CardHeader>
          <CardTitle>Dataset preview</CardTitle>
          <CardDescription>
            Watch saved episodes and fix their task. Follows the dataset the running operation
            records into.
          </CardDescription>
        </CardHeader>
        <Button variant="outline" size="sm" onClick={toggle}>
          {open ? "Hide" : "Show"}
        </Button>
      </div>

      {open && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
            <Select
              value={selected ?? ""}
              onChange={(e) => {
                setChosen(e.target.value || null)
                setChosenEpisode(null)
              }}
              className="sm:flex-1"
            >
              {!selected && <SelectOption value="" label="No datasets on this host" />}
              {selected && !datasets.some((d) => d.repoId === selected) && (
                <SelectOption value={selected} />
              )}
              {datasets.map((d) => (
                <SelectOption
                  key={d.repoId}
                  value={d.repoId}
                  label={d.repoId === liveRepoId ? `${d.repoId} (recording)` : d.repoId}
                />
              ))}
            </Select>
            <div className="flex gap-2">
              {liveRepoId && !following && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setChosen(null)
                    setChosenEpisode(null)
                  }}
                >
                  Follow session
                </Button>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setReloadNonce((n) => n + 1)}
                disabled={loading}
                aria-label="Refresh"
              >
                {loading ? <Loader2 className="animate-spin" /> : <RefreshCw />}
              </Button>
            </div>
          </div>

          {error && <p className="text-xs text-red-300/80">{error}</p>}
          {following && (
            <p className="text-xs text-white/45">
              Recording in progress — a take appears here once it is saved.
            </p>
          )}

          {shown && episodes.length === 0 && !error && (
            <p className="text-sm text-white/45">No saved episodes yet.</p>
          )}

          {shown && episode && (
            <div className="grid gap-4 lg:grid-cols-[16rem_1fr]">
              <EpisodeList
                episodes={episodes}
                selected={episode.index}
                onSelect={setChosenEpisode}
              />
              <div className="flex min-w-0 flex-col gap-4">
                <EpisodePlayer
                  key={`${shown.repoId}:${episode.index}:${previewFps}`}
                  repoId={shown.repoId}
                  episode={episode}
                  cameras={shown.cameras}
                  datasetFps={shown.fps}
                  previewFps={previewFps}
                  onPreviewFps={(fps) => {
                    writePreviewFps(fps)
                    setPreviewFps(fps)
                  }}
                />
                <TaskEditor
                  key={`${shown.repoId}:${episode.index}:${episode.tasks.join("\n")}`}
                  repoId={shown.repoId}
                  episode={episode}
                  knownTasks={shown.tasks}
                  onRenamed={onRenamed}
                />
              </div>
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

function EpisodeList({
  episodes,
  selected,
  onSelect,
}: {
  episodes: DatasetEpisode[]
  selected: number
  onSelect: (index: number) => void
}) {
  return (
    <div className="flex max-h-80 flex-col gap-1 overflow-y-auto pr-1">
      {episodes.map((e) => (
        <button
          key={e.index}
          type="button"
          onClick={() => onSelect(e.index)}
          title={`episode_index ${e.index}`}
          className={cn(
            "rounded-lg border px-3 py-2 text-left transition-colors",
            e.index === selected
              ? "border-[#eff483]/40 bg-[#eff483]/10"
              : "border-white/10 bg-white/[0.02] hover:border-white/25 hover:bg-white/[0.05]"
          )}
        >
          <div className="flex items-center justify-between gap-2 text-sm">
            <span className="font-medium">Episode {e.index + 1}</span>
            <span className="font-mono text-xs text-white/45">{formatSeconds(e.durationS)}</span>
          </div>
          <div className="truncate text-xs text-white/55">{e.tasks.join(" | ") || "—"}</div>
        </button>
      ))}
    </div>
  )
}

// HTMLMediaElement.HAVE_FUTURE_DATA: enough buffered to advance playback.
const HAVE_FUTURE_DATA = 3
// A camera further than this from the clock is re-seeked (a few preview frames).
const DRIFT_S = 0.25

/**
 * Every camera of one episode, played in lockstep from a shared transport.
 * Each video element plays its own span of its source (a shared mp4 or the
 * host's preview cut); the first camera is the clock. When any camera runs
 * out of data or drifts, every camera pauses until they are all buffered and
 * aligned, then all resume — instead of re-seeking the laggard every frame,
 * which restarts its download and keeps it behind on a slow link.
 */
function EpisodePlayer({
  repoId,
  episode,
  cameras,
  datasetFps,
  previewFps,
  onPreviewFps,
}: {
  repoId: string
  episode: DatasetEpisode
  cameras: string[]
  datasetFps: number
  previewFps: number
  onPreviewFps: (fps: number) => void
}) {
  const keys = useMemo(() => cameras.filter((k) => episode.videos[k]), [cameras, episode])
  const sources = useMemo(
    () =>
      new Map(keys.map((k) => [k, episodeVideoSource(repoId, episode, k, datasetFps, previewFps)])),
    [keys, repoId, episode, datasetFps, previewFps]
  )
  const videos = useRef(new Map<string, HTMLVideoElement>())
  // One stable ref callback per camera (a new function each render would
  // detach and re-attach — and so release — the element on every render).
  const videoRefs = useMemo(
    () =>
      new Map(
        keys.map((key) => [
          key,
          (el: HTMLVideoElement | null) => {
            if (!el) return
            videos.current.set(key, el)
            // Leaving an episode (or the card) must stop its download: a
            // removed <video> keeps loading until its src is cleared, and on a
            // slow link the abandoned episode's streams (and the browser's six
            // connections per host) starve the next one.
            return () => {
              videos.current.delete(key)
              el.pause()
              el.removeAttribute("src")
              el.load()
            }
          },
        ])
      ),
    [keys]
  )
  const [playing, setPlaying] = useState(false)
  // The sources the operator has pressed Play on. Until then nothing loads
  // (see the <video> preload below); after it, loading must continue while
  // playback is held for buffering, which a paused preload="none" element
  // may not do.
  const [armedFor, setArmedFor] = useState<typeof sources | null>(null)
  const armed = armedFor === sources
  const [buffering, setBuffering] = useState(false)
  const [time, setTime] = useState(0)
  const duration = episode.durationS
  const choices = PREVIEW_FPS_CHOICES.filter(
    (fps) => fps === 0 || previewStep(datasetFps, fps) >= 2
  )

  function each(fn: (video: HTMLVideoElement, from: number) => void) {
    for (const key of keys) {
      const video = videos.current.get(key)
      const source = sources.get(key)
      if (video && source) fn(video, source.span.from)
    }
  }

  function seek(t: number) {
    const clamped = Math.min(Math.max(t, 0), duration)
    each((video, from) => {
      video.currentTime = from + clamped
    })
    setTime(clamped)
  }

  function play() {
    setArmedFor(sources)
    if (time >= duration - 0.05) seek(0)
    each((video) => {
      void video.play().catch(() => undefined)
    })
    setPlaying(true)
  }

  function pause() {
    each((video) => video.pause())
    setPlaying(false)
  }

  // While playing, follow the clock camera: stop at the episode's end (the
  // file may continue into the next episode); hold everything while a camera
  // buffers or realigns.
  useEffect(() => {
    if (!playing || keys.length === 0) return
    let frame = 0
    let held = false
    const tick = () => {
      const all = keys.flatMap((key) => {
        const video = videos.current.get(key)
        const source = sources.get(key)
        return video && source ? [{ video, from: source.span.from }] : []
      })
      const master = all[0]
      if (master) {
        const t = master.video.currentTime - master.from
        if (t >= duration || master.video.ended) {
          for (const { video } of all) video.pause()
          setPlaying(false)
          setBuffering(false)
          setTime(duration)
          return
        }
        const drifted = all
          .slice(1)
          .filter(
            ({ video, from }) => !video.seeking && Math.abs(video.currentTime - from - t) > DRIFT_S
          )
        const starved = all.some(
          ({ video }) => video.seeking || video.readyState < HAVE_FUTURE_DATA
        )
        if (starved || drifted.length > 0) {
          if (!held) {
            held = true
            for (const { video } of all) video.pause()
            setBuffering(true)
          }
          for (const { video, from } of drifted) video.currentTime = from + t
        } else if (held) {
          held = false
          for (const { video } of all) void video.play().catch(() => undefined)
          setBuffering(false)
        }
        setTime(Math.max(0, t))
      }
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(frame)
      setBuffering(false)
    }
  }, [playing, keys, sources, duration])

  if (keys.length === 0) {
    return <p className="text-sm text-white/45">This episode has no video.</p>
  }

  return (
    <div className="flex flex-col gap-3">
      <div
        className={cn(
          "grid gap-2",
          keys.length === 1 ? "grid-cols-1" : "grid-cols-1 sm:grid-cols-2",
          keys.length >= 3 && "xl:grid-cols-3"
        )}
      >
        {keys.map((key) => {
          const source = sources.get(key)!
          return (
            <figure key={key} className="flex min-w-0 flex-col gap-1">
              <video
                ref={videoRefs.get(key)}
                src={source.url}
                muted
                playsInline
                // Nothing loads until Play. The card follows the live session
                // and selects each newly saved episode, so an eager preload
                // pulled every camera's cut (~50 MB) after every save, over
                // the next take: on an Orin NX recording at 60 fps with the
                // headset and panel streaming, that extra network and serve
                // work on the camera cores starved the dataset encoders and
                // discarded takes started within ~40 s of a save.
                preload={armed ? "auto" : "none"}
                onLoadedMetadata={(e) => {
                  e.currentTarget.currentTime = source.span.from + time
                }}
                className="aspect-video w-full rounded-lg border border-white/10 bg-black object-contain"
              />
              <figcaption className="font-mono text-xs text-white/45">
                {cameraLabel(key)}
              </figcaption>
            </figure>
          )
        })}
      </div>
      <div className="flex items-center gap-3">
        <Button
          variant="outline"
          size="icon"
          onClick={playing ? pause : play}
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? <Pause /> : <Play />}
        </Button>
        <input
          type="range"
          min={0}
          max={duration}
          step={0.01}
          value={time}
          onChange={(e) => seek(Number(e.target.value))}
          className="min-w-0 flex-1 accent-[#eff483]"
          aria-label="Episode time"
        />
        <span className="w-24 text-right font-mono text-xs text-white/55">
          {buffering ? (
            <span className="inline-flex items-center gap-1">
              <Loader2 className="size-3 animate-spin" /> buffering
            </span>
          ) : (
            `${time.toFixed(1)} / ${duration.toFixed(1)} s`
          )}
        </span>
        {choices.length > 1 && (
          <Select
            value={String(choices.includes(previewFps) ? previewFps : 0)}
            onChange={(e) => onPreviewFps(Number(e.target.value))}
            className="h-8 w-32 shrink-0"
            aria-label="Preview quality"
            title="Lighter previews keep every Nth frame, so they load over slow links"
          >
            {choices.map((fps) => (
              <SelectOption
                key={fps}
                value={String(fps)}
                label={fps === 0 ? `Full ${datasetFps} fps` : `${fps} fps`}
              />
            ))}
          </Select>
        )}
      </div>
    </div>
  )
}

function TaskEditor({
  repoId,
  episode,
  knownTasks,
  onRenamed,
}: {
  repoId: string
  episode: DatasetEpisode
  knownTasks: string[]
  onRenamed: (episode: DatasetEpisode) => void
}) {
  const toast = useToast()
  const current = episode.tasks.join(" | ")
  const [text, setText] = useState(episode.tasks.length === 1 ? episode.tasks[0] : "")
  const [saving, setSaving] = useState(false)
  const unchanged = episode.tasks.length === 1 && text.trim() === episode.tasks[0]
  const listId = `dataset-tasks-${repoId.replace(/[^a-zA-Z0-9_-]/g, "_")}`

  async function save() {
    if (unchanged || !text.trim() || saving) return
    setSaving(true)
    try {
      onRenamed(await setEpisodeTask(repoId, episode.index, text.trim()))
    } catch (e) {
      if (e instanceof ApiRequestError && e.status === 409) {
        toast.warning("The recorder is saving an episode — try again in a moment.")
      } else {
        toast.error(`Could not rename the task: ${errorText(e)}`)
      }
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2 text-xs text-white/45">
        <span>Task</span>
        {episode.tasks.length > 1 && <Badge variant="warning">{episode.tasks.length} tasks</Badge>}
        <span className="text-white/70">{current || "—"}</span>
      </div>
      <div className="flex items-center gap-2">
        <Input
          value={text}
          list={listId}
          placeholder="New task for this episode"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void save()
          }}
          className="h-8 flex-1"
        />
        <datalist id={listId}>
          {knownTasks.map((t) => (
            <option key={t} value={t} />
          ))}
        </datalist>
        <Button
          variant="outline"
          size="sm"
          disabled={unchanged || !text.trim() || saving}
          onClick={() => void save()}
        >
          {saving && <Loader2 className="animate-spin" />}
          Save task
        </Button>
      </div>
    </div>
  )
}
