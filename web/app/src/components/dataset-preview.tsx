import { useEffect, useMemo, useRef, useState } from "react"
import { Loader2, Pause, Play, RefreshCw } from "lucide-react"
import { cn } from "@/lib/utils"
import {
  ApiRequestError,
  datasetVideoUrl,
  fetchDatasetEpisodes,
  fetchDatasets,
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
                  key={`${shown.repoId}:${episode.index}`}
                  repoId={shown.repoId}
                  episode={episode}
                  cameras={shown.cameras}
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

/**
 * Every camera of one episode, played in lockstep from a shared transport.
 * Each video element seeks within its own span of the (possibly shared) mp4;
 * the first camera is the clock and the others are nudged back when they
 * drift more than a frame or two.
 */
function EpisodePlayer({
  repoId,
  episode,
  cameras,
}: {
  repoId: string
  episode: DatasetEpisode
  cameras: string[]
}) {
  const keys = useMemo(() => cameras.filter((k) => episode.videos[k]), [cameras, episode])
  const videos = useRef(new Map<string, HTMLVideoElement>())
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const duration = episode.durationS

  function each(fn: (video: HTMLVideoElement, from: number) => void) {
    for (const key of keys) {
      const video = videos.current.get(key)
      if (video) fn(video, episode.videos[key].from)
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
  // file may continue into the next episode) and keep the others aligned.
  useEffect(() => {
    if (!playing || keys.length === 0) return
    let frame = 0
    const tick = () => {
      const master = videos.current.get(keys[0])
      if (master) {
        const t = master.currentTime - episode.videos[keys[0]].from
        if (t >= duration || master.ended) {
          for (const key of keys) videos.current.get(key)?.pause()
          setPlaying(false)
          setTime(duration)
          return
        }
        for (const key of keys.slice(1)) {
          const video = videos.current.get(key)
          const target = episode.videos[key].from + t
          if (video && Math.abs(video.currentTime - target) > 0.08) video.currentTime = target
        }
        setTime(Math.max(0, t))
      }
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [playing, keys, duration, episode])

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
        {keys.map((key) => (
          <figure key={key} className="flex min-w-0 flex-col gap-1">
            <video
              ref={(el) => {
                if (el) videos.current.set(key, el)
                else videos.current.delete(key)
              }}
              src={datasetVideoUrl(repoId, episode.index, key, episode.videos[key])}
              muted
              playsInline
              preload="metadata"
              onLoadedMetadata={(e) => {
                e.currentTarget.currentTime = episode.videos[key].from + time
              }}
              className="aspect-video w-full rounded-lg border border-white/10 bg-black object-contain"
            />
            <figcaption className="font-mono text-xs text-white/45">{cameraLabel(key)}</figcaption>
          </figure>
        ))}
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
          className="flex-1 accent-[#eff483]"
          aria-label="Episode time"
        />
        <span className="w-24 text-right font-mono text-xs text-white/55">
          {time.toFixed(1)} / {duration.toFixed(1)} s
        </span>
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
