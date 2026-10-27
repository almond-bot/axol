import { useEffect, useState } from "react"
import { VideoOff } from "lucide-react"
import { Card, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import { wsBaseUrl } from "@/lib/supervisor"
import {
  CAMERA_STORAGE_KEY,
  CAMERA_STREAMS,
  cameraLabel,
  cameraNoun,
  cameraSocketPath,
  isCameraStream,
  parseCameraChoice,
  parseCameraFrame,
  parseCameraInfo,
  type CameraInfo,
  type CameraStream,
  type MarionetteState,
} from "@/lib/marionette"

function storedChoice(): CameraStream[] {
  try {
    return parseCameraChoice(localStorage.getItem(CAMERA_STORAGE_KEY))
  } catch {
    return parseCameraChoice(null)
  }
}

/**
 * Live frames from `/api/marionette/cameras` for some streams, as object URLs.
 *
 * While `enabled`, it keeps a socket open for only the wanted streams,
 * reconnecting a second after it drops. Each frame replaces its stream's
 * URL and the previous one is revoked; closing revokes them all.
 */
function useCameraFeed(enabled: boolean, streams: readonly CameraStream[]) {
  const [frames, setFrames] = useState<Partial<Record<CameraStream, string>>>({})
  const [info, setInfo] = useState<CameraInfo | null>(null)
  const wanted = streams.join(",")

  useEffect(() => {
    if (!enabled || !wanted) return
    const ids = wanted.split(",") as CameraStream[]
    const urls = new Map<CameraStream, string>()
    let socket: WebSocket | null = null
    let closed = false
    let retry: ReturnType<typeof setTimeout> | undefined

    const connect = () => {
      const ws = new WebSocket(`${wsBaseUrl()}${cameraSocketPath(ids)}`)
      ws.binaryType = "arraybuffer"
      socket = ws
      ws.onmessage = (event) => {
        if (typeof event.data === "string") {
          const next = parseCameraInfo(event.data)
          if (next) setInfo(next)
          return
        }
        const frame = parseCameraFrame(event.data as ArrayBuffer)
        if (!frame || !isCameraStream(frame.stream) || !ids.includes(frame.stream)) return
        const stream = frame.stream
        const url = URL.createObjectURL(new Blob([frame.jpeg], { type: "image/jpeg" }))
        const previous = urls.get(stream)
        urls.set(stream, url)
        setFrames((current) => ({ ...current, [stream]: url }))
        if (previous) URL.revokeObjectURL(previous)
      }
      ws.onclose = () => {
        if (!closed) retry = setTimeout(connect, 1000)
      }
    }
    connect()
    return () => {
      closed = true
      clearTimeout(retry)
      socket?.close()
      for (const url of urls.values()) URL.revokeObjectURL(url)
      setFrames({})
      setInfo(null)
    }
  }, [enabled, wanted])

  return { frames, info }
}

/** The head and wrist cameras, each pane shown or hidden by the operator. */
export function CameraPanes({
  running,
  cameras,
}: {
  running: boolean
  /** The session's camera report, from the state stream. */
  cameras: MarionetteState["cameras"] | null
}) {
  const [shown, setShown] = useState<CameraStream[]>(storedChoice)
  const { frames, info } = useCameraFeed(running, shown)
  const available = info?.available ?? cameras?.available ?? null
  const error = info?.error ?? cameras?.error ?? null

  const toggle = (stream: CameraStream) => {
    const next = shown.includes(stream) ? shown.filter((s) => s !== stream) : [...shown, stream]
    const ordered = CAMERA_STREAMS.map((s) => s.id).filter((id) => next.includes(id))
    setShown(ordered)
    try {
      localStorage.setItem(CAMERA_STORAGE_KEY, JSON.stringify(ordered))
    } catch {
      // The choice lasts until the page closes.
    }
  }

  const placeholder = (stream: CameraStream) => {
    if (!running) return "Camera off"
    // An empty list is a camera source still starting, not a robot without cameras.
    if (available?.length && !available.includes(stream)) return `No ${cameraNoun(stream)}`
    return `Waiting for the ${cameraNoun(stream)}`
  }

  return (
    <Card data-testid="marionette-cameras">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <CardHeader>
          <CardTitle>Cameras</CardTitle>
        </CardHeader>
        <div className="flex gap-1.5" role="group" aria-label="Cameras to show">
          {CAMERA_STREAMS.map(({ id, label }) => {
            const on = shown.includes(id)
            return (
              <button
                key={id}
                type="button"
                aria-pressed={on}
                onClick={() => toggle(id)}
                className={cn(
                  "rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
                  on
                    ? "border-[#eff483]/50 bg-[#eff483]/15 text-[#eff483]"
                    : "border-white/10 text-white/45 hover:text-white/75"
                )}
              >
                {label}
              </button>
            )
          })}
        </div>
      </div>
      {shown.length === 0 ? (
        <p className="text-sm text-white/45">Pick a camera to show.</p>
      ) : (
        // Two panes share a row. With three, the first gets a row of its own.
        <div className={cn("grid gap-3", shown.length > 1 && "sm:grid-cols-2")}>
          {shown.map((stream, i) => {
            const src = frames[stream]
            return (
              <figure
                key={stream}
                data-testid={`camera-${stream}`}
                className={cn(
                  "relative aspect-video overflow-hidden rounded-lg border border-white/10 bg-black",
                  shown.length === 3 && i === 0 && "sm:col-span-2"
                )}
              >
                {src ? (
                  <img
                    src={src}
                    alt={`${cameraLabel(stream)} camera`}
                    className="size-full object-contain"
                  />
                ) : (
                  <div className="flex size-full flex-col items-center justify-center gap-1.5 px-3 text-center text-xs text-white/40">
                    <VideoOff className="size-4" />
                    {placeholder(stream)}
                  </div>
                )}
                <figcaption className="absolute top-2 left-2 rounded bg-black/60 px-1.5 py-0.5 text-[11px] text-white/75">
                  {cameraLabel(stream)}
                </figcaption>
              </figure>
            )
          })}
        </div>
      )}
      {running && error && <p className="text-xs text-amber-300">{error}</p>}
      {running && !error && cameras?.source === "synthetic" && (
        <p className="text-xs text-white/40">Showing test images.</p>
      )}
    </Card>
  )
}
