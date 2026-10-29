import { useEffect, useRef, useState, type ReactNode } from "react"
import { Loader2, VideoOff } from "lucide-react"
import { cn } from "@/lib/utils"
import { wsBaseUrl } from "@/lib/supervisor"
import {
  CAMERA_STREAMS,
  MAIN_VIEW_STORAGE_KEY,
  MARIONETTE_VIEWS,
  cameraNoun,
  cameraSocketPath,
  isCameraStream,
  parseCameraFrame,
  parseCameraInfo,
  parseMainView,
  type CameraInfo,
  type CameraStream,
  type MarionetteState,
  type MarionetteView,
} from "@/lib/marionette"

const STREAM_IDS = CAMERA_STREAMS.map((s) => s.id)
// The small views' places in the row under the large one, in order.
const SMALL_COLUMNS = ["lg:col-start-1", "lg:col-start-2", "lg:col-start-3"]

function storedMainView(): MarionetteView {
  try {
    return parseMainView(localStorage.getItem(MAIN_VIEW_STORAGE_KEY))
  } catch {
    return parseMainView(null)
  }
}

/**
 * Live frames of every camera from `/api/marionette/cameras`, as object URLs.
 *
 * While `enabled`, it keeps one socket open, reconnecting a second after it
 * drops. The `large` stream comes sharper; changing it tells the open socket
 * instead of reconnecting, so no pane goes blank. Each frame replaces its
 * stream's URL and the previous one is revoked; closing revokes them all.
 */
function useCameraFeed(enabled: boolean, large: CameraStream | null) {
  const [frames, setFrames] = useState<Partial<Record<CameraStream, string>>>({})
  const [info, setInfo] = useState<CameraInfo | null>(null)
  const socket = useRef<WebSocket | null>(null)
  const largeRef = useRef(large)

  useEffect(() => {
    largeRef.current = large
    const ws = socket.current
    if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ large: large ?? "" }))
  }, [large])

  useEffect(() => {
    if (!enabled) return
    const urls = new Map<CameraStream, string>()
    let closed = false
    let retry: ReturnType<typeof setTimeout> | undefined

    const connect = () => {
      const ws = new WebSocket(`${wsBaseUrl()}${cameraSocketPath(STREAM_IDS, largeRef.current)}`)
      ws.binaryType = "arraybuffer"
      socket.current = ws
      // The large view may have changed while connecting.
      ws.onopen = () => ws.send(JSON.stringify({ large: largeRef.current ?? "" }))
      ws.onmessage = (event) => {
        if (typeof event.data === "string") {
          const next = parseCameraInfo(event.data)
          if (next) setInfo(next)
          return
        }
        const frame = parseCameraFrame(event.data as ArrayBuffer)
        if (!frame || !isCameraStream(frame.stream)) return
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
      socket.current?.close()
      socket.current = null
      for (const url of urls.values()) URL.revokeObjectURL(url)
      setFrames({})
      setInfo(null)
    }
  }, [enabled])

  return { frames, info }
}

/**
 * The head and wrist cameras and the 3D view: one large, the other three
 * small in a row under it. The operator picks the large one, and the choice
 * is kept. The four panes stay mounted as they swap places, so the 3D view
 * never reloads.
 */
export function MarionetteStage({
  running,
  ready,
  cameras,
  robot,
}: {
  running: boolean
  ready: boolean
  /** The session's camera report, from the state stream. */
  cameras: MarionetteState["cameras"] | null
  /** The 3D view of the robot. */
  robot: ReactNode
}) {
  const [main, setMain] = useState<MarionetteView>(storedMainView)
  const { frames, info } = useCameraFeed(running, isCameraStream(main) ? main : null)
  const available = info?.available ?? cameras?.available ?? null
  const error = info?.error ?? cameras?.error ?? null
  const small = MARIONETTE_VIEWS.filter((view) => view.id !== main)

  const show = (view: MarionetteView) => {
    setMain(view)
    try {
      localStorage.setItem(MAIN_VIEW_STORAGE_KEY, view)
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
    <section
      data-testid="marionette-stage"
      aria-label="Cameras"
      className="grid min-h-0 grid-cols-3 gap-2 lg:grid-rows-[minmax(0,1fr)_clamp(96px,18vh,200px)]"
    >
      {MARIONETTE_VIEWS.map(({ id, label }) => {
        const large = id === main
        const src = isCameraStream(id) ? frames[id] : undefined
        return (
          <figure
            key={id}
            data-testid={`view-${id}`}
            data-large={large}
            className={cn(
              "relative m-0 min-h-0 overflow-hidden rounded-lg border bg-black",
              large
                ? "order-first col-span-3 aspect-video border-white/15 lg:row-start-1 lg:aspect-auto"
                : cn(
                    "aspect-video border-white/10 lg:row-start-2 lg:aspect-auto",
                    SMALL_COLUMNS[small.findIndex((view) => view.id === id)]
                  )
            )}
          >
            {id === "robot" ? (
              <div className="absolute inset-0">{robot}</div>
            ) : src ? (
              <img
                src={src}
                alt={`${label} camera`}
                className="absolute inset-0 size-full object-contain"
              />
            ) : (
              <div
                className={cn(
                  "absolute inset-0 flex flex-col items-center justify-center gap-1.5 px-3 text-center text-white/40",
                  large ? "text-sm" : "text-xs"
                )}
              >
                <VideoOff className="size-4" />
                {placeholder(id)}
              </div>
            )}
            {large ? (
              <div
                role="group"
                aria-label="Large view"
                className="absolute top-2 left-2 flex rounded-lg border border-white/10 bg-black/65 p-0.5 backdrop-blur-sm"
              >
                {MARIONETTE_VIEWS.map((view) => (
                  <button
                    key={view.id}
                    type="button"
                    aria-pressed={view.id === main}
                    onClick={() => show(view.id)}
                    className={cn(
                      "rounded-md px-2.5 py-1 text-xs font-medium transition-colors",
                      view.id === main
                        ? "bg-[#eff483]/15 text-[#eff483]"
                        : "text-white/60 hover:text-white/90"
                    )}
                  >
                    {view.label}
                  </button>
                ))}
              </div>
            ) : (
              <>
                <figcaption className="pointer-events-none absolute bottom-1.5 left-1.5 rounded bg-black/60 px-1.5 py-0.5 text-[11px] text-white/75">
                  {label}
                </figcaption>
                <button
                  type="button"
                  aria-label={`Show ${isCameraStream(id) ? `the ${cameraNoun(id)}` : "the 3D view"} large`}
                  title="Show large"
                  onClick={() => show(id)}
                  className="absolute inset-0 cursor-zoom-in rounded-lg outline-offset-[-2px] transition-colors hover:bg-white/[0.04]"
                />
              </>
            )}
            {large && (
              <div className="pointer-events-none absolute inset-x-0 bottom-3 flex flex-col items-center gap-1.5">
                {running && !ready && (
                  <span className="flex items-center gap-2 rounded-full border border-white/10 bg-black/60 px-3 py-1.5 text-xs text-white/70">
                    <Loader2 className="size-3.5 animate-spin" /> Starting Marionette…
                  </span>
                )}
                {!running && (
                  <span className="rounded-full border border-white/10 bg-black/60 px-3 py-1.5 text-xs text-white/70">
                    Start Marionette to move the robot.
                  </span>
                )}
                {running && error && (
                  <span className="rounded-full border border-amber-400/30 bg-black/70 px-3 py-1.5 text-xs text-amber-300">
                    {error}
                  </span>
                )}
                {running && !error && cameras?.source === "synthetic" && id !== "robot" && (
                  <span className="rounded-full border border-white/10 bg-black/60 px-3 py-1.5 text-xs text-white/50">
                    Showing test images.
                  </span>
                )}
              </div>
            )}
          </figure>
        )
      })}
    </section>
  )
}
