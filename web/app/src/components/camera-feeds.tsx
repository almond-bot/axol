import { useEffect, useMemo, useRef, useState } from "react"
import { Loader2, ShieldCheck, VideoOff } from "lucide-react"
import { useAxolVideo, type CameraStreams } from "@almond/axol-vr-client"
import { authorizeCert } from "@/lib/cert-accept"
import { serverHttpBase } from "@/lib/supervisor"
import { Button } from "@/components/ui/button"

/**
 * Live camera feeds in the control panel, for driving a session with the
 * headset off.
 *
 * The panel joins the running operation's VR server (`wss://host:vrPort/ws`)
 * as one more view-only client and negotiates the same WebRTC camera tracks
 * the headset would receive — no new backend video path, and the relay's
 * encoders are shared. A stereo camera streaming both eyes arrives as a
 * single side-by-side track (`{name}_sbs`); only its left half is shown.
 * When both eyes arrive as separate tracks, only the left one is shown.
 *
 * The VR server's self-signed certificate must be accepted per origin, so a
 * connection that keeps failing offers the same authorize-popup flow the VR
 * app uses (see lib/cert-accept.ts).
 */
export function CameraFeeds({ host, vrPort }: { host: string; vrPort: number }) {
  const wsRef = useRef<WebSocket | null>(null)
  const [wsOpen, setWsOpen] = useState(false)
  // Consecutive failed connection attempts without ever opening — the
  // signature of an unaccepted self-signed cert (or a server still starting).
  const [failedAttempts, setFailedAttempts] = useState(0)

  // Bare hostname of the serve machine: the stored host may carry a scheme
  // or the control-panel port; same-origin panels have no host at all.
  const hostname = useMemo(() => {
    const base = serverHttpBase(host)
    if (base) {
      try {
        return new URL(base).hostname
      } catch {
        // fall through to the page's own host
      }
    }
    return window.location.hostname
  }, [host])
  const vrOrigin = `https://${hostname}:${vrPort}`

  // Own the WebSocket to the VR server, reconnecting while mounted: the
  // server only comes up partway through the operation's startup (after the
  // teleop stack connects), and its video manager is registered later still —
  // reconnecting also re-triggers useAxolVideo's one-shot webrtc-request.
  useEffect(() => {
    let closed = false
    let timer: ReturnType<typeof setTimeout> | null = null

    function connect() {
      let ws: WebSocket
      try {
        ws = new WebSocket(`wss://${hostname}:${vrPort}/ws`)
      } catch {
        timer = setTimeout(connect, 3000)
        return
      }
      let opened = false
      wsRef.current = ws
      ws.onopen = () => {
        opened = true
        setWsOpen(true)
        setFailedAttempts(0)
      }
      ws.onclose = () => {
        if (wsRef.current === ws) wsRef.current = null
        if (closed) return
        setWsOpen(false)
        if (!opened) setFailedAttempts((n) => n + 1)
        timer = setTimeout(connect, 3000)
      }
    }

    connect()
    return () => {
      closed = true
      if (timer) clearTimeout(timer)
      const ws = wsRef.current
      wsRef.current = null
      if (ws) {
        ws.onclose = null
        ws.close()
      }
    }
  }, [hostname, vrPort])

  const { streams, available } = useAxolVideo(wsRef, true)

  // `available === false` means this socket's webrtc-request landed before the
  // op registered its video manager (or video is off). Recycle the socket on a
  // timer so the next request can succeed once video comes up.
  useEffect(() => {
    if (available !== false) return
    const t = setTimeout(() => wsRef.current?.close(), 5000)
    return () => clearTimeout(t)
  }, [available])

  const feeds = useMemo(() => selectFeeds(streams), [streams])
  // A repeatedly-failing connection is either the unaccepted self-signed cert
  // or the VR server still starting; offer the one-tap cert authorize either
  // way (harmless while starting).
  const certHint = !wsOpen && failedAttempts >= 2

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-white/10 bg-white/[0.02] p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-xs tracking-widest text-white/40 uppercase">
          Camera feeds
        </span>
        {feeds.length > 0 && <span className="font-mono text-[0.65rem] text-white/40">live</span>}
      </div>

      {feeds.length > 0 ? (
        <div className="grid grid-cols-2 gap-2">
          {feeds.map((feed) => (
            <FeedTile
              key={feed.name}
              feed={feed}
              wide={feeds.length % 2 === 1 && feed === feeds[0]}
            />
          ))}
        </div>
      ) : available === false ? (
        <p className="flex items-center gap-2 text-xs text-white/45">
          <VideoOff className="size-3.5 shrink-0" />
          No camera stream from this operation — check that streaming is enabled in the camera
          settings.
        </p>
      ) : (
        <p className="flex items-center gap-2 text-xs text-white/45">
          <Loader2 className="size-3.5 shrink-0 animate-spin" />
          {wsOpen ? "Waiting for video…" : "Connecting to the camera stream…"}
        </p>
      )}

      {certHint && (
        <div className="flex flex-wrap items-center gap-2 text-xs text-white/45">
          <span>
            If the feed never connects, authorize the camera server&apos;s certificate once:
          </span>
          <Button variant="outline" size="sm" onClick={() => authorizeCert(vrOrigin)}>
            <ShieldCheck />
            Authorize
          </Button>
        </div>
      )}
    </div>
  )
}

interface Feed {
  /** Camera slot name shown as the label (e.g. "overhead", "left_arm"). */
  name: string
  stream: MediaStream
  /** Side-by-side stereo track: display only the left half (the left eye). */
  leftHalf: boolean
}

/**
 * Pick one displayable feed per camera from the negotiated tracks.
 *
 * - `{name}_sbs` (both eyes packed side-by-side) → shown cropped to the left
 *   eye, labelled `{name}`; any accompanying per-eye tracks are dropped.
 * - `{name}_left` / `{name}_right` (per-eye tracks) → only the left is shown.
 * - anything else (mono or single-eye) → shown as-is.
 */
function selectFeeds(streams: CameraStreams): Feed[] {
  const names = new Set(Object.keys(streams))
  const feeds: Feed[] = []
  for (const [name, stream] of Object.entries(streams)) {
    if (name.endsWith("_sbs")) {
      feeds.push({ name: name.slice(0, -4), stream, leftHalf: true })
      continue
    }
    if (name.endsWith("_left")) {
      const base = name.slice(0, -5)
      if (names.has(`${base}_sbs`)) continue
      feeds.push({ name: base, stream, leftHalf: false })
      continue
    }
    if (name.endsWith("_right")) {
      const base = name.slice(0, -6)
      if (names.has(`${base}_sbs`) || names.has(`${base}_left`)) continue
      feeds.push({ name: base, stream, leftHalf: false })
      continue
    }
    if (names.has(`${name}_sbs`)) continue
    feeds.push({ name, stream, leftHalf: false })
  }
  // Stable, overhead-first ordering so the primary feed leads the grid.
  return feeds.sort((a, b) =>
    a.name === "overhead" ? -1 : b.name === "overhead" ? 1 : a.name.localeCompare(b.name)
  )
}

function FeedTile({ feed, wide }: { feed: Feed; wide: boolean }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  // Displayed aspect (width/height of the visible region). ZED streams are
  // 16:10 per eye; corrected from the track's real dimensions on metadata.
  const [aspect, setAspect] = useState(1.6)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    video.srcObject = feed.stream
    video.play().catch(() => {
      // Autoplay of a muted video is allowed everywhere modern; best-effort.
    })
  }, [feed.stream])

  return (
    <div
      className={`relative overflow-hidden rounded-lg border border-white/10 bg-black ${
        wide ? "col-span-2" : ""
      }`}
      style={{ aspectRatio: aspect }}
    >
      {/* For a side-by-side track the container is sized to one eye, and the
          height-fitted video is naturally twice its width — so exactly the
          left eye is visible. */}
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        onLoadedMetadata={() => {
          const v = videoRef.current
          if (v && v.videoWidth > 0 && v.videoHeight > 0) {
            setAspect(v.videoWidth / (feed.leftHalf ? 2 : 1) / v.videoHeight)
          }
        }}
        className={feed.leftHalf ? "h-full w-auto max-w-none" : "h-full w-full object-contain"}
      />
      <span className="absolute bottom-1 left-1.5 rounded bg-black/60 px-1.5 py-0.5 font-mono text-[0.6rem] text-white/70">
        {feed.name.replace(/_/g, " ")}
      </span>
    </div>
  )
}
