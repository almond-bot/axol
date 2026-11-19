/**
 * The Sensors page: the lidar's live map with the robot on it, the live
 * scan, and the camera feeds.
 *
 * The map and the scan come from `axol serve` itself, so they show whether
 * or not an operation runs. The cameras come from the running operation's
 * VR server: this page's own (Start streams just the cameras), or teleop's,
 * Jog's or data collection's when one of those runs.
 */
import { useEffect, useRef, useState } from "react"
import {
  AlertTriangle,
  Loader2,
  Maximize2,
  Minimize2,
  RotateCcw,
  ScanEye,
  VideoOff,
} from "lucide-react"
import { CameraFeeds } from "@/components/camera-feeds"
import { LidarControls } from "@/components/lidar/lidar-controls"
import { useJointPush } from "@/components/lidar/joint-push"
import { SensorScene, type SceneView } from "@/components/sensors/sensor-scene"
import { Button } from "@/components/ui/button"
import { Select, SelectOption } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { useLidarConfigured, useLidarStream } from "@/lib/lidar-stream"
import {
  lidarStatusLine,
  loadLidarSettings,
  saveLidarSettings,
  type LidarSettings,
} from "@/lib/lidar"
import { fetchSavedMaps, mapStatusLine } from "@/lib/lidar-map"
import { apiUrl, type AxolModel, type GripperType } from "@/lib/supervisor"

const LIVE = "__live__"

function toneClass(tone: "ok" | "wait" | "bad") {
  return tone === "ok"
    ? "text-white/50"
    : tone === "bad"
      ? "flex items-center gap-1 text-amber-300"
      : "flex items-center gap-1 text-white/45"
}

export default function SensorsWorkspace({
  robot,
  video,
  cameras,
}: {
  robot: { model: AxolModel; gripper: GripperType } | null
  /** Where camera video comes from: the Host Computer's VR server. */
  video: { host: string; vrPort: number }
  /** An operation is streaming the cameras. */
  cameras: boolean
}) {
  const configured = useLidarConfigured()
  const { cloud, status, map, mapStatus, send } = useLidarStream(configured === true, 12000, {
    map: true,
  })
  const [settings, setSettings] = useState<LidarSettings>(loadLidarSettings)
  const [follow, setFollow] = useState(true)
  const [large, setLarge] = useState(false)
  const [savedMaps, setSavedMaps] = useState<string[]>([])
  const [vrSocket, setVrSocket] = useState<WebSocket | null>(null)
  const joints = useJointPush(vrSocket)
  const viewRef = useRef<SceneView | null>(null)

  useEffect(() => {
    if (configured !== true) return
    let cancelled = false
    fetchSavedMaps(apiUrl("/api/lidar/maps")).then((maps) => {
      if (!cancelled) setSavedMaps(maps)
    })
    return () => {
      cancelled = true
    }
  }, [configured])

  // Frame the robot once the map knows where it is.
  const placed = useRef(-1)
  useEffect(
    () =>
      map.subscribe(() => {
        if (map.pose && placed.current !== map.epoch) {
          placed.current = map.epoch
          viewRef.current?.reset()
        }
      }),
    [map]
  )

  const change = (next: LidarSettings) => {
    setSettings(next)
    saveLidarSettings(next)
  }

  const lidarLine = lidarStatusLine(status)
  const mapLine = mapStatusLine(mapStatus)
  const source = mapStatus?.source ?? null
  const lidarDown = status.state === "error"
  // Draw the scans on the map once it is running.
  const mapEpoch =
    mapStatus && !mapStatus.error && mapStatus.state !== "starting" ? mapStatus.epoch : null

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <div className="flex flex-col gap-2 rounded-lg border border-white/10 bg-white/[0.02] p-3">
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
          <div className="flex flex-wrap items-center gap-3">
            <span className="font-mono text-xs tracking-widest text-white/40 uppercase">Map</span>
            {configured === true && (
              <>
                <Select
                  aria-label="Map"
                  value={source ?? LIVE}
                  onChange={(e) =>
                    send({
                      type: "map-source",
                      source: e.target.value === LIVE ? null : e.target.value,
                    })
                  }
                  className="h-8 w-48"
                  title="Build a map as you drive, or find the robot on a saved map"
                >
                  <SelectOption value={LIVE} label="Live map" />
                  {savedMaps.map((name) => (
                    <SelectOption key={name} value={name} label={`Saved: ${name}`} />
                  ))}
                  {source && !savedMaps.includes(source) && (
                    <SelectOption value={source} label={`Saved: ${source}`} />
                  )}
                </Select>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => send({ type: "reset-map" })}
                  title={
                    source
                      ? "Search for the robot on the map again"
                      : "Clear the map and start again from here"
                  }
                >
                  {source ? "Find robot" : "Reset map"}
                </Button>
              </>
            )}
          </div>
          <div className="flex items-center gap-3 text-xs">
            {configured === true && (
              <label className="flex items-center gap-2 text-white/55">
                <Switch checked={follow} onChange={setFollow} aria-label="Follow robot" />
                Follow robot
              </label>
            )}
            <button
              type="button"
              onClick={() => viewRef.current?.top()}
              title="View from above"
              className="text-white/40 transition-colors hover:text-white/80"
            >
              <ScanEye className="size-4" />
            </button>
            <button
              type="button"
              onClick={() => viewRef.current?.reset()}
              title="Reset view"
              className="text-white/40 transition-colors hover:text-white/80"
            >
              <RotateCcw className="size-4" />
            </button>
            <button
              type="button"
              onClick={() => setLarge((v) => !v)}
              title={large ? "Smaller view" : "Larger view"}
              className="text-white/40 transition-colors hover:text-white/80"
            >
              {large ? <Minimize2 className="size-4" /> : <Maximize2 className="size-4" />}
            </button>
          </div>
        </div>

        {configured === true && (
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
            <span className={toneClass(lidarDown ? "bad" : mapLine.tone)}>
              {!lidarDown && mapLine.tone === "bad" && <AlertTriangle className="size-3.5" />}
              {!lidarDown && mapLine.tone === "wait" && (
                <Loader2 className="size-3.5 animate-spin" />
              )}
              {lidarDown ? "No map without the lidar" : mapLine.text}
            </span>
            <span
              className={`font-mono text-[0.65rem] ${lidarLine.tone === "ok" ? "text-white/35" : "text-white/45"}`}
            >
              Lidar: {lidarLine.tone === "bad" ? "no scan" : lidarLine.text}
            </span>
          </div>
        )}

        <div
          className={`relative overflow-hidden rounded-md bg-[#0b0c0e] ${
            large ? "h-[clamp(520px,82vh,1100px)]" : "h-[clamp(380px,62vh,760px)]"
          }`}
        >
          <SensorScene
            cloud={cloud}
            map={map}
            useMap={configured === true}
            mapEpoch={mapEpoch}
            settings={settings}
            mount={status.mount}
            robot={robot}
            joints={joints}
            follow={follow}
            viewRef={viewRef}
          />
          {configured === false && (
            <div className="pointer-events-none absolute inset-x-0 bottom-3 flex justify-center px-3">
              <span className="rounded-md border border-white/10 bg-black/70 px-3 py-1.5 text-xs text-white/70">
                No lidar is set up on this robot. Run <code>axol lidar.setup</code> to add one.
              </span>
            </div>
          )}
          {lidarDown && (
            <div className="pointer-events-none absolute inset-x-0 bottom-3 flex justify-center px-3">
              <span className="rounded-md border border-amber-400/25 bg-black/70 px-3 py-1.5 text-xs text-amber-200">
                {status.error ?? "The lidar isn't sending."} If it still fails, run{" "}
                <code>axol lidar.info</code> on the robot.
              </span>
            </div>
          )}
        </div>

        {configured === true && <LidarControls settings={settings} onChange={change} />}
      </div>

      {cameras ? (
        <CameraFeeds host={video.host} vrPort={video.vrPort} onSocket={setVrSocket} />
      ) : (
        <div className="flex items-center gap-3 rounded-lg border border-white/10 bg-white/[0.02] p-4 text-sm text-white/55">
          <VideoOff className="size-4 shrink-0 text-white/35" />
          Press Start to stream the cameras. They also show here while teleop, Jog or data
          collection runs.
        </div>
      )}
    </div>
  )
}
