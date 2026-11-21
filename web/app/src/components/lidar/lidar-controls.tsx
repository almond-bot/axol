import { Slider } from "@/components/ui/slider"
import { Select, SelectOption } from "@/components/ui/select"
import { type LidarColorMode, type LidarSettings } from "@/lib/lidar"

/** History, point size, ceiling cut and coloring for a lidar view. */
export function LidarControls({
  settings,
  onChange,
}: {
  settings: LidarSettings
  onChange: (next: LidarSettings) => void
}) {
  const set = <K extends keyof LidarSettings>(key: K, value: LidarSettings[K]) =>
    onChange({ ...settings, [key]: value })
  return (
    <div className="grid gap-x-6 gap-y-2 text-xs text-white/55 sm:grid-cols-2">
      <label className="flex items-center gap-3" title="How many seconds of scans to show">
        <span className="w-20 shrink-0">History</span>
        <Slider
          aria-label="History"
          value={settings.history}
          min={0.1}
          max={5}
          step={0.1}
          format={(v) => `${v.toFixed(1)} s`}
          onChange={(v) => set("history", v)}
        />
      </label>
      <label className="flex items-center gap-3" title="Size of each point">
        <span className="w-20 shrink-0">Point size</span>
        <Slider
          aria-label="Point size"
          value={settings.pointSize}
          min={0.3}
          max={5}
          step={0.1}
          format={(v) => `${v.toFixed(1)} cm`}
          onChange={(v) => set("pointSize", v)}
        />
      </label>
      <label
        className="flex items-center gap-3"
        title="Hide points above this height, to see under a ceiling"
      >
        <span className="w-20 shrink-0">Hide above</span>
        <Slider
          aria-label="Hide above"
          value={settings.hideAbove}
          min={0.3}
          max={6}
          step={0.1}
          format={(v) => `${v.toFixed(1)} m`}
          onChange={(v) => set("hideAbove", v)}
        />
      </label>
      <label
        className="flex items-center gap-3"
        title="Color points by height or by how much light they reflect"
      >
        <span className="w-20 shrink-0">Color by</span>
        <Select
          aria-label="Color by"
          value={settings.colorBy}
          onChange={(e) => set("colorBy", e.target.value as LidarColorMode)}
          className="h-8"
        >
          <SelectOption value="height" label="Height" />
          <SelectOption value="reflectivity" label="Reflectivity" />
        </Select>
      </label>
    </div>
  )
}
