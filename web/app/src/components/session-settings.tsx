import { useMemo } from "react"
import { Minus, Plus, SlidersHorizontal } from "lucide-react"
import {
  formatSettingValue,
  nextSettingValue,
  useAxolSettings,
  type AxolSettingDef,
  type AxolSettingValue,
} from "@almond/axol-vr-client"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"

/**
 * Live session settings for the running operation, on the control panel.
 *
 * The same knobs the in-headset HUD exposes (box mode, re-engage behaviour,
 * grip force, reach scale, arm speed, …), rendered generically from the
 * schema the teleop server publishes over its WebSocket (`settings`
 * messages) and changed with `set` messages on the same socket — so an
 * operator at the desk and one in the headset see one shared state. Renders
 * nothing until the server announces its settings (an older operation has
 * none).
 *
 * `socket` is the open VR-server connection owned by `CameraFeeds` (see its
 * `onSocket`), null while disconnected.
 */
export function SessionSettings({ socket }: { socket: WebSocket | null }) {
  // useAxolSettings wants a ref; a fresh object per socket re-runs its
  // subscription effect on every (re)connect.
  const socketRef = useMemo(() => ({ current: socket }), [socket])
  const { settings, setSetting } = useAxolSettings(socketRef, socket !== null)

  if (!settings || settings.schema.length === 0) return null

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-white/10 bg-white/[0.02] p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-xs tracking-widest text-white/40 uppercase">
          Session settings
        </span>
        <span className="flex items-center gap-1 font-mono text-[0.65rem] text-white/40">
          <SlidersHorizontal className="size-3" />
          live
        </span>
      </div>
      <div className="divide-y divide-white/5">
        {settings.schema.map((def) => (
          <SettingRow
            key={def.key}
            def={def}
            value={settings.values[def.key]}
            onChange={(v) => setSetting(def.key, v)}
          />
        ))}
      </div>
    </div>
  )
}

function SettingRow({
  def,
  value,
  onChange,
}: {
  def: AxolSettingDef
  value: AxolSettingValue | undefined
  onChange: (v: AxolSettingValue) => void
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-2" title={def.help}>
      <div className="min-w-0">
        <div className="text-sm text-white/85">{def.label}</div>
        <div className="truncate text-xs text-white/40">{def.help}</div>
      </div>
      {def.type === "boolean" ? (
        <Switch checked={value === true} onChange={(v) => onChange(v)} aria-label={def.label} />
      ) : def.type === "select" ? (
        <div className="flex shrink-0 overflow-hidden rounded-md border border-white/10">
          {def.options.map((opt) => (
            <button
              key={opt}
              type="button"
              onClick={() => onChange(opt)}
              className={cn(
                "px-2.5 py-1 font-mono text-xs transition-colors",
                value === opt
                  ? "bg-[#eff483]/80 text-[#121212]"
                  : "text-white/60 hover:bg-white/[0.06] hover:text-white/90"
              )}
            >
              {opt}
            </button>
          ))}
        </div>
      ) : (
        <div className="flex shrink-0 items-center gap-1">
          <Button
            variant="outline"
            size="icon"
            className="size-7"
            aria-label={`${def.label} down`}
            disabled={nextSettingValue(def, value, -1) === undefined}
            onClick={() => {
              const next = nextSettingValue(def, value, -1)
              if (next !== undefined) onChange(next)
            }}
          >
            <Minus />
          </Button>
          <span className="w-20 text-center font-mono text-xs text-white/80 tabular-nums">
            {formatSettingValue(def, value)}
          </span>
          <Button
            variant="outline"
            size="icon"
            className="size-7"
            aria-label={`${def.label} up`}
            disabled={nextSettingValue(def, value, 1) === undefined}
            onClick={() => {
              const next = nextSettingValue(def, value, 1)
              if (next !== undefined) onChange(next)
            }}
          >
            <Plus />
          </Button>
        </div>
      )}
    </div>
  )
}
