import { useState } from "react"
import { Camera, Loader2, Plug, Check, AlertTriangle } from "lucide-react"
import {
  fetchBoxInfo,
  zedCameraCount,
  type ServerInfo,
  type ZedSpec,
  type ZedTopology,
} from "@/lib/supervisor"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

const selectClass =
  "h-9 w-full rounded-md border border-input bg-white/[0.02] px-3 text-sm text-foreground outline-none focus-visible:border-ring/70 disabled:opacity-50"

const CAMERAS: { key: keyof ZedSpec["cameras"]; label: string; port: number }[] = [
  { key: "overhead", label: "Overhead", port: 30000 },
  { key: "left_arm", label: "Left arm", port: 30002 },
  { key: "right_arm", label: "Right arm", port: 30004 },
]

export function ZedOrchestration({
  spec,
  onChange,
  hostInfo,
  disabled,
}: {
  spec: ZedSpec
  onChange: (patch: Partial<ZedSpec>) => void
  hostInfo: ServerInfo | null
  disabled: boolean
}) {
  const [boxIfaces, setBoxIfaces] = useState<string[]>([])
  const [probe, setProbe] = useState<{
    state: "idle" | "loading" | "ok" | "err"
    message?: string
  }>({ state: "idle" })

  async function testBox() {
    if (!spec.boxUrl.trim()) return
    setProbe({ state: "loading" })
    try {
      const info = await fetchBoxInfo(spec.boxUrl.trim())
      setBoxIfaces(info.ethIfaces ?? [])
      setProbe({ state: "ok", message: info.hostname })
      // Adopt the box's detected interface if the user hasn't picked one yet.
      if (!spec.boxIface && info.ethIface) onChange({ boxIface: info.ethIface })
      // For LAN topology, default the stream IP to the box's address.
      if (spec.topology === "lan" && !spec.zedHost) {
        onChange({ zedHost: hostFromUrl(spec.boxUrl) })
      }
    } catch (e) {
      setProbe({ state: "err", message: String(e) })
    }
  }

  const camCount = zedCameraCount(spec)

  return (
    <Card>
      <CardHeader className="flex-row items-start justify-between gap-3">
        <div className="min-w-0">
          <CardTitle className="flex items-center gap-2">
            <Camera className="size-4 text-[#eff483]" />
            ZED cameras
          </CardTitle>
          <CardDescription>
            Auto-sync clocks and start the ZED streams before recording.
          </CardDescription>
        </div>
        <Toggle
          checked={spec.enabled}
          disabled={disabled}
          onChange={(v) => onChange({ enabled: v })}
        />
      </CardHeader>

      {spec.enabled && (
        <CardContent className="gap-5">
          <Field label="Network topology">
            <select
              className={selectClass}
              value={spec.topology}
              disabled={disabled}
              onChange={(e) => onChange({ topology: e.target.value as ZedTopology })}
            >
              <option value="direct" className="bg-[#1a1a1a]">
                Direct cable (auto 192.168.10.1 / .2)
              </option>
              <option value="lan" className="bg-[#1a1a1a]">
                Shared LAN (DHCP)
              </option>
            </select>
            <p className="text-xs text-white/40">
              {spec.topology === "direct"
                ? "One cable between the two NICs. The box takes 192.168.10.1, this host 192.168.10.2 — assigned for you."
                : "Both machines on the same network. Point the receiver at the box's streaming IP below."}
            </p>
          </Field>

          <Field label="ZED box address" hint="its `axol serve` (default :8090)">
            <div className="flex gap-2">
              <Input
                value={spec.boxUrl}
                disabled={disabled}
                onChange={(e) => onChange({ boxUrl: e.target.value })}
                placeholder="192.168.1.50:8090"
                spellCheck={false}
                autoCapitalize="off"
                autoCorrect="off"
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={disabled || !spec.boxUrl.trim() || probe.state === "loading"}
                onClick={testBox}
                className="shrink-0"
              >
                {probe.state === "loading" ? <Loader2 className="animate-spin" /> : <Plug />}
                Test
              </Button>
            </div>
            {probe.state === "ok" && (
              <p className="flex items-center gap-1.5 text-xs text-emerald-400/90">
                <Check className="size-3" />
                Connected to {probe.message}
              </p>
            )}
            {probe.state === "err" && (
              <p className="flex items-center gap-1.5 text-xs text-red-400">
                <AlertTriangle className="size-3" />
                {probe.message}
              </p>
            )}
          </Field>

          <div className="grid grid-cols-2 gap-4">
            <Field label="Host interface">
              <IfaceField
                value={spec.hostIface}
                candidates={hostInfo?.ethIfaces ?? []}
                disabled={disabled}
                onChange={(v) => onChange({ hostIface: v })}
              />
            </Field>
            <Field label="ZED box interface">
              <IfaceField
                value={spec.boxIface}
                candidates={boxIfaces}
                disabled={disabled}
                onChange={(v) => onChange({ boxIface: v })}
              />
            </Field>
          </div>

          {spec.topology === "lan" && (
            <Field label="ZED stream IP" hint="where the box streams from">
              <Input
                value={spec.zedHost}
                disabled={disabled}
                onChange={(e) => onChange({ zedHost: e.target.value })}
                placeholder="192.168.1.50"
                spellCheck={false}
              />
            </Field>
          )}

          <div className="flex flex-col gap-3 rounded-lg border border-[#eff483]/25 bg-[#eff483]/[0.04] p-3">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs tracking-widest text-[#eff483]/80 uppercase">
                Camera serials
              </span>
              <Badge variant={camCount > 0 ? "success" : "warning"}>{camCount} set</Badge>
            </div>
            <p className="text-xs text-white/45">
              Serial number of each ZED-X One. Enter every camera you have wired — at least one is
              required.
            </p>
            {CAMERAS.map((cam) => (
              <Field key={cam.key} label={cam.label} hint={`port ${cam.port}`} inline>
                <Input
                  value={spec.cameras[cam.key]}
                  disabled={disabled}
                  inputMode="numeric"
                  onChange={(e) =>
                    onChange({
                      cameras: { ...spec.cameras, [cam.key]: e.target.value },
                    })
                  }
                  placeholder="serial"
                  className="max-w-[180px]"
                />
              </Field>
            ))}
          </div>
        </CardContent>
      )}
    </Card>
  )
}

function hostFromUrl(url: string): string {
  return url
    .trim()
    .replace(/^\w+:\/\//, "")
    .split("/")[0]
    .split(":")[0]
}

function IfaceField({
  value,
  candidates,
  disabled,
  onChange,
}: {
  value: string
  candidates: string[]
  disabled: boolean
  onChange: (v: string) => void
}) {
  // Offer detected interfaces as a dropdown, but always allow a manual value
  // (the machine may be remote / non-Linux, where detection returns nothing).
  const known = candidates.includes(value) || value === ""
  if (candidates.length > 0 && known) {
    return (
      <select
        className={selectClass}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="" className="bg-[#1a1a1a]">
          Select…
        </option>
        {candidates.map((c) => (
          <option key={c} value={c} className="bg-[#1a1a1a]">
            {c}
          </option>
        ))}
        <option value="__custom__" className="bg-[#1a1a1a]">
          Other…
        </option>
      </select>
    )
  }
  return (
    <Input
      value={value === "__custom__" ? "" : value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      placeholder="eth0"
      spellCheck={false}
      autoCapitalize="off"
      autoCorrect="off"
    />
  )
}

function Field({
  label,
  hint,
  inline,
  children,
}: {
  label: string
  hint?: string
  inline?: boolean
  children: React.ReactNode
}) {
  if (inline) {
    return (
      <div className="flex items-center justify-between gap-4">
        <div className="flex min-w-0 items-baseline gap-2">
          <Label>{label}</Label>
          {hint && <span className="text-xs text-white/35">{hint}</span>}
        </div>
        {children}
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <Label>{label}</Label>
        {hint && <span className="text-xs text-white/35">{hint}</span>}
      </div>
      {children}
    </div>
  )
}

function Toggle({
  checked,
  disabled,
  onChange,
}: {
  checked: boolean
  disabled: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative h-6 w-11 shrink-0 rounded-full border transition-colors disabled:opacity-50",
        checked ? "border-[#eff483]/50 bg-[#eff483]/80" : "border-white/15 bg-white/[0.06]"
      )}
    >
      <span
        className={cn(
          "absolute top-0.5 left-0.5 size-4.5 rounded-full transition-transform",
          checked ? "translate-x-5 bg-[#121212]" : "translate-x-0 bg-white/80"
        )}
      />
    </button>
  )
}
