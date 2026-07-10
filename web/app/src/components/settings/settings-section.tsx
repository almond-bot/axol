import { useMemo, useRef, useState } from "react"
import { ChevronDown, Download, Loader2, Plug, RotateCcw, Search, Upload } from "lucide-react"
import {
  flattenFields,
  type AdvancedSection,
  type CameraDevice,
  type CameraSpec,
  type FormValue,
  type SettingsCategory,
  type SettingsField,
  type SettingsPatch,
  type SettingsSnapshot,
  type SettingValue,
  type UsbStatus,
} from "@/lib/supervisor"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectOption } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { useToast } from "@/components/ui/toast"
import { Card } from "@/components/ui/card"
import { FieldRow, FlatSchemaForm } from "@/components/config-form"
import { materializeCameraSpec } from "@/lib/camera-spec"
import { CamerasPanel } from "./cameras-panel"
import { PosePanel } from "./pose-panel"
import { cn } from "@/lib/utils"

export type SettingsTab = string // "cameras" | "usb" | "pose" | "advanced" | a category key

interface Draft {
  values: Record<string, SettingValue>
  cameras: CameraSpec
  advanced: Record<string, FormValue>
}

/**
 * The shared settings, tabbed directly on the control panel page: everything
 * that isn't a per-run input — Cameras, Quest USB, Robot, Teleop & VR, Rest
 * pose, Recording, Inference, System, plus per-op Advanced overrides. Values
 * persist on the serve host (~/.almond/settings.json) and are folded into
 * every operation start, so they're shared across operations and operator
 * devices. Edits stage locally until **Save**.
 */
export function SettingsSection({
  open,
  onOpenChange,
  tab,
  onTabChange,
  snapshot,
  supportError,
  cameras,
  onSave,
  devices,
  detecting,
  onRefresh,
  usb,
  usbBusy,
  onUsbConnect,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  tab: SettingsTab
  onTabChange: (tab: SettingsTab) => void
  /** Server-stored settings + schema; null while loading or on an old host. */
  snapshot: SettingsSnapshot | null
  /** Why the settings API is unavailable (old serve host), else null. */
  supportError: string | null
  /** Current camera spec (server-stored, with localStorage fallback). */
  cameras: CameraSpec
  onSave: (patch: SettingsPatch) => Promise<void>
  devices: CameraDevice[] | null
  detecting: boolean
  onRefresh: () => void
  usb: UsbStatus | null
  usbBusy: boolean
  onUsbConnect: () => void
}) {
  const toast = useToast()
  const [draft, setDraft] = useState<Draft | null>(null)
  const [saving, setSaving] = useState(false)
  const [seedKey, setSeedKey] = useState("")
  const fileRef = useRef<HTMLInputElement>(null)

  const patch = useMemo(
    () => (draft ? computePatch(snapshot, cameras, draft, devices) : null),
    [draft, snapshot, cameras, devices]
  )
  const dirty = patch != null && Object.keys(patch).length > 0

  // Seed / reseed the draft from the stored state ("adjust state when props
  // change" — set during render, not in an effect). Do not seed while the
  // settings request is still in flight: an empty early draft compared with a
  // populated response looks like the operator deleted every stored value,
  // which makes `dirty` true and prevents the response from ever reaching the
  // form. On an old host, `supportError` resolves the request and enables the
  // camera-only fallback draft.
  const seedDraft = (): Draft => ({
    values: { ...(snapshot?.values ?? {}) },
    cameras,
    advanced: { ...(snapshot?.advanced ?? {}) },
  })
  const settingsResolved = snapshot !== null || supportError !== null
  const nextSeedKey = settingsResolved
    ? JSON.stringify([snapshot?.values, snapshot?.advanced, cameras])
    : null
  if (nextSeedKey !== null && nextSeedKey !== seedKey && (draft == null || !dirty)) {
    setSeedKey(nextSeedKey)
    setDraft(seedDraft())
  }

  const schema = useMemo(() => snapshot?.schema ?? [], [snapshot])

  async function save() {
    if (!patch) return
    setSaving(true)
    try {
      await onSave(patch)
    } catch (e) {
      toast.error(String(e).replace(/^Error:\s*/, ""))
    } finally {
      setSaving(false)
    }
  }

  function discard() {
    if (nextSeedKey === null) return
    setSeedKey(nextSeedKey)
    setDraft(seedDraft())
  }

  function setValue(key: string, value: SettingValue | null) {
    setDraft((d) => {
      if (!d) return d
      const values = { ...d.values }
      if (value === null) delete values[key]
      else values[key] = value
      return { ...d, values }
    })
  }

  function exportFile() {
    if (!draft) return
    const blob = new Blob(
      [
        JSON.stringify(
          { values: draft.values, cameras: draft.cameras, advanced: draft.advanced },
          null,
          2
        ),
      ],
      { type: "application/json" }
    )
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "axol-settings.json"
    a.click()
    URL.revokeObjectURL(url)
  }

  function importFile(text: string) {
    try {
      const data = JSON.parse(text)
      if (!data || typeof data !== "object") throw new Error("invalid settings file")
      setDraft((d) =>
        d
          ? {
              values: { ...(data.values ?? {}) },
              cameras: (data.cameras as CameraSpec) ?? d.cameras,
              advanced: { ...(data.advanced ?? {}) },
            }
          : d
      )
    } catch (e) {
      toast.error(`Import failed: ${e}`)
    }
  }

  // Tab order: the hardware attached to the host first (Cameras, Quest USB),
  // then the behaviour categories with the pose editor after Teleop & VR,
  // then the per-op Advanced overrides.
  const tabs: { key: SettingsTab; label: string }[] = [
    { key: "cameras", label: "Cameras" },
    { key: "usb", label: "Quest USB" },
  ]
  for (const cat of schema) {
    tabs.push({ key: cat.key, label: cat.label })
    if (cat.key === "teleop") tabs.push({ key: "pose", label: "Rest pose" })
  }
  tabs.push({ key: "advanced", label: "Advanced" })

  const activeCategory = schema.find((c) => c.key === tab)
  const poseFields =
    schema.find((c) => c.key === "teleop")?.settings.filter((s) => s.ui.widget === "pose") ?? []

  return (
    <Card className="gap-0 p-0">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => onOpenChange(!open)}
        className={cn(
          "flex w-full items-center justify-between gap-3 px-5 py-3.5 text-left",
          open && "border-b border-white/10"
        )}
      >
        <span className="font-mono text-xs tracking-widest text-white/40 uppercase">Settings</span>
        <span className="ml-auto hidden text-xs text-white/35 sm:block">
          shared by all operations on this robot
        </span>
        <ChevronDown
          className={cn("size-4 shrink-0 text-white/45 transition-transform", open && "rotate-180")}
        />
      </button>

      {open && (
        <>
          {/* Horizontal tab bar */}
          <nav className="flex gap-1 overflow-x-auto border-b border-white/10 px-3 py-2">
            {tabs.map((t) => (
              <button
                key={t.key}
                type="button"
                onClick={() => onTabChange(t.key)}
                className={cn(
                  "rounded-lg px-3 py-1.5 text-sm whitespace-nowrap transition-colors",
                  t.key === tab
                    ? "bg-[#eff483]/10 text-[#eff483]"
                    : "text-white/60 hover:bg-white/[0.05] hover:text-white/85"
                )}
              >
                {t.label}
              </button>
            ))}
          </nav>

          <div className="p-5">
            {!draft ? (
              <p className="text-sm text-white/40">Loading settings…</p>
            ) : supportError && tab !== "cameras" && tab !== "usb" ? (
              <UpdateRequired error={supportError} />
            ) : tab === "cameras" ? (
              <CamerasPanel
                spec={draft.cameras}
                onChange={(spec) => setDraft((d) => (d ? { ...d, cameras: spec } : d))}
                devices={devices}
                detecting={detecting}
                onRefresh={onRefresh}
              />
            ) : tab === "usb" ? (
              <UsbPanel usb={usb} usbBusy={usbBusy} onUsbConnect={onUsbConnect} />
            ) : tab === "pose" ? (
              <PosePanel fields={poseFields} values={draft.values} onChange={setValue} />
            ) : tab === "advanced" ? (
              <AdvancedPanel
                sections={snapshot?.advancedSchema ?? []}
                overrides={draft.advanced}
                onChange={(key, value) =>
                  setDraft((d) => (d ? { ...d, advanced: { ...d.advanced, [key]: value } } : d))
                }
                onReset={(key) =>
                  setDraft((d) => {
                    if (!d) return d
                    const advanced = { ...d.advanced }
                    delete advanced[key]
                    return { ...d, advanced }
                  })
                }
              />
            ) : activeCategory ? (
              <CategoryPanel category={activeCategory} values={draft.values} onChange={setValue} />
            ) : (
              <p className="text-sm text-white/40">Loading settings…</p>
            )}
          </div>

          <div className="flex items-center justify-between gap-2 border-t border-white/10 px-5 py-3">
            <div className="flex items-center gap-1">
              <Button variant="ghost" size="sm" onClick={exportFile}>
                <Download />
                Export
              </Button>
              <Button variant="ghost" size="sm" onClick={() => fileRef.current?.click()}>
                <Upload />
                Import
              </Button>
              <input
                ref={fileRef}
                type="file"
                accept="application/json"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (file) file.text().then(importFile)
                  e.target.value = ""
                }}
              />
            </div>
            <div className="flex items-center gap-2">
              {dirty && (
                <>
                  <span className="text-xs text-white/40">Unsaved changes</span>
                  <Button variant="outline" size="sm" onClick={discard}>
                    Discard
                  </Button>
                </>
              )}
              <Button size="sm" onClick={save} disabled={!dirty || saving}>
                {saving && <Loader2 className="animate-spin" />}
                Save
              </Button>
            </div>
          </div>
        </>
      )}
    </Card>
  )
}

/** Diff the draft against what's stored, producing the minimal PUT payload. */
function computePatch(
  snapshot: SettingsSnapshot | null,
  storedCameras: CameraSpec,
  draft: Draft,
  devices: CameraDevice[] | null
): SettingsPatch {
  const patch: SettingsPatch = {}

  const before = snapshot?.values ?? {}
  const valuesPatch: Record<string, SettingValue | null> = {}
  for (const [k, v] of Object.entries(draft.values)) {
    if (JSON.stringify(before[k]) !== JSON.stringify(v)) valuesPatch[k] = v
  }
  for (const k of Object.keys(before)) {
    if (!(k in draft.values)) valuesPatch[k] = null
  }
  if (Object.keys(valuesPatch).length > 0) patch.values = valuesPatch

  const camerasOut = materializeCameraSpec(draft.cameras, devices)
  if (JSON.stringify(camerasOut) !== JSON.stringify(storedCameras)) {
    patch.cameras = camerasOut
    patch.camerasSet = true
  }

  const beforeAdv = snapshot?.advanced ?? {}
  const advPatch: Record<string, FormValue | null> = {}
  for (const [k, v] of Object.entries(draft.advanced)) {
    if (JSON.stringify(beforeAdv[k]) !== JSON.stringify(v)) advPatch[k] = v
  }
  for (const k of Object.keys(beforeAdv)) {
    if (!(k in draft.advanced)) advPatch[k] = null
  }
  if (Object.keys(advPatch).length > 0) patch.advanced = advPatch

  return patch
}

function UpdateRequired({ error }: { error: string }) {
  return (
    <div className="flex flex-col gap-2 rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-4 text-sm">
      <span className="font-medium text-amber-300/90">Serve host update required</span>
      <p className="text-white/55">
        This Axol host doesn&apos;t support shared settings yet — update it from the banner on the
        control panel (or run <span className="font-mono">axol update</span>), then reconnect.
        Cameras still work in the meantime.
      </p>
      <code className="rounded bg-black/30 p-2 text-xs break-words text-white/45">{error}</code>
    </div>
  )
}

/** The Quest-over-USB pose link: live status + connect, from the old tile. */
function UsbPanel({
  usb,
  usbBusy,
  onUsbConnect,
}: {
  usb: UsbStatus | null
  usbBusy: boolean
  onUsbConnect: () => void
}) {
  const dotClass = !usb
    ? "bg-white/30"
    : !usb.installed
      ? "bg-amber-400"
      : usb.ready
        ? "bg-emerald-400"
        : usb.state === "none"
          ? "bg-white/30"
          : "bg-amber-400"
  const label = !usb
    ? "—"
    : !usb.installed
      ? "adb not installed"
      : usb.ready
        ? "Controller over USB"
        : usb.state === "device"
          ? "Headset ready"
          : usb.state === "none"
            ? "No headset"
            : usb.state === "unauthorized"
              ? "Authorize on headset"
              : usb.state
  // Per-state hint, mirroring the help line every other settings field has.
  const hint = !usb
    ? "Waiting for the host to report the USB link state."
    : !usb.installed
      ? "Install adb on the host, then reconnect the headset."
      : usb.ready
        ? "Controller poses stream over the cable; a dropped cable fails over to Wi-Fi instantly."
        : usb.state === "device"
          ? "Headset detected — the tunnel connects automatically; use Connect to retry."
          : usb.state === "none"
            ? "Plug the Quest into the Axol host with a USB-C cable."
            : usb.state === "unauthorized"
              ? "Put the headset on and accept the “Allow USB debugging?” prompt."
              : "Reconnect the headset's USB cable."
  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-white/45">
        Stream controller poses over a USB cable instead of Wi-Fi for lower latency. The link is set
        up automatically when an authorized headset is plugged into the host; camera video keeps
        using the LAN.
      </p>
      <div className="flex max-w-xl flex-col gap-1.5">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className={cn("size-2 shrink-0 rounded-full", dotClass)} />
            <Label>{label}</Label>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={onUsbConnect}
            disabled={usbBusy || usb?.installed === false}
          >
            {usbBusy ? <Loader2 className="animate-spin" /> : <Plug />}
            {usb?.ready ? "Reconnect" : "Connect"}
          </Button>
        </div>
        <p className="max-w-prose text-xs text-white/35">{hint}</p>
      </div>
    </div>
  )
}

/** One settings category rendered as proper controls (not bare text boxes). */
function CategoryPanel({
  category,
  values,
  onChange,
}: {
  category: SettingsCategory
  values: Record<string, SettingValue>
  onChange: (key: string, value: SettingValue | null) => void
}) {
  const fields = category.settings.filter((s) => s.ui.widget !== "pose")
  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-white/45">{category.description}</p>
      <div className="grid gap-x-8 gap-y-5 sm:grid-cols-2">
        {fields.map((f) => (
          <SettingRow key={f.key} field={f} value={values[f.key]} onChange={onChange} />
        ))}
      </div>
    </div>
  )
}

function SettingRow({
  field,
  value,
  onChange,
}: {
  field: SettingsField
  value: SettingValue | undefined
  onChange: (key: string, value: SettingValue | null) => void
}) {
  const set = value !== undefined
  const id = `setting-${field.key}`

  const labelNode = (
    <div className="flex min-w-0 items-center gap-2">
      <Label htmlFor={id}>{field.label}</Label>
      {set && (
        <>
          <span className="size-1.5 rounded-full bg-[#eff483]" title="Changed from default" />
          <button
            type="button"
            onClick={() => onChange(field.key, null)}
            title="Reset to default"
            className="text-white/30 hover:text-white/70"
          >
            <RotateCcw className="size-3" />
          </button>
        </>
      )}
    </div>
  )

  if (field.type === "boolean") {
    const checked = set ? Boolean(value) : Boolean(field.default)
    return (
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between gap-4">
          {labelNode}
          <Switch
            checked={checked}
            onChange={(v) => onChange(field.key, v)}
            aria-label={field.label}
          />
        </div>
        <p className="max-w-prose text-xs text-white/35">{field.help}</p>
      </div>
    )
  }

  if (field.ui.widget === "slider") {
    const current = set ? Number(value) : Number(field.default ?? field.ui.min ?? 0)
    return (
      <div className="flex flex-col gap-1.5">
        {labelNode}
        <Slider
          value={current}
          min={field.ui.min ?? 0}
          max={field.ui.max ?? 1}
          step={field.ui.step ?? 0.05}
          onChange={(v) => onChange(field.key, v)}
          format={(v) => v.toFixed(2)}
          aria-label={field.label}
        />
        <p className="max-w-prose text-xs text-white/35">{field.help}</p>
      </div>
    )
  }

  if (field.type === "select") {
    return (
      <div className="flex flex-col gap-1.5">
        {labelNode}
        <Select
          id={id}
          value={set ? String(value) : String(field.default ?? "")}
          onChange={(e) => onChange(field.key, e.target.value)}
        >
          {(field.options ?? []).map((opt) => (
            <SelectOption key={opt} value={opt} />
          ))}
        </Select>
        <p className="max-w-prose text-xs text-white/35">{field.help}</p>
      </div>
    )
  }

  const text = set ? String(value) : ""
  const placeholder = field.default != null ? String(field.default) : (field.defaultText ?? "unset")
  return (
    <div className="flex flex-col gap-1.5">
      {labelNode}
      <Input
        id={id}
        inputMode={field.type === "number" ? "decimal" : undefined}
        value={text}
        placeholder={placeholder}
        onChange={(e) => {
          const raw = e.target.value
          if (raw === "") return onChange(field.key, null)
          if (field.type === "number") {
            const n = Number(raw)
            return onChange(field.key, Number.isFinite(n) ? n : raw)
          }
          onChange(field.key, raw)
        }}
      />
      <p className="max-w-prose text-xs text-white/35">{field.help}</p>
    </div>
  )
}

/**
 * The unified Advanced tree: every remaining config field, organized by
 * subsystem (Axol, Teleop, Kinematics, VR server, LeRobot robot). One value
 * here is the source of truth for **all** operations — the server translates
 * each canonical key to the right config path per op at start. Curated
 * settings, cameras and per-run fields are pruned server-side so every knob
 * has exactly one home.
 */
function AdvancedPanel({
  sections,
  overrides,
  onChange,
  onReset,
}: {
  sections: AdvancedSection[]
  overrides: Record<string, FormValue>
  onChange: (key: string, value: FormValue) => void
  onReset: (key: string) => void
}) {
  const [sectionKey, setSectionKey] = useState<string | null>(null)
  const section = sections.find((s) => s.key === sectionKey) ?? sections[0]

  const [query, setQuery] = useState("")
  const q = query.trim().toLowerCase()
  const matches = useMemo(
    () =>
      q
        ? sections
            .flatMap((s) => flattenFields(s.nodes))
            .filter((f) => f.key.toLowerCase().includes(q) || f.label.toLowerCase().includes(q))
        : null,
    [q, sections]
  )

  const common = { overrides, disabled: false, onChange, onReset }
  const editedCount = Object.keys(overrides).length

  if (sections.length === 0) {
    return <p className="text-sm text-white/40">No advanced settings available on this host.</p>
  }
  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-white/45">
        Everything not covered by the other tabs. One value set here applies to{" "}
        <span className="text-white/65">every operation</span> that uses the subsystem.
      </p>
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative min-w-48 flex-1">
          <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-white/30" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search all config…"
            className="pl-9"
          />
        </div>
        {editedCount > 0 && (
          <span className="rounded-full bg-[#eff483]/15 px-2 py-0.5 font-mono text-[0.65rem] text-[#eff483]">
            {editedCount} edited
          </span>
        )}
      </div>
      {matches ? (
        <div className="flex max-w-xl flex-col gap-4">
          {matches.length === 0 ? (
            <p className="text-sm text-white/35">No matching config.</p>
          ) : (
            matches.map((f) => <FieldRow key={f.key} field={f} showPath {...common} />)
          )}
        </div>
      ) : (
        <>
          <div className="flex flex-wrap gap-1">
            {sections.map((s) => (
              <button
                key={s.key}
                type="button"
                onClick={() => setSectionKey(s.key)}
                className={cn(
                  "rounded-md px-2.5 py-1 text-xs whitespace-nowrap transition-colors",
                  s.key === section?.key
                    ? "bg-white/[0.08] text-white"
                    : "text-white/55 hover:bg-white/[0.04] hover:text-white/80"
                )}
              >
                {s.label}
              </button>
            ))}
          </div>
          {section && <FlatSchemaForm nodes={section.nodes} {...common} />}
        </>
      )}
    </div>
  )
}
