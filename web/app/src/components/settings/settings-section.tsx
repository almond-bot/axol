import { useMemo, useRef, useState } from "react"
import { Download, Loader2, Plug, RotateCcw, Search, Upload } from "lucide-react"
import {
  OPERATIONS,
  filterSchema,
  flattenFields,
  managedKeysForOp,
  type CameraDevice,
  type CameraSpec,
  type CommandSpec,
  type FormValue,
  type OperationId,
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
  opOverrides: Record<string, Record<string, FormValue>>
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
  tab,
  onTabChange,
  snapshot,
  supportError,
  cameras,
  onSave,
  specs,
  devices,
  detecting,
  onRefresh,
  usb,
  usbBusy,
  onUsbConnect,
}: {
  tab: SettingsTab
  onTabChange: (tab: SettingsTab) => void
  /** Server-stored settings + schema; null while loading or on an old host. */
  snapshot: SettingsSnapshot | null
  /** Why the settings API is unavailable (old serve host), else null. */
  supportError: string | null
  /** Current camera spec (server-stored, with localStorage fallback). */
  cameras: CameraSpec
  onSave: (patch: SettingsPatch) => Promise<void>
  /** Command specs, for the Advanced per-op override forms. */
  specs: CommandSpec[]
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
  // change" — set during render, not in an effect). While the operator has
  // unsaved edits, incoming server state must not clobber them; once clean
  // (including right after a save), the draft follows the store again.
  const seedDraft = (): Draft => ({
    values: { ...(snapshot?.values ?? {}) },
    cameras,
    opOverrides: Object.fromEntries(
      Object.entries(snapshot?.opOverrides ?? {}).map(([op, v]) => [op, { ...v }])
    ),
  })
  const nextSeedKey = JSON.stringify([snapshot?.values, snapshot?.opOverrides, cameras])
  if (nextSeedKey !== seedKey && (draft == null || !dirty)) {
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
          { values: draft.values, cameras: draft.cameras, opOverrides: draft.opOverrides },
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
              opOverrides: { ...(data.opOverrides ?? {}) },
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
      <div className="flex items-center justify-between gap-3 border-b border-white/10 px-5 pt-4 pb-0">
        <span className="pb-3 font-mono text-xs tracking-widest text-white/40 uppercase">
          Settings
        </span>
        <span className="hidden pb-3 text-xs text-white/35 sm:block">
          shared by all operations on this robot
        </span>
      </div>

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
            specs={specs}
            schema={schema}
            overridesByOp={draft.opOverrides}
            onChange={(op, next) =>
              setDraft((d) => (d ? { ...d, opOverrides: { ...d.opOverrides, [op]: next } } : d))
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

  const beforeOps = snapshot?.opOverrides ?? {}
  const opsPatch: Record<string, Record<string, FormValue>> = {}
  for (const op of new Set([...Object.keys(beforeOps), ...Object.keys(draft.opOverrides)])) {
    const next = draft.opOverrides[op] ?? {}
    if (JSON.stringify(beforeOps[op] ?? {}) !== JSON.stringify(next)) opsPatch[op] = next
  }
  if (Object.keys(opsPatch).length > 0) patch.opOverrides = opsPatch

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
  return (
    <div className="flex max-w-prose flex-col gap-4">
      <p className="text-xs text-white/45">
        Plug the Quest into the Axol host over USB to stream controller poses over the cable instead
        of Wi-Fi (lower latency). The link is set up automatically when an authorized headset is
        detected — accept the <span className="text-white/70">Allow USB debugging?</span> prompt on
        the headset the first time. Camera video keeps using the LAN; a dropped cable fails over to
        Wi-Fi instantly.
      </p>
      <div className="flex items-center justify-between gap-4 rounded-lg border border-white/10 bg-white/[0.02] px-4 py-3">
        <div className="flex items-center gap-2 text-sm">
          <span className={cn("size-2 shrink-0 rounded-full", dotClass)} />
          <span className="text-white/75">{label}</span>
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
    <div className="flex max-w-xl flex-col gap-5">
      <p className="text-xs text-white/45">{category.description}</p>
      {fields.map((f) => (
        <SettingRow key={f.key} field={f} value={values[f.key]} onChange={onChange} />
      ))}
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
  const placeholder = field.default == null ? "unset" : String(field.default)
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
 * Per-op escape hatch: the full introspected config tree, minus everything a
 * shared setting or the Cameras tab already owns (single source of truth) and
 * minus the per-run fields the op panel asks for. Values persist server-side
 * and are folded in beneath the per-run args.
 */
function AdvancedPanel({
  specs,
  schema,
  overridesByOp,
  onChange,
}: {
  specs: CommandSpec[]
  schema: SettingsCategory[]
  overridesByOp: Record<string, Record<string, FormValue>>
  onChange: (op: string, next: Record<string, FormValue>) => void
}) {
  const [op, setOp] = useState<OperationId>("teleop")
  const spec = specs.find((s) => s.id === op)
  const meta = OPERATIONS.find((o) => o.id === op)

  const visibleSchema = useMemo(() => {
    if (!spec) return []
    const exclude = managedKeysForOp(schema, op)
    for (const f of meta?.fields ?? []) exclude.add(f)
    for (const f of flattenFields(spec.schema)) {
      if (f.required) exclude.add(f.key)
      // Camera fields are owned end-to-end by the Cameras tab.
      if (f.key.startsWith("robot_config.cameras.")) exclude.add(f.key)
    }
    return filterSchema(spec.schema, exclude)
  }, [spec, schema, op, meta])

  // Top-level sections of the selected op's config, as flat sub-tabs: root
  // fields group under "General", each nested config gets its own tab. No
  // collapsed trees to dig through — a tab's fields are all visible at once.
  const sections = useMemo(() => {
    const rootFields = visibleSchema.filter((n) => n.kind === "field")
    const out: { key: string; label: string; nodes: typeof visibleSchema }[] = []
    if (rootFields.length > 0) out.push({ key: "__general", label: "General", nodes: rootFields })
    for (const n of visibleSchema) {
      if (n.kind === "group") out.push({ key: n.key, label: n.label, nodes: n.children })
    }
    return out
  }, [visibleSchema])
  const [sectionKey, setSectionKey] = useState<string | null>(null)
  const section = sections.find((s) => s.key === sectionKey) ?? sections[0]

  const [query, setQuery] = useState("")
  const q = query.trim().toLowerCase()
  const matches = useMemo(
    () =>
      q
        ? flattenFields(visibleSchema).filter(
            (f) => f.key.toLowerCase().includes(q) || f.label.toLowerCase().includes(q)
          )
        : null,
    [q, visibleSchema]
  )

  const overrides = overridesByOp[op] ?? {}
  const editedCount = Object.keys(overrides).length
  const common = {
    overrides,
    disabled: false,
    onChange: (key: string, value: FormValue) => onChange(op, { ...overrides, [key]: value }),
    onReset: (key: string) => {
      const next = { ...overrides }
      delete next[key]
      onChange(op, next)
    },
  }

  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-white/45">
        Rarely-needed per-operation overrides for anything not covered by the other tabs. Values set
        here apply on top of the shared settings every time the operation runs.
      </p>
      <div className="flex flex-wrap items-center gap-3">
        <Select
          value={op}
          onChange={(e) => {
            setOp(e.target.value as OperationId)
            setSectionKey(null)
            setQuery("")
          }}
          className="max-w-56"
        >
          {OPERATIONS.map((o) => (
            <SelectOption key={o.id} value={o.id} label={o.label} />
          ))}
        </Select>
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
      {!spec ? (
        <p className="text-sm text-white/40">This operation isn&apos;t available on the host.</p>
      ) : matches ? (
        <div className="flex max-w-xl flex-col gap-4">
          {matches.length === 0 ? (
            <p className="text-sm text-white/35">No matching config.</p>
          ) : (
            matches.map((f) => <FieldRow key={f.key} field={f} showPath {...common} />)
          )}
        </div>
      ) : (
        <>
          {sections.length > 1 && (
            <div className="flex flex-wrap gap-1">
              {sections.map((s) => (
                <button
                  key={s.key}
                  type="button"
                  onClick={() => setSectionKey(s.key)}
                  className={cn(
                    "rounded-md px-2.5 py-1 text-xs capitalize whitespace-nowrap transition-colors",
                    s.key === section?.key
                      ? "bg-white/[0.08] text-white"
                      : "text-white/55 hover:bg-white/[0.04] hover:text-white/80"
                  )}
                >
                  {s.label}
                </button>
              ))}
            </div>
          )}
          {section && <FlatSchemaForm nodes={section.nodes} {...common} />}
        </>
      )}
    </div>
  )
}
