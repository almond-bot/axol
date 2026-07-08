import { useRef, useState } from "react"
import { createPortal } from "react-dom"
import { Info, RotateCcw } from "lucide-react"
import {
  defaultString,
  isModified,
  type FormValue,
  type SchemaField,
  type SchemaNode,
} from "@/lib/supervisor"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"

interface CommonProps {
  overrides: Record<string, FormValue>
  disabled: boolean
  onChange: (key: string, value: FormValue) => void
  onReset: (key: string) => void
}

/**
 * Renders a curated, flat list of fields (a hand-picked subset of an op's
 * full schema) — used by the operation panels and the diagnostics actions.
 */
export function CuratedForm({ fields, ...common }: CommonProps & { fields: SchemaField[] }) {
  if (fields.length === 0) {
    return <p className="text-sm text-white/40">No settings — just press Start.</p>
  }
  return (
    <div className="flex flex-col gap-4">
      {fields.map((f) => (
        <FieldRow key={f.key} field={f} {...common} />
      ))}
    </div>
  )
}

/**
 * A config subtree rendered flat: every field visible (no collapsed groups to
 * dig through), nested groups become plain section headings ("left › elbow")
 * with their fields in a compact grid. Used by the settings Advanced tab.
 */
export function FlatSchemaForm({
  nodes,
  path = [],
  ...common
}: CommonProps & { nodes: SchemaNode[]; path?: string[] }) {
  const fields = nodes.filter((n): n is SchemaField => n.kind === "field")
  const groups = nodes.filter((n) => n.kind === "group")
  return (
    <div className="flex flex-col gap-4">
      {fields.length > 0 && (
        <div className="flex flex-col gap-2">
          {path.length > 0 && (
            <span className="border-b border-white/10 pb-1 font-mono text-[0.65rem] tracking-widest text-white/40 uppercase">
              {path.join(" › ")}
            </span>
          )}
          <div className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
            {fields.map((f) => (
              <FieldRow key={f.key} field={f} {...common} />
            ))}
          </div>
        </div>
      )}
      {groups.map((g) => (
        <FlatSchemaForm key={g.key} nodes={g.children} path={[...path, g.label]} {...common} />
      ))}
    </div>
  )
}

export function FieldRow({
  field,
  showPath,
  overrides,
  disabled,
  onChange,
  onReset,
}: CommonProps & { field: SchemaField; showPath?: boolean }) {
  const has = field.key in overrides
  const value = has ? overrides[field.key] : undefined
  const modified = isModified(field, value)
  // Namespace the DOM id so keys like "root" don't collide with app-level
  // element ids (e.g. the React mount node <div id="root">, which a global
  // `#root { min-height: 100vh }` rule would otherwise stretch the input to).
  const fieldId = `cfg-${field.key}`

  const labelNode = (
    <div className="flex min-w-0 items-center gap-2">
      <Label htmlFor={fieldId} className="truncate">
        {showPath ? (
          <span className="font-mono text-xs text-white/55">{field.key}</span>
        ) : (
          <span className="capitalize">{field.label}</span>
        )}
      </Label>
      {field.required && <span className="text-xs text-[#eff483]">*</span>}
      {field.help && <HelpTip text={field.help} />}
      {modified && <span className="size-1.5 rounded-full bg-[#eff483]" />}
      {modified && (
        <button
          type="button"
          onClick={() => onReset(field.key)}
          disabled={disabled}
          title="Reset to default"
          className="text-white/30 hover:text-white/70"
        >
          <RotateCcw className="size-3" />
        </button>
      )}
    </div>
  )

  if (field.type === "boolean") {
    const checked = has ? Boolean(value) : Boolean(field.default)
    return (
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between gap-4">
          {labelNode}
          <Switch checked={checked} disabled={disabled} onChange={(v) => onChange(field.key, v)} />
        </div>
      </div>
    )
  }

  const text = has ? String(value ?? "") : defaultString(field)

  return (
    <div className="flex flex-col gap-1.5">
      {labelNode}
      {field.type === "select" ? (
        <select
          id={fieldId}
          value={text}
          disabled={disabled}
          onChange={(e) => onChange(field.key, e.target.value)}
          className="h-9 w-full rounded-md border border-input bg-white/[0.02] px-3 text-sm text-foreground outline-none focus-visible:border-ring/70 disabled:opacity-50"
        >
          {field.required && <option value="">Select…</option>}
          {(field.options ?? []).map((opt) => (
            <option key={opt} value={opt} className="bg-[#1a1a1a]">
              {opt}
            </option>
          ))}
        </select>
      ) : (
        <Input
          id={fieldId}
          inputMode={field.type === "number" ? "decimal" : undefined}
          value={text}
          placeholder={field.required ? "required" : defaultString(field)}
          disabled={disabled}
          onChange={(e) => onChange(field.key, e.target.value)}
        />
      )}
    </div>
  )
}

/**
 * An info dot next to a field label. Hovering reveals the field's docs (pulled
 * from the config dataclass / CLI help). Rendered through a portal so the popup
 * is never clipped by the surrounding card / scroll container.
 */
function HelpTip({ text }: { text: string }) {
  const ref = useRef<HTMLSpanElement>(null)
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null)

  function show() {
    const rect = ref.current?.getBoundingClientRect()
    if (rect) setPos({ x: rect.left + rect.width / 2, y: rect.bottom })
  }

  return (
    <span
      ref={ref}
      onMouseEnter={show}
      onMouseLeave={() => setPos(null)}
      className="inline-flex shrink-0 cursor-help text-white/30 hover:text-white/70"
    >
      <Info className="size-3.5" />
      {pos &&
        createPortal(
          <span
            style={{ left: pos.x, top: pos.y + 6 }}
            className="pointer-events-none fixed z-[60] w-72 max-w-[80vw] -translate-x-1/2 rounded-md border border-white/10 bg-[#1c1c1c] px-3 py-2 text-xs leading-snug text-white/75 shadow-xl"
          >
            {text}
          </span>,
          document.body
        )}
    </span>
  )
}
