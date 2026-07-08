import { useRef, useState } from "react"
import { createPortal } from "react-dom"
import { Info, RotateCcw } from "lucide-react"
import {
  defaultString,
  isModified,
  type FormValue,
  type SchemaField,
  type SchemaGroup,
  type SchemaNode,
} from "@/lib/supervisor"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { cn, sentenceCase } from "@/lib/utils"

interface CommonProps {
  overrides: Record<string, FormValue>
  disabled: boolean
  onChange: (key: string, value: FormValue) => void
  onReset: (key: string) => void
}

// -- vector fields (numeric arrays rendered one input per component) ---------

const XYZ = ["x", "y", "z"]

const componentLabel = (length: number, i: number) => (length === 3 ? XYZ[i] : String(i + 1))

const vectorDefault = (field: SchemaField): number[] =>
  Array.isArray(field.default) ? field.default : []

/** Parse one component's text: number when it is one, raw text while mid-edit. */
const parseComponent = (text: string): number | string => {
  const t = text.trim()
  return t !== "" && Number.isFinite(Number(t)) ? Number(t) : text
}

/** Current component values: the override if set, else the defaults. */
function vectorValue(field: SchemaField, value: FormValue | undefined): (number | string)[] {
  return Array.isArray(value) ? value : vectorDefault(field)
}

/** Write one component, dropping the override when it lands back on defaults. */
function setVectorComponent(
  field: SchemaField,
  value: FormValue | undefined,
  index: number,
  text: string,
  onChange: (key: string, value: FormValue) => void,
  onReset: (key: string) => void
) {
  const def = vectorDefault(field)
  const next = [...vectorValue(field, value)]
  next[index] = parseComponent(text)
  const isDefault = next.length === def.length && next.every((v, i) => String(v) === String(def[i]))
  if (isDefault) onReset(field.key)
  else onChange(field.key, next)
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

// -- uniform-subgroup tables --------------------------------------------------
//
// The per-joint gain configs are 7+ sibling groups with identical numeric
// fields (kp, kd, friction.*, mass, com, …). Rendering those as repeated field
// lists is unreadable; a table — one row per joint, one column per parameter,
// vectors split into per-component columns — is how you actually scan them.

interface TableColumn {
  /** Dotted subpath inside each subgroup (e.g. "friction.fc" or "com"). */
  sub: string
  label: string
  /** Component index for vector fields; null for scalars. */
  index: number | null
}

/**
 * The column set of a subgroup if it is purely numeric (numbers + numeric
 * vectors, possibly nested); null disqualifies it from table rendering.
 */
function numericColumns(group: SchemaGroup, prefix = ""): TableColumn[] | null {
  const out: TableColumn[] = []
  for (const node of group.children) {
    if (node.kind === "group") {
      const nested = numericColumns(node, prefix ? `${prefix}.${node.label}` : node.label)
      if (nested == null) return null
      out.push(...nested)
      continue
    }
    const sub = prefix ? `${prefix}.${node.label}` : node.label
    const leaf = sub.split(".").pop() ?? sub
    if (node.type === "number") {
      out.push({ sub, label: leaf, index: null })
    } else if (node.type === "vector") {
      const def = vectorDefault(node)
      def.forEach((_, i) =>
        out.push({ sub, label: `${leaf} ${componentLabel(def.length, i)}`, index: i })
      )
    } else {
      return null
    }
  }
  return out
}

const columnsSignature = (cols: TableColumn[]) =>
  cols.map((c) => `${c.sub}:${c.index ?? ""}`).join("|")

/** Look up the field at `sub` inside a subgroup (mirrors numericColumns). */
function fieldAt(group: SchemaGroup, sub: string): SchemaField | null {
  const [head, ...rest] = sub.split(".")
  for (const node of group.children) {
    if (node.label !== head) continue
    if (node.kind === "field") return rest.length === 0 ? node : null
    return fieldAt(node, rest.join("."))
  }
  return null
}

function GroupTable({
  groups,
  columns,
  path,
  ...common
}: CommonProps & { groups: SchemaGroup[]; columns: TableColumn[]; path: string[] }) {
  return (
    <div className="flex flex-col gap-2">
      {path.length > 0 && (
        <span className="border-b border-white/10 pb-1 font-mono text-[0.65rem] tracking-widest text-white/40 uppercase">
          {path.join(" › ")}
        </span>
      )}
      <div className="overflow-x-auto">
        <table className="w-full border-separate border-spacing-x-1 border-spacing-y-1">
          <thead>
            <tr>
              <th />
              {columns.map((c) => (
                <th
                  key={`${c.sub}:${c.index ?? ""}`}
                  title={c.sub}
                  className="px-1 pb-0.5 text-center text-[0.65rem] font-medium whitespace-nowrap text-white/45"
                >
                  {sentenceCase(c.label.replace(/_/g, " "))}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {groups.map((g) => (
              <tr key={g.key}>
                <td className="pr-2 text-xs whitespace-nowrap text-white/60">
                  {sentenceCase(g.label)}
                </td>
                {columns.map((c) => {
                  const field = fieldAt(g, c.sub)
                  return field ? (
                    <TableCell
                      key={`${c.sub}:${c.index ?? ""}`}
                      field={field}
                      index={c.index}
                      {...common}
                    />
                  ) : (
                    <td key={`${c.sub}:${c.index ?? ""}`} />
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function TableCell({
  field,
  index,
  overrides,
  disabled,
  onChange,
  onReset,
}: CommonProps & { field: SchemaField; index: number | null }) {
  const has = field.key in overrides
  const value = has ? overrides[field.key] : undefined
  let text: string
  let modified: boolean
  if (index == null) {
    text = has ? String(value ?? "") : defaultString(field)
    modified = isModified(field, value)
  } else {
    const arr = vectorValue(field, value)
    text = String(arr[index] ?? "")
    modified = has && String(arr[index]) !== String(vectorDefault(field)[index])
  }
  return (
    <td>
      <input
        value={text}
        inputMode="decimal"
        disabled={disabled}
        title={`${field.key}${index != null ? ` [${index}]` : ""}`}
        onChange={(e) => {
          const raw = e.target.value
          if (index == null) {
            if (raw === "") onReset(field.key)
            else onChange(field.key, raw)
          } else {
            setVectorComponent(field, value, index, raw, onChange, onReset)
          }
        }}
        className={cn(
          "h-8 w-full min-w-14 rounded border bg-white/[0.02] px-1.5 text-center text-xs text-foreground outline-none focus-visible:border-ring/70 disabled:opacity-50",
          modified ? "border-[#eff483]/50" : "border-white/10"
        )}
      />
    </td>
  )
}

/**
 * A config subtree rendered flat: every field visible (no collapsed groups to
 * dig through). Nested groups become plain section headings ("left › elbow")
 * with their fields in a compact grid — except runs of sibling groups with
 * identical numeric fields (the per-joint gains), which collapse into one
 * table: a row per joint, a column per parameter. Used by the settings
 * Advanced tab.
 */
export function FlatSchemaForm({
  nodes,
  path = [],
  ...common
}: CommonProps & { nodes: SchemaNode[]; path?: string[] }) {
  const fields = nodes.filter((n): n is SchemaField => n.kind === "field")
  const groups = nodes.filter((n): n is SchemaGroup => n.kind === "group")

  // Cluster sibling groups by their numeric-column signature; a cluster of 3+
  // renders as one table (keyed by its first member's position).
  const signatures = new Map<string, { cols: TableColumn[] | null; sig: string | null }>()
  const clusterSizes = new Map<string, number>()
  for (const g of groups) {
    const cols = numericColumns(g)
    const sig = cols && cols.length >= 2 ? columnsSignature(cols) : null
    signatures.set(g.key, { cols, sig })
    if (sig) clusterSizes.set(sig, (clusterSizes.get(sig) ?? 0) + 1)
  }
  const renderedSigs = new Set<string>()

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
      {groups.map((g) => {
        const { cols, sig } = signatures.get(g.key)!
        if (sig && (clusterSizes.get(sig) ?? 0) >= 3) {
          if (renderedSigs.has(sig)) return null
          renderedSigs.add(sig)
          const members = groups.filter((m) => signatures.get(m.key)!.sig === sig)
          return <GroupTable key={g.key} groups={members} columns={cols!} path={path} {...common} />
        }
        return (
          <FlatSchemaForm key={g.key} nodes={g.children} path={[...path, g.label]} {...common} />
        )
      })}
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
          <span>{sentenceCase(field.label)}</span>
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

  if (field.type === "vector") {
    const def = vectorDefault(field)
    const arr = vectorValue(field, value)
    return (
      <div className="flex flex-col gap-1.5">
        {labelNode}
        <div className="flex flex-wrap gap-2">
          {def.map((_, i) => (
            <label key={i} className="flex items-center gap-1.5">
              <span className="font-mono text-[0.65rem] text-white/40 uppercase">
                {componentLabel(def.length, i)}
              </span>
              <Input
                value={String(arr[i] ?? "")}
                inputMode="decimal"
                disabled={disabled}
                onChange={(e) =>
                  setVectorComponent(field, value, i, e.target.value, onChange, onReset)
                }
                className="h-8 w-24 px-2 text-xs"
              />
            </label>
          ))}
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
