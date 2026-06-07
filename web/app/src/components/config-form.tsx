import { useMemo, useState } from "react"
import { ChevronRight, RotateCcw, Search } from "lucide-react"
import {
  defaultString,
  flattenFields,
  isModified,
  type FormValue,
  type SchemaField,
  type SchemaNode,
} from "@/lib/supervisor"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { cn } from "@/lib/utils"

interface CommonProps {
  overrides: Record<string, FormValue>
  disabled: boolean
  onChange: (key: string, value: FormValue) => void
  onReset: (key: string) => void
}

export function ConfigForm({ schema, ...common }: CommonProps & { schema: SchemaNode[] }) {
  const [query, setQuery] = useState("")
  const allFields = useMemo(() => flattenFields(schema), [schema])

  const required = allFields.filter((f) => f.required)
  const rootFields = schema.filter((n): n is SchemaField => n.kind === "field" && !n.required)
  const groups = schema.filter((n) => n.kind === "group")

  const q = query.trim().toLowerCase()
  const matches = q
    ? allFields.filter((f) => f.key.toLowerCase().includes(q) || f.label.toLowerCase().includes(q))
    : null

  return (
    <div className="flex flex-col gap-4">
      <div className="relative">
        <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-white/30" />
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search all config…"
          className="pl-9"
        />
      </div>

      {matches ? (
        <div className="flex flex-col gap-4">
          {matches.length === 0 ? (
            <p className="text-sm text-white/35">No matching config.</p>
          ) : (
            matches.map((f) => <FieldRow key={f.key} field={f} showPath {...common} />)
          )}
        </div>
      ) : (
        <>
          {required.length > 0 && (
            <div className="flex flex-col gap-4 rounded-lg border border-[#eff483]/25 bg-[#eff483]/[0.04] p-3">
              <span className="font-mono text-xs tracking-widest text-[#eff483]/80 uppercase">
                Required
              </span>
              {required.map((f) => (
                <FieldRow key={f.key} field={f} {...common} />
              ))}
            </div>
          )}

          {rootFields.length > 0 && (
            <div className="flex flex-col gap-4">
              {rootFields.map((f) => (
                <FieldRow key={f.key} field={f} {...common} />
              ))}
            </div>
          )}

          {groups.map((g) => (
            <GroupSection key={g.key} group={g} depth={0} {...common} />
          ))}
        </>
      )}
    </div>
  )
}

function GroupSection({
  group,
  depth,
  ...common
}: CommonProps & { group: Extract<SchemaNode, { kind: "group" }>; depth: number }) {
  const [open, setOpen] = useState(false)

  const leaves = useMemo(() => flattenFields(group.children), [group])
  const modifiedCount = leaves.filter((f) => isModified(f, common.overrides[f.key])).length

  const fields = group.children.filter((n): n is SchemaField => n.kind === "field")
  const subgroups = group.children.filter((n) => n.kind === "group")

  return (
    <div
      className={cn(
        "rounded-lg border border-white/10",
        depth === 0 ? "bg-white/[0.02]" : "bg-transparent"
      )}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left"
      >
        <ChevronRight
          className={cn("size-4 shrink-0 text-white/40 transition-transform", open && "rotate-90")}
        />
        <span className="text-sm font-medium capitalize">{group.label}</span>
        <span className="text-xs text-white/30">{leaves.length}</span>
        {modifiedCount > 0 && (
          <span className="ml-auto rounded-full bg-[#eff483]/15 px-2 py-0.5 font-mono text-[0.65rem] text-[#eff483]">
            {modifiedCount} edited
          </span>
        )}
      </button>
      {open && (
        <div className="flex flex-col gap-4 border-t border-white/10 p-3">
          {fields.map((f) => (
            <FieldRow key={f.key} field={f} {...common} />
          ))}
          {subgroups.map((sg) => (
            <GroupSection
              key={sg.key}
              group={sg as Extract<SchemaNode, { kind: "group" }>}
              depth={depth + 1}
              {...common}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function FieldRow({
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

  const labelNode = (
    <div className="flex min-w-0 items-center gap-2">
      <Label htmlFor={field.key} className="truncate">
        {showPath ? (
          <span className="font-mono text-xs text-white/55">{field.key}</span>
        ) : (
          <span className="capitalize">{field.label}</span>
        )}
      </Label>
      {field.required && <span className="text-xs text-[#eff483]">*</span>}
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
        {field.help && <FieldHelp text={field.help} />}
      </div>
    )
  }

  const text = has ? String(value ?? "") : defaultString(field)

  return (
    <div className="flex flex-col gap-1.5">
      {labelNode}
      {field.type === "select" ? (
        <select
          id={field.key}
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
          id={field.key}
          inputMode={field.type === "number" ? "decimal" : undefined}
          value={text}
          placeholder={field.required ? "required" : defaultString(field)}
          disabled={disabled}
          onChange={(e) => onChange(field.key, e.target.value)}
        />
      )}
      {field.help && <FieldHelp text={field.help} />}
    </div>
  )
}

function FieldHelp({ text }: { text: string }) {
  return <p className="text-xs leading-snug text-white/35">{text}</p>
}

function Switch({
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
