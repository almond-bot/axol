import { useMemo, useState } from "react"
import { Search, Plug } from "lucide-react"
import { CATEGORY_ORDER, type CommandSpec } from "@/lib/supervisor"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"

/**
 * The full `axol` command surface, grouped by category and searchable. Acts as
 * the primary navigation for the control panel — selecting a command swaps the
 * configuration pane beside it.
 */
export function CommandCatalog({
  commands,
  selectedId,
  disabled,
  connected,
  onSelect,
  onOpenSetup,
}: {
  commands: CommandSpec[]
  selectedId: string
  disabled: boolean
  connected: boolean
  onSelect: (id: string) => void
  onOpenSetup: () => void
}) {
  const [query, setQuery] = useState("")
  const q = query.trim().toLowerCase()

  const groups = useMemo(() => {
    const filtered = q
      ? commands.filter(
          (c) =>
            c.label.toLowerCase().includes(q) ||
            c.cli.toLowerCase().includes(q) ||
            c.category.toLowerCase().includes(q)
        )
      : commands
    const known = CATEGORY_ORDER.map((cat) => ({
      cat,
      items: filtered.filter((c) => c.category === cat),
    }))
    const extraCats = [...new Set(filtered.map((c) => c.category))].filter(
      (c) => !CATEGORY_ORDER.includes(c)
    )
    const extra = extraCats.map((cat) => ({
      cat,
      items: filtered.filter((c) => c.category === cat),
    }))
    return [...known, ...extra].filter((g) => g.items.length > 0)
  }, [commands, q])

  return (
    <Card className="gap-0 self-start p-0 lg:sticky lg:top-20">
      <div className="border-b border-white/10 p-3">
        <div className="relative">
          <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-white/30" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search commands…"
            className="pl-9"
          />
        </div>
      </div>

      <div className="flex max-h-[calc(100vh-12rem)] flex-col gap-4 overflow-auto p-3">
        {!connected ? (
          <div className="flex flex-col items-center gap-3 px-2 py-8 text-center">
            <p className="text-sm text-white/45">Connect to a machine running axol serve.</p>
            <Button variant="outline" size="sm" onClick={onOpenSetup}>
              <Plug />
              Connect
            </Button>
          </div>
        ) : groups.length === 0 ? (
          <p className="px-1 py-4 text-sm text-white/35">No matching commands.</p>
        ) : (
          groups.map((g) => (
            <div key={g.cat} className="flex flex-col gap-1">
              <div className="px-1 pb-1 font-mono text-[0.65rem] tracking-widest text-white/35 uppercase">
                {g.cat}
              </div>
              {g.items.map((cmd) => (
                <CatalogRow
                  key={cmd.id}
                  cmd={cmd}
                  selected={cmd.id === selectedId}
                  disabled={disabled && cmd.id !== selectedId}
                  onSelect={() => onSelect(cmd.id)}
                />
              ))}
            </div>
          ))
        )}
      </div>
    </Card>
  )
}

function CatalogRow({
  cmd,
  selected,
  disabled,
  onSelect,
}: {
  cmd: CommandSpec
  selected: boolean
  disabled: boolean
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={disabled}
      className={cn(
        "flex w-full items-center gap-2 rounded-lg border px-2.5 py-2 text-left transition-all",
        selected
          ? "border-[#eff483]/40 bg-[#eff483]/10"
          : "border-transparent hover:border-white/15 hover:bg-white/[0.04]",
        disabled && "cursor-not-allowed opacity-40"
      )}
    >
      <div className="min-w-0 flex-1">
        <div className={cn("truncate text-sm font-medium", !selected && "text-white/85")}>
          {cmd.label}
        </div>
        <div className="truncate font-mono text-[0.7rem] text-white/40">axol {cmd.cli}</div>
      </div>
      {!cmd.available && <Badge variant="warning">N/A</Badge>}
    </button>
  )
}
