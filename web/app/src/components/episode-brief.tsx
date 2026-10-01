import { useState } from "react"
import { ChevronDown, ChevronUp } from "lucide-react"

import { Button } from "@/components/ui/button"
import type {
  EpisodeBrief,
  EpisodeBriefGrid,
  EpisodeBriefItem,
  EpisodeBriefPlacement,
} from "@/lib/supervisor"

/** How each placement role is drawn in the layout grid (and its legend). */
const ROLE_STYLES: Record<string, { label: string; cls: string }> = {
  object: {
    label: "Object",
    cls: "border-[#eff483]/70 bg-[#eff483]/15 text-[#f6f9b8]",
  },
  target: {
    label: "Destination",
    cls: "border-emerald-400/70 bg-emerald-400/15 text-emerald-100 ring-1 ring-emerald-400/40",
  },
  zone: {
    label: "Keep clear / place here",
    cls: "border-dashed border-emerald-400/70 bg-emerald-400/5 text-emerald-100",
  },
  reference: {
    label: "Reference",
    cls: "border-sky-400/60 bg-sky-400/10 text-sky-100",
  },
  container: {
    label: "Container",
    cls: "border-white/30 bg-white/[0.07] text-white/85",
  },
  distractor: {
    label: "Distractor",
    cls: "border-white/15 bg-transparent text-white/50",
  },
}
const NEUTRAL_ROLE = { label: "", cls: "border-white/20 bg-white/[0.05] text-white/75" }

function roleStyle(role: string | undefined) {
  return ROLE_STYLES[role ?? ""] ?? NEUTRAL_ROLE
}

/**
 * The op's instruction card for the current episode (the snapshot's `brief`):
 * e.g. a scripted scene — the task, what goes where, and a top-down diagram
 * of the layout. The setup folds away with a click; while recording it starts
 * folded (the operator needs the instruction then, not the setup list).
 */
export function EpisodeBriefCard({ brief, phase }: { brief: EpisodeBrief; phase: string }) {
  // The operator's fold choice holds for the phase it was made in; a new
  // phase goes back to that phase's default.
  const [fold, setFold] = useState<{ phase: string; open: boolean } | null>(null)
  const open = fold?.phase === phase ? fold.open : phase !== "recording"
  const hasDetail = Boolean(brief.items?.length || brief.grid?.items.length || brief.note)
  const done = brief.tone === "done"

  return (
    <div
      data-testid="episode-brief"
      className={`flex flex-col gap-3 rounded-md border p-3 ${
        done ? "border-emerald-400/30 bg-emerald-400/[0.04]" : "border-white/10 bg-black/20"
      }`}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="flex min-w-0 flex-col gap-0.5">
          {brief.eyebrow && (
            <span className="font-mono text-[0.65rem] tracking-widest text-white/45 uppercase">
              {brief.eyebrow}
            </span>
          )}
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-heading text-base font-semibold text-white/90">
              {brief.title}
            </span>
            {brief.tag && (
              <span
                className={`rounded border px-1.5 py-0.5 font-mono text-[0.6rem] tracking-widest uppercase ${
                  done ? "border-emerald-400/50 text-emerald-200" : "border-white/25 text-white/60"
                }`}
              >
                {brief.tag}
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3">
          {brief.progress && <BriefProgress progress={brief.progress} />}
          {hasDetail && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setFold({ phase, open: !open })}
              aria-expanded={open}
            >
              {open ? <ChevronUp /> : <ChevronDown />}
              {open ? "Hide setup" : "Show setup"}
            </Button>
          )}
        </div>
      </div>
      {brief.headline && (
        <p className="text-lg leading-snug font-medium text-white">{brief.headline}</p>
      )}
      {open && hasDetail && (
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
          {brief.items && brief.items.length > 0 && (
            <dl className="grid min-w-0 flex-1 grid-cols-[max-content_1fr] gap-x-4 gap-y-2 text-sm">
              {brief.items.map((item) => (
                <BriefItem key={item.label} item={item} />
              ))}
            </dl>
          )}
          {brief.grid && brief.grid.items.length > 0 && <BriefGrid grid={brief.grid} />}
        </div>
      )}
      {open && brief.note && <p className="text-xs leading-relaxed text-white/50">{brief.note}</p>}
    </div>
  )
}

function BriefProgress({ progress }: { progress: NonNullable<EpisodeBrief["progress"]> }) {
  const pct = progress.total > 0 ? Math.min(100, (100 * progress.done) / progress.total) : 0
  return (
    <div className="flex flex-col items-end gap-1">
      <span className="font-mono text-[0.65rem] text-white/50">
        {progress.label ?? `${progress.done} / ${progress.total}`}
      </span>
      <div
        className="h-1 w-32 overflow-hidden rounded-full bg-white/10"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={progress.total}
        aria-valuenow={progress.done}
      >
        <div className="h-full rounded-full bg-emerald-400/80" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function BriefItem({ item }: { item: EpisodeBriefItem }) {
  const values = Array.isArray(item.value) ? item.value : [item.value]
  return (
    <>
      <dt className="pt-0.5 font-mono text-[0.65rem] tracking-widest text-white/45 uppercase">
        {item.label}
      </dt>
      <dd className="min-w-0">
        {item.emphasis ? (
          <span className="inline-block rounded-md border border-[#eff483]/50 bg-[#eff483]/10 px-2 py-0.5 font-semibold text-[#f6f9b8]">
            {values.join(", ")}
          </span>
        ) : values.length > 1 ? (
          <ul className="flex flex-col gap-0.5 text-white/80">
            {values.map((v) => (
              <li key={v}>{v}</li>
            ))}
          </ul>
        ) : (
          <span className="text-white/80">{values[0]}</span>
        )}
      </dd>
    </>
  )
}

/**
 * Top-down layout diagram: the grid's rows run away from the viewer (first
 * row farthest), the footer marks the near edge (e.g. the robot).
 */
function BriefGrid({ grid }: { grid: EpisodeBriefGrid }) {
  const roles = [...new Set(grid.items.map((p) => p.role ?? ""))].filter((r) => ROLE_STYLES[r])
  return (
    <div className="flex w-full shrink-0 flex-col gap-2 lg:w-[26rem]">
      <div
        className="grid gap-1 text-[0.7rem]"
        style={{ gridTemplateColumns: `2.5rem repeat(${grid.cols.length}, minmax(0, 1fr))` }}
      >
        <span />
        {grid.cols.map((c) => (
          <span
            key={c.key}
            className="text-center font-mono text-[0.6rem] tracking-widest text-white/40 uppercase"
          >
            {c.label}
          </span>
        ))}
        {grid.rows.map((r) => (
          <GridRow key={r.key} row={r} grid={grid} />
        ))}
        {grid.footer && (
          <>
            <span />
            <span
              className="mt-1 rounded border border-white/15 bg-white/[0.06] py-0.5 text-center font-mono text-[0.6rem] tracking-widest text-white/55 uppercase"
              style={{ gridColumn: `span ${grid.cols.length}` }}
            >
              {grid.footer}
            </span>
          </>
        )}
      </div>
      {roles.length > 0 && (
        <div className="flex flex-wrap gap-x-3 gap-y-1 pl-10 text-[0.65rem] text-white/50">
          {roles.map((r) => (
            <span key={r} className="flex items-center gap-1">
              <span className={`inline-block size-2.5 rounded-sm border ${ROLE_STYLES[r].cls}`} />
              {ROLE_STYLES[r].label}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function GridRow({ row, grid }: { row: EpisodeBriefGrid["rows"][number]; grid: EpisodeBriefGrid }) {
  return (
    <>
      <span className="self-center font-mono text-[0.6rem] tracking-widest text-white/40 uppercase">
        {row.label}
      </span>
      {grid.cols.map((c) => {
        const here = grid.items.filter((p) => p.row === row.key && p.col === c.key)
        return (
          <div
            key={c.key}
            data-cell={`${c.key}-${row.key}`}
            className="flex min-h-14 flex-col gap-1 rounded border border-white/[0.07] bg-white/[0.02] p-1"
          >
            {here.map((p, i) => (
              <Placement key={`${p.label}-${i}`} placement={p} />
            ))}
          </div>
        )
      })}
    </>
  )
}

function Placement({ placement }: { placement: EpisodeBriefPlacement }) {
  const s = roleStyle(placement.role)
  const title = [placement.label, placement.detail].filter(Boolean).join(" — ")
  return (
    <div
      title={title}
      data-role={placement.role}
      className={`flex items-start gap-1 rounded border px-1 py-0.5 leading-tight ${s.cls}`}
    >
      {placement.rotation != null && (
        // A box outline turned to the placement's orientation (0° = wide).
        // Rotation is counter-clockwise from above; CSS turns clockwise.
        <span
          aria-hidden
          className="mt-0.5 inline-block h-2 w-4 shrink-0 rounded-[1px] border border-current"
          style={{ transform: `rotate(${-placement.rotation}deg)` }}
        />
      )}
      <span className="min-w-0">
        <span className="font-medium">{placement.label}</span>
        {placement.detail && (
          <span className="block text-[0.6rem] opacity-70">{placement.detail}</span>
        )}
      </span>
    </div>
  )
}
