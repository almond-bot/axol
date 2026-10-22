/**
 * "Choose your Axol": the robot-model picker, as a character-select screen.
 *
 * A roster of every Axol version on the right, a live turntable of the
 * highlighted one on the stage, and its stats. Without Jelly the operator picks
 * what Axol is mounted on (`kinematics.mount`); Axol on Jelly is never picked —
 * it is the robot whenever Jelly is attached — so its card shows as detected,
 * or locked until Jelly is plugged in.
 */

import { lazy, Suspense, useState, type ReactNode } from "react"
import { Check, Loader2, Lock, Radar } from "lucide-react"
import { cn } from "@/lib/utils"
import { AXOL_MOUNTS, type AxolModel, type AxolMount } from "@/lib/supervisor"

// three.js only loads when the Robot tab is open (see pose-panel.tsx).
const PoseViewer = lazy(() => import("./pose-viewer"))

interface Character {
  model: AxolModel
  name: string
  /** The short line under the name on the roster card. */
  kind: string
  pitch: string
  /** Each stat: a 0..1 bar and what the bar means. */
  stats: { label: string; value: number; reading: string }[]
}

const LIFT_MAX_MM = 680

const ROSTER: Record<AxolModel, Character> = {
  classic: {
    model: "classic",
    name: "Axol Legacy",
    kind: "Original model",
    pitch: "The original Axol model. Keep it if this robot was set up before the new mounts.",
    stats: [
      { label: "Lift", value: 0, reading: "None" },
      { label: "Mobility", value: 0.12, reading: "Fixed" },
    ],
  },
  standalone: {
    model: "standalone",
    name: "Axol Standalone",
    kind: "Just Axol",
    pitch: "Shoulders, head camera and arms, bolted wherever the work is.",
    stats: [
      { label: "Lift", value: 0, reading: "None" },
      { label: "Mobility", value: 0.12, reading: "Fixed" },
    ],
  },
  ox_cart: {
    model: "ox_cart",
    name: "Axol × Ox",
    kind: "Lift cart",
    pitch: "Rides a rolling cart with a 457 mm lift: wheel it to the job, raise it to the work.",
    stats: [
      { label: "Lift", value: 457 / LIFT_MAX_MM, reading: "457 mm" },
      { label: "Mobility", value: 0.55, reading: "Rolls" },
    ],
  },
  owl: {
    model: "owl",
    name: "Axol × Owl",
    kind: "Column mount",
    pitch: "A compact column on a foot plate: a steady, fixed station.",
    stats: [
      { label: "Lift", value: 0, reading: "None" },
      { label: "Mobility", value: 0.12, reading: "Fixed" },
    ],
  },
  mobile: {
    model: "mobile",
    name: "Axol × Jelly",
    kind: "Mobile base",
    pitch:
      "The mobile Axol: a self-driving base and a 680 mm two-stage lift. It takes over the moment Jelly is attached.",
    stats: [
      { label: "Lift", value: 1, reading: "680 mm" },
      { label: "Mobility", value: 1, reading: "Drives" },
    ],
  },
}

// Elbows a little bent: a ready stance for the turntable.
const STANCE: Record<string, number> = { left_e1_0: 0.7, right_e1_0: -0.7 }

export function MountPicker({
  mount,
  robotModel,
  onSelect,
  showActive = true,
}: {
  /** The saved (or draft) `kinematics.mount`; null before one is picked. */
  mount: AxolMount | null
  /** What the host runs now: `mobile` while Jelly is enabled. */
  robotModel: AxolModel
  onSelect: (mount: AxolMount) => void
  /** Badge the version the host runs ("In play"); off before a first pick. */
  showActive?: boolean
}) {
  const jelly = robotModel === "mobile"
  // The stage shows what the operator points at, else what the robot is.
  const [hovered, setHovered] = useState<AxolModel | null>(null)
  const staged = ROSTER[hovered ?? robotModel]

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between gap-3">
        <span className="font-mono text-[0.65rem] tracking-widest text-white/40 uppercase">
          Choose your Axol
        </span>
        {jelly && (
          <span className="text-xs text-white/45">
            Jelly is attached, so this robot is Axol × Jelly. Your pick applies when it is off.
          </span>
        )}
      </div>
      <div className="grid gap-3 md:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <Stage character={staged} active={showActive && staged.model === robotModel} />
        <div role="radiogroup" aria-label="Robot model" className="grid gap-2">
          {AXOL_MOUNTS.map((m) => (
            <RosterCard
              key={m}
              character={ROSTER[m]}
              selected={m === mount}
              dimmed={jelly}
              onSelect={() => onSelect(m)}
              onHover={setHovered}
              silhouette={<Silhouette model={m} />}
            />
          ))}
          <JellyCard detected={jelly} onHover={setHovered} />
        </div>
      </div>
    </div>
  )
}

function Stage({ character, active }: { character: Character; active: boolean }) {
  return (
    <div className="relative flex min-h-80 flex-col overflow-hidden rounded-xl border border-white/10 bg-[radial-gradient(ellipse_at_50%_35%,rgba(239,244,131,0.12),transparent_65%),linear-gradient(to_bottom,rgba(0,0,0,0.15),rgba(0,0,0,0.5))]">
      <div className="h-64 md:h-72">
        <Suspense
          fallback={
            <div className="flex h-full items-center justify-center">
              <Loader2 className="size-5 animate-spin text-white/30" />
            </div>
          }
        >
          {/* Keyed: the viewer loads its URDF once per mount. */}
          <PoseViewer
            key={character.model}
            robotModel={character.model}
            jointValues={STANCE}
            autoRotate
            distance={1.25}
          />
        </Suspense>
      </div>
      <div className="flex flex-col gap-2 border-t border-white/10 bg-black/40 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-lg font-semibold tracking-tight text-white">{character.name}</span>
          {active && (
            <span className="rounded-sm bg-[#eff483] px-1.5 py-0.5 font-mono text-[0.6rem] font-bold tracking-widest text-black uppercase">
              In play
            </span>
          )}
        </div>
        <p className="text-xs leading-relaxed text-white/55">{character.pitch}</p>
        <div className="grid gap-1.5 pt-1">
          {character.stats.map((s) => (
            <div key={s.label} className="grid grid-cols-[4.5rem_1fr_4rem] items-center gap-2">
              <span className="font-mono text-[0.6rem] tracking-widest text-white/40 uppercase">
                {s.label}
              </span>
              <div className="h-1.5 overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-[#eff483] transition-[width] duration-500"
                  style={{ width: `${Math.max(s.value, 0.02) * 100}%` }}
                />
              </div>
              <span className="text-right font-mono text-[0.65rem] text-white/60">{s.reading}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function RosterCard({
  character,
  selected,
  dimmed,
  onSelect,
  onHover,
  silhouette,
}: {
  character: Character
  selected: boolean
  dimmed: boolean
  onSelect: () => void
  onHover: (model: AxolModel | null) => void
  silhouette: ReactNode
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      onClick={onSelect}
      onMouseEnter={() => onHover(character.model)}
      onMouseLeave={() => onHover(null)}
      onFocus={() => onHover(character.model)}
      onBlur={() => onHover(null)}
      className={cn(
        "group flex items-center gap-3 rounded-lg border px-3 py-2 text-left transition-all duration-150",
        "hover:-translate-y-px hover:border-white/30 hover:bg-white/[0.06]",
        "focus-visible:ring-2 focus-visible:ring-[#eff483]/60 focus-visible:outline-none",
        selected
          ? "border-[#eff483]/70 bg-[#eff483]/[0.08] shadow-[0_0_18px_-6px_rgba(239,244,131,0.6)]"
          : "border-white/10 bg-white/[0.02]",
        dimmed && !selected && "opacity-60"
      )}
    >
      <div
        className={cn(
          "flex h-12 w-10 shrink-0 items-end justify-center rounded-md bg-black/40",
          selected ? "text-[#eff483]" : "text-white/40 group-hover:text-white/70"
        )}
      >
        {silhouette}
      </div>
      <div className="flex min-w-0 flex-1 flex-col">
        <span className="truncate text-sm font-medium text-white">{character.name}</span>
        <span className="truncate font-mono text-[0.6rem] tracking-widest text-white/40 uppercase">
          {character.kind}
        </span>
      </div>
      {selected && <Check className="size-4 shrink-0 text-[#eff483]" aria-hidden />}
    </button>
  )
}

function JellyCard({
  detected,
  onHover,
}: {
  detected: boolean
  onHover: (model: AxolModel | null) => void
}) {
  const character = ROSTER.mobile
  return (
    // Not a radio: Jelly is detected, never picked. Focusable so keyboard
    // users can preview it too.
    <div
      tabIndex={0}
      onMouseEnter={() => onHover("mobile")}
      onMouseLeave={() => onHover(null)}
      onFocus={() => onHover("mobile")}
      onBlur={() => onHover(null)}
      className={cn(
        "flex items-center gap-3 rounded-lg border px-3 py-2 transition-all duration-150",
        "focus-visible:ring-2 focus-visible:ring-[#eff483]/60 focus-visible:outline-none",
        detected
          ? "border-[#eff483]/70 bg-[#eff483]/[0.08] shadow-[0_0_18px_-6px_rgba(239,244,131,0.6)]"
          : "border-dashed border-white/15 bg-white/[0.01] hover:border-white/30"
      )}
    >
      <div
        className={cn(
          "flex h-12 w-10 shrink-0 items-end justify-center rounded-md bg-black/40",
          detected ? "text-[#eff483]" : "text-white/25"
        )}
      >
        <Silhouette model="mobile" />
      </div>
      <div className="flex min-w-0 flex-1 flex-col">
        <span
          className={cn("truncate text-sm font-medium", detected ? "text-white" : "text-white/60")}
        >
          {character.name}
        </span>
        <span className="truncate font-mono text-[0.6rem] tracking-widest text-white/40 uppercase">
          {detected ? "Detected · automatic" : "Attach Jelly to unlock"}
        </span>
      </div>
      {detected ? (
        <Radar className="size-4 shrink-0 text-[#eff483]" aria-label="Detected" />
      ) : (
        <Lock className="size-4 shrink-0 text-white/30" aria-label="Locked" />
      )}
    </div>
  )
}

/** A tiny line drawing of each version: shoulder bar, head, arms, and what
 *  they stand on. 40 x 48, drawn in currentColor. */
function Silhouette({ model }: { model: AxolModel }) {
  const body: Record<AxolModel, ReactNode> = {
    classic: <rect x="16" y="13" width="8" height="26" rx="1.5" />,
    standalone: null,
    owl: (
      <>
        <rect x="17" y="13" width="6" height="18" rx="1" />
        <rect x="11" y="31" width="18" height="2.5" rx="1" />
      </>
    ),
    ox_cart: (
      <>
        <rect x="17" y="13" width="6" height="25" rx="1" />
        <path d="M8 41 L20 37 L32 41" fill="none" stroke="currentColor" strokeWidth="2" />
        <circle cx="8" cy="44" r="2" />
        <circle cx="32" cy="44" r="2" />
      </>
    ),
    mobile: (
      <>
        <rect x="17" y="13" width="6" height="22" rx="1" />
        <rect x="7" y="35" width="26" height="7" rx="2" />
        <circle cx="11" cy="44" r="2.5" />
        <circle cx="29" cy="44" r="2.5" />
      </>
    ),
  }
  return (
    <svg viewBox="0 0 40 48" className="h-11 w-9" fill="currentColor" aria-hidden>
      <rect x="10" y="3" width="20" height="3.5" rx="1.5" />
      <rect x="10" y="7" width="20" height="5" rx="2.5" />
      <rect x="5" y="9" width="4" height="22" rx="2" />
      <rect x="31" y="9" width="4" height="22" rx="2" />
      {body[model]}
    </svg>
  )
}
