/**
 * The robot-model pick a host needs before its first Axol run, shown on the
 * control panel's Overview until one is saved. After that the picker lives in
 * Settings → Robot (see mount-picker.tsx).
 */

import { useState } from "react"
import { Check, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { ROSTER } from "@/lib/axol-roster"
import { AXOL_MOUNTS, type AxolMount } from "@/lib/supervisor"
import { cn } from "@/lib/utils"
import { MountPicker, Silhouette } from "./mount-picker"

const NAMES: Record<AxolMount, string> = {
  classic: "Axol Legacy",
  standalone: "Axol Standalone",
  ox_cart: "Axol on Ox Cart",
  owl: "Axol on Owl Mount",
}

export function MountRequired({
  locked,
  onSave,
}: {
  /** No Axol connected yet: the mounts can be previewed, not picked. */
  locked: boolean
  onSave: (mount: AxolMount) => Promise<void>
}) {
  // A card click only stages the pick; Save commits it.
  const [picked, setPicked] = useState<AxolMount | null>(null)
  const [saving, setSaving] = useState(false)

  async function save() {
    if (!picked || saving) return
    setSaving(true)
    try {
      await onSave(picked)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Card className="gap-3 border-[#eff483]/30 p-5">
      <div className="flex flex-col gap-1">
        <span className="text-base font-semibold text-white">What is Axol mounted on?</span>
        <p className="text-xs text-white/50">
          Axol needs to know its mount so the arms don&apos;t hit it. You can change this later in
          Axol settings → Robot. When Jelly is attached, Axol on Jelly is used automatically.
        </p>
      </div>
      <MountPicker
        mount={picked}
        robotModel="classic"
        showActive={false}
        locked={locked}
        onSelect={setPicked}
      />
      <div className="flex items-center justify-end gap-3">
        <span className="text-xs text-white/45">
          {locked ? "" : picked ? `${NAMES[picked]} selected` : "Select a mount, then press Save."}
        </span>
        <Button size="sm" onClick={() => void save()} disabled={locked || !picked || saving}>
          {saving && <Loader2 className="animate-spin" />}
          Save
        </Button>
      </div>
    </Card>
  )
}

/**
 * The mount choice inline on Overview: one button per mount, then Save. The
 * full picker, with the 3D view, is on Devices and in Axol settings → Robot.
 */
export function MountQuickPick({ onSave }: { onSave: (mount: AxolMount) => Promise<void> }) {
  const [picked, setPicked] = useState<AxolMount | null>(null)
  const [saving, setSaving] = useState(false)

  async function save() {
    if (!picked || saving) return
    setSaving(true)
    try {
      await onSave(picked)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div role="radiogroup" aria-label="Mount" className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        {AXOL_MOUNTS.map((m) => {
          const selected = m === picked
          return (
            <button
              key={m}
              type="button"
              role="radio"
              aria-checked={selected}
              onClick={() => setPicked(m)}
              className={cn(
                "group flex items-center gap-3 rounded-lg border px-3 py-2 text-left transition-colors",
                "focus-visible:ring-2 focus-visible:ring-[#eff483]/60 focus-visible:outline-none",
                selected
                  ? "border-[#eff483]/70 bg-[#eff483]/[0.08]"
                  : "border-white/10 bg-white/[0.02] hover:border-white/30 hover:bg-white/[0.05]"
              )}
            >
              <span
                className={cn(
                  "flex h-10 w-8 shrink-0 items-end justify-center rounded-md bg-black/40",
                  selected ? "text-[#eff483]" : "text-white/40 group-hover:text-white/70"
                )}
              >
                <Silhouette model={m} />
              </span>
              <span className="min-w-0 flex-1 text-sm font-medium text-white">
                {ROSTER[m].name}
              </span>
              {selected && <Check className="size-4 shrink-0 text-[#eff483]" aria-hidden />}
            </button>
          )
        })}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className="text-xs text-white/45">
          When Jelly is attached, Axol on Jelly is used automatically.
        </span>
        <Button size="sm" onClick={() => void save()} disabled={!picked || saving}>
          {saving && <Loader2 className="animate-spin" />}
          Save mount
        </Button>
      </div>
    </div>
  )
}
