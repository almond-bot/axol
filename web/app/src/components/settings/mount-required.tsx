/**
 * The robot-model pick a host needs before its first Axol run, shown above the
 * settings on the main page until one is saved. After that the picker lives in
 * Settings → Robot (see mount-picker.tsx).
 */

import { useState } from "react"
import { Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import type { AxolMount } from "@/lib/supervisor"
import { MountPicker } from "./mount-picker"

const NAMES: Record<AxolMount, string> = {
  classic: "Axol Legacy",
  standalone: "Axol Standalone",
  ox_cart: "Axol × Ox",
  owl: "Axol × Owl",
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
        <span className="text-base font-semibold text-white">Which Axol is this?</span>
        <p className="text-xs text-white/50">
          Pick what this Axol is mounted on before the first run: it sets the body IK keeps the arms
          clear of. Change it any time in Settings → Robot. With Jelly attached the robot is Axol ×
          Jelly automatically.
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
          {locked
            ? "Connect the Axol to choose."
            : picked
              ? `${NAMES[picked]} selected`
              : "Select a version, then save."}
        </span>
        <Button size="sm" onClick={() => void save()} disabled={locked || !picked || saving}>
          {saving && <Loader2 className="animate-spin" />}
          Save
        </Button>
      </div>
    </Card>
  )
}
