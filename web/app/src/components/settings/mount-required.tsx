/**
 * The robot-model pick a host needs before its first Axol run, shown on the
 * control panel's Overview until one is saved. After that the picker lives in
 * Settings → Robot (see mount-picker.tsx).
 */

import { useState } from "react"
import { Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { type AxolMount } from "@/lib/supervisor"
import { MountPicker } from "./mount-picker"

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
  return (
    <Card className="gap-3 border-[#eff483]/30 p-5">
      <div className="flex flex-col gap-1">
        <span className="text-base font-semibold text-white">What is Axol mounted on?</span>
        <p className="text-xs text-white/50">
          Axol needs to know its mount so the arms don&apos;t hit it. You can change this later in
          Axol settings → Robot. When Jelly is attached, Axol on Jelly is used automatically.
        </p>
      </div>
      <MountChooser locked={locked} onSave={onSave} />
    </Card>
  )
}

/**
 * The full picker with its Save row, without a card: inline on Overview, and
 * inside MountRequired on Devices.
 */
export function MountChooser({
  locked = false,
  onSave,
}: {
  locked?: boolean
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
    <div className="flex flex-col gap-3">
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
    </div>
  )
}
