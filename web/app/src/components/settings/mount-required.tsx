/**
 * The robot-model pick a host needs before its first Axol run, shown above the
 * settings on the main page until one is saved. After that the picker lives in
 * Settings → Robot (see mount-picker.tsx).
 */

import { useState } from "react"
import { Loader2 } from "lucide-react"
import { Card } from "@/components/ui/card"
import type { AxolMount } from "@/lib/supervisor"
import { MountPicker } from "./mount-picker"

export function MountRequired({ onSave }: { onSave: (mount: AxolMount) => Promise<void> }) {
  const [saving, setSaving] = useState<AxolMount | null>(null)

  async function pick(mount: AxolMount) {
    if (saving) return
    setSaving(mount)
    try {
      await onSave(mount)
    } finally {
      setSaving(null)
    }
  }

  return (
    <Card className="gap-3 border-[#eff483]/30 p-5">
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <span className="text-base font-semibold text-white">Which Axol is this?</span>
          {saving && <Loader2 className="size-4 animate-spin text-white/40" />}
        </div>
        <p className="text-xs text-white/50">
          Pick what this Axol is mounted on before the first run: it sets the body IK keeps the arms
          clear of. Change it any time in Settings → Robot. With Jelly attached the robot is Axol ×
          Jelly automatically.
        </p>
      </div>
      <MountPicker
        mount={saving}
        robotModel="classic"
        showActive={false}
        onSelect={(m) => void pick(m)}
      />
    </Card>
  )
}
