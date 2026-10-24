/**
 * Which Axol a sim run simulates: a compact roster of every version, unlocked
 * — sim needs no hardware, and the pick applies to that run alone (it is sent
 * as the run's `kinematics.robot_model`), never to the robot's saved model.
 */

import { cn } from "@/lib/utils"
import { ROSTER } from "@/lib/axol-roster"
import type { AxolModel, GripperType } from "@/lib/supervisor"
import { Silhouette } from "@/components/settings/mount-picker"
import { Switch } from "@/components/ui/switch"

/** The sim run's robot: a version, and (on Jelly) the gripper. */
export interface SimRobot {
  model: AxolModel
  gripper: GripperType
}

const ORDER: AxolModel[] = ["classic", "standalone", "ox_cart", "owl", "mobile"]

export function SimRobotPicker({
  value,
  onChange,
  disabled = false,
}: {
  value: SimRobot
  onChange: (next: SimRobot) => void
  /** A run is live: show its robot, change nothing. */
  disabled?: boolean
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-sm font-medium text-white">Sim robot</span>
        <span className="text-xs text-white/40">
          Applies to this run only. The robot&apos;s saved model stays the same.
        </span>
      </div>
      <div
        role="radiogroup"
        aria-label="Sim robot"
        className="grid grid-cols-3 gap-2 sm:grid-cols-5"
      >
        {ORDER.map((model) => {
          const selected = model === value.model
          return (
            <button
              key={model}
              type="button"
              role="radio"
              aria-checked={selected}
              disabled={disabled}
              onClick={() => onChange({ ...value, model })}
              className={cn(
                "flex flex-col items-center gap-1 rounded-lg border px-2 py-2 text-center transition-all duration-150",
                "focus-visible:ring-2 focus-visible:ring-[#eff483]/60 focus-visible:outline-none",
                "disabled:cursor-default",
                selected
                  ? "border-[#eff483]/70 bg-[#eff483]/[0.08] text-[#eff483] shadow-[0_0_14px_-6px_rgba(239,244,131,0.6)]"
                  : "border-white/10 bg-white/[0.02] text-white/40 enabled:hover:border-white/30 enabled:hover:text-white/70",
                disabled && !selected && "opacity-50"
              )}
            >
              <Silhouette model={model} />
              <span
                className={cn("text-xs leading-tight", selected ? "text-white" : "text-white/70")}
              >
                {ROSTER[model].name}
              </span>
            </button>
          )
        })}
      </div>
      {value.model === "mobile" && (
        <div className="flex items-center justify-between gap-4">
          <span className="text-xs text-white/55">Parcel gripper</span>
          <Switch
            checked={value.gripper === "parcel"}
            disabled={disabled}
            onChange={(on) => onChange({ ...value, gripper: on ? "parcel" : "parallel" })}
            aria-label="Parcel gripper"
          />
        </div>
      )}
    </div>
  )
}
