import { useMemo } from "react"
import { JOINTS, JOINT_COLORS, jointLabel, type JointName } from "@/lib/telemetry"
import type { FormValue } from "@/lib/supervisor"
import { cn } from "@/lib/utils"

/** The seven gravity-compensatable arm joints (no gripper). */
const ARM_JOINTS = JOINTS.filter((j) => j !== "GRIPPER")

/** Parse a draccus-style list value ("[shoulder_1, wrist_3]") into a set. */
function parseSelection(value: FormValue | undefined): Set<JointName> {
  const text = typeof value === "string" ? value.trim() : ""
  if (!text) return new Set(ARM_JOINTS)
  const names = new Set(
    text
      .replace(/^\[|\]$/g, "")
      .split(",")
      .map((s) => s.trim().toUpperCase())
      .filter(Boolean)
  )
  const picked = ARM_JOINTS.filter((j) => names.has(j))
  return picked.length > 0 ? new Set(picked) : new Set(ARM_JOINTS)
}

/**
 * Checkbox picker for gravity-comp's `free_joints` — which arm joints are
 * gravity-compensated (freely movable by hand); unchecked joints hold their
 * position. Mirrors the diagnostics joint picker. All checked (the default)
 * clears the override so the command frees all seven; a subset is sent as the
 * draccus list `[shoulder_1, wrist_3]`. The last checked joint can't be
 * removed (an empty set is a config error).
 */
export function ArmJointPicker({
  value,
  disabled,
  onChange,
  onReset,
}: {
  value: FormValue | undefined
  disabled: boolean
  onChange: (value: string) => void
  onReset: () => void
}) {
  const selected = useMemo(() => parseSelection(value), [value])

  function toggle(joint: JointName) {
    const next = new Set(selected)
    if (next.has(joint)) {
      if (next.size === 1) return // an empty set is invalid — keep the last one
      next.delete(joint)
    } else {
      next.add(joint)
    }
    if (next.size === ARM_JOINTS.length)
      onReset() // all = command default
    else
      onChange(
        `[${ARM_JOINTS.filter((j) => next.has(j))
          .map((j) => j.toLowerCase())
          .join(", ")}]`
      )
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium text-white/80">Free joints</span>
        <span className="text-xs text-white/35">
          {selected.size === ARM_JOINTS.length ? "all" : `${selected.size} of ${ARM_JOINTS.length}`}
        </span>
        {selected.size < ARM_JOINTS.length && (
          <button
            type="button"
            onClick={onReset}
            disabled={disabled}
            className="text-xs text-[#eff483]/80 hover:text-[#eff483]"
          >
            Select all
          </button>
        )}
      </div>
      <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
        {ARM_JOINTS.map((joint) => {
          const on = selected.has(joint)
          return (
            <label
              key={joint}
              className={cn(
                "flex cursor-pointer items-center gap-2 rounded-md border px-2 py-1.5 text-xs capitalize transition-colors",
                on
                  ? "border-white/20 bg-white/[0.05] text-white/85"
                  : "border-white/10 text-white/35 hover:border-white/20",
                disabled && "pointer-events-none opacity-50"
              )}
            >
              <input
                type="checkbox"
                checked={on}
                disabled={disabled}
                onChange={() => toggle(joint)}
                className="sr-only"
              />
              <span
                className={cn("inline-block size-2 rounded-full", !on && "opacity-30")}
                style={{ background: JOINT_COLORS[joint] }}
              />
              {jointLabel(joint)}
            </label>
          )
        })}
      </div>
      <p className="text-xs text-white/35">
        Checked joints are held weightless and can be moved by hand; unchecked joints hold their
        position.
      </p>
    </div>
  )
}
