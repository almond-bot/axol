import { lazy, Suspense, useMemo, useState } from "react"
import { Loader2, RotateCcw } from "lucide-react"
import type { SettingsField, SettingValue } from "@/lib/supervisor"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import type { JointLimits } from "./pose-viewer"

// The 3D viewer pulls in three.js — lazy so opening the control panel (or even
// the settings dialog) doesn't download it until the Rest pose tab is shown.
const PoseViewer = lazy(() => import("./pose-viewer"))

// Arm joints in ARM_JOINTS order (constants.py) — index i of a rest_pose array
// drives the URDF joint `${side}_${SUFFIX[i]}` (see urdf_arm_joint_names).
const JOINT_SUFFIXES = ["s1_0", "s2_0", "s3_0", "e1_0", "e2_0", "w1_0", "w2_0"] as const
const JOINT_LABELS = [
  "Shoulder 1",
  "Shoulder 2",
  "Shoulder 3",
  "Elbow",
  "Wrist 1",
  "Wrist 2",
  "Wrist 3",
] as const

const SIDES = [
  { side: "left", label: "Left arm", key: "teleop.rest_pose_left" },
  { side: "right", label: "Right arm", key: "teleop.rest_pose_right" },
] as const

const deg = (rad: number) => `${((rad * 180) / Math.PI).toFixed(0)}°`

function parsePose(value: SettingValue | null | undefined): number[] | null {
  if (Array.isArray(value)) return value.map(Number)
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value)
      if (Array.isArray(parsed)) return parsed.map(Number)
    } catch {
      return null
    }
  }
  return null
}

/**
 * Interactive rest-pose editor: a live URDF view of the robot next to per-joint
 * angle sliders for both arms. The pose previewed here is what the arms move to
 * on teleop start/reset and data collection.
 */
export function PosePanel({
  fields,
  values,
  onChange,
}: {
  /** The rest-pose settings fields (teleop.rest_pose_left / _right). */
  fields: SettingsField[]
  /** Draft settings values (sparse; unset keys mean "use the default"). */
  values: Record<string, SettingValue>
  onChange: (key: string, value: SettingValue | null) => void
}) {
  const [limits, setLimits] = useState<JointLimits>({})

  const fieldByKey = useMemo(() => new Map(fields.map((f) => [f.key, f])), [fields])

  // Current pose per side: the draft value if set, else the config default.
  const poses = useMemo(() => {
    const out: Record<string, number[]> = {}
    for (const { side, key } of SIDES) {
      const f = fieldByKey.get(key)
      out[side] = parsePose(values[key]) ?? parsePose(f?.default ?? null) ?? Array(7).fill(0)
    }
    return out
  }, [fieldByKey, values])

  const jointValues = useMemo(() => {
    const out: Record<string, number> = {}
    for (const { side } of SIDES) {
      JOINT_SUFFIXES.forEach((suffix, i) => {
        out[`${side}_${suffix}`] = poses[side][i] ?? 0
      })
    }
    return out
  }, [poses])

  function setJoint(side: "left" | "right", key: string, index: number, value: number) {
    const next = [...poses[side]]
    next[index] = Number(value.toFixed(4))
    onChange(key, next)
  }

  return (
    <div className="flex flex-col gap-5">
      <p className="text-xs text-white/45">
        The rest pose is where the arms start from (and return to) in teleop and data collection.
        Drag the sliders to preview the pose on the robot model, then save.
      </p>

      <div className="h-72 overflow-hidden rounded-xl border border-white/10 bg-black/30">
        <Suspense
          fallback={
            <div className="flex h-full items-center justify-center">
              <Loader2 className="size-5 animate-spin text-white/30" />
            </div>
          }
        >
          <PoseViewer jointValues={jointValues} onLoaded={setLimits} />
        </Suspense>
      </div>

      <div className="grid gap-5 sm:grid-cols-2">
        {SIDES.map(({ side, label, key }) => {
          const modified = values[key] !== undefined
          return (
            <div
              key={side}
              className="flex flex-col gap-3 rounded-lg border border-white/10 bg-white/[0.02] p-3"
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{label}</span>
                {modified && (
                  <button
                    type="button"
                    onClick={() => onChange(key, null)}
                    className="flex items-center gap-1 text-xs text-white/40 hover:text-white/70"
                    title="Reset this arm to the default rest pose"
                  >
                    <RotateCcw className="size-3" />
                    Reset
                  </button>
                )}
              </div>
              {JOINT_SUFFIXES.map((suffix, i) => {
                const jointName = `${side}_${suffix}`
                const lim = limits[jointName] ?? { lower: -Math.PI, upper: Math.PI }
                return (
                  <div key={suffix} className="flex items-center gap-3">
                    <span className="w-20 shrink-0 text-xs text-white/55">{JOINT_LABELS[i]}</span>
                    <Slider
                      value={poses[side][i] ?? 0}
                      min={lim.lower}
                      max={lim.upper}
                      step={0.005}
                      onChange={(v) => setJoint(side, key, i, v)}
                      format={deg}
                      aria-label={`${label} ${JOINT_LABELS[i]}`}
                    />
                  </div>
                )
              })}
            </div>
          )
        })}
      </div>

      <div className="flex justify-end">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => SIDES.forEach(({ key }) => onChange(key, null))}
        >
          <RotateCcw />
          Reset both arms to defaults
        </Button>
      </div>
    </div>
  )
}
