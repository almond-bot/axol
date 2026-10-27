import { Zap } from "lucide-react"
import { Card, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import type { JogArm, JogPlan, JogState } from "@/lib/jog"

const JOINT_LABELS = [
  "Shoulder 1",
  "Shoulder 2",
  "Shoulder 3",
  "Elbow",
  "Wrist 1",
  "Wrist 2",
  "Wrist 3",
]

const deg = (rad: number, digits = 1) => `${((rad * 180) / Math.PI).toFixed(digits)}°`

/**
 * One condensed table of everything moving: both arms, the lift and the base.
 * `activeArm` highlights one arm's column, and `plan` adds the planned joints.
 */
export function RobotStatus({
  state,
  activeArm = null,
  plan = null,
}: {
  state: JogState | null
  activeArm?: JogArm | null
  plan?: JogPlan | null
}) {
  const arms = state?.arms
  const lift = state?.lift
  const base = state?.base
  const row = "contents"
  const name = "text-white/50"
  const value = "text-right font-mono text-white/80 tabular-nums"
  const armHeader = (side: JogArm) =>
    cn("text-right font-medium", side === activeArm ? "text-[#eff483]" : "text-white/50")
  return (
    <Card data-testid="jog-status">
      <CardHeader>
        <CardTitle>Robot state</CardTitle>
      </CardHeader>
      <div
        className={cn(
          "grid gap-x-3 gap-y-1 text-xs",
          plan ? "grid-cols-[1fr_auto_auto_auto]" : "grid-cols-[1fr_auto_auto]"
        )}
      >
        <span className={name}>Joint</span>
        <span className={armHeader("left")}>Left</span>
        <span className={armHeader("right")}>Right</span>
        {plan && (
          <span className="text-right font-medium text-[#ff9a3c]">
            Plan {plan.arm === "left" ? "L" : "R"}
          </span>
        )}
        {JOINT_LABELS.map((label, i) => (
          <div key={label} className={row}>
            <span className={name}>{label}</span>
            <span data-testid={`joint-left-${i}`} className={value}>
              {arms ? deg(arms.left[i]) : "–"}
            </span>
            <span data-testid={`joint-right-${i}`} className={value}>
              {arms ? deg(arms.right[i]) : "–"}
            </span>
            {plan && (
              <span className="text-right font-mono text-[#ff9a3c] tabular-nums">
                {deg(plan.joints[i])}
              </span>
            )}
          </div>
        ))}
      </div>
      <div className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 border-t border-white/10 pt-3 text-xs">
        <span className={name}>Lift</span>
        <span data-testid="status-lift" className={value}>
          {lift?.raised != null
            ? `${(lift.raised * 100).toFixed(1)} % · ${(lift.velocity * 100).toFixed(2)} %/s`
            : lift
              ? "Not homed"
              : "–"}
        </span>
        <span className={name}>Base position{base?.estimated ? " (est.)" : ""}</span>
        <span data-testid="status-odometry" className={value}>
          {base ? `${base.x.toFixed(2)}, ${base.y.toFixed(2)} m · ${deg(base.yaw, 0)}` : "–"}
        </span>
        <span className={name}>Base speed</span>
        <span className={value}>
          {base
            ? `${Math.hypot(base.velocity[0], base.velocity[1]).toFixed(2)} m/s · ${deg(base.velocity[2], 0)}/s`
            : "–"}
        </span>
        <span className={name}>Wheel command</span>
        <span className={value}>
          {base ? base.command.map((c) => `${Math.round(c * 100)}`).join(" / ") + " %" : "–"}
        </span>
        {base?.fault && (
          <span className="col-span-2 text-amber-300">
            <Zap className="mr-1 inline size-3.5" />A wheel is faulted.
          </span>
        )}
      </div>
    </Card>
  )
}
