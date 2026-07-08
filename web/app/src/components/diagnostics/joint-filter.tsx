import { cn } from "@/lib/utils"
import { JOINTS, JOINT_COLORS, jointLabel, type JointName } from "@/lib/telemetry"

/**
 * Toggleable joint chips — one fixed color slot per joint (filtering never
 * repaints the survivors). An empty selection is allowed; charts show an
 * empty-state notice.
 */
export function JointFilter({
  hidden,
  onChange,
}: {
  hidden: Set<JointName>
  onChange: (hidden: Set<JointName>) => void
}) {
  function toggle(joint: JointName) {
    const next = new Set(hidden)
    if (next.has(joint)) next.delete(joint)
    else next.add(joint)
    onChange(next)
  }

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {JOINTS.map((joint) => {
        const off = hidden.has(joint)
        return (
          <button
            key={joint}
            type="button"
            onClick={() => toggle(joint)}
            aria-pressed={!off}
            title={off ? "Show joint" : "Hide joint"}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs capitalize transition-colors",
              off
                ? "border-white/10 text-white/30 hover:border-white/25"
                : "border-white/15 bg-white/[0.04] text-white/75 hover:bg-white/[0.08]"
            )}
          >
            <span
              className="inline-block size-2 rounded-full"
              style={{ background: off ? "rgba(255,255,255,0.15)" : JOINT_COLORS[joint] }}
            />
            {jointLabel(joint)}
          </button>
        )
      })}
      {hidden.size > 0 && (
        <button
          type="button"
          onClick={() => onChange(new Set())}
          className="px-1.5 py-1 text-xs text-[#eff483]/80 hover:text-[#eff483]"
        >
          Show all
        </button>
      )}
    </div>
  )
}
