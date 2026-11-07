import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

/** What Stop does with arms that are away from rest. */
export type StopArmsChoice = "rest" | "hold" | "gravity"

const CHOICES: { id: StopArmsChoice; label: string; hint: string }[] = [
  {
    id: "rest",
    label: "Go to rest",
    hint: "The arms move to the rest pose, then the motors turn off.",
  },
  {
    id: "hold",
    label: "Hold position",
    hint: "The arms stay where they are with the motors on.",
  },
  {
    id: "gravity",
    label: "Gravity comp",
    hint: "The arms stay where they are, then go weightless so you can move them by hand.",
  },
]

/**
 * Asked when teleop stops with the arms away from rest, so they never drop
 * when the motors turn off.
 */
export function StopArmsDialog({
  open,
  busy,
  onChoose,
  onCancel,
}: {
  open: boolean
  busy: boolean
  onChoose: (choice: StopArmsChoice) => void
  onCancel: () => void
}) {
  if (!open) return null
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onCancel}
    >
      <Card
        role="dialog"
        aria-modal="true"
        aria-labelledby="stop-arms-title"
        className="w-full max-w-sm gap-4 bg-[#1a1a1a] p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex flex-col gap-1">
          <h3 id="stop-arms-title" className="font-heading text-base font-semibold">
            The arms are not at rest
          </h3>
          <p className="text-sm leading-relaxed text-white/45">
            Choose what the arms do when teleop stops.
          </p>
        </div>
        <div className="flex flex-col gap-2">
          {CHOICES.map((c) => (
            <Button
              key={c.id}
              variant={c.id === "rest" ? "default" : "secondary"}
              className="h-auto flex-col items-start gap-0.5 py-2 text-left whitespace-normal"
              disabled={busy}
              onClick={() => onChoose(c.id)}
            >
              <span className="font-medium">{c.label}</span>
              <span className="text-xs font-normal opacity-70">{c.hint}</span>
            </Button>
          ))}
        </div>
        <div className="flex justify-end">
          <Button variant="ghost" size="sm" onClick={onCancel} disabled={busy}>
            Keep running
          </Button>
        </div>
      </Card>
    </div>
  )
}
