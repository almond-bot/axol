import { type InputHTMLAttributes, useState } from "react"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"

/** One suggestion offered inside a text field (typing stays free-form). */
export interface FieldSuggestion {
  value: string
  /** Secondary text shown next to the value (e.g. "12 episodes"). */
  label?: string
}

/**
 * A text input with an attached suggestion dropdown — one visual control, not
 * an input plus a separate picker. Focusing (or typing) opens a styled list
 * of suggestions filtered by the current text; clicking or Enter fills the
 * field, while any free-form text remains valid. Used for dataset repo ids on
 * every operation panel and for the recent hosts in the setup dialog. Extra
 * input attributes (spellCheck, autoCapitalize, …) pass through to the input;
 * `className` styles the wrapper so it can size itself inside a flex row.
 */
export function SuggestInput({
  id,
  value,
  placeholder,
  disabled,
  suggestions,
  onChange,
  className,
  ...inputProps
}: Omit<InputHTMLAttributes<HTMLInputElement>, "value" | "onChange" | "disabled"> & {
  id: string
  value: string
  placeholder?: string
  disabled?: boolean
  suggestions: FieldSuggestion[]
  onChange: (value: string) => void
  className?: string
}) {
  const [open, setOpen] = useState(false)
  const [highlight, setHighlight] = useState(-1)

  // Filter by the typed text; an exact match (a suggestion was just picked,
  // or the field reopened on a stored value) shows the full list again so
  // the operator can switch values without clearing the field first.
  const text = value.trim().toLowerCase()
  const exact = suggestions.some((s) => s.value.toLowerCase() === text)
  const shown =
    text === "" || exact
      ? suggestions
      : suggestions.filter((s) => s.value.toLowerCase().includes(text))

  function pick(v: string) {
    onChange(v)
    setOpen(false)
    setHighlight(-1)
  }

  return (
    <div className={cn("relative", className)}>
      <Input
        {...inputProps}
        id={id}
        value={value}
        placeholder={placeholder}
        disabled={disabled}
        autoComplete="off"
        onChange={(e) => {
          onChange(e.target.value)
          setOpen(true)
          setHighlight(-1)
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => {
          setOpen(false)
          setHighlight(-1)
        }}
        onKeyDown={(e) => {
          if (!open || shown.length === 0) return
          if (e.key === "ArrowDown") {
            e.preventDefault()
            setHighlight((h) => (h + 1) % shown.length)
          } else if (e.key === "ArrowUp") {
            e.preventDefault()
            setHighlight((h) => (h <= 0 ? shown.length - 1 : h - 1))
          } else if (e.key === "Enter" && highlight >= 0) {
            e.preventDefault()
            pick(shown[highlight].value)
          } else if (e.key === "Escape") {
            setOpen(false)
            setHighlight(-1)
          }
        }}
      />
      {open && shown.length > 0 && (
        <ul className="absolute top-full right-0 left-0 z-20 mt-1 max-h-52 overflow-auto rounded-md border border-white/10 bg-[#1c1c1c] py-1 shadow-xl">
          {shown.map((s, i) => (
            <li key={s.value}>
              <button
                type="button"
                // mousedown fires before the input's blur, which would
                // otherwise close the list under the click.
                onMouseDown={(e) => {
                  e.preventDefault()
                  pick(s.value)
                }}
                onMouseEnter={() => setHighlight(i)}
                className={cn(
                  "flex w-full items-baseline justify-between gap-3 px-3 py-1.5 text-left text-sm",
                  i === highlight ? "bg-white/10 text-foreground" : "text-white/80"
                )}
              >
                <span className="truncate">{s.value}</span>
                {s.label && <span className="shrink-0 text-xs text-white/40">{s.label}</span>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
