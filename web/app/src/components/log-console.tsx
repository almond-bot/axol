import { useEffect, useRef } from "react"
import { Terminal } from "lucide-react"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"

/** Auto-scrolling log viewer shared by the operation panels and setup page. */
export function LogConsole({ lines }: { lines: string[] }) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [lines])

  return (
    <Card className="min-h-0 flex-1 gap-3 p-0">
      <div className="flex items-center gap-2 border-b border-white/10 px-4 py-3">
        <Terminal className="size-4 text-white/40" />
        <span className="font-heading text-sm font-semibold">Logs</span>
      </div>
      <div
        ref={scrollRef}
        className="max-h-[60vh] min-h-[280px] overflow-auto px-4 pb-4 font-mono text-xs leading-relaxed"
      >
        {lines.length === 0 ? (
          <p className="text-white/30">No output yet.</p>
        ) : (
          lines.map((line, i) => (
            <div
              key={i}
              className={cn(
                "break-words whitespace-pre-wrap",
                line.startsWith("[serve]") ? "text-[#eff483]/70" : "text-white/70"
              )}
            >
              {line}
            </div>
          ))
        )}
      </div>
    </Card>
  )
}
