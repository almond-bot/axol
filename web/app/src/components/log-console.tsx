import { useEffect, useRef } from "react"
import { Download, Terminal } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"

/** Auto-scrolling log viewer shared by the operation panels and setup page. */
export function LogConsole({ lines }: { lines: string[] }) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [lines])

  function downloadLogs() {
    const blob = new Blob([lines.join("\n") + "\n"], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const stamp = new Date().toISOString().replace(/[:.]/g, "-")
    const a = document.createElement("a")
    a.href = url
    a.download = `axol-logs-${stamp}.log`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  return (
    <Card className="min-h-0 flex-1 gap-3 p-0">
      <div className="flex items-center gap-2 border-b border-white/10 px-4 py-3">
        <Terminal className="size-4 text-white/40" />
        <span className="font-heading text-sm font-semibold">Logs</span>
        <Button
          variant="ghost"
          size="icon"
          className="ml-auto size-7 text-white/30 hover:bg-white/[0.04] hover:text-white/70"
          onClick={downloadLogs}
          disabled={lines.length === 0}
          aria-label="Download logs"
          title="Download logs"
        >
          <Download />
        </Button>
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
