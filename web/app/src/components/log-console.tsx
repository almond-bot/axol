import { useEffect, useRef, useState } from "react"
import { ArrowDown, Download, Terminal } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"

/** Auto-scrolling log viewer shared by the operation panels and setup page. */
export function LogConsole({ lines }: { lines: string[] }) {
  const scrollRef = useRef<HTMLDivElement>(null)
  // Whether new lines should pull the view to the bottom. True while the user
  // is at (or near) the bottom; cleared once they scroll up so they can read
  // back through history without being yanked down by incoming logs.
  const stickToBottom = useRef(true)
  const [showJumpToBottom, setShowJumpToBottom] = useState(false)

  // Distance from the bottom (px) still treated as "following"; absorbs
  // sub-pixel rounding and wrapped-line reflow.
  const STICK_THRESHOLD_PX = 24

  function handleScroll() {
    const el = scrollRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight <= STICK_THRESHOLD_PX
    stickToBottom.current = atBottom
    setShowJumpToBottom(!atBottom)
  }

  useEffect(() => {
    const el = scrollRef.current
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight
  }, [lines])

  function scrollToBottom() {
    const el = scrollRef.current
    if (!el) return
    stickToBottom.current = true
    setShowJumpToBottom(false)
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" })
  }

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
      <div className="relative min-h-0">
        <div
          ref={scrollRef}
          onScroll={handleScroll}
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
        {showJumpToBottom && (
          <Button
            variant="secondary"
            size="icon"
            className="absolute right-4 bottom-4 size-8 rounded-full border border-white/10 bg-black/60 text-white/70 shadow-lg backdrop-blur hover:bg-black/80 hover:text-white"
            onClick={scrollToBottom}
            aria-label="Scroll to bottom"
            title="Scroll to bottom"
          >
            <ArrowDown />
          </Button>
        )}
      </div>
    </Card>
  )
}
