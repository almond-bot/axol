import { useId, useMemo, useState } from "react"
import { TASK_SETTING_KEY, taskPicker, useAxolSettings } from "@almond/axol-vr-client"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

/**
 * The task of the next episode during a `collect-data` session, on the
 * control panel.
 *
 * The session publishes its task picker as the live setting `task` over the
 * VR server's WebSocket (the same channel as `SessionSettings`), so the
 * headset HUD and this card share one selection. Picking a task sends a
 * `set`. Adding a task sends the new text, which the server appends to the
 * list and selects. Renders nothing until the server announces a picker
 * (other operations have none).
 *
 * `locked` is true while an episode records or saves: the server keeps the
 * task fixed for the whole episode.
 */
export function TaskPicker({ socket, locked }: { socket: WebSocket | null; locked: boolean }) {
  // useAxolSettings wants a ref; a fresh object per socket re-runs its
  // subscription effect on every (re)connect.
  const socketRef = useMemo(() => ({ current: socket }), [socket])
  const { settings, setSetting } = useAxolSettings(socketRef, socket !== null)
  const [draft, setDraft] = useState("")
  const selectId = useId()
  const inputId = useId()

  const picker = taskPicker(settings)
  if (!picker) return null

  const newTask = draft.trim()
  function addTask() {
    if (!newTask || locked) return
    setSetting(TASK_SETTING_KEY, newTask)
    setDraft("")
  }

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-white/10 bg-white/[0.02] p-3">
      <label
        htmlFor={selectId}
        className="font-mono text-xs tracking-widest text-white/40 uppercase"
      >
        Task
      </label>
      <select
        id={selectId}
        value={picker.current}
        disabled={locked}
        onChange={(e) => setSetting(TASK_SETTING_KEY, e.target.value)}
        className="h-9 w-full rounded-md border border-input bg-white/[0.02] px-3 text-sm text-foreground outline-none focus-visible:border-ring/70 disabled:opacity-50"
      >
        {picker.tasks.map((task) => (
          <option key={task} value={task} className="bg-[#1a1a1a]">
            {task}
          </option>
        ))}
      </select>
      <div className="flex items-center gap-2">
        <Input
          id={inputId}
          aria-label="New task"
          value={draft}
          placeholder="New task"
          disabled={locked}
          maxLength={1000}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") addTask()
          }}
          className="h-8 flex-1"
        />
        <Button variant="outline" size="sm" disabled={locked || !newTask} onClick={addTask}>
          Add task
        </Button>
      </div>
      <p className="text-xs text-white/40">
        {locked
          ? "You can change the task when this episode ends."
          : "The next episode records this task."}
      </p>
    </div>
  )
}
