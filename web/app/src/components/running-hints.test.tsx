import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { SessionInfo } from "@/lib/supervisor"
import { RunningHints } from "./operation-panel"

// operation-panel's camera feeds import the workspace @almond/axol-vr-client
// package, which CI runs these tests before building; the hints don't use them.
vi.mock("@/components/camera-feeds", () => ({ CameraFeeds: () => null }))

// Lets React's act() flush updates outside a test renderer.
;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

let root: Root
let container: HTMLDivElement

beforeEach(() => {
  container = document.createElement("div")
  document.body.appendChild(container)
  root = createRoot(container)
})

afterEach(() => {
  act(() => root.unmount())
  container.remove()
})

const session: SessionInfo = {
  id: "s1",
  command: "waypoints",
  args: { teach: "vr" },
  status: "running",
  exitCode: null,
  error: null,
  startedAt: 0,
  pid: null,
}

function render(props: { usesHeadset: boolean; vrWaypoints?: boolean }) {
  act(() =>
    root.render(
      <RunningHints
        {...props}
        mantisMode={false}
        mantisSource="quest"
        dataCollection={false}
        session={session}
        isSim={false}
        host="axol.local"
        viewerPort={8002}
      />
    )
  )
}

describe("RunningHints", () => {
  it("tells a VR waypoints run to connect the headset and names its buttons", () => {
    render({ usesHeadset: true, vrWaypoints: true })
    expect(container.textContent).toContain("connect to axol.local")
    expect(container.textContent).toContain("A records a waypoint")
  })

  it("keeps hand-guided waypoints free of headset hints", () => {
    render({ usesHeadset: false })
    expect(container.textContent).not.toContain("headset")
  })
})
