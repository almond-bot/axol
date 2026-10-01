import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { DatasetEpisodes } from "@/lib/supervisor"
import { DatasetPreview } from "./dataset-preview"

const EPISODES: DatasetEpisodes = {
  repoId: "axol/pick",
  root: "/data/axol/pick",
  fps: 60,
  cameras: ["observation.images.overhead", "observation.images.left_arm"],
  tasks: ["pick"],
  episodes: [
    {
      index: 0,
      length: 120,
      durationS: 2,
      tasks: ["pick"],
      videos: {
        "observation.images.overhead": { from: 0, to: 2 },
        "observation.images.left_arm": { from: 0, to: 2 },
      },
    },
  ],
  unreadableFiles: 0,
}

vi.mock("@/lib/supervisor", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/supervisor")>()),
  fetchDatasets: vi.fn(async () => [
    { repoId: "axol/pick", root: "/data/axol/pick", episodes: 1, fps: 60 },
  ]),
  fetchDatasetEpisodes: vi.fn(async () => EPISODES),
}))
vi.mock("@/components/ui/toast", () => ({
  useToast: () => ({ success: () => undefined, error: () => undefined }),
}))
;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

let root: Root
let container: HTMLDivElement

beforeEach(() => {
  localStorage.setItem("datasetPreviewOpen", "1")
  // jsdom has no media pipeline.
  vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined)
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => undefined)
  vi.spyOn(HTMLMediaElement.prototype, "load").mockImplementation(() => undefined)
  container = document.createElement("div")
  document.body.appendChild(container)
  root = createRoot(container)
})

afterEach(() => {
  act(() => root.unmount())
  container.remove()
  localStorage.clear()
  vi.restoreAllMocks()
})

async function render() {
  await act(async () => {
    root.render(<DatasetPreview connected liveDataset={null} episodesRecorded={null} />)
  })
}

function preloads(): (string | null)[] {
  return [...container.querySelectorAll("video")].map((v) => v.getAttribute("preload"))
}

describe("DatasetPreview", () => {
  it("loads no episode video until Play is pressed", async () => {
    await render()
    // The selected episode's cameras are laid out, but none downloads: the
    // card selects each newly saved take during a live session.
    expect(preloads()).toEqual(["none", "none"])

    const play = container.querySelector('button[aria-label="Play"]') as HTMLButtonElement
    await act(async () => play.click())

    expect(preloads()).toEqual(["auto", "auto"])
  })
})
