import { act } from "react"
import { createRoot, type Root } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { setServerBase } from "@/lib/supervisor"
import { FirmwareUploadField } from "./diagnostic-actions"
;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

function response(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

describe("FirmwareUploadField", () => {
  let container: HTMLDivElement
  let root: Root
  const onChange = vi.fn()
  const onUploading = vi.fn()

  beforeEach(() => {
    setServerBase("robot.local")
    container = document.createElement("div")
    document.body.appendChild(container)
    root = createRoot(container)
    onChange.mockReset()
    onUploading.mockReset()
  })

  afterEach(() => {
    act(() => root.unmount())
    container.remove()
    vi.unstubAllGlobals()
  })

  async function pick(file: File) {
    act(() => {
      root.render(
        <FirmwareUploadField
          help="firmware.bin"
          disabled={false}
          onUploading={onUploading}
          onChange={onChange}
        />
      )
    })
    const input = container.querySelector("input[type=file]") as HTMLInputElement
    Object.defineProperty(input, "files", { value: [file], configurable: true })
    await act(async () => {
      input.dispatchEvent(new Event("change", { bubbles: true }))
    })
  }

  it("uploads the file and hands the stored path to the run args", async () => {
    const fetchMock = vi.fn(async () =>
      response({
        path: "/home/axol/.almond/lift-firmware/jelly_legs-0.9-abc.bin",
        version: "0.9",
        built: "Oct 10 2026 12:00:00",
        buildId: "0x12345678",
        size: 65536,
      })
    )
    vi.stubGlobal("fetch", fetchMock)

    await pick(new File([new Uint8Array([1, 2, 3])], "firmware.bin"))

    expect(fetchMock).toHaveBeenCalledOnce()
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe("https://robot.local:8001/api/lift/firmware")
    expect(init.method).toBe("POST")
    expect(onChange).toHaveBeenLastCalledWith(
      "/home/axol/.almond/lift-firmware/jelly_legs-0.9-abc.bin"
    )
    expect(onUploading.mock.calls).toEqual([[true], [false]])
    expect(container.textContent).toContain("firmware 0.9")
  })

  it("shows the host's rejection and leaves the argument unset", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response({ error: "not a Jelly Legs firmware image" }, 400))
    )

    await pick(new File([new Uint8Array([1])], "other.bin"))

    expect(onChange).toHaveBeenCalledWith(null)
    expect(onChange).not.toHaveBeenCalledWith(expect.stringContaining("/"))
    expect(container.textContent).toContain("not a Jelly Legs firmware image")
  })
})
