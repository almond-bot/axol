import { describe, expect, it } from "vitest"

import { cn, sentenceCase } from "./utils"

describe("UI utilities", () => {
  it("uses canonical acronym casing", () => {
    expect(sentenceCase("repo id")).toBe("Repo ID")
    expect(sentenceCase("teleop hz")).toBe("Teleop Hz")
    expect(sentenceCase("com x offset")).toBe("CoM x offset")
  })

  it("merges conflicting Tailwind classes", () => {
    expect(cn("p-2 text-red-500", undefined, "p-4")).toBe("text-red-500 p-4")
  })
})
