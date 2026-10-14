import { describe, expect, it } from "vitest"

import { showsDatasetPreview } from "./dataset-operations"

describe("dataset preview visibility", () => {
  it("shows for the operations that record a dataset", () => {
    for (const id of ["collect-data", "collect-dagger", "run-policy"]) {
      expect(showsDatasetPreview(id)).toBe(true)
    }
  })

  it("hides for every other operation", () => {
    for (const id of ["teleop", "gravity-comp", "waypoints", "replay-dataset", ""]) {
      expect(showsDatasetPreview(id)).toBe(false)
    }
  })
})
