import { describe, expect, it } from "vitest"

import {
  BOTH_CLICK_DEBOUNCE_MS,
  BOTH_CLICK_TOGETHER_MS,
  initialBothStickClickState,
  stepBothStickClick,
} from "./bothStickClick"

describe("stepBothStickClick", () => {
  it("fires once when both sticks are clicked together", () => {
    const s = initialBothStickClickState()
    expect(stepBothStickClick(s, true, false, 1000)).toBe(false)
    expect(stepBothStickClick(s, true, true, 1100)).toBe(true)
    // Held: no repeat.
    expect(stepBothStickClick(s, true, true, 1200)).toBe(false)
  })

  it("ignores a stick held for the lift when the other is pressed", () => {
    const s = initialBothStickClickState()
    stepBothStickClick(s, true, false, 1000)
    expect(stepBothStickClick(s, true, true, 1000 + BOTH_CLICK_TOGETHER_MS + 1)).toBe(false)
  })

  it("debounces a second rising edge", () => {
    const s = initialBothStickClickState()
    expect(stepBothStickClick(s, true, true, 1000)).toBe(true)
    stepBothStickClick(s, true, false, 1050)
    expect(stepBothStickClick(s, true, true, 1100)).toBe(false)
    stepBothStickClick(s, false, false, 1200)
    const later = 1000 + BOTH_CLICK_DEBOUNCE_MS + 10
    expect(stepBothStickClick(s, true, true, later)).toBe(true)
  })
})
