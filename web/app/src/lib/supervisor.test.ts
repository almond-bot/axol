import { beforeEach, describe, expect, it } from "vitest"

import {
  HARDWARE_PROFILE_ARG,
  cameraCount,
  computeArgs,
  curatedFields,
  defaultString,
  filterSchema,
  flattenFields,
  isModified,
  isRobotFreeRun,
  loadLocalHardwareProfile,
  loadOpSettings,
  missingCameraSerials,
  missingRequired,
  operationsFromCommands,
  parseHardwareProfile,
  participatingCameraSerials,
  perRunFields,
  runFieldVisible,
  saveLocalHardwareProfile,
  saveOpSettings,
  serverHttpBase,
  type CameraSpec,
  type CommandSpec,
  type OperationId,
  type SchemaField,
  type SchemaNode,
} from "./supervisor"

const required: SchemaField = {
  kind: "field",
  key: "repo_id",
  label: "Repo ID",
  type: "text",
  default: null,
  required: true,
}
const optional: SchemaField = {
  kind: "field",
  key: "sim",
  label: "Sim",
  type: "boolean",
  default: false,
  required: false,
}
const schema: SchemaNode[] = [
  required,
  { kind: "group", key: "advanced", label: "Advanced", children: [optional] },
]

describe("supervisor pure helpers", () => {
  beforeEach(() => localStorage.clear())

  it.each([
    ["robot.local", "https://robot.local:8001"],
    ["http://robot.local:9000/path", "http://robot.local:9000"],
    ["", ""],
    ["http://[", ""],
  ])("normalizes server address %s", (input, expected) => {
    expect(serverHttpBase(input)).toBe(expected)
  })

  it("counts, trims, and checks configured cameras", () => {
    const cameras: CameraSpec = {
      serials: { overhead: " 1 ", left_arm: "2", right_arm: " " },
      stream_resolution: "SVGA",
      record_resolution: "SVGA",
    }
    expect(cameraCount(cameras)).toBe(2)
    expect(participatingCameraSerials(cameras)).toEqual(["1", "2"])
    expect(missingCameraSerials(cameras, [{ serial: 2, model: "one", kind: "mono" }])).toEqual([
      "1",
    ])
    // Mantis reuses the Axol wrist serials unless it has its own.
    expect(cameraCount(cameras, true)).toBe(1)
    // A camera disabled on the stream branch still participates via recording.
    expect(participatingCameraSerials({ ...cameras, stream: { left_arm: false } })).toEqual([
      "1",
      "2",
    ])
    expect(
      participatingCameraSerials({
        ...cameras,
        stream: { left_arm: false },
        record: { left_arm: false },
      })
    ).toEqual(["1"])
  })

  it("flattens and filters nested schemas", () => {
    expect(flattenFields(schema).map((field) => field.key)).toEqual(["repo_id", "sim"])
    expect(flattenFields(filterSchema(schema, new Set(["repo_id"])))).toEqual([optional])
    expect(filterSchema(schema, new Set(["repo_id", "sim"]))).toEqual([])
  })

  it("sends required and changed values only", () => {
    expect(missingRequired([required, optional], {})).toEqual(["repo_id"])
    expect(missingRequired([required], { repo_id: "org/data" })).toEqual([])
    expect(isModified(optional, false)).toBe(false)
    expect(isModified(optional, true)).toBe(true)
    expect(computeArgs([required, optional], { repo_id: "org/data", sim: false })).toEqual({
      repo_id: "org/data",
    })
    expect(computeArgs([required, optional], { repo_id: "org/data", sim: true })).toEqual({
      repo_id: "org/data",
      sim: true,
    })
  })

  it("derives operation metadata and robot-free flags", () => {
    const command = {
      id: "custom",
      label: "Custom",
      description: "Custom operation",
      simCapable: true,
      requiresHardware: false,
      available: true,
      error: null,
      schema: [],
      required: [],
      cli: "custom",
      category: "Operate",
      isOperation: true,
      simFlag: "simulate",
      robotFreeFlags: ["cart_only"],
      perRunFields: ["simulate"],
    } satisfies CommandSpec
    const [meta] = operationsFromCommands([command])
    expect(meta.fields).toEqual(["simulate"])
    expect(isRobotFreeRun(meta, { simulate: true })).toBe(true)
    expect(isRobotFreeRun(meta, { cart_only: true })).toBe(true)
    expect(isRobotFreeRun(meta, {})).toBe(false)
  })

  it("persists the system-wide hardware profile locally", () => {
    expect(parseHardwareProfile("axol")).toBe("axol")
    expect(parseHardwareProfile("mantis")).toBe("mantis")
    expect(parseHardwareProfile("other")).toBeNull()
    expect(loadLocalHardwareProfile()).toBe("axol")
    saveLocalHardwareProfile("mantis")
    expect(loadLocalHardwareProfile()).toBe("mantis")
  })

  it("hides the device flag and Axol-only run modes per profile", () => {
    expect(runFieldVisible(HARDWARE_PROFILE_ARG, "axol")).toBe(false)
    expect(runFieldVisible("sim", "axol")).toBe(true)
    expect(runFieldVisible("sim", "mantis")).toBe(false)
    expect(runFieldVisible("jelly_only", "mantis")).toBe(false)
    expect(runFieldVisible("repo_id", "mantis")).toBe(true)
  })

  it("resolves curated and per-run fields with required ones first", () => {
    const mantisFlag: SchemaField = {
      kind: "field",
      key: HARDWARE_PROFILE_ARG,
      label: "Mantis",
      type: "boolean",
      default: false,
      required: false,
    }
    const command = {
      id: "custom",
      label: "Custom",
      description: "Custom operation",
      simCapable: true,
      requiresHardware: false,
      available: true,
      error: null,
      schema: [...schema, mantisFlag],
      required: [],
      cli: "custom",
      category: "Operate",
      isOperation: true,
      simFlag: "sim",
      robotFreeFlags: [],
      perRunFields: ["sim", HARDWARE_PROFILE_ARG, "unknown"],
    } satisfies CommandSpec
    const [meta] = operationsFromCommands([command])
    expect(curatedFields(command, meta)).toEqual([optional, mantisFlag])
    expect(perRunFields(command, meta)).toEqual([required, optional])
    expect(perRunFields(command, meta, "mantis")).toEqual([required])
    expect(defaultString(required)).toBe("")
    expect(defaultString(optional)).toBe("false")
  })

  it("round-trips per-operation settings and drops the stale device flag", () => {
    const op = "teleop" as OperationId
    expect(loadOpSettings(op)).toEqual({})
    saveOpSettings(op, { sim: true, [HARDWARE_PROFILE_ARG]: true })
    expect(loadOpSettings(op)).toEqual({ sim: true })
    localStorage.setItem("axolOp:teleop", "{not json")
    expect(loadOpSettings(op)).toEqual({})
  })
})
