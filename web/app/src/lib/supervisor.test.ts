import { beforeEach, describe, expect, it } from "vitest"

import {
  HARDWARE_PROFILE_ARG,
  CUSTOM_POLICY_TYPE,
  applyPolicyTypeRules,
  cameraCount,
  computeArgs,
  curatedFields,
  defaultString,
  episodeVideoSource,
  filterSchema,
  flattenFields,
  isCustomPolicyRun,
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
  previewStep,
  runFieldVisible,
  saveLocalHardwareProfile,
  saveOpSettings,
  serverHttpBase,
  settingShown,
  visibleAdvancedSections,
  type AdvancedSection,
  type CameraSpec,
  type CommandSpec,
  type DatasetEpisode,
  type OperationId,
  type SchemaField,
  type SchemaNode,
  type SettingsCategory,
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

  it("requires a policy path only for LeRobot policy types", () => {
    const field = (key: string, required: boolean): SchemaField => ({
      kind: "field",
      key,
      label: key,
      type: "text",
      default: null,
      required,
    })
    const fields = [field("policy_type", true), field("task", true), field("policy_path", false)]
    const lerobot = applyPolicyTypeRules(fields, false)
    expect(lerobot.find((f) => f.key === "policy_path")?.required).toBe(true)
    const custom = applyPolicyTypeRules(fields, true)
    const path = custom.find((f) => f.key === "policy_path")
    expect(path?.required).toBe(false)
    expect(path?.help).toMatch(/policy server/)
    expect(custom.map((f) => f.key)).toEqual(["policy_type", "task", "policy_path"])
    // Ops without a policy type are left alone.
    const other = [field("policy_path", false)]
    expect(applyPolicyTypeRules(other, false)).toBe(other)
    expect(isCustomPolicyRun({ policy_type: CUSTOM_POLICY_TYPE })).toBe(true)
    expect(isCustomPolicyRun({ policy_type: "act" })).toBe(false)
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

  it("attaches the host's widget hints to per-run fields", () => {
    const speed: SchemaField = {
      kind: "field",
      key: "speed_scale",
      label: "Speed scale",
      type: "number",
      default: 1,
      required: false,
    }
    const slider = { widget: "slider" as const, min: 0.1, max: 1, step: 0.05 }
    const command = {
      id: "waypoints",
      label: "Waypoints",
      description: "",
      simCapable: true,
      requiresHardware: false,
      available: true,
      error: null,
      schema: [speed, optional],
      required: [],
      cli: "waypoints",
      category: "Operate",
      isOperation: true,
      perRunFields: ["speed_scale", "sim"],
      fieldUi: { speed_scale: slider },
    } satisfies CommandSpec
    const [meta] = operationsFromCommands([command])
    expect(perRunFields(command, meta)).toEqual([{ ...speed, ui: slider }, optional])
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

describe("dataset preview sources", () => {
  const episode: DatasetEpisode = {
    index: 3,
    length: 120,
    durationS: 2,
    tasks: ["pick"],
    videos: { "observation.images.left_arm": { from: 6, to: 8 } },
  }

  it.each([
    [60, 15, 4],
    [60, 30, 2],
    [60, 0, 1],
    [30, 30, 1],
    [30, 15, 2],
    [10, 15, 1],
  ])("keeps every Nth frame (%i fps at %i → %i)", (dataset, preview, step) => {
    expect(previewStep(dataset, preview)).toBe(step)
  })

  it("asks for the host's cut, rebased to the episode", () => {
    const light = episodeVideoSource("org/ds", episode, "observation.images.left_arm", 60, 15)
    expect(light.span).toEqual({ from: 0, to: 2 })
    const url = new URL(light.url, "http://host")
    expect(url.searchParams.get("fps")).toBe("15")
    expect(url.searchParams.get("episode")).toBe("3")
    expect(url.hash).toBe("#t=0.000,2.000")

    // Full rate is the cut too (index first, so it starts before it downloads).
    const full = episodeVideoSource("org/ds", episode, "observation.images.left_arm", 60, 0)
    expect(new URL(full.url, "http://host").searchParams.get("fps")).toBe("60")
    expect(full.span).toEqual({ from: 0, to: 2 })
  })
})

describe("showWhen gates", () => {
  const parcelOnly = { key: "axol.gripper", equals: "parcel" }
  const schema: SettingsCategory[] = [
    {
      key: "robot",
      label: "Robot",
      description: "",
      settings: [
        {
          key: "axol.gripper",
          label: "Gripper",
          type: "select",
          help: "",
          options: ["parallel", "parcel"],
          default: "parallel",
          ui: {},
          targets: {},
        },
      ],
    },
  ]
  const advanced: AdvancedSection[] = [
    {
      key: "teleop",
      label: "Teleop",
      nodes: [
        { ...optional, key: "teleop.frequency" },
        { ...optional, key: "teleop.box_flush_deg", showWhen: parcelOnly },
      ],
    },
  ]

  it("reads the staged value, else the default", () => {
    expect(settingShown(undefined, {}, schema)).toBe(true)
    expect(settingShown(parcelOnly, {}, schema)).toBe(false)
    expect(settingShown(parcelOnly, { "axol.gripper": "parcel" }, schema)).toBe(true)
  })

  it("reads a reset (null) value as the default", () => {
    const parcelDefault: SettingsCategory[] = [
      {
        ...schema[0],
        settings: [{ ...schema[0].settings[0], default: "parcel" }],
      },
    ]
    const reset = { "axol.gripper": null } as unknown as Record<string, string>
    expect(settingShown(parcelOnly, reset, parcelDefault)).toBe(true)
    expect(settingShown(parcelOnly, reset, schema)).toBe(false)
  })

  it("prunes gated Advanced fields", () => {
    const keys = (sections: AdvancedSection[]) =>
      sections.flatMap((s) => flattenFields(s.nodes)).map((f) => f.key)
    expect(keys(visibleAdvancedSections(advanced, {}, schema))).toEqual(["teleop.frequency"])
    expect(keys(visibleAdvancedSections(advanced, { "axol.gripper": "parcel" }, schema))).toEqual([
      "teleop.frequency",
      "teleop.box_flush_deg",
    ])
  })
})
