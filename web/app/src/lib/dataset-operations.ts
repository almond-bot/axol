/** The operations that record a dataset, so the dataset preview has
 *  something to show; every other operation hides it. */
export const DATASET_OPERATIONS: ReadonlySet<string> = new Set([
  "collect-data",
  "collect-dagger",
  "run-policy",
])

/** Whether the control panel shows the dataset preview for an operation. */
export function showsDatasetPreview(operationId: string): boolean {
  return DATASET_OPERATIONS.has(operationId)
}
