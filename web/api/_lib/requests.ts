/** Helpers for the Teleop Data Collection routes, kept out of the route files so
 *  each route exports only its HTTP handlers. */

/** `returnUrl` with `data_collection=<outcome>`, replacing an earlier one. */
export function returnTo(returnUrl: string, outcome: "paid" | "cancelled"): string {
  const url = new URL(returnUrl)
  url.searchParams.set("data_collection", outcome)
  return url.toString()
}
