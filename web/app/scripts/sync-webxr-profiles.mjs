// Copy the WebXR input-profile descriptors out of @webxr-input-profiles/assets
// into public/webxr-profiles/ so they ship with the bundle and are served
// same-origin. @react-three/xr otherwise fetches these from a jsdelivr CDN at
// runtime (profilesList.json + <profile>/profile.json) to map controller
// gamepad layouts, which hangs forever on an offline LAN. createXRStore is
// pointed at this local copy via baseAssetPath (see App.tsx).
//
// Only the .json descriptors are copied — the .glb controller meshes are not,
// because the store runs with controller/hand models disabled, so they are
// never requested. Runs from predev/prebuild; output is gitignored.

import { createRequire } from "node:module"
import { cp, mkdir, rm, readdir } from "node:fs/promises"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

const require = createRequire(import.meta.url)
const here = dirname(fileURLToPath(import.meta.url))

const pkgJson = require.resolve("@webxr-input-profiles/assets/package.json")
const sourceDir = join(dirname(pkgJson), "dist", "profiles")
const destDir = join(here, "..", "public", "webxr-profiles")

async function main() {
  await rm(destDir, { recursive: true, force: true })
  await mkdir(destDir, { recursive: true })
  // Copy the whole tree but keep only .json files (drop .glb meshes).
  await cp(sourceDir, destDir, {
    recursive: true,
    filter: (src) => !src.endsWith(".glb"),
  })
  const count = (await readdir(destDir)).length
  console.log(`webxr profiles: copied ${count} entries -> ${destDir}`)
}

main().catch((err) => {
  console.error("failed to sync webxr profiles:", err)
  process.exit(1)
})
