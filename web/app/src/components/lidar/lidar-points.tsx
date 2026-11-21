/**
 * Three.js pieces shared by the lidar views: the live point cloud and a
 * marker for the lidar itself. Both draw in Jelly's base frame (Z up, floor
 * at z = 0, +X forward); the caller places that frame in its scene.
 */
import { useEffect, useMemo, useRef } from "react"
import { useFrame, useThree } from "@react-three/fiber"
import { BufferAttribute, BufferGeometry, CanvasTexture, MathUtils, type Texture } from "three"
import { type LidarCloud, type LidarMount, type LidarSettings } from "@/lib/lidar"

let disc: Texture | null = null

/** A soft-edged white disc, so points draw round instead of square. */
function discTexture(): Texture {
  if (disc) return disc
  const size = 64
  const canvas = document.createElement("canvas")
  canvas.width = canvas.height = size
  const ctx = canvas.getContext("2d")!
  ctx.beginPath()
  ctx.arc(size / 2, size / 2, size / 2 - 1, 0, Math.PI * 2)
  ctx.fillStyle = "#ffffff"
  ctx.fill()
  disc = new CanvasTexture(canvas)
  return disc
}

/**
 * The live scans, redrawn whenever a frame arrives or a setting changes.
 * With `mapEpoch`, they are drawn in that live map's frame (see
 * `LidarCloud.fill`).
 */
export function LidarPoints({
  cloud,
  settings,
  mapEpoch = null,
}: {
  cloud: LidarCloud
  settings: LidarSettings
  mapEpoch?: number | null
}) {
  const invalidate = useThree((s) => s.invalidate)
  const geometry = useMemo(() => new BufferGeometry(), [])
  const texture = useMemo(() => discTexture(), [])
  const buffers = useRef<{ positions: Float32Array; colors: Float32Array }>({
    positions: new Float32Array(0),
    colors: new Float32Array(0),
  })
  const drawn = useRef({
    revision: -1,
    settings: null as LidarSettings | null,
    mapEpoch: null as number | null,
  })

  // Views that render on demand redraw when a frame arrives.
  useEffect(() => cloud.subscribe(() => invalidate()), [cloud, invalidate])

  useEffect(() => () => geometry.dispose(), [geometry])

  useFrame(() => {
    const last = drawn.current
    if (
      last.revision === cloud.revision &&
      last.settings === settings &&
      last.mapEpoch === mapEpoch
    )
      return
    last.revision = cloud.revision
    last.settings = settings
    last.mapEpoch = mapEpoch
    const prev = buffers.current
    const out = cloud.fill(settings, prev.positions, prev.colors, mapEpoch)
    if (out.positions !== prev.positions || !geometry.getAttribute("position")) {
      buffers.current = { positions: out.positions, colors: out.colors }
      geometry.setAttribute("position", new BufferAttribute(out.positions, 3))
      geometry.setAttribute("color", new BufferAttribute(out.colors, 3))
    } else {
      geometry.getAttribute("position").needsUpdate = true
      geometry.getAttribute("color").needsUpdate = true
    }
    geometry.setDrawRange(0, out.count)
  })

  return (
    <points geometry={geometry} frustumCulled={false}>
      <pointsMaterial
        vertexColors
        sizeAttenuation
        map={texture}
        alphaTest={0.5}
        size={settings.pointSize / 100}
      />
    </points>
  )
}

/** The lidar puck at its mount, so the operator sees where the scan comes from. */
export function LidarMarker({ mount }: { mount: LidarMount }) {
  return (
    <group
      position={[mount.x, mount.y, mount.z]}
      rotation={[
        MathUtils.degToRad(mount.roll),
        MathUtils.degToRad(mount.pitch),
        MathUtils.degToRad(mount.yaw),
        "ZYX",
      ]}
    >
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.033, 0.033, 0.06, 24]} />
        <meshStandardMaterial color="#3a3f46" roughness={0.5} />
      </mesh>
      {/* Forward tick: the lidar's +X. */}
      <mesh position={[0.05, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[0.012, 0.03, 12]} />
        <meshBasicMaterial color="#ef4444" />
      </mesh>
    </group>
  )
}
