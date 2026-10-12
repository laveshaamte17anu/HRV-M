// Extract the mean R, G, B intensities from three skin patches
// (subject-left cheek, subject-right cheek, central forehead).
//
// The signal is the three color-channel means over ROI skin pixels, which is
// rotation-invariant — cropping an axis-aligned rectangle straight from the
// source frame gives a skin patch that the downstream BVP methods (GREEN,
// POS, CHROM) can all consume identically.
//
// Multi-region averaging halves MAE on the LGI-PPGI benchmark in the
// Álvarez Casado 2025 paper — bigger gain than any single algorithm
// change. We use polygon fills over MediaPipe Face Mesh landmarks rather
// than bounding rectangles to avoid catching non-skin pixels (eyebrows,
// lip edge, hairline) at extreme head angles.

import type { NormalizedLandmark } from '@mediapipe/tasks-vision'

// Cheek-apex polygons on the MediaPipe 478-point Face Mesh. These bound the
// zygomatic region — high, well clear of the nasolabial fold, the lips, and
// the nose — so the channel means sample clean cheek skin with no hair or
// lip edge contamination.
//
// Subject-left cheek (viewer's-right side of the image).
const LEFT_CHEEK_RING = [345, 352, 411, 425, 266, 371, 355]
// Subject-right cheek (viewer's-left side of the image).
const RIGHT_CHEEK_RING = [116, 123, 187, 205, 36, 142, 126]
// Glabella / lower-central-forehead patch. Bounded below by the
// inner-eyebrow landmarks (107 left, 336 right) and above by the
// mid-forehead row (66, 69, 151, 299, 296). Intentionally stays in the
// lower half of the forehead so the polygon never reaches the hairline,
// which varies by haircut and would contaminate the green mean with
// hair pixels for many users. The cardiocam ROI visibility overlay lets
// us verify this empirically per user.
const FOREHEAD_RING = [107, 66, 69, 151, 299, 296, 336, 9]

/** Per-ROI RGB triple (mean channel intensities over the polygon skin
 *  pixels, 0-255). */
export interface RgbMean {
  r: number
  g: number
  b: number
  /** Number of pixels contributing to the means. 0 if the polygon fell
   *  off the frame. */
  area: number
}

export interface CheekSample {
  /** Equal-weighted mean red across all available ROIs, 0-255. */
  r: number
  /** Equal-weighted mean green across all available ROIs, 0-255. */
  g: number
  /** Equal-weighted mean blue across all available ROIs, 0-255. */
  b: number
  /** Alias for g — keeps the old green-only pipeline trivially backwards compatible. */
  greenMean: number
  leftArea: number
  rightArea: number
  /** Number of pixels that contributed to the forehead mean. 0 if the
   *  forehead polygon fell outside the frame or degenerated to zero area. */
  foreheadArea: number
  /** Per-region means. Forehead is null when its polygon was empty. */
  left: RgbMean
  right: RgbMean
  forehead: RgbMean | null
}

/**
 * Sample the three skin ROIs (both cheeks + central forehead) from the
 * current video frame and return their equal-weighted per-channel means.
 * Returns null if BOTH cheek ROIs fail — forehead alone is not enough
 * because it's often occluded by hair/bangs on real users.
 *
 * Forehead is included when its polygon yields pixels; if the forehead
 * falls off-frame (e.g. phone held low) we still return a valid sample
 * using just the cheeks.
 */
export function sampleCheekRgb(
  landmarks: NormalizedLandmark[],
  ctx: CanvasRenderingContext2D,
  frameWidth: number,
  frameHeight: number
): CheekSample | null {
  const pointAt = (i: number) => ({
    x: landmarks[i].x * frameWidth,
    y: landmarks[i].y * frameHeight,
  })
  const leftPoly = LEFT_CHEEK_RING.map(pointAt)
  const rightPoly = RIGHT_CHEEK_RING.map(pointAt)
  const foreheadPoly = FOREHEAD_RING.map(pointAt)

  const left = meanRgbInPolygon(ctx, leftPoly, frameWidth, frameHeight)
  const right = meanRgbInPolygon(ctx, rightPoly, frameWidth, frameHeight)
  const forehead = meanRgbInPolygon(ctx, foreheadPoly, frameWidth, frameHeight)
  if (left === null || right === null) return null

  // Equal-weight average across available ROIs. Both cheeks are always
  // present once we're past the null check above; forehead is optional.
  const parts = [left, right]
  if (forehead !== null) parts.push(forehead)
  let r = 0
  let g = 0
  let b = 0
  for (const p of parts) {
    r += p.r
    g += p.g
    b += p.b
  }
  r /= parts.length
  g /= parts.length
  b /= parts.length

  return {
    r,
    g,
    b,
    greenMean: g,
    leftArea: left.area,
    rightArea: right.area,
    foreheadArea: forehead?.area ?? 0,
    left,
    right,
    forehead,
  }
}

/** Old name kept as a thin alias for any external callers. */
export const sampleCheekGreen = sampleCheekRgb

function meanRgbInPolygon(
  ctx: CanvasRenderingContext2D,
  poly: { x: number; y: number }[],
  frameWidth: number,
  frameHeight: number
): { r: number; g: number; b: number; area: number } | null {
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity
  for (const p of poly) {
    if (p.x < minX) minX = p.x
    if (p.x > maxX) maxX = p.x
    if (p.y < minY) minY = p.y
    if (p.y > maxY) maxY = p.y
  }
  minX = Math.max(0, Math.floor(minX))
  minY = Math.max(0, Math.floor(minY))
  maxX = Math.min(frameWidth - 1, Math.ceil(maxX))
  maxY = Math.min(frameHeight - 1, Math.ceil(maxY))
  const w = maxX - minX + 1
  const h = maxY - minY + 1
  if (w <= 0 || h <= 0) return null

  let imageData: ImageData
  try {
    imageData = ctx.getImageData(minX, minY, w, h)
  } catch {
    return null
  }
  const buf = imageData.data

  let rSum = 0
  let gSum = 0
  let bSum = 0
  let count = 0
  for (let py = 0; py < h; py++) {
    const worldY = minY + py + 0.5
    for (let px = 0; px < w; px++) {
      const worldX = minX + px + 0.5
      if (pointInPolygon(worldX, worldY, poly)) {
        const i = (py * w + px) * 4
        rSum += buf[i]
        gSum += buf[i + 1]
        bSum += buf[i + 2]
        count++
      }
    }
  }
  if (count === 0) return null
  return { r: rSum / count, g: gSum / count, b: bSum / count, area: count }
}

function pointInPolygon(
  x: number,
  y: number,
  poly: { x: number; y: number }[]
): boolean {
  let inside = false
  const n = poly.length
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = poly[i].x
    const yi = poly[i].y
    const xj = poly[j].x
    const yj = poly[j].y
    const intersect =
      yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi
    if (intersect) inside = !inside
  }
  return inside
}

export { LEFT_CHEEK_RING, RIGHT_CHEEK_RING, FOREHEAD_RING }
