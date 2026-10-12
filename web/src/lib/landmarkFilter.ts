import type { NormalizedLandmark } from '@mediapipe/tasks-vision'

// Alpha-Beta filter over the full face-mesh landmark array.
//
// MediaPipe's landmarks wobble by ~1-2 pixels frame-to-frame even on a
// perfectly still face — that wobble propagates into our polygon ROIs,
// slightly shifting which pixels we average. Since the rPPG signal is
// a very subtle color change, ROI jitter shows up as noise in the BVP.
//
// Alpha-Beta is a simplified 2-state constant-velocity Kalman. Low
// alpha/beta absorb jitter while letting sustained motion through (a
// real head turn doesn't get smoothed away, because velocity builds
// up and position tracks). This mirrors the technique described in
// Álvarez Casado et al. 2025 (arXiv:2508.18787v1).
//
// We filter the 2D projections only. Z is ignored (we don't use it).
export class LandmarkFilter {
  private readonly alpha: number
  private readonly beta: number
  private positions: Float32Array | null = null
  private velocities: Float32Array | null = null
  private lastTimeMs: number | null = null

  constructor(alpha = 0.35, beta = 0.02) {
    this.alpha = alpha
    this.beta = beta
  }

  reset() {
    this.positions = null
    this.velocities = null
    this.lastTimeMs = null
  }

  /** Filter one frame's landmarks. timestampMs is monotonic; dt is
   *  computed from it. Returns a NEW NormalizedLandmark[] with filtered
   *  x/y (and original z passed through untouched). */
  stabilize(
    landmarks: NormalizedLandmark[],
    timestampMs: number
  ): NormalizedLandmark[] {
    const n = landmarks.length
    if (
      this.positions === null ||
      this.positions.length !== n * 2 ||
      this.lastTimeMs === null
    ) {
      // First frame (or count changed): initialize to raw measurements.
      this.positions = new Float32Array(n * 2)
      this.velocities = new Float32Array(n * 2)
      for (let i = 0; i < n; i++) {
        this.positions[2 * i] = landmarks[i].x
        this.positions[2 * i + 1] = landmarks[i].y
      }
      this.lastTimeMs = timestampMs
      return landmarks
    }

    let dt = (timestampMs - this.lastTimeMs) / 1000
    // Clamp dt in case of a bad timestamp (e.g. backgrounded tab). Anything
    // outside [1/120, 1/5] seconds implies something is off; treat as 1/30.
    if (!Number.isFinite(dt) || dt < 1 / 120 || dt > 1 / 5) dt = 1 / 30
    this.lastTimeMs = timestampMs

    const pos = this.positions
    const vel = this.velocities as Float32Array
    const out: NormalizedLandmark[] = new Array(n)
    for (let i = 0; i < n; i++) {
      const ix = 2 * i
      const iy = ix + 1
      // Predict.
      const px = pos[ix] + vel[ix] * dt
      const py = pos[iy] + vel[iy] * dt
      // Residual against measurement.
      const rx = landmarks[i].x - px
      const ry = landmarks[i].y - py
      // Update.
      pos[ix] = px + this.alpha * rx
      pos[iy] = py + this.alpha * ry
      vel[ix] += (this.beta / dt) * rx
      vel[iy] += (this.beta / dt) * ry
      // Emit.
      out[i] = {
        x: pos[ix],
        y: pos[iy],
        z: landmarks[i].z,
      } as NormalizedLandmark
    }
    return out
  }
}
