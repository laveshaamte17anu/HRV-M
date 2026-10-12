import { describe, it, expect } from 'vitest'
import { LandmarkFilter } from './landmarkFilter'
import type { NormalizedLandmark } from '@mediapipe/tasks-vision'

function pt(x: number, y: number): NormalizedLandmark {
  return { x, y, z: 0 } as NormalizedLandmark
}

describe('LandmarkFilter', () => {
  it('first frame passes through unchanged', () => {
    const f = new LandmarkFilter()
    const lms = [pt(0.5, 0.5), pt(0.1, 0.9)]
    const out = f.stabilize(lms, 0)
    expect(out[0].x).toBe(0.5)
    expect(out[1].y).toBe(0.9)
  })

  it('attenuates single-frame jitter by ~alpha', () => {
    // With alpha=0.35, a ±X jitter on frame 2 is followed by a correction of
    // 0.35·X — so after one frame the stabilized position has moved 35% of
    // the way from the predicted position (which equals frame 1's position,
    // since velocity is still zero) toward the noisy measurement.
    const f = new LandmarkFilter(0.35, 0.02)
    const frame1 = [pt(0.5, 0.5)]
    f.stabilize(frame1, 0)
    const frame2 = [pt(0.6, 0.5)] // +0.1 jitter on x
    const out = f.stabilize(frame2, 33)
    // Filtered output should be around 0.5 + 0.35 * 0.1 = 0.535.
    expect(out[0].x).toBeGreaterThan(0.5)
    expect(out[0].x).toBeLessThan(0.56)
  })

  it('tracks sustained motion (constant velocity)', () => {
    // Simulate a constant-velocity pan: x moves by +0.01 per frame.
    const f = new LandmarkFilter(0.35, 0.02)
    let t = 0
    const dt = 33
    let x = 0.5
    for (let i = 0; i < 120; i++) {
      f.stabilize([pt(x, 0.5)], t)
      t += dt
      x += 0.01
    }
    // After ~4 seconds of steady motion, the filter should have caught up
    // to within a small bias of the ground truth.
    const out = f.stabilize([pt(x, 0.5)], t)
    expect(out[0].x).toBeGreaterThan(x - 0.05)
    expect(out[0].x).toBeLessThan(x + 0.05)
  })

  it('reset restores initial state', () => {
    const f = new LandmarkFilter()
    f.stabilize([pt(0.1, 0.1)], 0)
    f.stabilize([pt(0.9, 0.9)], 33)
    f.reset()
    // After reset, first frame again passes through unchanged.
    const out = f.stabilize([pt(0.5, 0.5)], 0)
    expect(out[0].x).toBe(0.5)
    expect(out[0].y).toBe(0.5)
  })
})
