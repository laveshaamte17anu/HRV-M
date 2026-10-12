import { describe, it, expect } from 'vitest'
import {
  chromTransform,
  estimateBpm,
  omitTransform,
  posTransform,
} from './signalProcessing'

/**
 * Synthesize a color-skin time series with a heart-rate modulation. The key
 * observation: the hemoglobin absorption spectrum makes the pulse-induced
 * change in green larger than in red or blue. Roughly, we pick per-channel
 * pulse amplitudes proportional to (R: 0.33, G: 0.77, B: 0.53) — the
 * commonly used rPPG pulse direction unit vector from de Haan & van Leest.
 *
 * A DC illumination "drift" is added the SAME to all channels — POS cancels
 * this because it lives in the mean-skin-tone direction. Green-channel-only
 * would see the drift.
 */
function synthRgbSignal(bpm: number, fps: number, L: number) {
  const r = new Float64Array(L)
  const g = new Float64Array(L)
  const b = new Float64Array(L)
  const ts = new Float64Array(L)
  const hz = bpm / 60
  // Baseline skin tone (arbitrary reasonable values).
  const rBase = 180
  const gBase = 130
  const bBase = 110
  // Pulse direction (hemoglobin absorption weights). Green dominates.
  const dR = 0.33
  const dG = 0.77
  const dB = 0.53
  for (let i = 0; i < L; i++) {
    ts[i] = i / fps
    // Pulse: 1% peak-to-peak modulation (realistic for rPPG).
    const pulse = 0.01 * Math.sin(2 * Math.PI * hz * ts[i])
    // Specular drift: slow linear ramp applied equally to all channels.
    const drift = 0.08 * (i / L)
    r[i] = rBase * (1 + dR * pulse + drift)
    g[i] = gBase * (1 + dG * pulse + drift)
    b[i] = bBase * (1 + dB * pulse + drift)
  }
  return { r, g, b, ts }
}

describe('posTransform', () => {
  const fps = 30
  const L = 200 // need > 48 samples (1.6*30) for the first POS output

  it('extracts a BVP with the synthetic heart-rate frequency', () => {
    const { r, g, b, ts } = synthRgbSignal(72, fps, L)
    const bvp = posTransform(r, g, b, fps)
    // First 48 samples are zero (warm-up); trim.
    const trimmed = bvp.slice(48)
    const tsTrim = ts.slice(48)
    const est = estimateBpm(trimmed, tsTrim, 1024)
    expect(est.bpm).toBeGreaterThan(70)
    expect(est.bpm).toBeLessThan(74)
    expect(est.snr).toBeGreaterThan(5)
  })

  it('survives a non-stationary illumination flicker that green-only fails on', () => {
    // Add a second sinusoid in the skin-tone direction at a different
    // frequency — e.g. a 1.7 Hz (~102 BPM) lighting flicker equal across
    // channels. Linear detrend cannot remove this; GREEN's FFT peak will
    // land on the flicker, while POS's orthogonal projection cancels it
    // because the perturbation is along the mean-skin-tone axis.
    const { r, g, b, ts } = synthRgbSignal(75, fps, L)
    const flickerHz = 1.7 // ~102 BPM, in-band so it competes with 75 BPM
    for (let i = 0; i < L; i++) {
      const flick = 0.03 * Math.sin(2 * Math.PI * flickerHz * ts[i])
      r[i] *= 1 + flick
      g[i] *= 1 + flick
      b[i] *= 1 + flick
    }
    const bvp = posTransform(r, g, b, fps).slice(48)
    const green = g.slice(48)
    const tsTrim = ts.slice(48)
    const posEst = estimateBpm(bvp, tsTrim, 1024)
    const greenEst = estimateBpm(green, tsTrim, 1024)
    // POS should land near the true 75 BPM; GREEN locks onto the ~102 BPM flicker.
    expect(Math.abs(posEst.bpm - 75)).toBeLessThan(3)
    expect(Math.abs(greenEst.bpm - 75)).toBeGreaterThan(Math.abs(posEst.bpm - 75))
  })

  it('returns an all-zero buffer when the input is shorter than the window', () => {
    const short = 20
    const r = new Float64Array(short).fill(180)
    const g = new Float64Array(short).fill(130)
    const b = new Float64Array(short).fill(110)
    const bvp = posTransform(r, g, b, fps)
    let sum = 0
    for (const v of bvp) sum += Math.abs(v)
    expect(sum).toBeCloseTo(0, 9)
  })

  it('throws when channel lengths disagree', () => {
    const r = new Float64Array(100)
    const g = new Float64Array(99)
    const b = new Float64Array(100)
    expect(() => posTransform(r, g, b, 30)).toThrow()
  })
})

describe('chromTransform', () => {
  const fps = 30
  const L = 200

  function synth(bpm: number) {
    const r = new Float64Array(L)
    const g = new Float64Array(L)
    const b = new Float64Array(L)
    const ts = new Float64Array(L)
    const hz = bpm / 60
    const rBase = 180
    const gBase = 130
    const bBase = 110
    // Hemoglobin weights same as POS synth — green-dominant pulse.
    const dR = 0.33
    const dG = 0.77
    const dB = 0.53
    for (let i = 0; i < L; i++) {
      ts[i] = i / fps
      const pulse = 0.01 * Math.sin(2 * Math.PI * hz * ts[i])
      r[i] = rBase * (1 + dR * pulse)
      g[i] = gBase * (1 + dG * pulse)
      b[i] = bBase * (1 + dB * pulse)
    }
    return { r, g, b, ts }
  }

  it('recovers a synthetic 72 BPM within one bin', () => {
    const { r, g, b, ts } = synth(72)
    const bvp = chromTransform(r, g, b)
    const est = estimateBpm(bvp, ts, 1024)
    expect(est.bpm).toBeGreaterThan(70)
    expect(est.bpm).toBeLessThan(74)
  })

  it('throws when channel lengths disagree', () => {
    expect(() =>
      chromTransform(new Float64Array(5), new Float64Array(4), new Float64Array(5))
    ).toThrow()
  })
})

describe('omitTransform', () => {
  const fps = 30
  const L = 200

  function synth(bpm: number) {
    const r = new Float64Array(L)
    const g = new Float64Array(L)
    const b = new Float64Array(L)
    const ts = new Float64Array(L)
    const hz = bpm / 60
    const rBase = 180
    const gBase = 130
    const bBase = 110
    const dR = 0.33
    const dG = 0.77
    const dB = 0.53
    for (let i = 0; i < L; i++) {
      ts[i] = i / fps
      const pulse = 0.01 * Math.sin(2 * Math.PI * hz * ts[i])
      r[i] = rBase * (1 + dR * pulse)
      g[i] = gBase * (1 + dG * pulse)
      b[i] = bBase * (1 + dB * pulse)
    }
    return { r, g, b, ts }
  }

  it('recovers a synthetic 72 BPM within one bin', () => {
    const { r, g, b, ts } = synth(72)
    const bvp = omitTransform(r, g, b)
    const est = estimateBpm(bvp, ts, 1024)
    expect(est.bpm).toBeGreaterThan(70)
    expect(est.bpm).toBeLessThan(74)
  })

  it('handles empty input gracefully', () => {
    const bvp = omitTransform(
      new Float64Array(0),
      new Float64Array(0),
      new Float64Array(0)
    )
    expect(bvp.length).toBe(0)
  })

  it('throws when channel lengths disagree', () => {
    expect(() =>
      omitTransform(
        new Float64Array(5),
        new Float64Array(4),
        new Float64Array(5)
      )
    ).toThrow()
  })
})
