import { describe, it, expect } from 'vitest'
import {
  designButterBandpass1,
  estimateBpm,
  filtfilt,
  smoothPriorDetrend,
} from './signalProcessing'

describe('smoothPriorDetrend', () => {
  it('removes a slow quadratic baseline while preserving the cardiac sinusoid', () => {
    const n = 240
    const fps = 30
    const data = new Float64Array(n)
    const hr = 72 / 60 // Hz
    for (let i = 0; i < n; i++) {
      const t = i / fps
      // Pulse + quadratic baseline (e.g. slow auto-exposure drift).
      data[i] = Math.sin(2 * Math.PI * hr * t) + 0.5 * t * t
    }
    const out = smoothPriorDetrend(data, 100)
    // Detrended signal should be near zero-mean.
    let mean = 0
    for (let i = 0; i < n; i++) mean += out[i]
    mean /= n
    expect(Math.abs(mean)).toBeLessThan(0.15)
    // And should still contain the 72 BPM peak — confirm via FFT.
    const ts = new Float64Array(n)
    for (let i = 0; i < n; i++) ts[i] = i / fps
    const est = estimateBpm(out, ts, 2048)
    expect(est.bpm).toBeGreaterThan(70)
    expect(est.bpm).toBeLessThan(74)
  })

  it('preserves a pure cardiac-band sinusoid in the interior (away from edges)', () => {
    // Tarvainen detrend has mild edge transients because the smoothness
    // prior couples the endpoints. The interior should be essentially
    // unchanged for a signal well above the cutoff (λ=100 at N=240 cuts
    // frequencies below ~0.2 Hz; a 1.2 Hz sinusoid is far above).
    const n = 240
    const fps = 30
    const data = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      data[i] = Math.sin((2 * Math.PI * 1.2 * i) / fps)
    }
    const out = smoothPriorDetrend(data)
    let maxDelta = 0
    for (let i = 30; i < n - 30; i++) {
      maxDelta = Math.max(maxDelta, Math.abs(out[i] - data[i]))
    }
    expect(maxDelta).toBeLessThan(0.05)
  })

  it('handles short inputs gracefully', () => {
    const out = smoothPriorDetrend(new Float64Array([1, 2]))
    expect(out.length).toBe(2)
  })
})

describe('designButterBandpass1 + filtfilt', () => {
  const fps = 30

  it('attenuates out-of-band frequencies and preserves in-band ones', () => {
    const { b, a } = designButterBandpass1(0.75, 2.5, fps)
    const n = 600 // 20 seconds — plenty of cycles
    const inBand = new Float64Array(n) // 1.2 Hz = 72 BPM, in the pass band
    const outBand = new Float64Array(n) // 0.2 Hz = 12 BPM, well below
    for (let i = 0; i < n; i++) {
      const t = i / fps
      inBand[i] = Math.sin(2 * Math.PI * 1.2 * t)
      outBand[i] = Math.sin(2 * Math.PI * 0.2 * t)
    }
    const inFiltered = filtfilt(b, a, inBand)
    const outFiltered = filtfilt(b, a, outBand)
    // Measure RMS of the last half of each (skipping warm-up).
    const rms = (v: Float64Array) => {
      const start = Math.floor(v.length * 0.5)
      let s = 0
      let c = 0
      for (let i = start; i < v.length; i++) {
        s += v[i] * v[i]
        c++
      }
      return Math.sqrt(s / c)
    }
    // In-band signal should survive largely intact.
    expect(rms(inFiltered)).toBeGreaterThan(0.4)
    // Out-of-band signal should be strongly attenuated.
    expect(rms(outFiltered)).toBeLessThan(0.3)
  })

  it('is zero-phase (a cosine pulse stays centered)', () => {
    const { b, a } = designButterBandpass1(0.75, 2.5, fps)
    const n = 300
    const signal = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      signal[i] = Math.sin((2 * Math.PI * 1.2 * i) / fps)
    }
    const filtered = filtfilt(b, a, signal)
    // Find zero crossings in the pre-filter signal and the filtered one; for
    // a zero-phase filter they should coincide to within a sample. We only
    // check the central portion to skip boundary transients.
    const crossings = (v: Float64Array) => {
      const out: number[] = []
      for (let i = 60; i < v.length - 60; i++) {
        if (v[i - 1] < 0 && v[i] >= 0) out.push(i)
      }
      return out
    }
    const a1 = crossings(signal)
    const a2 = crossings(filtered)
    expect(a2.length).toBeGreaterThan(0)
    for (let i = 0; i < Math.min(a1.length, a2.length); i++) {
      expect(Math.abs(a1[i] - a2[i])).toBeLessThanOrEqual(2)
    }
  })
})

describe('chromTransform sliding-window form', () => {
  it('recovers a synthetic 72 BPM', async () => {
    const { chromTransform } = await import('./signalProcessing')
    const fps = 30
    const L = 240 // 8 seconds, ample for several 1.6 s windows
    const r = new Float64Array(L)
    const g = new Float64Array(L)
    const b = new Float64Array(L)
    const ts = new Float64Array(L)
    const hz = 72 / 60
    const dR = 0.33
    const dG = 0.77
    const dB = 0.53
    for (let i = 0; i < L; i++) {
      ts[i] = i / fps
      const pulse = 0.01 * Math.sin(2 * Math.PI * hz * ts[i])
      r[i] = 180 * (1 + dR * pulse)
      g[i] = 130 * (1 + dG * pulse)
      b[i] = 110 * (1 + dB * pulse)
    }
    const bvp = chromTransform(r, g, b, fps)
    // Trim the first ~2 seconds of warm-up / leading overlap-add zeros.
    const trimmed = bvp.slice(60)
    const tsTrim = ts.slice(60)
    const est = estimateBpm(trimmed, tsTrim, 2048)
    expect(est.bpm).toBeGreaterThan(70)
    expect(est.bpm).toBeLessThan(74)
  })
})
