import { describe, it, expect } from 'vitest'
import {
  detrend,
  estimateBloodPressure,
  estimateBpm,
  hammingWindow,
  l2Norm,
} from './signalProcessing'

describe('detrend', () => {
  it('removes a linear trend', () => {
    const n = 100
    const a = new Float64Array(n)
    for (let i = 0; i < n; i++) a[i] = 3 + 0.5 * i // pure line
    const out = detrend(a)
    for (let i = 0; i < n; i++) expect(out[i]).toBeCloseTo(0, 9)
  })

  it('keeps a sinusoid zero-mean', () => {
    const n = 128
    const a = new Float64Array(n)
    for (let i = 0; i < n; i++) a[i] = Math.sin((2 * Math.PI * i) / n)
    const out = detrend(a)
    let sum = 0
    for (let i = 0; i < n; i++) sum += out[i]
    expect(sum / n).toBeCloseTo(0, 8)
  })
})

describe('hammingWindow', () => {
  it('matches numpy.hamming at endpoints and midpoint', () => {
    const w = hammingWindow(100)
    expect(w[0]).toBeCloseTo(0.08, 5)
    expect(w[w.length - 1]).toBeCloseTo(0.08, 5)
    expect(w[49]).toBeCloseTo(1, 1) // ~1 at the center
  })
})

describe('estimateBpm', () => {
  const nFft = 1024
  const fps = 30
  const L = 100

  const synth = (bpm: number): { buf: Float64Array; ts: Float64Array } => {
    const hz = bpm / 60
    const buf = new Float64Array(L)
    const ts = new Float64Array(L)
    for (let i = 0; i < L; i++) {
      ts[i] = i / fps
      buf[i] = Math.sin(2 * Math.PI * hz * ts[i])
    }
    return { buf, ts }
  }

  it('detects a 75 BPM sinusoid within one zero-padded bin (~1.8 BPM)', () => {
    const { buf, ts } = synth(75)
    const est = estimateBpm(buf, ts, nFft)
    expect(est.bpm).toBeGreaterThan(73)
    expect(est.bpm).toBeLessThan(77)
    expect(est.fps).toBeCloseTo(fps, 1)
    expect(est.snr).toBeGreaterThan(5) // clean sinusoid => high SNR
  })

  it('detects a 63 BPM sinusoid — the case that used to snap to 63 via coarse bin', () => {
    const { buf, ts } = synth(63)
    const est = estimateBpm(buf, ts, nFft)
    expect(est.bpm).toBeGreaterThan(61)
    expect(est.bpm).toBeLessThan(65)
  })

  it('SNR on pure noise is much lower than on a clean sinusoid', () => {
    // A 100-sample random buffer can occasionally rank a single bin up to
    // ~5x the median by chance, so we don't pin the threshold. The real
    // guarantee: SNR on noise is dramatically lower than on a signal.
    const buf = new Float64Array(L)
    const ts = new Float64Array(L)
    let seed = 42
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0
      return seed / 0xffffffff
    }
    for (let i = 0; i < L; i++) {
      ts[i] = i / fps
      buf[i] = rand() - 0.5
    }
    const noiseEst = estimateBpm(buf, ts, nFft)
    const signalEst = estimateBpm(synth(75).buf, synth(75).ts, nFft)
    expect(signalEst.snr).toBeGreaterThan(noiseEst.snr * 2)
  })

  it('fps correction: N-1 denominator, not N', () => {
    // If we used N/T the fps would be ~30.3, not 30. Confirm we read the
    // (N-1)/T convention that Phase 2 corrected the Python code to.
    const { buf, ts } = synth(75)
    const est = estimateBpm(buf, ts, nFft)
    expect(est.fps).toBeGreaterThan(29.9)
    expect(est.fps).toBeLessThan(30.1)
  })
})

describe('l2Norm', () => {
  it('returns 5 for [3, 4]', () => {
    expect(l2Norm(new Float64Array([3, 4]))).toBeCloseTo(5, 9)
  })
})

describe('estimateBpm with prior bias', () => {
  const fps = 30
  const L = 100

  // Build a signal with two peaks: a real pulse at 75 BPM and a slightly
  // stronger spurious peak at 110 BPM. Unbiased argmax picks 110; the
  // prior bias centered at 75 should pull us back to 75.
  const twoPeaks = (): { buf: Float64Array; ts: Float64Array } => {
    const buf = new Float64Array(L)
    const ts = new Float64Array(L)
    for (let i = 0; i < L; i++) {
      ts[i] = i / fps
      // Spurious peak is ~20% stronger than the real one.
      buf[i] =
        Math.sin(2 * Math.PI * 1.25 * ts[i]) + // 75 BPM
        1.2 * Math.sin(2 * Math.PI * 1.833 * ts[i]) // 110 BPM
    }
    return { buf, ts }
  }

  it('without a prior, lands on the stronger spurious peak', () => {
    const { buf, ts } = twoPeaks()
    const est = estimateBpm(buf, ts, 1024)
    expect(est.bpm).toBeGreaterThan(105)
    expect(est.bpm).toBeLessThan(115)
  })

  it('with a prior at 75 and sigma 12, snaps back to the real peak', () => {
    const { buf, ts } = twoPeaks()
    // Wide search window so both peaks are visible to the scorer; the
    // Gaussian bias is what moves the argmax, not the window clipping.
    const est = estimateBpm(buf, ts, 1024, 45, 150, 75, 50, 12)
    expect(est.bpm).toBeGreaterThan(72)
    expect(est.bpm).toBeLessThan(78)
  })
})

describe('estimateBloodPressure', () => {
  const fps = 30
  const L = 120

  it('calculates plausible resting blood pressure for a 72 BPM pulse', () => {
    const buf = new Float64Array(L)
    for (let i = 0; i < L; i++) {
      const t = i / fps
      buf[i] = Math.sin(2 * Math.PI * 1.2 * t)
    }
    const bp = estimateBloodPressure(buf, 72, fps)
    expect(bp.systolic).not.toBeNull()
    expect(bp.diastolic).not.toBeNull()
    expect(bp.systolic!).toBeGreaterThanOrEqual(105)
    expect(bp.systolic!).toBeLessThanOrEqual(130)
    expect(bp.diastolic!).toBeGreaterThanOrEqual(65)
    expect(bp.diastolic!).toBeLessThanOrEqual(85)
    expect(bp.pulsePressure!).toBeGreaterThanOrEqual(30)
    expect(bp.category).toBeDefined()
  })

  it('returns invalid for low sample count or out of bound HR', () => {
    const shortBuf = new Float64Array(20)
    const bp = estimateBloodPressure(shortBuf, 72, fps)
    expect(bp.systolic).toBeNull()
    expect(bp.diastolic).toBeNull()
    expect(bp.category).toBe('Unknown')
  })
})
