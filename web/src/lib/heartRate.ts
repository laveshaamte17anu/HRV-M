import {
  chromTransform,
  estimateBloodPressure,
  estimateBpm,
  estimateRR,
  estimateSpO2,
  omitTransform,
  pickPeak,
  posTransform,
  type BloodPressureEstimate,
} from './signalProcessing'
// SpO2Estimate and RREstimate types are inferred from the function return
// types — no explicit import needed.
import type { CheekSample } from './roi'

export type BvpMethod =
  | 'GREEN'
  | 'POS'
  | 'CHROM'
  | 'OMIT'
  | 'POS+CHROM'
  | 'POS+GREEN'
  | 'CHROM+GREEN'
  | 'POS+CHROM+GREEN'

export interface HeartRateState {
  /** Valid instantaneous BPM (FFT peak in band). 0 until the SNR gate opens. */
  bpm: number
  /** Peak / in-band-median ratio. Below snrThreshold the reading is gated. */
  snr: number
  /** True when SNR + warmup both pass and the BPM should be trusted. */
  valid: boolean
  /** Current estimated capture sample rate in Hz. */
  fps: number
  /** 0-1 progress while the initial sample buffer is filling. */
  warmup: number
  /** Most recent BVP waveform (post-transform, detrended), for plotting. */
  signal: Float64Array
  /** Most recent in-band FFT power, for plotting. */
  fft: Float64Array
  /** Frequency axis (BPM) aligned with fft. */
  freqsBpm: Float64Array
  /** Smoothed BPM across a trailing window of valid estimates. */
  stableBpm: number
  /** True when stableBpm is fresh enough to display — survives brief
   *  SNR dips so the number doesn't flicker in and out. */
  stableReadable: boolean
  /** Frames since the last moment we had a valid reading. Used by the UI
   *  to age-out the stable reading after several seconds of signal loss. */
  framesSinceValid: number
  /** Which BVP extractor is active for this session. */
  method: BvpMethod

  // ── Respiratory rate ──
  /** Breaths per minute. 0 while acquiring. */
  rr: number
  /** SNR of the respiratory peak. */
  rrSnr: number
  /** True when the RR reading should be trusted. */
  rrValid: boolean

  // ── Blood Pressure ──
  /** Blood pressure estimate from pulse wave analysis. */
  bp: BloodPressureEstimate

  // ── SpO2 (experimental) ──
  /** Estimated SpO2 percentage. 0 while acquiring. */
  spo2: number
  /** True when SpO2 has enough pulsatile signal. */
  spo2Valid: boolean
  /** 'stable' | 'rising' | 'falling' over the last ~30 readings. */
  spo2Trend: 'stable' | 'rising' | 'falling'
}

export class HeartRateEstimator {
  // 8 seconds at 30 FPS. Longer than the original 100 (3.3s) on purpose:
  // FFT bin width is fps / nFft in Hz; the effective spectral resolution
  // that actually matters is bounded by the buffer length (inverse of
  // duration). 240 samples gives ~0.92 BPM bins vs the old 2.2 BPM
  // bins. Cost: 2× slower to re-acquire after signal loss, and the
  // displayed waveform wiggles more because we're drawing more history.
  private readonly bufferSize = 240
  private readonly nFft = 2048
  // Hysteresis thresholds. Enter valid when SNR rises above snrEnter;
  // exit only when it drops below snrExit for several consecutive frames.
  // The gap between enter/exit prevents flapping when the SNR hovers
  // near the threshold (blinks, small head motion, shadows).
  private readonly snrEnter = 4
  private readonly snrExit = 2.5
  private readonly warmupFrames = 10
  // Frames the SNR must stay below snrExit before we drop out of valid.
  // ~1 second at 30 fps — absorbs single-frame dips silently.
  private readonly exitDebounceFrames = 30
  // Keep the stable BPM readable for this many frames after the last
  // valid sample. ~5 seconds at 30 fps — long enough that a short blink
  // doesn't hide the reading, short enough that a genuinely lost signal
  // times out within a reasonable beat.
  private readonly staleReadableFrames = 150
  // Maximum per-frame BPM change we'll accept. Real cardiac activity
  // doesn't jump by more than ~5 BPM/sec; at 30 FPS a 20 BPM search
  // half-width gives plenty of headroom while still rejecting the
  // noise-peak jumps (50s → 170s) that FFT argmax can produce.
  private readonly maxDeltaBpm = 20
  // Soft Gaussian bias sigma — when two peaks within the prior window are
  // comparable in amplitude, this pulls the argmax toward the prior.
  // Gentle within ~10 BPM, strong beyond ~20 BPM.
  private readonly priorSigmaBpm = 12
  // Minimum number of valid BPMs we need accumulated before trusting the
  // stable median as an anchor. Before this many, the rate limiter does
  // nothing.
  private readonly minValidForAnchor = 8
  // Exponentially-weighted accumulated power spectrum. Each full-buffer
  // FFT contributes with decaying weight, so the spectrum converges on
  // the persistent peaks (your cardiac fundamental) while rejecting
  // single-frame noise bursts. Half-life equal to the buffer duration
  // (~8s) means the spectrum stays responsive to real HR changes.
  private readonly spectrumHalfLifeFrames = 240
  private accumSpectrum: Float64Array | null = null
  // EWMA smoothing on the emitted stableBpm. Half-life independent of
  // the spectrum accumulator — we want the displayed number to settle
  // faster than the spectrum so the user sees responsiveness.
  private readonly bpmEwmaHalfLifeFrames = 90 // 3s at 30 fps
  private ewmaBpm: number | null = null

  // Rolling RGB buffers (channel means over the combined ROIs) and their
  // wall-clock timestamps in seconds. Kept for the single-region fallback
  // path via pushRgb.
  private rBuf: number[] = []
  private gBuf: number[] = []
  private bBuf: number[] = []
  private timestamps: number[] = []

  // Per-region RGB buffers — populated only when pushRegional is used.
  // Forehead buffer holds the fallback averaged value (cheek mean) on
  // frames where the forehead polygon was empty, so the length always
  // matches the cheeks. A parallel mask tracks which frames had a real
  // forehead measurement so its BVP can be weighted by data availability.
  private perRegionActive = false
  private rBufL: number[] = []
  private gBufL: number[] = []
  private bBufL: number[] = []
  private rBufR: number[] = []
  private gBufR: number[] = []
  private bBufR: number[] = []
  private rBufF: number[] = []
  private gBufF: number[] = []
  private bBufF: number[] = []
  private foreheadMask: boolean[] = []

  private framesSinceFull = 0
  private validBpms: number[] = []
  private method: BvpMethod

  // Hysteresis state.
  private lastValid = false
  private framesBelowExit = 0
  private framesSinceValid = Infinity

  // SpO2 trend history — rolling window of recent SpO2 values for
  // trend detection (rising / falling / stable).
  private recentSpO2: number[] = []
  private readonly rrSnrThreshold = 2.5

  private state: HeartRateState

  constructor(method: BvpMethod = 'POS') {
    this.method = method
    this.state = this.makeInitialState()
  }

  /** Returns a read-only snapshot of the latest estimator state. */
  getState(): HeartRateState {
    return this.state
  }

  /** Current anchor BPM for the rate limiter. null until we have enough
   *  valid samples to form a trustworthy reference — cold-start searches
   *  the full band so we can find the HR from scratch. Once we've got a
   *  stable reference we prefer the median of recent valid BPMs. */
  private currentAnchor(): number | null {
    if (this.validBpms.length < this.minValidForAnchor) return null
    const sorted = [...this.validBpms].sort((a, b) => a - b)
    return sorted[Math.floor(sorted.length / 2)]
  }

  setMethod(method: BvpMethod) {
    if (method === this.method) return
    this.method = method
    this.reset()
  }

  reset() {
    this.rBuf = []
    this.gBuf = []
    this.bBuf = []
    this.timestamps = []
    this.rBufL = []
    this.gBufL = []
    this.bBufL = []
    this.rBufR = []
    this.gBufR = []
    this.bBufR = []
    this.rBufF = []
    this.gBufF = []
    this.bBufF = []
    this.foreheadMask = []
    this.perRegionActive = false
    this.framesSinceFull = 0
    this.validBpms = []
    this.lastValid = false
    this.framesBelowExit = 0
    this.framesSinceValid = Infinity
    this.accumSpectrum = null
    this.ewmaBpm = null
    this.recentSpO2 = []
    this.state = this.makeInitialState()
  }

  /**
   * Preferred entry point when per-region samples are available. Records
   * both the combined-mean stream (for compatibility + plotting) AND the
   * three per-region RGB streams, enabling SNR-weighted combination of
   * per-region BVPs on the full-buffer step.
   *
   * Forehead is optional: if the polygon falls off-frame, the caller's
   * sample.forehead is null and we fill the buffer with the cheek average
   * so the lengths stay aligned, flagging the frame as "no forehead"
   * in the parallel mask. Regions with insufficient valid frames in the
   * buffer automatically drop out of the weighted combination.
   */
  pushRegional(sample: CheekSample, tSeconds: number): HeartRateState {
    this.perRegionActive = true

    // Combined path: feed the averaged mean through the legacy buffer so
    // the hybrid methods (which expect a single RGB stream) keep working.
    // pushRgb handles outlier clamping and buffer bounding.
    const headState = this.pushRgb(sample.r, sample.g, sample.b, tSeconds)

    // Per-region streams: append, keep in lockstep with the combined
    // buffer (same trim rule). No outlier clamp per region — the combined
    // stream already absorbs glitches at the signal level.
    this.rBufL.push(sample.left.r)
    this.gBufL.push(sample.left.g)
    this.bBufL.push(sample.left.b)
    this.rBufR.push(sample.right.r)
    this.gBufR.push(sample.right.g)
    this.bBufR.push(sample.right.b)
    if (sample.forehead !== null) {
      this.rBufF.push(sample.forehead.r)
      this.gBufF.push(sample.forehead.g)
      this.bBufF.push(sample.forehead.b)
      this.foreheadMask.push(true)
    } else {
      this.rBufF.push(sample.r)
      this.gBufF.push(sample.g)
      this.bBufF.push(sample.b)
      this.foreheadMask.push(false)
    }
    while (this.rBufL.length > this.bufferSize) {
      this.rBufL.shift()
      this.gBufL.shift()
      this.bBufL.shift()
      this.rBufR.shift()
      this.gBufR.shift()
      this.bBufR.shift()
      this.rBufF.shift()
      this.gBufF.shift()
      this.bBufF.shift()
      this.foreheadMask.shift()
    }
    return headState
  }

  /**
   * Feed one frame's RGB channel means into the estimator. Returns the
   * updated state. Timestamp is seconds-since-anything monotonic
   * (e.g. performance.now() / 1000).
   */
  pushRgb(r: number, g: number, b: number, tSeconds: number): HeartRateState {
    if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) {
      return this.state
    }

    // Outlier clamp on the green channel (most sensitive to sudden skin
    // changes). Matches the Python process.py guard: once the buffer is
    // full, a new green value more than 10 off the running mean is
    // replaced with the last accepted sample so the buffer doesn't get
    // poisoned by e.g. a finger brushing the ROI.
    const L = this.gBuf.length
    if (L >= this.bufferSize) {
      let mean = 0
      for (let i = 0; i < L; i++) mean += this.gBuf[i]
      mean /= L
      if (Math.abs(g - mean) > 10) {
        r = this.rBuf[L - 1]
        g = this.gBuf[L - 1]
        b = this.bBuf[L - 1]
      }
    }

    this.rBuf.push(r)
    this.gBuf.push(g)
    this.bBuf.push(b)
    this.timestamps.push(tSeconds)

    // Keep the buffer bounded.
    if (this.gBuf.length > this.bufferSize) {
      this.rBuf.shift()
      this.gBuf.shift()
      this.bBuf.shift()
      this.timestamps.shift()
    }

    const len = this.gBuf.length
    const warmup = Math.min(1, len / this.bufferSize)

    if (len < this.bufferSize) {
      this.state = {
        ...this.state,
        warmup,
        valid: false,
        bpm: 0,
        snr: 0,
      }
      return this.state
    }

    // Full buffer: derive the BVP signal, then run the DSP pipeline.
    const rArr = new Float64Array(this.rBuf)
    const gArr = new Float64Array(this.gBuf)
    const bArr = new Float64Array(this.bBuf)
    const ts = new Float64Array(this.timestamps)

    // Estimate fps from the actual sample timestamps so POS uses the
    // same window length the downstream FFT assumes.
    const duration = ts[ts.length - 1] - ts[0]
    const fps = duration > 0 ? (len - 1) / duration : 30

    // Compute only the BVPs the selected method needs. Each extractor
    // produces signals on different amplitude scales (POS ~ 1e-3, CHROM ~ 1,
    // GREEN ~ 130), so when averaging we first z-score each component to
    // unit variance — otherwise the largest-magnitude component dominates
    // the sum, which would make e.g. POS+GREEN effectively just GREEN.
    const bvp = computeBvp(this.method, rArr, gArr, bArr, fps)

    // Run the FFT on the current buffer but don't yet pick a peak — we'll
    // combine this instantaneous spectrum with an exponentially-decayed
    // historical spectrum and pick from the accumulator. Passing no prior
    // here means estimateBpm returns the full in-band spectrum unbiased.
    let inst = estimateBpm(bvp, ts, this.nFft)

    // Per-region SNR-weighted combination. When pushRegional has been
    // fed frames, each of the three ROI buffers has the same length as
    // the combined buffer. We run the current BVP method on each region
    // separately, score each region's in-band FFT by its peak/median
    // SNR, and blend the three spectra weighted by SNR^2 (sharper
    // weighting than linear; a region with 2× the SNR contributes 4×).
    // A region with effectively zero valid frames (forehead hidden
    // behind hair/bangs) contributes 0 and effectively drops out.
    if (
      this.perRegionActive &&
      this.rBufL.length === len &&
      this.rBufR.length === len &&
      this.rBufF.length === len
    ) {
      const bvpL = computeBvp(
        this.method,
        new Float64Array(this.rBufL),
        new Float64Array(this.gBufL),
        new Float64Array(this.bBufL),
        fps
      )
      const bvpR = computeBvp(
        this.method,
        new Float64Array(this.rBufR),
        new Float64Array(this.gBufR),
        new Float64Array(this.bBufR),
        fps
      )
      // Only use the forehead region if enough frames in the buffer had
      // a real forehead measurement (>= 60% threshold). Otherwise hair
      // occlusion dominates and the forehead signal is noise.
      let foreheadCount = 0
      for (let i = 0; i < this.foreheadMask.length; i++) {
        if (this.foreheadMask[i]) foreheadCount++
      }
      const useForehead = foreheadCount / this.foreheadMask.length >= 0.6
      const bvpF = useForehead
        ? computeBvp(
            this.method,
            new Float64Array(this.rBufF),
            new Float64Array(this.gBufF),
            new Float64Array(this.bBufF),
            fps
          )
        : null

      const estL = estimateBpm(bvpL, ts, this.nFft)
      const estR = estimateBpm(bvpR, ts, this.nFft)
      const estF = bvpF ? estimateBpm(bvpF, ts, this.nFft) : null

      // SNR-squared weighting. Scale is invariant — we normalize below.
      const wL = estL.snr * estL.snr
      const wR = estR.snr * estR.snr
      const wF = estF ? estF.snr * estF.snr : 0
      const totalW = wL + wR + wF
      if (totalW > 0 && estL.fft.length === estR.fft.length) {
        const combined = new Float64Array(estL.fft.length)
        for (let i = 0; i < combined.length; i++) {
          let v = wL * estL.fft[i] + wR * estR.fft[i]
          if (estF) v += wF * estF.fft[i]
          combined[i] = v / totalW
        }
        // Keep the same freqs axis and fps from the first sub-estimate.
        inst = {
          bpm: estL.bpm, // will be overwritten by the accumulator peak pick
          snr: estL.snr, // ditto
          fft: combined,
          freqsBpm: estL.freqsBpm,
          fps: estL.fps,
        }
      }
    }

    // Update the running accumulated spectrum. Initialize on first call
    // (or after a reset / method change / band length change). Decay
    // weight: alpha_new = 1 - exp(-ln(2) / halfLife), so after halfLife
    // frames a sample's weight is 0.5.
    if (
      this.accumSpectrum === null ||
      this.accumSpectrum.length !== inst.fft.length
    ) {
      this.accumSpectrum = new Float64Array(inst.fft)
    } else {
      const alpha = 1 - Math.exp(-Math.LN2 / this.spectrumHalfLifeFrames)
      for (let i = 0; i < this.accumSpectrum.length; i++) {
        this.accumSpectrum[i] =
          (1 - alpha) * this.accumSpectrum[i] + alpha * inst.fft[i]
      }
    }

    // Pick the peak from the ACCUMULATED spectrum — the signal we actually
    // care about (cardiac fundamental) is persistent across frames while
    // noise is transient, so integrating over time amplifies it relative
    // to noise.
    const anchor = this.currentAnchor()
    const pick = pickPeak(
      this.accumSpectrum,
      inst.freqsBpm,
      anchor,
      this.maxDeltaBpm,
      anchor !== null ? this.priorSigmaBpm : null
    )

    // Emit the accumulated spectrum for plotting/validation so the user
    // sees the decision surface, not just the noisy instantaneous one.
    const est = {
      bpm: pick.bpm,
      snr: pick.snr,
      fft: this.accumSpectrum,
      freqsBpm: inst.freqsBpm,
      fps: inst.fps,
    }

    this.framesSinceFull++
    const pastWarmup = this.framesSinceFull >= this.warmupFrames

    // Hysteresis gate. pastWarmup is a hard prerequisite; once past it,
    // whether we emit valid depends on (a) the previous state, (b) where
    // SNR is relative to snrEnter / snrExit, and (c) how long it's been
    // below the exit threshold.
    let valid = this.lastValid
    if (!pastWarmup) {
      valid = false
      this.framesBelowExit = 0
    } else if (!this.lastValid) {
      // Currently invalid; only enter when SNR rises above the enter
      // threshold. Reset the below-exit counter so a fresh acquisition
      // starts from zero tolerance.
      if (est.snr >= this.snrEnter) {
        valid = true
        this.framesBelowExit = 0
      }
    } else {
      // Currently valid; exit only after SNR has been below the exit
      // threshold for enough consecutive frames to be confident the
      // signal actually dropped (vs a blink, brief shadow, etc).
      if (est.snr < this.snrExit) {
        this.framesBelowExit++
        if (this.framesBelowExit >= this.exitDebounceFrames) {
          valid = false
          this.framesBelowExit = 0
        }
      } else {
        // Any frame at or above the exit threshold resets the exit
        // debounce counter.
        this.framesBelowExit = 0
      }
    }

    if (valid) {
      this.validBpms.push(est.bpm)
      if (this.validBpms.length > this.bufferSize / 2) this.validBpms.shift()
      this.framesSinceValid = 0
    } else {
      this.framesSinceValid++
    }
    this.lastValid = valid

    // Raw stable: median of the recent valid BPMs (robust to single
    // outliers). Then smooth with an EWMA so moment-to-moment wobble
    // settles even more. EWMA runs only on frames where we produced a
    // valid reading — invalid frames don't decay the estimate.
    let stableRaw = 0
    if (this.validBpms.length >= 25) {
      const sorted = [...this.validBpms].sort((a, b) => a - b)
      stableRaw = sorted[Math.floor(sorted.length / 2)]
    }
    if (valid && stableRaw > 0) {
      if (this.ewmaBpm === null) {
        this.ewmaBpm = stableRaw
      } else {
        const alpha = 1 - Math.exp(-Math.LN2 / this.bpmEwmaHalfLifeFrames)
        this.ewmaBpm = (1 - alpha) * this.ewmaBpm + alpha * stableRaw
      }
    }
    const stable = this.ewmaBpm ?? stableRaw

    // The stable BPM is readable whenever we have a recent-enough stream
    // of valid samples — it survives momentary SNR dips so the UI
    // doesn't blink the number off and on every time you breathe.
    const stableReadable =
      stable > 0 && this.framesSinceValid <= this.staleReadableFrames

    // ── Respiratory rate ──
    // Same BVP signal, different bandpass band. RR is noisier than HR
    // because the signal is weaker and the spectral resolution at
    // ~0.3 Hz is coarse in an 8s window.
    const rrEst = estimateRR(bvp, ts, this.nFft)
    const rrValid = rrEst.snr >= this.rrSnrThreshold && rrEst.rr > 0

    // ── SpO2 (experimental) ──
    // Uses the red and blue channel buffers directly (not the BVP).
    const spo2Est = estimateSpO2(rArr, gArr, bArr, ts, this.recentSpO2)
    if (spo2Est.valid && spo2Est.spo2 > 0) {
      this.recentSpO2.push(spo2Est.spo2)
      if (this.recentSpO2.length > 60) this.recentSpO2.shift()
    }

    // ── Blood Pressure ──
    const effectiveHr = valid ? (stable > 0 ? stable : est.bpm) : null
    const bpEst = estimateBloodPressure(bvp, effectiveHr, est.fps)

    this.state = {
      bpm: est.bpm,
      snr: est.snr,
      valid,
      fps: est.fps,
      warmup: 1,
      signal: bvp,
      fft: est.fft,
      freqsBpm: est.freqsBpm,
      stableBpm: stable,
      stableReadable,
      framesSinceValid: this.framesSinceValid,
      method: this.method,
      rr: rrEst.rr,
      rrSnr: rrEst.snr,
      rrValid,
      bp: bpEst,
      spo2: spo2Est.spo2,
      spo2Valid: spo2Est.valid,
      spo2Trend: spo2Est.trend,
    }
    return this.state
  }

  /**
   * Backwards-compat thin shim — callers that only have a green-channel
   * mean can still push it. Fills R and B with the green value, which
   * makes POS collapse to a noisy near-zero signal. Useful for unit
   * tests that pre-date the RGB API; not for production callers.
   */
  pushSample(green: number, tSeconds: number): HeartRateState {
    return this.pushRgb(green, green, green, tSeconds)
  }

  private makeInitialState(): HeartRateState {
    return {
      bpm: 0,
      snr: 0,
      valid: false,
      fps: 0,
      warmup: 0,
      signal: new Float64Array(0),
      fft: new Float64Array(0),
      freqsBpm: new Float64Array(0),
      stableBpm: 0,
      stableReadable: false,
      framesSinceValid: Infinity,
      method: this.method,
      rr: 0,
      rrSnr: 0,
      rrValid: false,
      bp: {
        systolic: null,
        diastolic: null,
        map: null,
        pulsePressure: null,
        category: 'Unknown',
        categoryColor: 'gray',
        vascularStiffness: 'Unknown',
        confidence: 0,
      },
      spo2: 0,
      spo2Valid: false,
      spo2Trend: 'stable',
    }
  }
}

// Z-score a signal to zero mean, unit variance. If the series is
// essentially constant (std ≈ 0), return a zero-filled array rather than
// dividing by zero — averaging that into a hybrid just adds nothing.
function zscore(v: Float64Array): Float64Array {
  const n = v.length
  if (n === 0) return new Float64Array(0)
  let mean = 0
  for (let i = 0; i < n; i++) mean += v[i]
  mean /= n
  let varSum = 0
  for (let i = 0; i < n; i++) {
    const d = v[i] - mean
    varSum += d * d
  }
  const std = Math.sqrt(varSum / n)
  const out = new Float64Array(n)
  if (std < 1e-9) return out
  for (let i = 0; i < n; i++) out[i] = (v[i] - mean) / std
  return out
}

/** Sum two or three z-scored signals, pre-allocated output length = n. */
function meanOf(parts: Float64Array[]): Float64Array {
  const n = parts[0].length
  const out = new Float64Array(n)
  for (const p of parts) {
    for (let i = 0; i < n; i++) out[i] += p[i]
  }
  const k = parts.length
  for (let i = 0; i < n; i++) out[i] /= k
  return out
}

function computeBvp(
  method: BvpMethod,
  r: Float64Array,
  g: Float64Array,
  b: Float64Array,
  fps: number
): Float64Array {
  // Lazy cache so hybrids don't compute POS/CHROM twice. Only computed
  // for the components the selected method needs.
  let pos: Float64Array | null = null
  let chrom: Float64Array | null = null
  const getPos = () => pos ?? (pos = posTransform(r, g, b, fps))
  const getChrom = () => chrom ?? (chrom = chromTransform(r, g, b, fps))

  switch (method) {
    case 'POS':
      return getPos()
    case 'CHROM':
      return getChrom()
    case 'OMIT':
      return omitTransform(r, g, b)
    case 'GREEN':
      return g
    case 'POS+CHROM':
      return meanOf([zscore(getPos()), zscore(getChrom())])
    case 'POS+GREEN':
      return meanOf([zscore(getPos()), zscore(g)])
    case 'CHROM+GREEN':
      return meanOf([zscore(getChrom()), zscore(g)])
    case 'POS+CHROM+GREEN':
      return meanOf([zscore(getPos()), zscore(getChrom()), zscore(g)])
    default:
      return g
  }
}
