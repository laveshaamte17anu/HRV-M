interface Props {
  /** BPM to pace the pulse animation. 0 disables it. */
  bpm: number
  size?: number
  className?: string
}

// Animated heart SVG. We drive the pulse cadence via inline CSS
// animation-duration so the beat matches the measured BPM: each "pulse"
// keyframe takes 60/bpm seconds. Below a valid reading we just hold a
// soft 1.2s pulse so the icon still feels alive.
export default function HeartIcon({ bpm, size = 20, className = '' }: Props) {
  const seconds = bpm > 40 && bpm < 200 ? 60 / bpm : 1.2
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      className={className}
      style={{
        display: 'inline-block',
        animationName: 'heart-pulse',
        animationDuration: `${seconds}s`,
        animationIterationCount: 'infinite',
        animationTimingFunction: 'cubic-bezier(0.4, 0, 0.6, 1)',
        transformOrigin: 'center',
      }}
    >
      <path
        d="M12 21s-7-4.35-7-10a4.5 4.5 0 0 1 8-2.8A4.5 4.5 0 0 1 19 11c0 5.65-7 10-7 10z"
        fill="currentColor"
      />
    </svg>
  )
}
