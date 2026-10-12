import { Link } from 'react-router-dom'
import Footer from '../components/Footer'
import HeartIcon from '../components/HeartIcon'

export default function Landing() {
  return (
    <div className="min-h-full flex flex-col">
      <main className="flex-1 flex flex-col items-center px-6 py-14 md:py-20">
        <div className="max-w-2xl text-center">
          <div className="inline-flex items-center gap-2 rounded-full bg-heart/10 text-heart px-4 py-1 text-sm font-medium mb-6">
            <HeartIcon bpm={72} size={14} />
            Remote photoplethysmography
          </div>

          <h1 className="text-4xl md:text-6xl font-semibold tracking-tight text-ink">
            Your heart rate &amp;
            <br />
            blood pressure,
            <br />
            from your camera.
          </h1>

          <p className="mt-6 text-lg text-ink2 max-w-xl mx-auto">
            Take a 30-second webcam recording to measure your heart rate, blood pressure, and respiratory vitals with research-grade optical rPPG.
          </p>

          <Link
            to="/measure"
            className="mt-10 inline-flex items-center justify-center rounded-full bg-heart px-8 py-3.5 text-white font-medium shadow-lg shadow-heart/25 hover:bg-heart2 transition gap-2"
          >
            Start 30s Scan →
          </Link>

          
        </div>
      </main>

      <Footer />
    </div>
  )
}