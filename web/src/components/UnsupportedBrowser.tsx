import Footer from './Footer'

interface Props {
  missing: string[]
}

export default function UnsupportedBrowser({ missing }: Props) {
  const isHttp = typeof window !== 'undefined' && window.location.protocol === 'http:'
  const isNotLocalhost =
    typeof window !== 'undefined' &&
    window.location.hostname !== 'localhost' &&
    window.location.hostname !== '127.0.0.1'
  const isSecureContextIssue = isHttp && isNotLocalhost

  const httpsUrl =
    typeof window !== 'undefined'
      ? `https://${window.location.host}${window.location.pathname}${window.location.search}`
      : ''

  return (
    <div className="min-h-full flex flex-col">
      <main className="flex-1 flex flex-col items-center justify-center px-6 py-16 text-center">
        <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-ink">
          {isSecureContextIssue ? 'HTTPS Connection Required' : 'Browser not supported'}
        </h1>

        {isSecureContextIssue ? (
          <div className="mt-4 max-w-xl text-ink2 space-y-4">
            <p>
              Mobile browsers (Chrome / Safari) strictly block camera access over plain HTTP connections.
              Please open this site using <strong>HTTPS</strong>:
            </p>
            <div className="pt-2">
              <a
                href={httpsUrl}
                className="inline-flex items-center justify-center rounded-full bg-heart px-8 py-3 text-white font-medium shadow-lg hover:bg-heart2 transition"
              >
                Open with HTTPS →
              </a>
            </div>
            <p className="text-xs text-ink2/80 pt-2">
              Note: If your browser displays &ldquo;Your connection is not private&rdquo;, tap <strong>Advanced</strong> &rarr; <strong>Proceed</strong>.
            </p>
          </div>
        ) : (
          <>
            <p className="mt-4 max-w-xl text-ink2">
              cardiocam needs a modern browser with camera access and GPU
              compute. Your current browser is missing:
            </p>
            <ul className="mt-4 text-ink font-medium">
              {missing.map((m) => (
                <li key={m}>· {m}</li>
              ))}
            </ul>
            <p className="mt-8 max-w-md text-sm text-ink2">
              Try a recent version of Chrome, Safari, Firefox, or Edge on a
              laptop or phone with a front-facing camera.
            </p>
          </>
        )}
      </main>
      <Footer />
    </div>
  )
}
