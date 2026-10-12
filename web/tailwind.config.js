/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Consumer-health palette (soft, warm, Apple-Health-adjacent)
        canvas: '#FBFAF8',
        ink: '#1F2937',
        ink2: '#4B5563',
        heart: '#F43F5E',
        heart2: '#FB7185',
        pulse: '#14B8A6',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      fontVariantNumeric: {
        tabular: 'tabular-nums',
      },
    },
  },
  plugins: [],
}
