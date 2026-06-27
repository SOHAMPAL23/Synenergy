/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Outfit', 'Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        // Dark navy base
        surface: {
          50:  '#f0f4ff',
          100: '#e0e9ff',
          200: '#c7d6fe',
          300: '#a6bbfc',
          400: '#8296f8',
          500: '#6070f1',
          600: '#4a4ee4',
          700: '#3d3ec8',
          800: '#3235a3',
          900: '#2c3180',
          950: '#1a1d4e',
        },
        // Background layers
        bg: {
          primary:   '#0a0c1a',
          secondary: '#0f1228',
          card:      '#131629',
          border:    '#1e2340',
          hover:     '#1a1f3c',
        },
        // Cyan accent
        cyan: {
          400: '#22d3ee',
          500: '#06b6d4',
          600: '#0891b2',
        },
        // Neon electric blue  
        electric: {
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
        },
        // Anomaly reds
        danger: {
          400: '#f87171',
          500: '#ef4444',
          600: '#dc2626',
        },
        // Success green
        success: {
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
        },
        // Warning amber
        warning: {
          400: '#fbbf24',
          500: '#f59e0b',
          600: '#d97706',
        },
        // Purple accent
        violet: {
          400: '#a78bfa',
          500: '#8b5cf6',
          600: '#7c3aed',
        },
      },
      backgroundImage: {
        'grid-pattern': "linear-gradient(rgba(99,102,241,0.05) 1px, transparent 1px), linear-gradient(to right, rgba(99,102,241,0.05) 1px, transparent 1px)",
        'hero-gradient': 'linear-gradient(135deg, #0a0c1a 0%, #0f1228 50%, #131629 100%)',
        'card-gradient': 'linear-gradient(135deg, rgba(19,22,41,0.9) 0%, rgba(15,18,40,0.7) 100%)',
        'electric-gradient': 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
        'cyan-gradient': 'linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)',
        'success-gradient': 'linear-gradient(135deg, #22c55e 0%, #06b6d4 100%)',
        'danger-gradient': 'linear-gradient(135deg, #ef4444 0%, #f59e0b 100%)',
      },
      boxShadow: {
        'glow-blue':  '0 0 20px rgba(59,130,246,0.3), 0 0 40px rgba(59,130,246,0.1)',
        'glow-cyan':  '0 0 20px rgba(6,182,212,0.3), 0 0 40px rgba(6,182,212,0.1)',
        'glow-green': '0 0 20px rgba(34,197,94,0.3), 0 0 40px rgba(34,197,94,0.1)',
        'glow-red':   '0 0 20px rgba(239,68,68,0.3),  0 0 40px rgba(239,68,68,0.1)',
        'card':       '0 4px 24px rgba(0,0,0,0.4), 0 1px 4px rgba(0,0,0,0.3)',
        'card-hover': '0 8px 40px rgba(0,0,0,0.5), 0 2px 8px rgba(0,0,0,0.4)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in':    'fadeIn 0.5s ease-out',
        'slide-up':   'slideUp 0.4s ease-out',
        'glow':       'glow 2s ease-in-out infinite alternate',
        'float':      'float 6s ease-in-out infinite',
      },
      keyframes: {
        fadeIn:  { from: { opacity: '0' }, to: { opacity: '1' } },
        slideUp: { from: { opacity: '0', transform: 'translateY(16px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
        glow:    { from: { boxShadow: '0 0 10px rgba(59,130,246,0.2)' }, to: { boxShadow: '0 0 30px rgba(59,130,246,0.5)' } },
        float:   { '0%, 100%': { transform: 'translateY(0px)' }, '50%': { transform: 'translateY(-10px)' } },
      },
      backdropBlur: { xs: '2px' },
      borderRadius: { xl: '0.75rem', '2xl': '1rem', '3xl': '1.5rem' },
    },
  },
  plugins: [],
}
