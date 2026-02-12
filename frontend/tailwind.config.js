/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                glass: {
                    50: 'rgba(255, 255, 255, 0.03)',
                    100: 'rgba(255, 255, 255, 0.06)',
                    200: 'rgba(255, 255, 255, 0.1)',
                    300: 'rgba(255, 255, 255, 0.15)',
                    400: 'rgba(255, 255, 255, 0.2)',
                },
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
            },
            animation: {
                'fade-in': 'fadeIn 0.6s ease-out',
                'slide-up': 'slideUp 0.5s ease-out',
                'pulse-slow': 'pulseSlow 4s ease-in-out infinite',
                'float': 'float 6s ease-in-out infinite',
                'glow-pulse': 'glowPulse 3s ease-in-out infinite',
            },
            keyframes: {
                fadeIn: {
                    '0%': { opacity: '0' },
                    '100%': { opacity: '1' },
                },
                slideUp: {
                    '0%': { transform: 'translateY(20px)', opacity: '0' },
                    '100%': { transform: 'translateY(0)', opacity: '1' },
                },
                pulseSlow: {
                    '0%, 100%': { opacity: '0.4' },
                    '50%': { opacity: '0.7' },
                },
                float: {
                    '0%, 100%': { transform: 'translateY(0)' },
                    '50%': { transform: 'translateY(-10px)' },
                },
                glowPulse: {
                    '0%, 100%': { boxShadow: '0 0 20px rgba(139, 92, 246, 0.15)' },
                    '50%': { boxShadow: '0 0 40px rgba(139, 92, 246, 0.3)' },
                },
            },
            backdropBlur: {
                xs: '2px',
                '2xl': '40px',
            },
            boxShadow: {
                'glow-sm': '0 0 15px rgba(139, 92, 246, 0.15)',
                'glow-md': '0 0 30px rgba(139, 92, 246, 0.2)',
                'glow-lg': '0 0 60px rgba(139, 92, 246, 0.25)',
                'glass': '0 8px 32px rgba(0, 0, 0, 0.3)',
                'glass-lg': '0 16px 48px rgba(0, 0, 0, 0.4)',
            },
        },
    },
    plugins: [],
}
