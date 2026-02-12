import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Sparkles, Home } from 'lucide-react'

function Navbar() {
    const location = useLocation()

    return (
        <motion.nav
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="fixed top-0 left-0 right-0 z-50 px-4 pt-4"
        >
            <div className="max-w-5xl mx-auto">
                <div className="navbar-glass flex items-center justify-between px-6 py-3">
                    {/* Logo */}
                    <Link to="/" className="flex items-center gap-2.5 group">
                        <motion.span
                            className="text-2xl"
                            whileHover={{ rotate: 15, scale: 1.1 }}
                            transition={{ type: "spring", stiffness: 300 }}
                        >
                            ✨
                        </motion.span>
                        <span className="text-base font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-200 to-pink-200 tracking-tight">
                            EmojiAI
                        </span>
                    </Link>

                    {/* Navigation Links */}
                    <div className="flex items-center gap-2">
                        <Link
                            to="/"
                            className={`nav-link flex items-center gap-1.5 px-4 py-2 text-sm font-medium ${location.pathname === '/' ? 'nav-link-active' : ''
                                }`}
                        >
                            <Home size={14} />
                            Home
                        </Link>
                        <Link
                            to="/predict"
                            className={`nav-link flex items-center gap-1.5 px-4 py-2 text-sm font-medium ${location.pathname === '/predict' ? 'nav-link-active' : ''
                                }`}
                        >
                            <Sparkles size={14} />
                            Predict
                        </Link>
                    </div>
                </div>
            </div>
        </motion.nav>
    )
}

export default Navbar
