import { useState, useCallback, useEffect } from 'react'
import axios from 'axios'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, Sparkles, Copy, Check, Brain, Heart, Zap, AlertCircle } from 'lucide-react'

// Debounce utility function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

const API_BASE = import.meta.env.VITE_API_URL || '/api'

function PredictorPage() {
    const [inputText, setInputText] = useState('')
    const [predictions, setPredictions] = useState([])
    const [analysis, setAnalysis] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [copiedIndex, setCopiedIndex] = useState(null)

    // Debounced prediction function
    const debouncedPredict = useCallback(
        debounce(async (text) => {
            if (!text.trim()) {
                setPredictions([])
                setAnalysis(null)
                return
            }

            setLoading(true)
            setError(null)

            try {
                const url = `${API_BASE}/predict`
                console.log('Requesting:', url, '| API_BASE:', API_BASE || '(empty - using relative)')
                const response = await axios.post(url, {
                    text: text,
                    top_k: 3
                })
                setPredictions(response.data.emojis)
                setAnalysis(response.data.analysis)
            } catch (err) {
                console.error('Prediction error:', err)
                if (err.response) {
                    setError(`Server error (${err.response.status}): ${err.response.data?.detail || 'Backend returned an error.'}`)
                } else if (err.request) {
                    setError(`Cannot reach the backend server. ${API_BASE ? `Tried: ${API_BASE}` : 'No API URL configured — set VITE_API_URL.'}`)
                } else {
                    setError("Failed to fetch predictions. Please try again later.")
                }
                setPredictions([])
                setAnalysis(null)
            } finally {
                setLoading(false)
            }
        }, 500), // 500ms debounce
        []
    )

    useEffect(() => {
        debouncedPredict(inputText)
    }, [inputText, debouncedPredict])

    const handlePredict = async (e) => {
        e.preventDefault()
        debouncedPredict(inputText)
    }

    const copyToClipboard = (emoji, index) => {
        navigator.clipboard.writeText(emoji)
        setCopiedIndex(index)
        setTimeout(() => setCopiedIndex(null), 2000)
    }

    const getSentimentColor = (sentiment) => {
        switch (sentiment) {
            case 'positive': return 'text-emerald-400'
            case 'negative': return 'text-rose-400'
            default: return 'text-gray-400'
        }
    }

    const getSentimentEmoji = (sentiment) => {
        switch (sentiment) {
            case 'positive': return '😊'
            case 'negative': return '😔'
            default: return '😐'
        }
    }

    return (
        <div className="min-h-screen text-white p-4 md:p-10 flex flex-col items-center justify-center relative overflow-hidden pt-24">

            {/* Floating Orbs */}
            <div className="orb orb-1"></div>
            <div className="orb orb-2"></div>
            <div className="orb orb-3"></div>
            <div className="orb orb-4"></div>

            <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
                className="w-full max-w-3xl z-10"
            >
                {/* Header */}
                <div className="text-center mb-12">
                    <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ type: "spring", stiffness: 150, damping: 15, delay: 0.1 }}
                    >
                        <h1 className="text-5xl md:text-7xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-purple-200 via-pink-200 to-indigo-200 mb-3 glow-text tracking-tight">
                            Emoji Predictor
                        </h1>
                    </motion.div>
                    <motion.p
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3, duration: 0.5 }}
                        className="text-base text-gray-400 font-light tracking-wide"
                    >
                        Advanced NLP-powered context-aware predictions
                    </motion.p>
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.5, duration: 0.5 }}
                        className="flex justify-center gap-3 mt-5"
                    >
                        <span className="glass-badge"><Brain size={11} /> Deep Learning</span>
                        <span className="glass-badge"><Heart size={11} /> Sentiment</span>
                        <span className="glass-badge"><Zap size={11} /> Transformer AI</span>
                    </motion.div>
                </div>

                {/* Input Card */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4, duration: 0.6 }}
                    className="glass-panel-strong p-6 mb-8"
                >
                    <div className="relative">
                        <div className="relative flex items-start">
                            <Search className="absolute left-4 top-4 text-gray-500 w-5 h-5 z-10" />
                            <textarea
                                value={inputText}
                                onChange={(e) => setInputText(e.target.value)}
                                placeholder="Start typing to see emoji predictions..."
                                className="glass-input w-full py-4 pl-12 pr-4 text-lg min-h-[120px] resize-none"
                                rows={4}
                            />
                        </div>
                        {loading && (
                            <div className="absolute right-4 top-4">
                                <div className="glass-spinner-small"></div>
                            </div>
                        )}
                    </div>
                </motion.div>

                <AnimatePresence mode="wait">
                    {/* Error */}
                    {error && (
                        <motion.div
                            initial={{ opacity: 0, y: -10, scale: 0.98 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.98 }}
                            className="glass-panel p-4 mb-6 flex items-center gap-3"
                            style={{ borderColor: 'rgba(244, 63, 94, 0.2)' }}
                        >
                            <AlertCircle size={18} className="text-rose-400 shrink-0" />
                            <span className="text-rose-200 text-sm">{error}</span>
                        </motion.div>
                    )}

                    {/* NLP Analysis Panel */}
                    {analysis && (
                        <motion.div
                            initial={{ opacity: 0, y: 15 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.5 }}
                            className="glass-panel p-5 mb-6"
                        >
                            <h3 className="text-xs font-semibold text-purple-300/80 mb-4 flex items-center gap-2 uppercase tracking-widest">
                                <Brain size={14} /> NLP Analysis
                            </h3>

                            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                                {/* Sentiment */}
                                <div className="glass-analysis-item">
                                    <div className="text-gray-500 text-[0.65rem] mb-1.5 uppercase tracking-wider font-medium">Sentiment</div>
                                    <div className={`font-semibold flex items-center gap-2 ${getSentimentColor(analysis.sentiment?.sentiment)}`}>
                                        <span className="text-2xl">{getSentimentEmoji(analysis.sentiment?.sentiment)}</span>
                                        <span className="capitalize text-sm">{analysis.sentiment?.sentiment}</span>
                                    </div>
                                    <div className="text-[0.65rem] text-gray-500 mt-1.5">
                                        Polarity: {analysis.sentiment?.polarity}
                                    </div>
                                </div>

                                {/* Emotion */}
                                <div className="glass-analysis-item">
                                    <div className="text-gray-500 text-[0.65rem] mb-1.5 uppercase tracking-wider font-medium">Dominant Emotion</div>
                                    <div className="font-semibold text-pink-300 capitalize text-sm">
                                        {analysis.sentiment?.dominant_emotion || 'neutral'}
                                    </div>
                                    {analysis.sentiment?.emotions && Object.keys(analysis.sentiment.emotions).length > 0 && (
                                        <div className="text-[0.65rem] text-gray-500 mt-1.5 flex gap-2">
                                            {Object.entries(analysis.sentiment.emotions).slice(0, 2).map(([k, v]) => (
                                                <span key={k}>{k}: {(v * 100).toFixed(0)}%</span>
                                            ))}
                                        </div>
                                    )}
                                </div>

                                {/* Processed Tokens */}
                                <div className="glass-analysis-item">
                                    <div className="text-gray-500 text-[0.65rem] mb-1.5 uppercase tracking-wider font-medium">Processed Tokens</div>
                                    <div className="flex flex-wrap gap-1 mt-1">
                                        {analysis.preprocessing?.processed?.slice(0, 5).map((token, i) => (
                                            <span key={i} className="token-badge">{token}</span>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}

                    {/* Emoji Predictions */}
                    {predictions.length > 0 && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ duration: 0.4 }}
                            className="grid grid-cols-1 md:grid-cols-2 gap-4"
                        >
                            {predictions.map((pred, idx) => (
                                <motion.div
                                    key={`${pred.emoji}-${idx}`}
                                    initial={{ opacity: 0, y: 20, scale: 0.95 }}
                                    animate={{ opacity: 1, y: 0, scale: 1 }}
                                    transition={{ delay: idx * 0.15, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                                    whileHover={{ scale: 1.03, y: -4 }}
                                    className="glass-emoji-card p-5 flex items-center justify-between group cursor-default"
                                >
                                    <div className="flex items-center gap-4">
                                        <motion.span
                                            className="text-5xl filter drop-shadow-lg"
                                            initial={{ rotate: -10, scale: 0.5 }}
                                            animate={{ rotate: 0, scale: 1 }}
                                            transition={{ type: "spring", stiffness: 200, delay: idx * 0.15 + 0.2 }}
                                        >
                                            {pred.emoji}
                                        </motion.span>
                                        <div className="flex flex-col">
                                            <span className="font-semibold text-gray-200 capitalize tracking-wide">{pred.name}</span>
                                            <span className="text-xs text-purple-300/60 font-medium">{(pred.score * 100).toFixed(1)}% match</span>
                                            <div className="score-bar w-32 mt-1.5">
                                                <motion.div
                                                    className="score-bar-fill"
                                                    initial={{ width: 0 }}
                                                    animate={{ width: `${pred.score * 100}%` }}
                                                    transition={{ delay: idx * 0.15 + 0.4, duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
                                                ></motion.div>
                                            </div>
                                        </div>
                                    </div>

                                    <button
                                        onClick={() => copyToClipboard(pred.emoji, idx)}
                                        className="copy-btn opacity-0 group-hover:opacity-100"
                                        title="Copy to clipboard"
                                    >
                                        {copiedIndex === idx ? <Check size={16} className="text-emerald-400" /> : <Copy size={16} />}
                                    </button>
                                </motion.div>
                            ))}
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Empty State */}
                {predictions.length === 0 && !loading && !error && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.8 }}
                        className="text-center mt-16"
                    >
                        <motion.div
                            animate={{ y: [0, -8, 0] }}
                            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                            className="text-4xl mb-4"
                        >
                            ✨
                        </motion.div>
                        <p className="text-gray-500 font-light text-sm">
                            Try sentences like <span className="text-gray-400">"I am feeling great"</span> or <span className="text-gray-400">"Let's go to the beach"</span>
                        </p>
                    </motion.div>
                )}

                {/* Shimmer Loading */}
                {loading && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="glass-panel shimmer p-6 mt-4"
                    >
                        <div className="flex items-center gap-3">
                            <div className="glass-spinner"></div>
                            <span className="text-gray-400 text-sm font-light">Analyzing with NLP pipeline...</span>
                        </div>
                    </motion.div>
                )}
            </motion.div>
        </div>
    )
}

export default PredictorPage
