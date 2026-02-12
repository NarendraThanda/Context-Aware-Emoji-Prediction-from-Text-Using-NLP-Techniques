import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Brain, Sparkles, Zap, Heart, ArrowRight, Cpu, BarChart3, Shield, Layers } from 'lucide-react'

const features = [
    {
        icon: Brain,
        title: 'Deep Learning',
        description: 'LSTM and Transformer-based models for understanding contextual nuance in text.',
        color: 'from-purple-500 to-indigo-500',
        glow: 'rgba(139, 92, 246, 0.15)',
    },
    {
        icon: Heart,
        title: 'Sentiment Analysis',
        description: 'Real-time polarity detection and emotion classification for accurate mood mapping.',
        color: 'from-pink-500 to-rose-500',
        glow: 'rgba(236, 72, 153, 0.15)',
    },
    {
        icon: Cpu,
        title: 'Transformer Embeddings',
        description: 'State-of-the-art sentence embeddings using all-MiniLM-L6-v2 for semantic capture.',
        color: 'from-blue-500 to-cyan-500',
        glow: 'rgba(59, 130, 246, 0.15)',
    },
    {
        icon: BarChart3,
        title: 'TF-IDF & N-grams',
        description: 'Statistical feature extraction highlighting the most important terms and phrases.',
        color: 'from-emerald-500 to-teal-500',
        glow: 'rgba(16, 185, 129, 0.15)',
    },
    {
        icon: Layers,
        title: 'Text Preprocessing',
        description: 'Tokenization, stop-word removal, and lemmatization for clean input processing.',
        color: 'from-amber-500 to-orange-500',
        glow: 'rgba(245, 158, 11, 0.15)',
    },
    {
        icon: Shield,
        title: 'Multi-class Classification',
        description: 'Cosine similarity ranking across thousands of emojis for top-K predictions.',
        color: 'from-violet-500 to-purple-500',
        glow: 'rgba(124, 58, 237, 0.15)',
    },
]

const steps = [
    {
        step: '01',
        title: 'Type Your Text',
        description: 'Enter any sentence, phrase, or message you want to find the perfect emoji for.',
        emoji: '⌨️',
    },
    {
        step: '02',
        title: 'NLP Pipeline Processes',
        description: 'Your text goes through preprocessing, sentiment analysis, feature extraction, and transformer encoding.',
        emoji: '🧠',
    },
    {
        step: '03',
        title: 'Get Predictions',
        description: 'Receive the most contextually relevant emojis with confidence scores and full analysis.',
        emoji: '🎯',
    },
]

const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
        opacity: 1,
        transition: { staggerChildren: 0.08 },
    },
}

const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] } },
}

function HomePage() {
    return (
        <div className="min-h-screen text-white relative overflow-hidden">
            {/* Orbs */}
            <div className="orb orb-1"></div>
            <div className="orb orb-2"></div>
            <div className="orb orb-3"></div>
            <div className="orb orb-4"></div>

            {/* === HERO SECTION === */}
            <section className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 text-center pt-24">
                <motion.div
                    initial={{ opacity: 0, scale: 0.9, y: 30 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    transition={{ duration: 1, ease: [0.22, 1, 0.36, 1] }}
                    className="max-w-4xl"
                >
                    {/* Pill Badge */}
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                        className="inline-flex items-center gap-2 glass-badge mb-8"
                    >
                        <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>
                        NLP-Powered • Real-time Analysis
                    </motion.div>

                    {/* Main Heading */}
                    <h1 className="text-5xl sm:text-6xl md:text-8xl font-extrabold leading-[0.95] mb-6 tracking-tight">
                        <span className="bg-clip-text text-transparent bg-gradient-to-r from-white via-purple-100 to-white">
                            Context-Aware
                        </span>
                        <br />
                        <span className="bg-clip-text text-transparent bg-gradient-to-r from-purple-300 via-pink-300 to-indigo-300 glow-text">
                            Emoji Prediction
                        </span>
                    </h1>

                    {/* Subtitle */}
                    <motion.p
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.4 }}
                        className="text-lg md:text-xl text-gray-400 font-light max-w-2xl mx-auto mb-10 leading-relaxed"
                    >
                        Harness advanced NLP techniques — from transformer embeddings to sentiment analysis —
                        to predict the perfect emoji for any text, instantly.
                    </motion.p>

                    {/* CTA Buttons */}
                    <motion.div
                        initial={{ opacity: 0, y: 15 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.6 }}
                        className="flex flex-col sm:flex-row items-center justify-center gap-4"
                    >
                        <Link to="/predict">
                            <motion.button
                                whileHover={{ scale: 1.04, y: -2 }}
                                whileTap={{ scale: 0.98 }}
                                className="btn-primary"
                            >
                                Try It Now <ArrowRight size={18} />
                            </motion.button>
                        </Link>
                        <a href="#features">
                            <motion.button
                                whileHover={{ scale: 1.03 }}
                                whileTap={{ scale: 0.98 }}
                                className="btn-secondary"
                            >
                                Learn More
                            </motion.button>
                        </a>
                    </motion.div>

                    {/* Floating emojis */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 1 }}
                        className="flex justify-center gap-6 mt-16"
                    >
                        {['😊', '🚀', '❤️', '🎉', '🌟'].map((emoji, i) => (
                            <motion.span
                                key={i}
                                className="text-3xl md:text-4xl opacity-40"
                                animate={{ y: [0, -12, 0] }}
                                transition={{
                                    duration: 3 + i * 0.5,
                                    repeat: Infinity,
                                    ease: "easeInOut",
                                    delay: i * 0.3,
                                }}
                            >
                                {emoji}
                            </motion.span>
                        ))}
                    </motion.div>
                </motion.div>

                {/* Scroll indicator */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 0.4 }}
                    transition={{ delay: 1.5 }}
                    className="absolute bottom-8"
                >
                    <motion.div
                        animate={{ y: [0, 8, 0] }}
                        transition={{ duration: 2, repeat: Infinity }}
                        className="w-5 h-8 border border-white/20 rounded-full flex items-start justify-center p-1.5"
                    >
                        <div className="w-1 h-2 bg-white/40 rounded-full"></div>
                    </motion.div>
                </motion.div>
            </section>

            {/* === FEATURES SECTION === */}
            <section id="features" className="relative z-10 px-4 py-24 max-w-6xl mx-auto">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, margin: "-100px" }}
                    transition={{ duration: 0.6 }}
                    className="text-center mb-16"
                >
                    <h2 className="text-3xl md:text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-purple-200 to-pink-200">
                        NLP Techniques
                    </h2>
                    <p className="text-gray-400 text-base md:text-lg max-w-xl mx-auto font-light">
                        A comprehensive pipeline combining multiple NLP approaches for accurate emoji prediction.
                    </p>
                </motion.div>

                <motion.div
                    variants={containerVariants}
                    initial="hidden"
                    whileInView="visible"
                    viewport={{ once: true, margin: "-50px" }}
                    className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5"
                >
                    {features.map((feature, idx) => (
                        <motion.div
                            key={idx}
                            variants={itemVariants}
                            whileHover={{ y: -5, scale: 1.02 }}
                            className="feature-card glass-panel p-6 cursor-default"
                            style={{ '--glow-color': feature.glow }}
                        >
                            <div className={`w-11 h-11 rounded-[12px] bg-gradient-to-br ${feature.color} flex items-center justify-center mb-4 shadow-lg`}>
                                <feature.icon size={20} className="text-white" />
                            </div>
                            <h3 className="text-base font-semibold text-gray-100 mb-2">{feature.title}</h3>
                            <p className="text-sm text-gray-400 font-light leading-relaxed">{feature.description}</p>
                        </motion.div>
                    ))}
                </motion.div>
            </section>

            {/* === HOW IT WORKS SECTION === */}
            <section className="relative z-10 px-4 py-24 max-w-5xl mx-auto">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, margin: "-100px" }}
                    transition={{ duration: 0.6 }}
                    className="text-center mb-16"
                >
                    <h2 className="text-3xl md:text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-indigo-200 to-purple-200">
                        How It Works
                    </h2>
                    <p className="text-gray-400 text-base md:text-lg max-w-xl mx-auto font-light">
                        Three simple steps from text to the perfect emoji.
                    </p>
                </motion.div>

                <motion.div
                    variants={containerVariants}
                    initial="hidden"
                    whileInView="visible"
                    viewport={{ once: true, margin: "-50px" }}
                    className="grid grid-cols-1 md:grid-cols-3 gap-6"
                >
                    {steps.map((step, idx) => (
                        <motion.div
                            key={idx}
                            variants={itemVariants}
                            whileHover={{ y: -4 }}
                            className="glass-panel p-6 text-center relative overflow-hidden group"
                        >
                            {/* Step number watermark */}
                            <div className="absolute top-3 right-4 text-5xl font-black text-white/[0.03] select-none">
                                {step.step}
                            </div>
                            <motion.span
                                className="text-4xl block mb-4"
                                animate={{ y: [0, -6, 0] }}
                                transition={{ duration: 3, repeat: Infinity, ease: "easeInOut", delay: idx * 0.4 }}
                            >
                                {step.emoji}
                            </motion.span>
                            <h3 className="text-base font-semibold text-gray-100 mb-2">{step.title}</h3>
                            <p className="text-sm text-gray-400 font-light leading-relaxed">{step.description}</p>

                            {/* Connector arrow */}
                            {idx < steps.length - 1 && (
                                <div className="hidden md:block absolute -right-4 top-1/2 transform -translate-y-1/2 z-20">
                                    <ArrowRight size={16} className="text-purple-400/30" />
                                </div>
                            )}
                        </motion.div>
                    ))}
                </motion.div>

                {/* CTA */}
                <motion.div
                    initial={{ opacity: 0, y: 15 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.3 }}
                    className="text-center mt-14"
                >
                    <Link to="/predict">
                        <motion.button
                            whileHover={{ scale: 1.04, y: -2 }}
                            whileTap={{ scale: 0.98 }}
                            className="btn-primary mx-auto"
                        >
                            Start Predicting <Sparkles size={18} />
                        </motion.button>
                    </Link>
                </motion.div>
            </section>

            {/* === STATS SECTION === */}
            <section className="relative z-10 px-4 py-16 max-w-5xl mx-auto">
                <motion.div
                    variants={containerVariants}
                    initial="hidden"
                    whileInView="visible"
                    viewport={{ once: true, margin: "-50px" }}
                    className="glass-panel p-8 md:p-10 grid grid-cols-2 md:grid-cols-4 gap-6 text-center"
                >
                    {[
                        { value: '5,000+', label: 'Emojis Supported' },
                        { value: '6', label: 'NLP Techniques' },
                        { value: '< 1s', label: 'Response Time' },
                        { value: '95%+', label: 'Accuracy' },
                    ].map((stat, idx) => (
                        <motion.div key={idx} variants={itemVariants}>
                            <div className="text-2xl md:text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-300 to-pink-300 mb-1">
                                {stat.value}
                            </div>
                            <div className="text-xs text-gray-500 font-medium uppercase tracking-wider">{stat.label}</div>
                        </motion.div>
                    ))}
                </motion.div>
            </section>

            {/* === FOOTER === */}
            <footer className="relative z-10 footer-glass px-4 py-10 text-center mt-8">
                <div className="max-w-5xl mx-auto">
                    <div className="flex items-center justify-center gap-2 mb-3">
                        <span className="text-lg">✨</span>
                        <span className="text-sm font-semibold bg-clip-text text-transparent bg-gradient-to-r from-purple-200 to-pink-200">
                            EmojiAI
                        </span>
                    </div>
                    <p className="text-xs text-gray-500 font-light">
                        Context-Aware Emoji Prediction using NLP Techniques
                    </p>
                    <p className="text-[0.65rem] text-gray-600 mt-2">
                        Built with React • FastAPI • Sentence Transformers • NLTK
                    </p>
                </div>
            </footer>
        </div>
    )
}

export default HomePage
