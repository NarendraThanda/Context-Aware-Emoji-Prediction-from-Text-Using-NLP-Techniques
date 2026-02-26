<p align="center">
  <h1 align="center">🎯 Context-Aware Emoji Prediction from Text Using NLP Techniques</h1>
  <p align="center">
    A full-stack application that predicts contextually relevant emojis from natural language text using an advanced NLP pipeline — featuring an ensemble classifier, transformer embeddings, sentiment analysis, and a modern React UI.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white" />
  <img src="https://img.shields.io/badge/Vite-7-646CFF?logo=vite&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Deployed-Vercel-black?logo=vercel" />
</p>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training the Model](#training-the-model)
  - [Running Locally](#running-locally)
- [API Endpoints](#-api-endpoints)
- [NLP Pipeline Details](#-nlp-pipeline-details)
- [Model Performance](#-model-performance)
- [Deployment](#-deployment)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

This project implements a **Context-Aware Emoji Prediction System** that takes natural language text as input and predicts the most relevant emojis. It goes beyond simple keyword matching by using a multi-stage NLP pipeline that understands the **context, sentiment, and semantics** of the text.

The system uses a **trained ensemble classifier** (Logistic Regression + Calibrated SVM) as the primary prediction engine, with **sentence-transformer embeddings** as an intelligent fallback.

---

## ✨ Features

- **🔤 Text Preprocessing** — Tokenization, stop-word removal, and lemmatization
- **📊 Feature Extraction** — TF-IDF vectorization with word-level and character-level n-grams
- **🤖 Ensemble Classifier** — Logistic Regression + Calibrated SVM with weighted voting
- **🧠 Transformer Embeddings** — Sentence-BERT (`all-MiniLM-L6-v2`) for semantic similarity fallback
- **💡 Sentiment & Emotion Analysis** — Polarity, subjectivity, and emotion detection using TextBlob
- **🎯 Top-K Prediction** — Returns the top-K predicted emojis with confidence scores
- **📈 Evaluation Metrics** — Accuracy, Precision, Recall, F1-Score, and Top-K Accuracy
- **🖥️ Modern React Frontend** — Glassmorphism UI with Framer Motion animations
- **🚀 Full-Stack Launcher** — Single-command startup serving both frontend and backend
- **☁️ Vercel Deployment** — Serverless-ready with lightweight TF-IDF engine for the cloud

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                       │
│  (Vite + TailwindCSS + Framer Motion)                   │
│  ┌──────────┐  ┌──────────────┐                         │
│  │ HomePage │  │ PredictorPage│                         │
│  └──────────┘  └──────┬───────┘                         │
│                       │ POST /predict                   │
└───────────────────────┼─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                   FastAPI Backend                        │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              NLP Pipeline                         │   │
│  │  ┌─────────────┐  ┌──────────────┐               │   │
│  │  │   Text      │  │  Sentiment   │               │   │
│  │  │Preprocessor │  │  Analyzer    │               │   │
│  │  └──────┬──────┘  └──────┬───────┘               │   │
│  │         │                │                        │   │
│  │  ┌──────▼────────────────▼───────┐               │   │
│  │  │     Feature Extractor         │               │   │
│  │  │  (TF-IDF word + char n-grams) │               │   │
│  │  └──────────────┬────────────────┘               │   │
│  │                 │                                 │   │
│  │  ┌──────────────▼────────────────┐               │   │
│  │  │   Ensemble Classifier         │               │   │
│  │  │  (LR + SVM) ──► Top-K Emojis  │               │   │
│  │  └───────────────────────────────┘               │   │
│  │            ▲ fallback                             │   │
│  │  ┌─────────┴─────────────────────┐               │   │
│  │  │  Sentence-Transformer (BERT)  │               │   │
│  │  │  Cosine Similarity Matching   │               │   │
│  │  └───────────────────────────────┘               │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  📂 trained_model/  ←  emoji_classifier.pkl             │
│  📂 data/           ←  full_emoji.csv (1800+ emojis)    │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer      | Technology                                                       |
|------------|------------------------------------------------------------------|
| **Frontend** | React 19, Vite 7, TailwindCSS, Framer Motion, Lucide Icons    |
| **Backend**  | FastAPI, Uvicorn, Python 3.11+                                 |
| **NLP**      | NLTK, TextBlob, scikit-learn (TF-IDF, LR, SVM, RF)            |
| **Deep Learning** | PyTorch, Sentence-Transformers (`all-MiniLM-L6-v2`)      |
| **Data**     | Pandas, NumPy, SciPy                                          |
| **Deployment (Cloud)** | Vercel (Serverless Functions + Static Frontend)      |
| **Deployment (Self-hosted)** | Render (backend), Vercel (frontend)              |

---

## 📁 Project Structure

```
Context-Aware-Emoji-Prediction-from-Text-Using-NLP-Techniques/
│
├── 📄 main.py                  # Full-stack launcher (single command startup)
├── 📄 train_model.py           # Model training pipeline
├── 📄 evaluate_accuracy.py     # Accuracy evaluation & reporting
├── 📄 test_predictions.py      # Quick prediction test script
├── 📄 requirements.txt         # Python dependencies
├── 📄 vercel.json              # Vercel deployment configuration
├── 📄 .vercelignore            # Files to ignore during Vercel deployment
├── 📄 .gitignore               # Git ignore rules
│
├── 📂 backend/                 # FastAPI backend (local development)
│   ├── main.py                 # FastAPI app with full NLP pipeline
│   ├── nlp_pipeline.py         # NLP components (Preprocessor, Feature Extractor, etc.)
│   └── render.yaml             # Render deployment config
│
├── 📂 api/                     # Vercel serverless functions (cloud deployment)
│   ├── index.py                # GET /api — Health check endpoint
│   ├── predict.py              # POST /api/predict — Emoji prediction
│   ├── analyze.py              # POST /api/analyze — Text analysis
│   └── _nlp_engine.py          # Lightweight NLP engine (scikit-learn only)
│
├── 📂 frontend/                # React frontend (Vite)
│   ├── src/
│   │   ├── App.jsx             # Main app with routing
│   │   ├── main.jsx            # React entry point
│   │   ├── index.css           # Global styles & design system
│   │   ├── App.css             # App-level styles
│   │   ├── components/
│   │   │   └── Navbar.jsx      # Navigation component
│   │   └── pages/
│   │       ├── HomePage.jsx    # Landing page
│   │       └── PredictorPage.jsx # Emoji prediction interface
│   ├── package.json            # Node.js dependencies
│   ├── vite.config.js          # Vite configuration with API proxy
│   ├── tailwind.config.js      # TailwindCSS configuration
│   └── index.html              # HTML entry point
│
├── 📂 data/                    # Dataset
│   ├── full_emoji.csv          # Full emoji dataset (1800+ emojis)
│   └── emojiclean.ipynb        # Data cleaning notebook
│
└── 📂 trained_model/           # Trained model artifacts (git-ignored)
    ├── emoji_classifier.pkl    # Trained ensemble classifier
    ├── tfidf_word.pkl          # Word-level TF-IDF vectorizer
    ├── tfidf_char.pkl          # Character-level TF-IDF vectorizer
    ├── label_encoder.pkl       # Label encoder for emoji classes
    ├── emoji_lookup.json       # Emoji name → character lookup
    └── training_metrics.json   # Training performance metrics
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** — [Download here](https://www.python.org/downloads/)
- **Node.js 18+** — [Download here](https://nodejs.org/)
- **Git** — [Download here](https://git-scm.com/)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/NarendraThanda/Context-Aware-Emoji-Prediction-from-Text-Using-NLP-Techniques.git
   cd Context-Aware-Emoji-Prediction-from-Text-Using-NLP-Techniques
   ```

2. **Create a Python virtual environment** (recommended)

   ```bash
   python -m venv venv

   # Activate on Windows
   venv\Scripts\activate

   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies**

   ```bash
   cd frontend
   npm install
   cd ..
   ```

5. **Download NLTK data** (auto-downloads on first run, but you can do it manually)

   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
   ```

### Training the Model

Before running the application for the first time, you need to train the emoji prediction model:

```bash
python train_model.py
```

This will:
1. Load the emoji dataset from `data/full_emoji.csv`
2. Generate enriched training data with text augmentation
3. Extract TF-IDF features (word-level + character-level n-grams)
4. Train an ensemble of Logistic Regression and Calibrated SVM classifiers
5. Evaluate the model with accuracy, precision, recall, F1-score, and top-K accuracy
6. Save all model artifacts to `trained_model/`

> ⏱️ Training takes approximately **2–5 minutes** depending on your hardware.

### Running Locally

#### Option 1: Full-Stack Mode (Recommended)

This serves both the FastAPI backend and the React frontend on a **single port** (`http://localhost:8000`):

```bash
# Build the frontend first (one-time)
cd frontend
npm run build
cd ..

# Launch the full-stack app
python main.py
```

Then open **http://localhost:8000** in your browser.

#### Option 2: Development Mode (Hot Reload)

Run the backend and frontend separately for active development:

**Terminal 1 — Backend:**
```bash
cd backend
uvicorn main:app --reload --host localhost --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

The frontend dev server (default `http://localhost:5173`) is configured to proxy API requests to the backend on port `8000`.

---

## 📡 API Endpoints

| Method | Endpoint         | Description                          |
|--------|------------------|--------------------------------------|
| `GET`  | `/api`           | Health check & API info              |
| `POST` | `/predict`       | Predict emojis for given text        |
| `POST` | `/analyze`       | Get detailed NLP analysis            |
| `GET`  | `/metrics`       | View model training metrics          |

### Example: Predict Emojis

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love you so much", "top_k": 2}'
```

**Response:**
```json
{
  "emojis": [
    {"emoji": "🤟", "name": "love-you gesture", "score": 0.4521},
    {"emoji": "🥰", "name": "smiling face with hearts", "score": 0.2103}
  ],
  "model_used": "trained_classifier",
  "analysis": {
    "preprocessing": {
      "original": "I love you so much",
      "tokens": ["love", "much"],
      "processed": ["love", "much"],
      "final_text": "love much"
    },
    "sentiment": {
      "polarity": 0.5,
      "subjectivity": 0.6,
      "sentiment": "positive",
      "emotions": {"joy": 1.0},
      "dominant_emotion": "joy"
    },
    "features": {
      "top_ngrams": [["love", 0.707], ["much", 0.707]]
    }
  }
}
```

### Example: Analyze Text

**Request:**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I am feeling really anxious about tomorrow"}'
```

---

## 🧪 NLP Pipeline Details

The prediction system uses a **6-stage NLP pipeline**:

### Stage 1: Text Preprocessing
- **Tokenization** — Splitting text into individual words using NLTK
- **Stop-word Removal** — Filtering out common English words (the, is, at, etc.)
- **Lemmatization** — Reducing words to their base/root form (running → run)

### Stage 2: Sentiment Analysis
- **Polarity Detection** — Measures how positive or negative the text is (-1 to +1)
- **Subjectivity Analysis** — Measures how subjective vs. objective the text is (0 to 1)
- **Emotion Classification** — Detects dominant emotion (joy, sadness, anger, fear, surprise, disgust)

### Stage 3: Feature Extraction
- **TF-IDF Vectorization** — Converts text into numerical feature vectors
- **Word-level N-grams** — Captures unigrams and bigrams from words
- **Character-level N-grams** — Captures sub-word patterns for robustness

### Stage 4: Trained Ensemble Classifier
- **Logistic Regression** — Fast, interpretable linear classifier
- **Calibrated SVM (LinearSVC)** — High-accuracy support vector machine with probability calibration
- **Weighted Voting** — Combines predictions from both models using probability averaging

### Stage 5: Transformer Embeddings (Fallback)
- **Model:** `all-MiniLM-L6-v2` (Sentence-BERT)
- **Method:** Encodes both user text and emoji descriptions into 384-dim vectors
- **Matching:** Cosine similarity to find the closest emoji descriptions
- **Trigger:** Activates when ensemble confidence is below 15%, or when no trained model is available

### Stage 6: Top-K Selection
- Returns the top-K unique emojis ranked by confidence score
- Deduplication ensures no repeated emoji predictions

---

## 📈 Model Performance

| Metric              | Score     |
|---------------------|-----------|
| **Accuracy**        | 91.81%    |
| **Precision**       | 93.13%    |
| **Recall**          | 91.81%    |
| **F1-Score**        | 91.84%    |
| **Top-3 Accuracy**  | 95.97%    |
| **Top-5 Accuracy**  | 96.63%    |

**Dataset:**
- Total samples: 12,024
- Training samples: 9,619
- Test samples: 2,405
- Emoji classes: 1,816

To run the full accuracy evaluation yourself:

```bash
python evaluate_accuracy.py
```

---

## ☁️ Deployment

### Vercel (Recommended for Demo)

The project is configured for **one-click Vercel deployment** with:
- **Frontend:** Built and served as static files from `frontend/dist/`
- **Backend:** Serverless Python functions in `api/` using a lightweight TF-IDF engine (no PyTorch)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

> **Note:** The Vercel deployment uses `api/_nlp_engine.py` — a lightweight NLP engine that uses scikit-learn TF-IDF + cosine similarity instead of PyTorch/sentence-transformers, keeping the bundle under Vercel's 250MB limit.

### Render (Self-hosted Backend)

The backend can be deployed separately to Render:

```bash
# Configuration is in backend/render.yaml
# Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## 🖼️ Screenshots

### Home Page
The landing page features a modern glassmorphism design with animated hero section, feature cards, and NLP technique explanations.

### Prediction Page
The predictor page allows users to input text and instantly see predicted emojis with confidence scores, along with detailed NLP analysis (preprocessing steps, sentiment, and features).

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/NarendraThanda">Narendra Thanda</a>
</p>
