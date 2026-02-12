"""
Context-Aware Emoji Prediction API
Advanced NLP Pipeline with comprehensive techniques.
"""
import os
import sys

# Resolve paths relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import Optional

# Import NLP pipeline components
from nlp_pipeline import TextPreprocessor, FeatureExtractor, SentimentAnalyzer, EvaluationMetrics

app = FastAPI(
    title="Context-Aware Emoji Prediction API",
    description="Advanced NLP pipeline for emoji prediction using transformer embeddings, sentiment analysis, and deep learning techniques.",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
model = None
emoji_df = None
emoji_embeddings = None
preprocessor = None
sentiment_analyzer = None
feature_extractor = None


@app.on_event("startup")
async def load_model():
    global model, emoji_df, emoji_embeddings, preprocessor, sentiment_analyzer, feature_extractor
    
    try:
        print("=" * 50)
        print("Initializing Advanced NLP Pipeline...")
        print("=" * 50)
        
        # 1. Text Preprocessing Module
        print("\n[1/5] Loading Text Preprocessor...")
        preprocessor = TextPreprocessor(use_lemmatization=True)
        print("      ✓ Tokenization ready")
        print("      ✓ Stop-word removal ready")
        print("      ✓ Lemmatization ready")
        
        # 2. Sentiment Analyzer
        print("\n[2/5] Loading Sentiment Analyzer...")
        sentiment_analyzer = SentimentAnalyzer()
        print("      ✓ Polarity detection ready")
        print("      ✓ Emotion classification ready")
        
        # 3. Feature Extractor
        print("\n[3/5] Loading Feature Extractor...")
        feature_extractor = FeatureExtractor(max_features=5000, ngram_range=(1, 2))
        print("      ✓ TF-IDF vectorizer ready")
        print("      ✓ N-gram analysis ready")
        
        # 4. Transformer Model (Sentence Embeddings)
        print("\n[4/5] Loading Sentence Transformer Model...")
        print("      (This may take a moment on first run)")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("      ✓ Transformer embeddings ready")
        print("      ✓ Context modeling ready")
        
        # 5. Emoji Data
        print("\n[5/5] Loading Emoji Dataset...")
        emoji_df = pd.read_csv(os.path.join(BASE_DIR, "full_emoji.csv"))
        descriptions = emoji_df['name'].fillna("").tolist()
        emoji_embeddings = model.encode(descriptions, convert_to_tensor=True)
        print(f"      ✓ Loaded {len(emoji_df)} emojis")
        print("      ✓ Pre-computed embeddings ready")
        
        print("\n" + "=" * 50)
        print("✓ All NLP components loaded successfully!")
        print("=" * 50 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error loading components: {e}")
        raise e


@app.get("/api")
def read_root():
    return {
        "message": "Context-Aware Emoji Prediction API v2.0",
        "features": [
            "Text Preprocessing (Tokenization, Stop-words, Lemmatization)",
            "Feature Extraction (TF-IDF, N-grams)",
            "Transformer Embeddings (BERT-style)",
            "Sentiment & Emotion Analysis",
            "Multi-class Emoji Classification",
            "Top-K Prediction"
        ]
    }


@app.post("/predict")
def predict_emoji(payload: dict):
    """
    Advanced emoji prediction with full NLP pipeline.
    
    Input: {"text": "your text here", "top_k": 2}
    
    Returns emojis with preprocessing details, sentiment, and predictions.
    """
    global model, emoji_df, emoji_embeddings, preprocessor, sentiment_analyzer
    
    if model is None or emoji_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    user_text = payload.get("text", "")
    top_k = payload.get("top_k", 2)
    
    if not user_text:
        return {"emojis": [], "analysis": None}
    
    # === 1. Text Preprocessing ===
    preprocessing_result = preprocessor.preprocess(user_text)
    processed_text = preprocessing_result["final_text"]
    
    # === 2. Sentiment & Emotion Analysis ===
    sentiment_result = sentiment_analyzer.full_analysis(user_text)
    
    # === 3. Feature Extraction (for transparency) ===
    top_ngrams = []
    if processed_text:
        try:
            feature_extractor.fit([processed_text])
            top_ngrams = feature_extractor.get_top_ngrams(processed_text, n=3)
        except:
            pass
    
    # === 4. Generate Embeddings (Transformer) ===
    user_embedding = model.encode(user_text, convert_to_tensor=True)
    
    # === 5. Cosine Similarity for Multi-class Classification ===
    cosine_scores = util.cos_sim(user_embedding, emoji_embeddings)[0]
    
    # === 6. Top-K Prediction ===
    top_indices = cosine_scores.argsort(descending=True)[:top_k]
    
    results = []
    for idx in top_indices:
        idx = int(idx)
        score = float(cosine_scores[idx])
        results.append({
            "emoji": emoji_df.iloc[idx]['emoji'],
            "name": emoji_df.iloc[idx]['name'],
            "score": round(score, 4)
        })
    
    return {
        "emojis": results,
        "analysis": {
            "preprocessing": {
                "original": preprocessing_result["original"],
                "tokens": preprocessing_result["tokens"][:10],  # Limit for response size
                "processed": preprocessing_result["processed"][:10],
                "final_text": processed_text
            },
            "sentiment": sentiment_result,
            "features": {
                "top_ngrams": [(ng, round(score, 3)) for ng, score in top_ngrams]
            }
        }
    }


@app.post("/analyze")
def analyze_text(payload: dict):
    """
    Get detailed NLP analysis without prediction.
    Useful for understanding the preprocessing pipeline.
    """
    text = payload.get("text", "")
    if not text:
        return {"error": "No text provided"}
    
    # Full preprocessing
    preprocessing = preprocessor.preprocess(text)
    
    # Sentiment analysis
    sentiment = sentiment_analyzer.full_analysis(text)
    
    # Feature extraction
    try:
        feature_extractor.fit([preprocessing["final_text"]])
        ngrams = feature_extractor.get_top_ngrams(preprocessing["final_text"], n=5)
    except:
        ngrams = []
    
    return {
        "preprocessing": preprocessing,
        "sentiment": sentiment,
        "features": {
            "top_ngrams": ngrams
        }
    }


@app.get("/metrics")
def get_metrics():
    """
    Return evaluation metrics information.
    In a production system, these would be computed from test data.
    """
    return {
        "metrics_supported": [
            "Accuracy",
            "Precision",
            "Recall", 
            "F1-Score",
            "Top-K Accuracy"
        ],
        "note": "Metrics are computed during model training. See /train endpoint for training details."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
