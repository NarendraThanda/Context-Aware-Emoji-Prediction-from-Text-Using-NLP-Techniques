"""
Context-Aware Emoji Prediction API
Advanced NLP Pipeline with trained classifier + transformer fallback.
"""
import os
import sys
import pickle
import json

# Resolve paths relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "trained_model")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, ROOT_DIR)  # For unpickling ManualEnsemble from train_model
from train_model import ManualEnsemble  # noqa: F401 - needed for pickle

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from typing import Optional

# Import NLP pipeline components
from nlp_pipeline import TextPreprocessor, FeatureExtractor, SentimentAnalyzer, EvaluationMetrics

app = FastAPI(
    title="Context-Aware Emoji Prediction API",
    description="Advanced NLP pipeline for emoji prediction using trained ensemble classifier, transformer embeddings, sentiment analysis, and deep learning techniques.",
    version="3.0.0"
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
transformer_model = None
emoji_df = None
emoji_embeddings = None
preprocessor = None
sentiment_analyzer = None
feature_extractor = None

# Trained model components
trained_classifier = None
tfidf_word = None
tfidf_char = None
label_encoder = None
emoji_lookup = None
training_metrics = None
USE_TRAINED_MODEL = False


@app.on_event("startup")
async def load_model():
    global transformer_model, emoji_df, emoji_embeddings
    global preprocessor, sentiment_analyzer, feature_extractor
    global trained_classifier, tfidf_word, tfidf_char, label_encoder
    global emoji_lookup, training_metrics, USE_TRAINED_MODEL

    try:
        print("=" * 50)
        print("Initializing Advanced NLP Pipeline v3.0...")
        print("=" * 50)

        # 1. Text Preprocessing Module
        print("\n[1/6] Loading Text Preprocessor...")
        preprocessor = TextPreprocessor(use_lemmatization=True)
        print("      OK Tokenization ready")
        print("      OK Stop-word removal ready")
        print("      OK Lemmatization ready")

        # 2. Sentiment Analyzer
        print("\n[2/6] Loading Sentiment Analyzer...")
        sentiment_analyzer = SentimentAnalyzer()
        print("      OK Polarity detection ready")
        print("      OK Emotion classification ready")

        # 3. Feature Extractor
        print("\n[3/6] Loading Feature Extractor...")
        feature_extractor = FeatureExtractor(max_features=5000, ngram_range=(1, 2))
        print("      OK TF-IDF vectorizer ready")
        print("      OK N-gram analysis ready")

        # 4. Try to load TRAINED model first (higher accuracy)
        print("\n[4/6] Loading Trained Classifier...")
        try:
            classifier_path = os.path.join(MODEL_DIR, "emoji_classifier.pkl")
            tfidf_word_path = os.path.join(MODEL_DIR, "tfidf_word.pkl")
            tfidf_char_path = os.path.join(MODEL_DIR, "tfidf_char.pkl")
            le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
            lookup_path = os.path.join(MODEL_DIR, "emoji_lookup.json")
            metrics_path = os.path.join(MODEL_DIR, "training_metrics.json")

            if all(os.path.exists(p) for p in [classifier_path, tfidf_word_path, tfidf_char_path, le_path, lookup_path]):
                with open(classifier_path, 'rb') as f:
                    trained_classifier = pickle.load(f)
                with open(tfidf_word_path, 'rb') as f:
                    tfidf_word = pickle.load(f)
                with open(tfidf_char_path, 'rb') as f:
                    tfidf_char = pickle.load(f)
                with open(le_path, 'rb') as f:
                    label_encoder = pickle.load(f)
                with open(lookup_path, 'r', encoding='utf-8') as f:
                    emoji_lookup = json.load(f)

                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        training_metrics = json.load(f)

                USE_TRAINED_MODEL = True
                acc = training_metrics['ensemble']['accuracy'] if training_metrics else 'N/A'
                print(f"      OK Trained ensemble classifier loaded!")
                print(f"      OK Model accuracy: {acc}")
                print(f"      OK {len(label_encoder.classes_)} emoji classes")
            else:
                print("      WARN No trained model found. Run 'python train_model.py' first.")
                print("      WARN Falling back to transformer embeddings.")
        except Exception as e:
            print(f"      WARN Could not load trained model: {e}")
            print("      WARN Falling back to transformer embeddings.")

        # 5. Load Transformer Model as fallback (or primary if no trained model)
        print("\n[5/6] Loading Sentence Transformer Model...")
        try:
            from sentence_transformers import SentenceTransformer, util
            print("      (This may take a moment on first run)")
            transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("      OK Transformer embeddings ready")
            print("      OK Context modeling ready")
        except ImportError:
            print("      WARN sentence-transformers not installed.")
            if not USE_TRAINED_MODEL:
                print("      ERROR No prediction method available!")
                raise Exception("No model available. Run 'python train_model.py' or install sentence-transformers.")

        # 6. Emoji Data
        print("\n[6/6] Loading Emoji Dataset...")
        emoji_df = pd.read_csv(os.path.join(ROOT_DIR, "data", "full_emoji.csv"))

        # Pre-compute transformer embeddings if transformer is available
        if transformer_model is not None:
            emoji_context_map = {
                "grinning face": "happy joy smile laughing cheerful excited good great amazing wonderful",
                "face with tears of joy": "funny hilarious laughing crying lol comedy joke humor haha",
                "smiling face with heart-eyes": "love adore beautiful gorgeous amazing crush romantic attraction",
                "thinking face": "wondering thinking confused hmm curious pondering question",
                "loudly crying face": "sad crying upset devastated heartbroken tears emotional pain",
                "fire": "hot amazing lit awesome cool fire trending popular excellent",
                "red heart": "love heart romance relationship care affection passion valentine",
                "thumbs up": "good okay yes agree approve nice well done great job",
                "clapping hands": "congratulations bravo applause well done great achievement success",
                "folded hands": "please thank you prayer grateful hope wish bless",
                "face with rolling eyes": "annoyed bored sarcasm whatever really seriously",
                "smiling face with sunglasses": "cool awesome confident swagger stylish boss",
                "winking face": "flirting playful hint sly joke kidding wink",
                "angry face": "angry mad furious rage upset irritated annoyed hate",
                "fearful face": "scared afraid fear horror terrified frightened panic",
                "nauseated face": "sick disgusted gross yuck eww vomit ill",
                "sleeping face": "tired sleepy bored exhausted zzz rest nap",
                "partying face": "celebration party fun birthday congratulations hooray woohoo",
                "smiling face with halo": "innocent angel good pure sweet kind blessed",
                "face screaming in fear": "shocked scared horror surprised omg terrified",
                "rolling on the floor laughing": "hilarious dying funny rofl lmao too funny",
                "hugging face": "hug love warm embrace comfort friendly care",
                "star-struck": "amazing celebrity idol wow incredible starstruck fan wonderful",
                "money-mouth face": "money rich wealthy cash dollar expensive profit",
                "nerd face": "smart intelligent geek nerd study academic clever",
                "broken heart": "heartbreak sad breakup hurt pain loss rejection cry",
                "skull": "dead dying hilarious literally dead OMG I cant",
                "sparkles": "magic beautiful amazing shine new sparkle glitter special",
                "rocket": "launch startup fast speed progress technology moon success",
                "sun": "sunny weather bright warm morning sunshine beautiful day",
                "moon": "night evening sleep dark goodnight lunar celestial",
                "rainbow": "colorful diversity pride beautiful hope promise",
                "trophy": "winner champion success achievement first place victory",
                "musical note": "music song singing melody tune rhythm listening",
                "camera": "photo picture photography selfie memory snapshot",
                "book": "reading study learning education knowledge literature",
                "laptop": "computer work technology coding programming developer",
                "pizza": "food hungry eating delicious dinner lunch yummy",
                "coffee": "morning caffeine drink energy wake up tired work",
                "beer": "drinks alcohol party cheers celebration bar",
                "dog face": "dog puppy pet cute animal adorable woof",
                "cat face": "cat kitty pet meow animal cute feline",
                "Christmas tree": "christmas holiday festive merry december winter celebration",
                "gift": "present birthday surprise gift giving celebration",
                "balloon": "party celebration birthday fun festive happy",
                "crown": "king queen royal boss leader power royalty",
                "100": "perfect score hundred percent absolutely totally agree completely",
                "eyes": "looking watching see staring curious notice attention",
                "wave": "hello hi goodbye greeting hey welcome wave",
                "muscle": "strong strength gym workout fitness power exercise flex",
                "brain": "smart intelligent thinking genius mind knowledge clever",
                "butterfly": "beautiful nature transformation change growth pretty",
                "rose": "love romance flower beautiful valentine date romantic",
                "sunflower": "happy bright cheerful flower nature yellow sunshine",
                "earth": "world global planet earth international travel nature",
                "car": "driving travel road trip automobile vehicle transportation",
                "airplane": "travel flying vacation trip journey international flight",
                "house": "home family living housing domestic shelter comfort",
                "hospital": "sick medical health doctor nurse emergency illness",
                "school": "education learning study class teacher student academic",
                "warning": "caution alert danger warning careful attention beware",
                "check mark": "done complete yes correct approved finished success",
                "cross mark": "no wrong incorrect error rejected failed cancel",
            }

            descriptions = []
            for _, row in emoji_df.iterrows():
                name = str(row['name']).strip() if pd.notna(row['name']) else ""
                extra_context = ""
                for key, context in emoji_context_map.items():
                    if key.lower() in name.lower():
                        extra_context = context
                        break
                desc = f"{name}. {extra_context}" if extra_context else name
                descriptions.append(desc)

            emoji_embeddings = transformer_model.encode(
                descriptions, convert_to_tensor=True, show_progress_bar=True
            )
            print(f"      OK Loaded {len(emoji_df)} emojis with enhanced descriptions")
            print("      OK Pre-computed embeddings ready")
        else:
            print(f"      OK Loaded {len(emoji_df)} emojis")

        mode = "TRAINED CLASSIFIER (Higher Accuracy)" if USE_TRAINED_MODEL else "TRANSFORMER EMBEDDINGS"
        print("\n" + "=" * 50)
        print(f"OK All NLP components loaded successfully!")
        print(f"   Primary prediction mode: {mode}")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"\nERROR loading components: {e}")
        raise e


def _predict_with_trained_model(text, top_k=2):
    """Predict using the trained ensemble classifier."""
    from scipy.sparse import hstack

    X_word = tfidf_word.transform([text])
    X_char = tfidf_char.transform([text])
    X = hstack([X_word, X_char])

    proba = trained_classifier.predict_proba(X)[0]
    top_indices = proba.argsort()[::-1]

    seen = set()
    results = []
    for idx in top_indices:
        label_name = label_encoder.inverse_transform([idx])[0]
        emoji_char = emoji_lookup.get(label_name, None)
        if emoji_char and emoji_char not in seen:
            seen.add(emoji_char)
            results.append({
                "emoji": emoji_char,
                "name": label_name,
                "score": round(float(proba[idx]), 4)
            })
        if len(results) >= top_k:
            break

    return results


def _predict_with_transformer(text, top_k=2):
    """Predict using transformer embeddings + cosine similarity (fallback)."""
    from sentence_transformers import util

    user_embedding = transformer_model.encode(text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, emoji_embeddings)[0]

    top_indices = cosine_scores.argsort(descending=True)
    seen = set()
    results = []
    for idx in top_indices:
        idx = int(idx)
        emoji_char = emoji_df.iloc[idx]['emoji']
        if emoji_char not in seen:
            seen.add(emoji_char)
            results.append({
                "emoji": emoji_char,
                "name": emoji_df.iloc[idx]['name'],
                "score": round(float(cosine_scores[idx]), 4)
            })
        if len(results) >= top_k:
            break

    return results


@app.get("/api")
def read_root():
    return {
        "message": "Context-Aware Emoji Prediction API v3.0",
        "model_mode": "trained_classifier" if USE_TRAINED_MODEL else "transformer_embeddings",
        "features": [
            "Text Preprocessing (Tokenization, Stop-words, Lemmatization)",
            "Feature Extraction (TF-IDF word + char n-grams)",
            "Trained Ensemble Classifier (LR + SVM + RF)",
            "Transformer Embeddings fallback (BERT-style)",
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

    Uses trained classifier for higher accuracy, with transformer fallback.
    """
    global transformer_model, emoji_df, emoji_embeddings, preprocessor, sentiment_analyzer

    if (not USE_TRAINED_MODEL) and (transformer_model is None or emoji_df is None):
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

    # === 4. Prediction ===
    if USE_TRAINED_MODEL:
        # Use trained classifier (higher accuracy)
        results = _predict_with_trained_model(user_text, top_k)

        # If trained model confidence is too low, blend with transformer
        if results and results[0]["score"] < 0.15 and transformer_model is not None:
            transformer_results = _predict_with_transformer(user_text, top_k)
            # Merge: take trained model results but boost with transformer if needed
            if transformer_results:
                # If transformer has a very different top pick with high score, consider it
                seen = {r["emoji"] for r in results}
                for tr in transformer_results:
                    if tr["emoji"] not in seen and len(results) < top_k * 2:
                        results.append(tr)
                results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    else:
        results = _predict_with_transformer(user_text, top_k)

    return {
        "emojis": results,
        "model_used": "trained_classifier" if USE_TRAINED_MODEL else "transformer",
        "analysis": {
            "preprocessing": {
                "original": preprocessing_result["original"],
                "tokens": preprocessing_result["tokens"][:10],
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

    preprocessing = preprocessor.preprocess(text)
    sentiment = sentiment_analyzer.full_analysis(text)

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
    """Return evaluation metrics from training."""
    if training_metrics:
        return {
            "model_type": "Ensemble (Logistic Regression + Calibrated SVM)",
            "metrics": training_metrics,
            "status": "trained"
        }
    return {
        "metrics_supported": [
            "Accuracy", "Precision", "Recall",
            "F1-Score", "Top-K Accuracy"
        ],
        "note": "No trained model found. Run 'python train_model.py' to train.",
        "status": "untrained"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
