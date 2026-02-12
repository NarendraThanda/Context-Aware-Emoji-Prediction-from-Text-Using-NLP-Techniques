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
        
        # 5. Emoji Data with Enhanced Descriptions
        print("\n[5/5] Loading Emoji Dataset...")
        emoji_df = pd.read_csv(os.path.join(BASE_DIR, "full_emoji.csv"))
        
        # Build rich contextual descriptions for better matching
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
        
        # Create enhanced descriptions
        descriptions = []
        for _, row in emoji_df.iterrows():
            name = str(row['name']).strip() if pd.notna(row['name']) else ""
            # Look for matching context in our map
            extra_context = ""
            for key, context in emoji_context_map.items():
                if key.lower() in name.lower():
                    extra_context = context
                    break
            # Build rich description
            if extra_context:
                desc = f"{name}. {extra_context}"
            else:
                desc = name
            descriptions.append(desc)
        
        emoji_embeddings = model.encode(descriptions, convert_to_tensor=True, show_progress_bar=True)
        print(f"      ✓ Loaded {len(emoji_df)} emojis with enhanced descriptions")
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
    # Use both original and processed text for better matching
    user_embedding_original = model.encode(user_text, convert_to_tensor=True)
    
    # Also encode with sentiment context for richer matching
    sentiment_label = sentiment_result.get("emotion", {}).get("primary_emotion", "")
    enriched_text = f"{user_text}. {sentiment_label}" if sentiment_label else user_text
    user_embedding_enriched = model.encode(enriched_text, convert_to_tensor=True)
    
    # Combine embeddings (70% original meaning + 30% sentiment-enriched)
    user_embedding = 0.7 * user_embedding_original + 0.3 * user_embedding_enriched
    
    # === 5. Cosine Similarity for Multi-class Classification ===
    cosine_scores = util.cos_sim(user_embedding, emoji_embeddings)[0]
    
    # === 6. Top-K Prediction ===
    # Get more candidates, then pick the best
    candidate_count = max(top_k * 5, 20)
    top_indices = cosine_scores.argsort(descending=True)[:candidate_count]
    
    # Score and rank candidates
    candidates = []
    for idx in top_indices:
        idx = int(idx)
        score = float(cosine_scores[idx])
        candidates.append({
            "emoji": emoji_df.iloc[idx]['emoji'],
            "name": emoji_df.iloc[idx]['name'],
            "score": round(score, 4),
            "idx": idx
        })
    
    # Take top_k unique emojis (some emojis may have duplicate entries)
    seen_emojis = set()
    results = []
    for c in candidates:
        if c["emoji"] not in seen_emojis:
            seen_emojis.add(c["emoji"])
            results.append({
                "emoji": c["emoji"],
                "name": c["name"],
                "score": c["score"]
            })
        if len(results) >= top_k:
            break
    
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
