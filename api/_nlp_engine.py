"""
NLP Engine for Vercel Serverless Functions
Lightweight implementation using scikit-learn TF-IDF instead of PyTorch.
"""
import re
import os
import json
import math
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ========== Text Preprocessor ==========

# Common English stop words (avoids NLTK dependency)
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re",
    "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
    "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
    "won", "wouldn"
}

# Simple lemmatization rules (avoids NLTK WordNet dependency)
LEMMA_RULES = {
    "running": "run", "swimming": "swim", "loving": "love", "hating": "hate",
    "making": "make", "going": "go", "coming": "come", "getting": "get",
    "saying": "say", "looking": "look", "thinking": "think", "feeling": "feel",
    "giving": "give", "taking": "take", "seeing": "see", "knowing": "know",
    "wanting": "want", "using": "use", "finding": "find", "telling": "tell",
    "asking": "ask", "working": "work", "calling": "call", "trying": "try",
    "leaving": "leave", "playing": "play", "living": "live", "believing": "believe",
    "bringing": "bring", "happening": "happen", "writing": "write",
    "sitting": "sit", "standing": "stand", "losing": "lose", "paying": "pay",
    "meeting": "meet", "including": "include", "continuing": "continue",
    "setting": "set", "learning": "learn", "changing": "change",
    "leading": "lead", "understanding": "understand", "watching": "watch",
    "following": "follow", "stopping": "stop", "creating": "create",
    "speaking": "speak", "reading": "read", "spending": "spend",
    "growing": "grow", "opening": "open", "walking": "walk", "winning": "win",
    "teaching": "teach", "offering": "offer", "remembering": "remember",
    "considering": "consider", "appearing": "appear", "buying": "buy",
    "serving": "serve", "dying": "die", "sending": "send",
    "building": "build", "staying": "stay", "falling": "fall",
    "cutting": "cut", "reaching": "reach", "killing": "kill",
    "raising": "raise", "passing": "pass", "selling": "sell",
    "deciding": "decide", "returning": "return", "explaining": "explain",
    "hoping": "hope", "developing": "develop", "carrying": "carry",
    "breaking": "break", "receiving": "receive", "agreeing": "agree",
    "supporting": "support", "hitting": "hit", "producing": "produce",
    "eating": "eat", "covering": "cover", "catching": "catch",
    "drawing": "draw", "choosing": "choose",
}


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def simple_lemmatize(word: str) -> str:
    """Simple rule-based lemmatization."""
    if word in LEMMA_RULES:
        return LEMMA_RULES[word]
    # Common suffix rules
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]
    return word


def preprocess(text: str) -> dict:
    """Full preprocessing pipeline - returns dict with intermediate steps."""
    cleaned = clean_text(text)
    tokens = cleaned.split()
    filtered = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    processed = [simple_lemmatize(t) for t in filtered]

    return {
        "original": text,
        "cleaned": cleaned,
        "tokens": tokens,
        "filtered": filtered,
        "processed": processed,
        "final_text": " ".join(processed)
    }


# ========== Sentiment Analyzer ==========

EMOTION_KEYWORDS = {
    "joy": ["happy", "joy", "excited", "love", "wonderful", "great", "amazing",
            "fantastic", "awesome", "delighted", "cheerful", "glad", "pleased",
            "thrilled", "ecstatic", "blissful", "elated", "jubilant", "overjoyed"],
    "sadness": ["sad", "unhappy", "depressed", "sorry", "miss", "lonely", "crying",
                "heartbroken", "miserable", "gloomy", "grief", "melancholy", "sorrow",
                "disappointed", "devastated", "hopeless", "hurt"],
    "anger": ["angry", "mad", "furious", "hate", "annoyed", "frustrated", "rage",
              "irritated", "outraged", "hostile", "bitter", "resentful", "livid"],
    "fear": ["scared", "afraid", "worried", "anxious", "nervous", "terrified",
             "frightened", "panic", "dread", "horror", "alarmed", "uneasy"],
    "surprise": ["surprised", "shocked", "amazed", "unexpected", "wow",
                 "astonished", "stunned", "startled", "bewildered", "astounded"],
    "disgust": ["disgusted", "gross", "yuck", "awful", "terrible", "nasty",
                "revolting", "repulsive", "sickening", "vile"]
}


def analyze_sentiment(text: str) -> dict:
    """Full sentiment and emotion analysis."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0.1:
        sentiment_label = "positive"
    elif polarity < -0.1:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    # Emotion detection
    text_lower = text.lower()
    emotions = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            emotions[emotion] = score

    total = sum(emotions.values()) if emotions else 1
    emotions = {k: round(v / total, 2) for k, v in emotions.items()}
    dominant = max(emotions, key=emotions.get) if emotions else "neutral"

    return {
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3),
        "sentiment": sentiment_label,
        "emotions": emotions,
        "dominant_emotion": dominant
    }


# ========== Emoji Prediction Engine ==========

# Enhanced emoji context map for better matching
EMOJI_CONTEXT_MAP = {
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

# Cache for the engine (persisted across warm invocations)
_engine_cache = {}


def _load_emoji_data():
    """Load and cache the emoji dataset."""
    if "emojis" in _engine_cache:
        return _engine_cache["emojis"]

    data_path = os.path.join(os.path.dirname(__file__), "emoji_list.json")
    with open(data_path, "r", encoding="utf-8") as f:
        emojis = json.load(f)

    _engine_cache["emojis"] = emojis
    return emojis


def _get_tfidf_engine():
    """Get or build the TF-IDF engine (cached for warm invocations)."""
    if "tfidf" in _engine_cache and "matrix" in _engine_cache:
        return _engine_cache["tfidf"], _engine_cache["matrix"]

    emojis = _load_emoji_data()

    # Build rich descriptions for each emoji
    descriptions = []
    for entry in emojis:
        name = entry["name"]
        extra = ""
        for key, context in EMOJI_CONTEXT_MAP.items():
            if key.lower() in name.lower():
                extra = context
                break
        desc = f"{name} {extra}" if extra else name
        descriptions.append(desc)

    # Fit TF-IDF vectorizer on emoji descriptions
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    matrix = tfidf.fit_transform(descriptions)

    _engine_cache["tfidf"] = tfidf
    _engine_cache["matrix"] = matrix
    return tfidf, matrix


def predict_emojis(text: str, top_k: int = 2) -> dict:
    """
    Predict emojis using TF-IDF + cosine similarity.
    Returns prediction results along with NLP analysis.
    """
    emojis = _load_emoji_data()
    tfidf, matrix = _get_tfidf_engine()

    # 1. Preprocess text
    preprocessing_result = preprocess(text)
    processed_text = preprocessing_result["final_text"]

    # 2. Sentiment analysis
    sentiment_result = analyze_sentiment(text)

    # 3. Feature extraction (top n-grams for transparency)
    top_ngrams = []
    try:
        feat_tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        feat_tfidf.fit([processed_text])
        vec = feat_tfidf.transform([processed_text])
        names = feat_tfidf.get_feature_names_out()
        scores = vec.toarray()[0]
        top_idx = scores.argsort()[-3:][::-1]
        top_ngrams = [(names[i], round(float(scores[i]), 3)) for i in top_idx if scores[i] > 0]
    except Exception:
        pass

    # 4. Enrich query with sentiment context for better matching
    sentiment_label = sentiment_result.get("dominant_emotion", "")
    enriched = f"{text} {sentiment_label}" if sentiment_label != "neutral" else text

    # 5. Vectorize user text and compute cosine similarity
    user_vec = tfidf.transform([enriched])
    scores = cosine_similarity(user_vec, matrix)[0]

    # 6. Get top-K unique results
    top_indices = scores.argsort()[::-1]
    seen = set()
    results = []
    for idx in top_indices:
        emoji_entry = emojis[idx]
        if emoji_entry["emoji"] not in seen:
            seen.add(emoji_entry["emoji"])
            results.append({
                "emoji": emoji_entry["emoji"],
                "name": emoji_entry["name"],
                "score": round(float(scores[idx]), 4)
            })
        if len(results) >= top_k:
            break

    return {
        "emojis": results,
        "analysis": {
            "preprocessing": {
                "original": preprocessing_result["original"],
                "tokens": preprocessing_result["tokens"][:10],
                "processed": preprocessing_result["processed"][:10],
                "final_text": processed_text
            },
            "sentiment": sentiment_result,
            "features": {
                "top_ngrams": top_ngrams
            }
        }
    }


def analyze_text(text: str) -> dict:
    """Get detailed NLP analysis without prediction."""
    preprocessing = preprocess(text)
    sentiment = analyze_sentiment(text)

    ngrams = []
    try:
        feat_tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        feat_tfidf.fit([preprocessing["final_text"]])
        vec = feat_tfidf.transform([preprocessing["final_text"]])
        names = feat_tfidf.get_feature_names_out()
        scores = vec.toarray()[0]
        top_idx = scores.argsort()[-5:][::-1]
        ngrams = [(names[i], round(float(scores[i]), 3)) for i in top_idx if scores[i] > 0]
    except Exception:
        pass

    return {
        "preprocessing": preprocessing,
        "sentiment": sentiment,
        "features": {
            "top_ngrams": ngrams
        }
    }
