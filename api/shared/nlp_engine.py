"""
NLP Engine for Vercel Serverless Functions
============================================
Ultra-lightweight implementation using ONLY Python stdlib + textblob.
No scikit-learn, no numpy, no scipy — stays well under Vercel's 250MB limit.

Implements TF-IDF and cosine similarity from scratch.
"""
import re
import os
import json
import math
from collections import Counter, defaultdict
from textblob import TextBlob


# ========== Text Preprocessor ==========

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
    "won", "wouldn",
}

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


def _tokenize(text: str) -> list:
    """Tokenize text into unigrams and bigrams, filtering stop words."""
    words = text.lower().split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    tokens = list(words)  # unigrams
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]} {words[i+1]}")  # bigrams
    return tokens


def preprocess(text: str) -> dict:
    """Full preprocessing pipeline."""
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
        "final_text": " ".join(processed),
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
                "revolting", "repulsive", "sickening", "vile"],
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
        "dominant_emotion": dominant,
    }


# ========== Pure-Python TF-IDF Engine ==========

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

# Cache (persisted across warm Vercel invocations)
_engine_cache = {}


def _load_emoji_data():
    """Load and cache the emoji dataset from emoji_lookup.json."""
    if "emojis" in _engine_cache:
        return _engine_cache["emojis"]

    data_path = os.path.join(os.path.dirname(__file__), "emoji_lookup.json")
    with open(data_path, "r", encoding="utf-8") as f:
        lookup = json.load(f)

    emojis = [{"emoji": char, "name": name} for name, char in lookup.items()]
    _engine_cache["emojis"] = emojis
    return emojis


def _build_tfidf_index():
    """
    Build a pure-Python TF-IDF index over emoji descriptions.
    Returns (vocab, idf_map, doc_tfidf_vectors, doc_norms).
    """
    if "tfidf_index" in _engine_cache:
        return _engine_cache["tfidf_index"]

    emojis = _load_emoji_data()

    # Build descriptions
    descriptions = []
    for entry in emojis:
        name = entry["name"]
        extra = ""
        for key, context in EMOJI_CONTEXT_MAP.items():
            if key.lower() in name.lower():
                extra = context
                break
        descriptions.append(f"{name} {extra}" if extra else name)

    # Tokenize all documents
    doc_tokens = [_tokenize(desc) for desc in descriptions]
    num_docs = len(doc_tokens)

    # Build vocabulary and document frequency
    df = Counter()  # document frequency: how many docs contain each term
    for tokens in doc_tokens:
        unique_terms = set(tokens)
        for term in unique_terms:
            df[term] += 1

    # IDF: log(N / df) + 1  (smooth)
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log(num_docs / freq) + 1.0

    # Compute TF-IDF vectors for each document (as sparse dicts)
    doc_vectors = []
    doc_norms = []
    for tokens in doc_tokens:
        tf = Counter(tokens)
        vec = {}
        for term, count in tf.items():
            if term in idf:
                # sublinear TF: 1 + log(tf)
                tf_val = 1.0 + math.log(count) if count > 0 else 0.0
                vec[term] = tf_val * idf[term]

        # L2 norm
        norm = math.sqrt(sum(v * v for v in vec.values())) if vec else 1.0
        doc_vectors.append(vec)
        doc_norms.append(norm)

    index = (idf, doc_vectors, doc_norms)
    _engine_cache["tfidf_index"] = index
    return index


def _cosine_similarity(query_tokens, idf, doc_vectors, doc_norms):
    """Compute cosine similarity between query and all documents."""
    # Build query TF-IDF vector
    qtf = Counter(query_tokens)
    query_vec = {}
    for term, count in qtf.items():
        if term in idf:
            tf_val = 1.0 + math.log(count) if count > 0 else 0.0
            query_vec[term] = tf_val * idf[term]

    query_norm = math.sqrt(sum(v * v for v in query_vec.values())) if query_vec else 1.0

    # Compute cosine similarity against each document
    scores = []
    for doc_vec, doc_norm in zip(doc_vectors, doc_norms):
        dot = 0.0
        for term, q_val in query_vec.items():
            if term in doc_vec:
                dot += q_val * doc_vec[term]
        sim = dot / (query_norm * doc_norm) if (query_norm * doc_norm) > 0 else 0.0
        scores.append(sim)

    return scores


def _get_top_ngrams(text: str, n: int = 3) -> list:
    """Extract top n-grams from text with simple TF scores."""
    tokens = _tokenize(text)
    if not tokens:
        return []
    tf = Counter(tokens)
    total = len(tokens)
    scored = [(term, round(count / total, 3)) for term, count in tf.most_common(n)]
    return scored


# ========== Public API ==========

def predict_emojis(text: str, top_k: int = 2) -> dict:
    """
    Predict emojis using pure-Python TF-IDF + cosine similarity.
    """
    emojis = _load_emoji_data()
    idf, doc_vectors, doc_norms = _build_tfidf_index()

    # 1. Preprocess
    preprocessing_result = preprocess(text)
    processed_text = preprocessing_result["final_text"]

    # 2. Sentiment
    sentiment_result = analyze_sentiment(text)

    # 3. Top n-grams
    top_ngrams = _get_top_ngrams(processed_text, n=3)

    # 4. Enrich query with emotion context
    dominant = sentiment_result.get("dominant_emotion", "")
    enriched = f"{text} {dominant}" if dominant != "neutral" else text

    # 5. Tokenize and compute similarity
    query_tokens = _tokenize(enriched)
    scores = _cosine_similarity(query_tokens, idf, doc_vectors, doc_norms)

    # 6. Rank and deduplicate
    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    seen = set()
    results = []
    for idx, score in indexed_scores:
        entry = emojis[idx]
        if entry["emoji"] not in seen:
            seen.add(entry["emoji"])
            results.append({
                "emoji": entry["emoji"],
                "name": entry["name"],
                "score": round(score, 4),
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
                "final_text": processed_text,
            },
            "sentiment": sentiment_result,
            "features": {
                "top_ngrams": top_ngrams,
            },
        },
    }


def analyze_text(text: str) -> dict:
    """Get detailed NLP analysis without prediction."""
    preprocessing = preprocess(text)
    sentiment = analyze_sentiment(text)
    ngrams = _get_top_ngrams(preprocessing["final_text"], n=5)

    return {
        "preprocessing": preprocessing,
        "sentiment": sentiment,
        "features": {
            "top_ngrams": ngrams,
        },
    }
