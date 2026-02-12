"""
NLP Pipeline Module
Implements comprehensive text preprocessing and feature extraction techniques.
"""
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextPreprocessor:
    """
    Handles all text preprocessing tasks:
    - Tokenization
    - Stop-word removal
    - Lemmatization / Stemming
    """
    
    def __init__(self, use_lemmatization=True):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.use_lemmatization = use_lemmatization
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text
    
    def tokenize(self, text: str) -> list:
        """Tokenize text into words."""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: list) -> list:
        """Remove stop words from token list."""
        return [t for t in tokens if t not in self.stop_words]
    
    def lemmatize(self, tokens: list) -> list:
        """Apply lemmatization to tokens."""
        return [self.lemmatizer.lemmatize(t) for t in tokens]
    
    def stem(self, tokens: list) -> list:
        """Apply stemming to tokens."""
        return [self.stemmer.stem(t) for t in tokens]
    
    def preprocess(self, text: str) -> dict:
        """
        Full preprocessing pipeline.
        Returns dict with intermediate steps for transparency.
        """
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        filtered = self.remove_stopwords(tokens)
        
        if self.use_lemmatization:
            processed = self.lemmatize(filtered)
        else:
            processed = self.stem(filtered)
        
        return {
            "original": text,
            "cleaned": cleaned,
            "tokens": tokens,
            "filtered": filtered,
            "processed": processed,
            "final_text": " ".join(processed)
        }


class FeatureExtractor:
    """
    Handles feature extraction:
    - TF-IDF
    - N-gram analysis
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self.is_fitted = False
    
    def fit(self, texts: list):
        """Fit TF-IDF on corpus."""
        self.tfidf.fit(texts)
        self.is_fitted = True
    
    def extract_tfidf(self, text: str) -> np.ndarray:
        """Extract TF-IDF features from text."""
        if not self.is_fitted:
            # Fit on single text if not fitted
            self.tfidf.fit([text])
            self.is_fitted = True
        return self.tfidf.transform([text]).toarray()[0]
    
    def get_top_ngrams(self, text: str, n: int = 5) -> list:
        """Get top N n-grams by TF-IDF score."""
        if not self.is_fitted:
            self.fit([text])
        
        vector = self.tfidf.transform([text])
        feature_names = self.tfidf.get_feature_names_out()
        scores = vector.toarray()[0]
        
        top_indices = scores.argsort()[-n:][::-1]
        return [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]


class SentimentAnalyzer:
    """
    Sentiment and Emotion Analysis:
    - Polarity detection
    - Subjectivity analysis
    - Emotion classification
    """
    
    # Simple emotion keywords mapping
    EMOTION_KEYWORDS = {
        "joy": ["happy", "joy", "excited", "love", "wonderful", "great", "amazing", "fantastic"],
        "sadness": ["sad", "unhappy", "depressed", "sorry", "miss", "lonely", "crying"],
        "anger": ["angry", "mad", "furious", "hate", "annoyed", "frustrated"],
        "fear": ["scared", "afraid", "worried", "anxious", "nervous", "terrified"],
        "surprise": ["surprised", "shocked", "amazed", "unexpected", "wow"],
        "disgust": ["disgusted", "gross", "yuck", "awful", "terrible"]
    }
    
    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment polarity and subjectivity."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # Classify polarity
        if polarity > 0.1:
            sentiment_label = "positive"
        elif polarity < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        return {
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "sentiment": sentiment_label
        }
    
    def detect_emotions(self, text: str) -> dict:
        """Detect emotions based on keyword matching."""
        text_lower = text.lower()
        emotions = {}
        
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                emotions[emotion] = score
        
        # Normalize scores
        total = sum(emotions.values()) if emotions else 1
        emotions = {k: round(v / total, 2) for k, v in emotions.items()}
        
        # Get dominant emotion
        dominant = max(emotions, key=emotions.get) if emotions else "neutral"
        
        return {
            "emotions": emotions,
            "dominant_emotion": dominant
        }
    
    def full_analysis(self, text: str) -> dict:
        """Complete sentiment and emotion analysis."""
        sentiment = self.analyze_sentiment(text)
        emotions = self.detect_emotions(text)
        return {**sentiment, **emotions}


class EvaluationMetrics:
    """
    Evaluation metrics for classifier performance.
    """
    
    @staticmethod
    def accuracy(y_true: list, y_pred: list) -> float:
        """Calculate accuracy."""
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return round(correct / len(y_true), 4) if y_true else 0.0
    
    @staticmethod
    def precision_recall_f1(y_true: list, y_pred: list, average='macro') -> dict:
        """Calculate precision, recall, and F1-score."""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        try:
            precision = precision_score(y_true, y_pred, average=average, zero_division=0)
            recall = recall_score(y_true, y_pred, average=average, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        except:
            precision = recall = f1 = 0.0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }
    
    @staticmethod
    def top_k_accuracy(y_true: list, y_pred_probs: list, k: int = 5) -> float:
        """
        Calculate top-K accuracy.
        y_pred_probs: list of lists containing top-k predictions per sample
        """
        correct = sum(1 for true, preds in zip(y_true, y_pred_probs) if true in preds[:k])
        return round(correct / len(y_true), 4) if y_true else 0.0
