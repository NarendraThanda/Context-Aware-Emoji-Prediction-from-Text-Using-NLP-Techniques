"""
Context-Aware Emoji Prediction - Model Training Pipeline
=========================================================
Trains a supervised classifier for emoji prediction using:
1. Synthetic training data generation from emoji descriptions + augmentation
2. TF-IDF feature extraction with character n-grams
3. Ensemble of Logistic Regression, SVM, and Random Forest
4. Proper train/test split with evaluation metrics
5. Saves trained model + vectorizer for production use

Run: python train_model.py
"""

import os
import sys
import json
import time
import random
import pickle
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, top_k_accuracy_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from textblob import TextBlob

# ============================================================
# Paths
# ============================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "trained_model")
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")

os.makedirs(MODEL_DIR, exist_ok=True)


class ManualEnsemble:
    """Lightweight ensemble that averages probabilities from pre-fitted models."""
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.classes_ = models[0].classes_ if hasattr(models[0], 'classes_') else None

    def predict_proba(self, X):
        total_weight = sum(self.weights)
        proba = None
        for model, weight in zip(self.models, self.weights):
            p = model.predict_proba(X)
            if proba is None:
                proba = p * (weight / total_weight)
            else:
                proba += p * (weight / total_weight)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)



# ============================================================
# 1. Enhanced Emoji Context Map (much richer than before)
# ============================================================
EMOJI_CONTEXT_MAP = {
    # === FACES & EMOTIONS ===
    "grinning face": [
        "happy", "joy", "smile", "laughing", "cheerful", "excited", "good",
        "great", "amazing", "wonderful", "glad", "pleased", "yay", "woohoo",
        "I'm so happy right now", "feeling great today", "everything is wonderful",
        "this makes me smile", "having a great day", "so excited about this",
        "feeling blessed and happy", "what a good day", "life is good",
        "I'm in such a good mood", "feeling amazing", "great news everyone",
    ],
    "beaming face with smiling eyes": [
        "very happy", "delighted", "overjoyed", "thrilled", "gleeful",
        "so happy I could burst", "this is the best day ever", "pure happiness",
        "feeling wonderful", "I'm beaming", "couldn't be happier",
    ],
    "face with tears of joy": [
        "funny", "hilarious", "laughing", "crying", "lol", "comedy", "joke",
        "humor", "haha", "lmao", "rofl", "dying of laughter", "too funny",
        "I can't stop laughing", "this is hilarious", "funniest thing ever",
        "laughing so hard right now", "that joke killed me", "I'm crying laughing",
        "oh my god that's funny", "dead from laughing", "comedy gold",
    ],
    "rolling on the floor laughing": [
        "hilarious", "dying", "funny", "rofl", "lmao", "too funny",
        "I literally cannot stop laughing", "rolling on the floor",
        "funniest thing I've seen", "I can't breathe from laughing",
        "this killed me", "I'm literally dead", "oh my god hahaha",
    ],
    "smiling face with heart-eyes": [
        "love", "adore", "beautiful", "gorgeous", "amazing", "crush",
        "romantic", "attraction", "stunning", "breathtaking", "lovely",
        "you look so beautiful", "I adore this", "absolutely gorgeous",
        "this is stunning", "I'm in love with this", "so pretty",
        "can't stop staring", "perfect", "dreamy", "heart eyes",
    ],
    "smiling face with hearts": [
        "in love", "loving", "feeling loved", "affection", "adoring",
        "I love you so much", "you make me so happy", "feeling so loved",
        "my heart is full", "so much love", "you're everything to me",
    ],
    "thinking face": [
        "wondering", "thinking", "confused", "hmm", "curious", "pondering",
        "question", "not sure", "interesting", "let me think",
        "I wonder what that means", "hmm that's interesting", "let me think about it",
        "I'm not sure about this", "what do you think", "curious about this",
    ],
    "loudly crying face": [
        "sad", "crying", "upset", "devastated", "heartbroken", "tears",
        "emotional", "pain", "sobbing", "bawling", "weeping",
        "I'm so sad right now", "this makes me cry", "my heart hurts",
        "I can't stop crying", "this is devastating", "feeling so emotional",
        "so heartbroken", "the tears won't stop", "this is so sad",
        "I miss you", "I miss you so much", "missing you badly",
        "miss you every day", "I really miss you", "missing someone",
    ],
    "sad but relieved face": [
        "sad", "relieved", "disappointed but okay", "sighing",
        "at least it's over", "sad but moving on", "feeling down but okay",
        "a bit sad really", "could have been worse", "disappointed",
        "I miss you", "missing you",
    ],
    "angry face": [
        "angry", "mad", "furious", "rage", "upset", "irritated", "annoyed",
        "hate", "pissed", "livid", "outraged", "frustrated",
        "I am so angry right now", "this makes me furious", "I'm really mad",
        "how dare you", "this is unacceptable", "I can't believe this",
        "so frustrated with this", "I'm livid", "that's infuriating",
    ],
    "pouting face": [
        "very angry", "rage", "furious", "extremely mad", "seething",
        "I am absolutely furious", "this is outrageous", "beyond angry",
    ],
    "face with steam from nose": [
        "triumphant", "huffing", "frustrated", "determined angry",
        "steaming mad", "huffing and puffing", "irritated",
    ],
    "fearful face": [
        "scared", "afraid", "fear", "horror", "terrified", "frightened",
        "panic", "alarmed", "anxious", "nervous", "dread",
        "I'm so scared", "this is terrifying", "I'm frightened",
        "I'm having anxiety", "this scares me", "so nervous about this",
    ],
    "face screaming in fear": [
        "shocked", "scared", "horror", "surprised", "omg", "terrified",
        "oh my god", "I can't believe this", "this is horrifying",
        "I'm screaming", "absolutely terrified", "this is horrific",
    ],
    "nauseated face": [
        "sick", "disgusted", "gross", "yuck", "eww", "vomit", "ill",
        "that's disgusting", "I feel sick", "this is gross", "so nasty",
        "I'm going to be sick", "revolting", "makes me nauseous",
    ],
    "sleeping face": [
        "tired", "sleepy", "bored", "exhausted", "zzz", "rest", "nap",
        "I'm so tired", "need to sleep", "about to fall asleep",
        "so sleepy right now", "I need a nap", "can barely keep my eyes open",
        "exhausted after today", "I feel sleepy", "goodnight",
    ],
    "sleepy face": [
        "drowsy", "sleepy", "tired", "half asleep", "yawning",
        "I feel so drowsy", "can't stay awake", "so tired today",
    ],
    "partying face": [
        "celebration", "party", "fun", "birthday", "congratulations",
        "hooray", "woohoo", "celebrate", "festive", "cheers",
        "let's party", "time to celebrate", "happy birthday to you",
        "congratulations to you", "we did it", "party time",
        "let's have fun tonight", "celebration time",
    ],
    "smiling face with sunglasses": [
        "cool", "awesome", "confident", "swagger", "stylish", "boss",
        "deal with it", "too cool", "feeling confident", "looking good",
        "I'm the coolest", "feeling like a boss", "swag",
    ],
    "winking face": [
        "flirting", "playful", "hint", "sly", "joke", "kidding", "wink",
        "just kidding", "you know what I mean", "if you know you know",
        "wink wink", "nudge nudge", "being playful",
    ],
    "smiling face with halo": [
        "innocent", "angel", "good", "pure", "sweet", "kind", "blessed",
        "I'm an angel", "I didn't do anything", "so innocent", "pure soul",
    ],
    "nerd face": [
        "smart", "intelligent", "geek", "nerd", "study", "academic", "clever",
        "you are so smart", "what a genius", "nerdy and proud", "bookworm",
        "aced the test", "top of the class", "intellectual",
    ],
    "face with rolling eyes": [
        "annoyed", "bored", "sarcasm", "whatever", "really", "seriously",
        "oh please", "not this again", "are you kidding me", "eye roll",
        "so done with this", "boring", "I can't even",
    ],
    "hugging face": [
        "hug", "love", "warm", "embrace", "comfort", "friendly", "care",
        "sending you a hug", "virtual hug", "big hugs", "I need a hug",
        "hugging you tight", "comfort and warmth",
    ],
    "star-struck": [
        "amazing", "celebrity", "idol", "wow", "incredible", "starstruck",
        "fan", "wonderful", "I'm starstruck", "meeting my hero",
    ],
    "money-mouth face": [
        "money", "rich", "wealthy", "cash", "dollar", "expensive", "profit",
        "making money", "getting paid", "cha-ching", "jackpot", "lottery",
    ],
    "zany face": [
        "crazy", "wild", "silly", "goofy", "wacky", "ridiculous",
        "feeling crazy", "being silly", "that's wild", "going crazy",
    ],
    "face with hand over mouth": [
        "oops", "giggling", "shocked", "secret", "oh no", "whispering",
        "did I say that", "oopsie", "little secret", "can't believe I said that",
    ],
    "shushing face": [
        "quiet", "secret", "shh", "hush", "silence", "don't tell",
        "keep it a secret", "shh don't tell anyone", "be quiet", "between us",
    ],
    "face savoring food": [
        "delicious", "yummy", "tasty", "mmm", "food", "eating",
        "this food is delicious", "so yummy", "taste so good",
        "the food is amazing", "mmm so tasty", "best meal ever",
    ],
    "disappointed face": [
        "disappointed", "let down", "sad", "unhappy", "bummed",
        "I'm so disappointed", "what a letdown", "not what I expected",
        "feeling let down", "such a disappointment",
    ],
    "worried face": [
        "worried", "concerned", "nervous", "anxious", "uneasy",
        "I'm worried about this", "feeling anxious", "this concerns me",
        "I'm scared", "I am scared",
    ],
    "confounded face": [
        "confused", "frustrated", "ugh", "bewildered", "perplexed",
        "I don't understand", "this is so confusing", "what is going on",
    ],
    "persevering face": [
        "struggling", "persevering", "hanging in there", "tough times",
        "I won't give up", "keeping going", "it's hard but I'm trying",
    ],
    "pleading face": [
        "please", "begging", "puppy eyes", "cute", "pretty please",
        "please help me", "I'm begging you", "can you please", "so cute",
    ],
    "face blowing a kiss": [
        "kiss", "love", "flirty", "mwah", "blow a kiss", "sending love",
        "kisses to you", "love you", "mwah darling", "sending a kiss",
    ],
    "smiling face with open hands": [
        "welcome", "friendly", "open arms", "warm", "accepting",
        "welcome aboard", "come here", "open arms for you",
    ],

    # === HEARTS & LOVE ===
    "red heart": [
        "love", "heart", "romance", "relationship", "care", "affection",
        "passion", "valentine", "I love you", "my heart belongs to you",
        "love you forever", "sending love", "true love", "deep love",
        "you mean everything to me", "with all my heart",
        "I miss you", "miss you so much", "missing you dearly",
    ],
    "broken heart": [
        "heartbreak", "sad", "breakup", "hurt", "pain", "loss", "rejection",
        "cry", "my heart is broken", "we broke up", "this hurts so much",
        "heartbroken and lost", "can't believe it's over", "feeling shattered",
        "I miss you", "missing you", "miss you badly",
    ],
    "sparkling heart": [
        "love", "sparkle", "affection", "adore", "shiny love",
        "my love for you sparkles", "you light up my life",
    ],
    "growing heart": [
        "growing love", "increasing affection", "falling deeper in love",
        "my love for you grows", "love getting stronger",
    ],
    "two hearts": [
        "love", "couple", "romance", "together", "mutual love",
        "we are in love", "hearts together", "couple goals",
    ],
    "beating heart": [
        "heartbeat", "love", "alive", "excitement", "pulse",
        "my heart is beating fast", "heart racing", "so alive",
    ],
    "pink heart": [
        "cute love", "sweet", "caring", "gentle love", "soft love",
        "you're so sweet", "gentle affection",
    ],

    # === HANDS & GESTURES ===
    "thumbs up": [
        "good", "okay", "yes", "agree", "approve", "nice", "well done",
        "great job", "sounds good", "I agree", "perfect", "go ahead",
        "that's great", "nice work", "approved", "thumbs up",
        "great job well done", "good work", "excellent",
    ],
    "thumbs down": [
        "bad", "no", "disagree", "disapprove", "not good", "dislike",
        "that's not good", "I disagree", "thumbs down", "not cool",
    ],
    "clapping hands": [
        "congratulations", "bravo", "applause", "well done", "great",
        "achievement", "success", "amazing job", "standing ovation",
        "great job well done", "congratulations to you", "bravo",
        "well deserved", "round of applause",
    ],
    "folded hands": [
        "please", "thank you", "prayer", "grateful", "hope", "wish", "bless",
        "thank you so much", "I'm so grateful", "praying for you",
        "please help", "hoping for the best", "bless you", "namaste",
        "thanks a lot", "thank you very much", "so thankful",
        "grateful for everything", "thanks for your help",
        "really appreciate it", "many thanks", "deeply grateful",
    ],
    "raised fist": [
        "power", "solidarity", "fight", "resistance", "strong",
        "stay strong", "power to the people", "we fight together",
    ],
    "victory hand": [
        "peace", "victory", "win", "success", "two", "peace sign",
        "we won", "peace and love", "feeling victorious",
    ],
    "love-you gesture": [
        "love you", "I love you", "sign language love", "love gesture",
        "I love you so much", "love you forever",
    ],
    "waving hand": [
        "hello", "hi", "goodbye", "greeting", "hey", "welcome", "wave",
        "hey there", "hello everyone", "goodbye friend", "hi there",
        "see you later", "waving at you", "good morning",
    ],
    "raising hands": [
        "celebration", "praise", "hooray", "yay", "hallelujah",
        "raise the roof", "we did it", "hands up", "celebrating",
    ],
    "open hands": [
        "hug", "open", "giving", "receiving", "jazz hands",
        "giving you a hug", "open arms", "come here",
    ],
    "flexed biceps": [
        "strong", "strength", "gym", "workout", "fitness", "power",
        "exercise", "flex", "muscle", "💪", "getting stronger",
        "hit the gym", "workout complete", "feeling strong", "beast mode",
    ],
    "writing hand": [
        "writing", "note", "studying", "homework", "author",
        "taking notes", "writing an essay", "doing homework",
    ],
    "palms up together": [
        "offering", "open", "prayer", "please", "receiving",
        "here you go", "offering to you", "open palms",
    ],
    "crossed fingers": [
        "hope", "luck", "wish", "fingers crossed", "hoping",
        "fingers crossed for you", "hoping for the best", "good luck",
    ],

    # === NATURE & WEATHER ===
    "sun": [
        "sunny", "weather", "bright", "warm", "morning", "sunshine",
        "beautiful day", "summer", "hot", "bright and warm",
        "good morning sunshine", "what a sunny day", "beautiful weather",
        "the sun is shining", "summer vibes", "soaking up the sun",
        "what a beautiful day",
    ],
    "sun with face": [
        "sunny", "happy sun", "good morning", "bright day", "cheerful",
        "the sun is smiling", "good morning world",
    ],
    "cloud": [
        "cloudy", "overcast", "weather", "gray sky",
        "it's cloudy today", "overcast skies", "looks like rain",
    ],
    "rainbow": [
        "colorful", "diversity", "pride", "beautiful", "hope", "promise",
        "rainbow after the storm", "beautiful rainbow", "colorful sky",
    ],
    "star": [
        "star", "night", "shine", "bright", "twinkle",
        "you're a star", "shining bright", "starry night",
    ],
    "glowing star": [
        "brilliant", "shining", "magical", "wonderful", "excellent",
        "you're a shining star", "brilliant performance",
    ],
    "snowflake": [
        "cold", "winter", "snow", "frozen", "ice", "chilly",
        "it's snowing", "winter wonderland", "so cold outside",
    ],
    "fire": [
        "hot", "amazing", "lit", "awesome", "cool", "fire", "trending",
        "popular", "excellent", "this is fire", "so hot right now",
        "that's lit", "absolutely amazing", "on fire", "burning hot",
    ],
    "water wave": [
        "ocean", "sea", "wave", "beach", "surfing", "water",
        "at the beach", "ocean waves", "surfing today", "by the sea",
    ],
    "sunrise": [
        "morning", "new day", "sunrise", "dawn", "early",
        "good morning", "beautiful sunrise", "new beginnings",
    ],
    "sunset": [
        "evening", "sunset", "dusk", "beautiful sky", "golden hour",
        "beautiful sunset", "watching the sunset", "golden hour vibes",
        "what a beautiful day", "gorgeous sky", "beautiful evening",
    ],
    "full moon": [
        "night", "moon", "lunar", "moonlight", "nighttime",
        "beautiful moon tonight", "moonlight shining",
    ],
    "crescent moon": [
        "night", "goodnight", "moon", "evening", "dark",
        "goodnight everyone", "sweet dreams", "crescent moon tonight",
    ],

    # === ANIMALS ===
    "dog face": [
        "dog", "puppy", "pet", "cute", "animal", "adorable", "woof",
        "good boy", "my puppy", "cute dog", "doggo", "pupper",
    ],
    "cat face": [
        "cat", "kitty", "pet", "meow", "animal", "cute", "feline",
        "my cat", "kitty cat", "cute kitten", "meowing",
    ],
    "monkey face": [
        "monkey", "playful", "silly", "cheeky", "primate",
        "monkeying around", "silly monkey",
    ],
    "unicorn": [
        "magical", "fantasy", "unique", "special", "mythical",
        "you're a unicorn", "magical creature", "so special",
    ],
    "butterfly": [
        "beautiful", "nature", "transformation", "change", "growth",
        "pretty", "you're beautiful like a butterfly", "transformation journey",
    ],
    "bird": [
        "freedom", "flying", "nature", "tweet", "singing",
        "free as a bird", "birds singing", "nature sounds",
    ],

    # === FOOD & DRINK ===
    "pizza": [
        "food", "hungry", "eating", "delicious", "dinner", "lunch", "yummy",
        "let's get pizza", "I'm hungry", "pizza night", "food time",
        "the food is delicious",
    ],
    "hamburger": [
        "food", "burger", "hungry", "fast food", "eating",
        "let's get burgers", "burger time", "so hungry",
    ],
    "hot beverage": [
        "coffee", "tea", "morning", "caffeine", "drink", "energy", "wake up",
        "need my morning coffee", "tea time", "coffee break",
        "can barely keep my eyes open", "wake me up",
    ],
    "tropical drink": [
        "vacation", "cocktail", "party", "beach", "summer",
        "vacation vibes", "cocktail hour", "beach day",
    ],
    "beer mug": [
        "beer", "drinks", "alcohol", "party", "cheers", "celebration", "bar",
        "let's party tonight", "cheers mate", "beer o'clock", "drinks tonight",
    ],
    "clinking beer mugs": [
        "cheers", "toast", "celebration", "party", "drinking",
        "cheers to that", "let's celebrate", "party tonight",
    ],
    "wine glass": [
        "wine", "classy", "dinner", "elegant", "celebration",
        "wine night", "classy evening", "wine and dine",
    ],
    "birthday cake": [
        "birthday", "cake", "celebration", "candles", "party",
        "happy birthday to you", "birthday celebration", "birthday party",
        "make a wish", "blow out the candles",
    ],
    "cooking": [
        "cooking", "chef", "kitchen", "food preparation",
        "cooking dinner", "in the kitchen", "chef mode",
    ],
    "pot of food": [
        "cooking", "dinner", "soup", "stew", "homemade",
        "the food is delicious", "homemade dinner", "cooking something good",
    ],
    "ice cream": [
        "dessert", "sweet", "treat", "summer", "cold",
        "ice cream time", "sweet treat", "summer desserts",
    ],
    "candy": [
        "sweet", "candy", "treat", "sugar", "yummy",
        "candy time", "sweet tooth", "so sweet",
    ],

    # === OBJECTS & SYMBOLS ===
    "sparkles": [
        "magic", "beautiful", "amazing", "shine", "new", "sparkle",
        "glitter", "special", "clean", "fresh", "brand new",
        "sparkly and new", "magical moment", "feeling special",
    ],
    "rocket": [
        "launch", "startup", "fast", "speed", "progress", "technology",
        "moon", "success", "to the moon", "launching now", "let's go",
        "full speed ahead", "blast off",
    ],
    "trophy": [
        "winner", "champion", "success", "achievement", "first place",
        "victory", "great job well done", "we won", "champion",
    ],
    "musical note": [
        "music", "song", "singing", "melody", "tune", "rhythm", "listening",
        "love this song", "music to my ears", "singing along",
    ],
    "camera": [
        "photo", "picture", "photography", "selfie", "memory", "snapshot",
        "take a picture", "photo time", "say cheese",
    ],
    "book": [
        "reading", "study", "learning", "education", "knowledge", "literature",
        "reading a book", "study time", "love to read",
    ],
    "laptop": [
        "computer", "work", "technology", "coding", "programming", "developer",
        "working on my laptop", "coding session", "tech life",
    ],
    "crown": [
        "king", "queen", "royal", "boss", "leader", "power", "royalty",
        "feeling like a queen", "king vibes", "boss moves",
    ],
    "hundred points": [
        "perfect", "score", "hundred", "percent", "absolutely", "totally",
        "agree", "completely", "one hundred percent", "keeping it real",
        "facts", "absolutely right", "total perfection",
    ],
    "skull": [
        "dead", "dying", "hilarious", "literally dead", "OMG", "I cant",
        "I'm literally dead", "dying of laughter", "that killed me",
    ],
    "brain": [
        "smart", "intelligent", "thinking", "genius", "mind", "knowledge",
        "clever", "big brain move", "genius idea", "you are so smart",
        "what a brain", "intellectual",
    ],
    "gift": [
        "present", "birthday", "surprise", "gift", "giving", "celebration",
        "got a gift", "it's a surprise", "present for you",
    ],
    "balloon": [
        "party", "celebration", "birthday", "fun", "festive", "happy",
        "birthday party", "celebration time", "festive mood",
    ],
    "eyes": [
        "looking", "watching", "see", "staring", "curious", "notice",
        "attention", "check this out", "look at this", "did you see that",
    ],
    "right anger bubble": [
        "anger", "mad", "furious", "rage", "annoyed", "irritated",
        "I am so angry right now", "this makes me so mad",
    ],

    # === PLANTS & NATURE ===
    "rose": [
        "love", "romance", "flower", "beautiful", "valentine", "date",
        "romantic", "a rose for you", "romantic gesture", "beautiful flower",
    ],
    "sunflower": [
        "happy", "bright", "cheerful", "flower", "nature", "yellow",
        "sunshine", "bright and cheerful", "sunflower fields",
    ],
    "cherry blossom": [
        "spring", "beautiful", "japan", "delicate", "pretty",
        "cherry blossom season", "so pretty", "spring vibes",
    ],
    "bouquet": [
        "flowers", "gift", "romantic", "congratulations", "beautiful",
        "bouquet of flowers", "flower gift",
    ],
    "four leaf clover": [
        "lucky", "luck", "fortune", "irish", "blessed",
        "feeling lucky", "good luck to you", "lucky day",
    ],
    "Christmas tree": [
        "christmas", "holiday", "festive", "merry", "december", "winter",
        "celebration", "merry christmas", "holiday season",
    ],

    # === TRAVEL & PLACES ===
    "airplane": [
        "travel", "flying", "vacation", "trip", "journey", "international",
        "flight", "going on vacation", "flying somewhere", "airport",
    ],
    "car": [
        "driving", "travel", "road trip", "automobile", "vehicle",
        "transportation", "let's go for a drive", "road trip time",
    ],
    "house": [
        "home", "family", "living", "housing", "domestic", "shelter",
        "comfort", "going home", "home sweet home", "at home",
    ],
    "earth globe americas": [
        "world", "global", "planet", "earth", "international", "travel",
        "nature", "around the world", "global community",
    ],

    # === ACTIVITIES ===
    "soccer ball": [
        "football", "soccer", "sports", "game", "playing",
        "playing soccer", "football match", "goal",
    ],
    "basketball": [
        "basketball", "sports", "game", "playing", "hoops",
        "playing basketball", "shooting hoops",
    ],
    "tennis": [
        "tennis", "sports", "game", "playing", "racket",
        "tennis match", "playing tennis",
    ],
    "video game": [
        "gaming", "play", "video game", "gamer", "controller",
        "playing video games", "gamer life", "gaming session",
    ],

    # === MISC ===
    "check mark button": [
        "done", "complete", "yes", "correct", "approved", "finished",
        "success", "task done", "completed", "mission accomplished",
    ],
    "cross mark": [
        "no", "wrong", "incorrect", "error", "rejected", "failed", "cancel",
        "that's wrong", "incorrect answer",
    ],
    "warning": [
        "caution", "alert", "danger", "warning", "careful", "attention",
        "beware", "be careful", "watch out", "danger ahead",
    ],
    "exclamation mark": [
        "important", "attention", "urgent", "alert", "hey",
        "pay attention", "this is important", "urgent matter",
    ],
    "question mark": [
        "question", "confused", "asking", "what", "why", "how",
        "I have a question", "what do you mean", "can you explain",
    ],

    # === BABY & PEOPLE ===
    "baby": [
        "baby", "newborn", "child", "infant", "little one",
        "baby news", "little baby", "having a baby",
        "baby shower", "newborn baby", "baby announcement",
    ],
    "person facepalming": [
        "facepalm", "embarrassed", "disappointed", "can't believe",
        "oh no", "really", "are you serious", "I can't believe this",
    ],
    "person shrugging": [
        "shrug", "don't know", "whatever", "idk", "not sure",
        "I don't know", "no idea", "who knows", "whatever happens",
    ],
}


# ============================================================
# 2. Training Data Generation with Augmentation
# ============================================================

def augment_text(text):
    """Generate augmented versions of a text for more training data."""
    augmented = [text]
    words = text.split()

    # 1. Random word dropout (remove 1 word if enough words)
    if len(words) > 3:
        drop_idx = random.randint(0, len(words) - 1)
        augmented.append(" ".join(w for i, w in enumerate(words) if i != drop_idx))

    # 2. Random word swap
    if len(words) > 2:
        i, j = random.sample(range(len(words)), 2)
        swapped = words.copy()
        swapped[i], swapped[j] = swapped[j], swapped[i]
        augmented.append(" ".join(swapped))

    # 3. Add filler words
    fillers = ["really", "so", "very", "just", "actually", "totally", "like"]
    if len(words) > 1:
        pos = random.randint(0, len(words) - 1)
        filler = random.choice(fillers)
        aug = words[:pos] + [filler] + words[pos:]
        augmented.append(" ".join(aug))

    # 4. Repeat a key word for emphasis
    if len(words) > 1:
        key = random.choice(words)
        augmented.append(text + f" {key} {key}")

    return augmented


def generate_training_data(emoji_df, context_map, augment=True, samples_per_class=50):
    """
    Generate training data by mapping emoji names to context phrases.
    Uses augmentation to increase dataset size.
    """
    texts = []
    labels = []
    emoji_chars = []

    # Build a reverse lookup: emoji_name -> row index
    name_to_idx = {}
    for idx, row in emoji_df.iterrows():
        name = str(row['name']).strip().lower()
        name_to_idx[name] = idx

    matched_emojis = 0
    unmatched_keys = []

    for context_key, phrases in context_map.items():
        # Find matching emoji in dataset
        matched_idx = None
        context_lower = context_key.lower()

        # Try exact match first
        for idx, row in emoji_df.iterrows():
            name = str(row['name']).strip().lower()
            if context_lower == name or context_lower in name:
                matched_idx = idx
                break

        # Try partial match
        if matched_idx is None:
            for idx, row in emoji_df.iterrows():
                name = str(row['name']).strip().lower()
                # Check if any significant word matches
                key_words = context_lower.split()
                if any(kw in name for kw in key_words if len(kw) > 3):
                    matched_idx = idx
                    break

        if matched_idx is None:
            unmatched_keys.append(context_key)
            continue

        matched_emojis += 1
        emoji_char = emoji_df.iloc[matched_idx]['emoji']
        emoji_name = emoji_df.iloc[matched_idx]['name']

        for phrase in phrases:
            if augment:
                augmented_phrases = augment_text(phrase)
                for aug_phrase in augmented_phrases:
                    texts.append(aug_phrase)
                    labels.append(emoji_name)
                    emoji_chars.append(emoji_char)
            else:
                texts.append(phrase)
                labels.append(emoji_name)
                emoji_chars.append(emoji_char)

    print(f"   Matched {matched_emojis}/{len(context_map)} emoji categories")
    if unmatched_keys:
        print(f"   Unmatched keys (will use name-only fallback): {unmatched_keys[:5]}...")

    # Also add emoji names themselves as training data
    for idx, row in emoji_df.iterrows():
        name = str(row['name']).strip()
        if name and name != 'nan':
            texts.append(name.lower())
            labels.append(name)
            emoji_chars.append(row['emoji'])
            # Add the name with some context
            texts.append(f"I feel like {name.lower()}")
            labels.append(name)
            emoji_chars.append(row['emoji'])
            # Add more variations to ensure minimum samples per class
            texts.append(f"{name.lower()} expression")
            labels.append(name)
            emoji_chars.append(row['emoji'])
            texts.append(f"this is {name.lower()}")
            labels.append(name)
            emoji_chars.append(row['emoji'])
            texts.append(f"showing {name.lower()}")
            labels.append(name)
            emoji_chars.append(row['emoji'])

    # Ensure minimum 5 samples per class for stratification
    from collections import Counter
    label_counts = Counter(labels)
    min_samples = 5
    extra_texts, extra_labels, extra_emoji_chars = [], [], []
    for label, count in label_counts.items():
        if count < min_samples:
            # Find existing samples for this label
            indices = [i for i, l in enumerate(labels) if l == label]
            needed = min_samples - count
            for _ in range(needed):
                src_idx = random.choice(indices)
                extra_texts.append(texts[src_idx] + " " + random.choice(["yeah", "really", "totally", "wow", "oh"]))
                extra_labels.append(label)
                extra_emoji_chars.append(emoji_chars[src_idx])
    texts.extend(extra_texts)
    labels.extend(extra_labels)
    emoji_chars.extend(extra_emoji_chars)

    return texts, labels, emoji_chars


# ============================================================
# 3. Model Training
# ============================================================

def train_model(texts, labels, emoji_chars):
    """Train the ensemble classifier and return model + metrics."""

    print("\n" + "=" * 60)
    print("  MODEL TRAINING")
    print("=" * 60)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print(f"\n   Total samples: {len(texts)}")
    print(f"   Total classes: {len(le.classes_)}")
    print(f"   Avg samples per class: {len(texts) / len(le.classes_):.1f}")

    # Filter out classes with too few samples for stratification
    from collections import Counter
    class_counts = Counter(y)
    min_count = 2  # Need at least 2 for stratified split
    valid_mask = [class_counts[yi] >= min_count for yi in y]
    texts_filtered = [t for t, v in zip(texts, valid_mask) if v]
    y_filtered = np.array([yi for yi, v in zip(y, valid_mask) if v])

    # Train/Test split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
    )

    print(f"\n   Training set: {len(X_train_text)} samples")
    print(f"   Test set:     {len(X_test_text)} samples")

    # TF-IDF Vectorization with character n-grams for robustness
    print("\n   [1/4] Building TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 3),         # unigrams, bigrams, trigrams
        analyzer='word',
        sublinear_tf=True,          # Apply sublinear TF scaling
        min_df=1,
        max_df=0.95,
        strip_accents='unicode',
    )

    # Also build character-level features
    tfidf_char = TfidfVectorizer(
        max_features=10000,
        ngram_range=(2, 5),         # character ngrams 2-5
        analyzer='char_wb',
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
    )

    X_train_word = tfidf.fit_transform(X_train_text)
    X_test_word = tfidf.transform(X_test_text)

    X_train_char = tfidf_char.fit_transform(X_train_text)
    X_test_char = tfidf_char.transform(X_test_text)

    # Stack word + char features
    from scipy.sparse import hstack
    X_train = hstack([X_train_word, X_train_char])
    X_test = hstack([X_test_word, X_test_char])

    print(f"   Word features: {X_train_word.shape[1]}")
    print(f"   Char features: {X_train_char.shape[1]}")
    print(f"   Total features: {X_train.shape[1]}")

    # Train individual models
    print("\n   [2/3] Training Logistic Regression...")
    t0 = time.time()
    lr = LogisticRegression(
        C=5.0,
        max_iter=2000,
        solver='lbfgs',
        multi_class='multinomial',
        class_weight='balanced',
        n_jobs=-1,
    )
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    print(f"         Accuracy: {lr_acc:.4f} ({time.time() - t0:.1f}s)")

    print("\n   [3/3] Training Calibrated SVM...")
    t0 = time.time()
    svm_base = LinearSVC(
        C=1.0,
        max_iter=3000,
        class_weight='balanced',
        random_state=42,
    )
    svm = CalibratedClassifierCV(svm_base, cv=3)
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))
    print(f"         Accuracy: {svm_acc:.4f} ({time.time() - t0:.1f}s)")

    # Build manual soft-voting ensemble (avoids refitting)
    print("\n   Building Ensemble (Manual Soft Voting)...")

    ensemble = ManualEnsemble([lr, svm], weights=[3, 2])

    # Evaluate
    y_pred = ensemble.predict(X_test)
    y_pred_proba = ensemble.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Top-K accuracy
    top3_acc = top_k_accuracy_score(y_test, y_pred_proba, k=min(3, len(le.classes_)))
    top5_acc = top_k_accuracy_score(y_test, y_pred_proba, k=min(5, len(le.classes_)))

    metrics = {
        "individual_models": {
            "logistic_regression": round(lr_acc, 4),
            "calibrated_svm": round(svm_acc, 4),
        },
        "ensemble": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "top_3_accuracy": round(top3_acc, 4),
            "top_5_accuracy": round(top5_acc, 4),
        },
        "dataset": {
            "total_samples": len(texts),
            "training_samples": len(X_train_text),
            "test_samples": len(X_test_text),
            "num_classes": len(le.classes_),
        }
    }

    return ensemble, tfidf, tfidf_char, le, metrics


# ============================================================
# 4. Save Model & Artifacts
# ============================================================

def save_model(ensemble, tfidf, tfidf_char, le, emoji_df, metrics):
    """Save trained model and all necessary artifacts."""

    print("\n" + "=" * 60)
    print("  SAVING MODEL")
    print("=" * 60)

    # Save the ensemble model
    model_path = os.path.join(MODEL_DIR, "emoji_classifier.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble, f)
    print(f"   [OK] Saved model: {model_path}")

    # Save TF-IDF vectorizers
    tfidf_word_path = os.path.join(MODEL_DIR, "tfidf_word.pkl")
    with open(tfidf_word_path, 'wb') as f:
        pickle.dump(tfidf, f)
    print(f"   [OK] Saved word vectorizer: {tfidf_word_path}")

    tfidf_char_path = os.path.join(MODEL_DIR, "tfidf_char.pkl")
    with open(tfidf_char_path, 'wb') as f:
        pickle.dump(tfidf_char, f)
    print(f"   [OK] Saved char vectorizer: {tfidf_char_path}")

    # Save label encoder
    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"   [OK] Saved label encoder: {le_path}")

    # Save emoji lookup (name -> emoji char)
    emoji_lookup = {}
    for _, row in emoji_df.iterrows():
        name = str(row['name']).strip()
        emoji_lookup[name] = row['emoji']

    lookup_path = os.path.join(MODEL_DIR, "emoji_lookup.json")
    with open(lookup_path, 'w', encoding='utf-8') as f:
        json.dump(emoji_lookup, f, ensure_ascii=False, indent=2)
    print(f"   [OK] Saved emoji lookup: {lookup_path}")

    # Save metrics
    metrics_path = os.path.join(MODEL_DIR, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   [OK] Saved metrics: {metrics_path}")

    print(f"\n   All artifacts saved to: {MODEL_DIR}")


# ============================================================
# 5. Test Predictions
# ============================================================

def test_predictions(ensemble, tfidf, tfidf_char, le, emoji_df):
    """Test the trained model on sample inputs."""
    from scipy.sparse import hstack

    print("\n" + "=" * 60)
    print("  TEST PREDICTIONS")
    print("=" * 60)

    # Build emoji lookup
    emoji_lookup = {}
    for _, row in emoji_df.iterrows():
        name = str(row['name']).strip()
        emoji_lookup[name] = row['emoji']

    test_cases = [
        "I love you so much",
        "I am very sad today",
        "This is hilarious LOL",
        "Good morning sunshine",
        "Lets party tonight",
        "I am so angry right now",
        "Happy birthday to you",
        "I feel sleepy",
        "You are so smart",
        "The food is delicious",
        "I miss you",
        "Great job well done",
        "I am scared",
        "What a beautiful day",
        "Thank you so much",
        "Let's go to the gym",
        "I need coffee",
        "Congratulations on your success",
        "I'm so confused right now",
        "Gaming all night",
    ]

    print()
    for text in test_cases:
        X_word = tfidf.transform([text])
        X_char = tfidf_char.transform([text])
        X = hstack([X_word, X_char])

        proba = ensemble.predict_proba(X)[0]
        top_indices = proba.argsort()[::-1][:2]

        results = []
        for idx in top_indices:
            label_name = le.inverse_transform([idx])[0]
            emoji_char = emoji_lookup.get(label_name, "?")
            score = proba[idx]
            results.append(f"{emoji_char} ({score:.2f})")

        print(f"  {text:40s} => {' | '.join(results)}")

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 60)
    print("  Context-Aware Emoji Prediction")
    print("  MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Load emoji dataset
    print("\n[1/5] Loading emoji dataset...")
    csv_path = os.path.join(DATA_DIR, "full_emoji.csv")
    emoji_df = pd.read_csv(csv_path)
    print(f"   Loaded {len(emoji_df)} emojis")

    # Generate training data
    print("\n[2/5] Generating training data with augmentation...")
    texts, labels, emoji_chars = generate_training_data(
        emoji_df, EMOJI_CONTEXT_MAP, augment=True
    )
    print(f"   Generated {len(texts)} training samples")
    print(f"   Covering {len(set(labels))} emoji classes")

    # Train model
    print("\n[3/5] Training ensemble model...")
    ensemble, tfidf, tfidf_char, le, metrics = train_model(
        texts, labels, emoji_chars
    )

    # Print metrics
    print("\n" + "=" * 60)
    print("  TRAINING RESULTS")
    print("=" * 60)
    print(f"\n   Individual Model Accuracy:")
    for model_name, acc in metrics["individual_models"].items():
        bar = "█" * int(acc * 30) + "░" * (30 - int(acc * 30))
        print(f"     {model_name:25s} {bar} {acc:.2%}")

    print(f"\n   Ensemble Metrics:")
    ens = metrics["ensemble"]
    print(f"     Accuracy:      {ens['accuracy']:.2%}")
    print(f"     Precision:     {ens['precision']:.2%}")
    print(f"     Recall:        {ens['recall']:.2%}")
    print(f"     F1-Score:      {ens['f1_score']:.2%}")
    print(f"     Top-3 Acc:     {ens['top_3_accuracy']:.2%}")
    print(f"     Top-5 Acc:     {ens['top_5_accuracy']:.2%}")

    # Save model
    print("\n[4/5] Saving model artifacts...")
    save_model(ensemble, tfidf, tfidf_char, le, emoji_df, metrics)

    # Test predictions
    print("\n[5/5] Testing predictions...")
    test_predictions(ensemble, tfidf, tfidf_char, le, emoji_df)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print(f"  Model saved to: {MODEL_DIR}/")
    print("  Run 'python main.py' to start the app with the trained model")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
