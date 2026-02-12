import sys
import os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
import requests

tests = [
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
]

print("\nPrediction Results:")
print("=" * 60)
for text in tests:
    res = requests.post(
        "http://localhost:8000/predict",
        json={"text": text, "top_k": 2}
    ).json()
    emojis = " ".join([e["emoji"] for e in res["emojis"]])
    names = ", ".join([e["name"] for e in res["emojis"]])
    print(f"  {text:30s} => {emojis}  ({names})")
print("=" * 60)
