"""
Context-Aware Emoji Prediction - Accuracy Evaluation
=====================================================
Shows detailed accuracy metrics and per-category prediction results.

Run: python evaluate_accuracy.py
"""
import os
import sys
import json
import pickle
import numpy as np

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "trained_model")

# Need this for unpickling ManualEnsemble
sys.path.insert(0, ROOT_DIR)
from train_model import ManualEnsemble  # noqa: F401 - needed for pickle


def load_model():
    """Load all trained model artifacts."""
    paths = {
        "classifier": os.path.join(MODEL_DIR, "emoji_classifier.pkl"),
        "tfidf_word": os.path.join(MODEL_DIR, "tfidf_word.pkl"),
        "tfidf_char": os.path.join(MODEL_DIR, "tfidf_char.pkl"),
        "label_encoder": os.path.join(MODEL_DIR, "label_encoder.pkl"),
        "emoji_lookup": os.path.join(MODEL_DIR, "emoji_lookup.json"),
        "metrics": os.path.join(MODEL_DIR, "training_metrics.json"),
    }

    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        print(f"ERROR: Missing model files: {missing}")
        print("Run 'python train_model.py' first to train the model.")
        sys.exit(1)

    with open(paths["classifier"], 'rb') as f:
        classifier = pickle.load(f)
    with open(paths["tfidf_word"], 'rb') as f:
        tfidf_word = pickle.load(f)
    with open(paths["tfidf_char"], 'rb') as f:
        tfidf_char = pickle.load(f)
    with open(paths["label_encoder"], 'rb') as f:
        label_encoder = pickle.load(f)
    with open(paths["emoji_lookup"], 'r', encoding='utf-8') as f:
        emoji_lookup = json.load(f)
    with open(paths["metrics"], 'r') as f:
        metrics = json.load(f)

    return classifier, tfidf_word, tfidf_char, label_encoder, emoji_lookup, metrics


def predict(text, classifier, tfidf_word, tfidf_char, label_encoder, emoji_lookup, top_k=2):
    """Predict emojis for a given text."""
    from scipy.sparse import hstack

    X_word = tfidf_word.transform([text])
    X_char = tfidf_char.transform([text])
    X = hstack([X_word, X_char])

    proba = classifier.predict_proba(X)[0]
    top_indices = proba.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        label_name = label_encoder.inverse_transform([idx])[0]
        emoji_char = emoji_lookup.get(label_name, "?")
        score = float(proba[idx])
        results.append({
            "emoji": emoji_char,
            "name": label_name,
            "score": score,
        })

    return results


def main():
    print()
    print("=" * 70)
    print("  Context-Aware Emoji Prediction")
    print("  ACCURACY EVALUATION REPORT")
    print("=" * 70)

    # Load model
    print("\n  Loading trained model...")
    classifier, tfidf_word, tfidf_char, label_encoder, emoji_lookup, metrics = load_model()
    print("  Model loaded successfully!\n")

    # ========================================
    # Section 1: Training Metrics Summary
    # ========================================
    print("=" * 70)
    print("  1. TRAINING METRICS SUMMARY")
    print("=" * 70)

    ind = metrics.get("individual_models", {})
    ens = metrics.get("ensemble", {})
    ds = metrics.get("dataset", {})

    print(f"\n  Dataset:")
    print(f"    Total samples:    {ds.get('total_samples', 'N/A'):,}")
    print(f"    Training samples: {ds.get('training_samples', 'N/A'):,}")
    print(f"    Test samples:     {ds.get('test_samples', 'N/A'):,}")
    print(f"    Emoji classes:    {ds.get('num_classes', 'N/A'):,}")

    print(f"\n  Individual Model Accuracy:")
    for model_name, acc in ind.items():
        bar = "█" * int(acc * 40) + "░" * (40 - int(acc * 40))
        print(f"    {model_name:30s} {bar} {acc:.2%}")

    print(f"\n  Ensemble Performance:")
    print(f"    ┌──────────────────────────────────────────────────┐")
    print(f"    │  Accuracy:      {ens.get('accuracy', 0):.2%}                            │")
    print(f"    │  Precision:     {ens.get('precision', 0):.2%}                            │")
    print(f"    │  Recall:        {ens.get('recall', 0):.2%}                            │")
    print(f"    │  F1-Score:      {ens.get('f1_score', 0):.2%}                            │")
    print(f"    │  Top-3 Acc:     {ens.get('top_3_accuracy', 0):.2%}                            │")
    print(f"    │  Top-5 Acc:     {ens.get('top_5_accuracy', 0):.2%}                            │")
    print(f"    └──────────────────────────────────────────────────┘")

    # ========================================
    # Section 2: Test Predictions with Expected Results
    # ========================================
    print(f"\n{'=' * 70}")
    print("  2. PREDICTION TEST RESULTS")
    print("=" * 70)

    # Test cases with expected emojis (human-verified)
    test_cases = [
        # (input_text, expected_category, expected_emoji_names)
        ("I love you so much", "Love", ["love-you gesture", "smiling face with hearts", "red heart", "smiling face with heart-eyes"]),
        ("I am very sad today", "Sadness", ["sad but relieved face", "loudly crying face", "disappointed face", "sleepy face", "worried face"]),
        ("This is hilarious LOL", "Humor", ["rolling on the floor laughing", "face with tears of joy", "skull"]),
        ("Good morning sunshine", "Morning", ["sun", "sun with face", "sunrise", "smiling face with sunglasses", "waving hand"]),
        ("Lets party tonight", "Party", ["clinking beer mugs", "beer mug", "partying face"]),
        ("I am so angry right now", "Anger", ["angry face", "pouting face", "right anger bubble", "face with steam from nose"]),
        ("Happy birthday to you", "Birthday", ["birthday cake", "partying face", "balloon", "gift"]),
        ("I feel sleepy", "Sleepy", ["sleepy face", "sleeping face"]),
        ("You are so smart", "Intelligence", ["brain", "nerd face"]),
        ("The food is delicious", "Food", ["face savoring food", "pot of food", "cooking", "pizza"]),
        ("I miss you", "Missing", ["loudly crying face", "red heart", "broken heart", "sad but relieved face"]),
        ("Great job well done", "Appreciation", ["thumbs up", "clapping hands", "trophy"]),
        ("I am scared", "Fear", ["worried face", "fearful face", "face screaming in fear"]),
        ("What a beautiful day", "Beauty", ["sun", "sunset", "rainbow", "sparkles", "smiling face with sunglasses", "smiling face with heart-eyes"]),
        ("Thank you so much", "Gratitude", ["folded hands", "smiling face with heart-eyes", "red heart", "smiling face with hearts"]),
        ("Let's go to the gym", "Fitness", ["flexed biceps"]),
        ("I need coffee", "Coffee", ["hot beverage"]),
        ("Congratulations on your success", "Achievement", ["clapping hands", "trophy", "partying face", "check mark button"]),
        ("I'm so confused right now", "Confusion", ["thinking face", "confounded face", "confused face"]),
        ("Gaming all night", "Gaming", ["video game"]),
    ]

    correct_top1 = 0
    correct_top2 = 0
    total = len(test_cases)

    print(f"\n  {'Input Text':<40s}  {'Predicted':30s}  {'Match':6s}  {'Confidence':10s}")
    print(f"  {'─' * 40}  {'─' * 30}  {'─' * 6}  {'─' * 10}")

    for text, category, expected_names in test_cases:
        results = predict(text, classifier, tfidf_word, tfidf_char, label_encoder, emoji_lookup, top_k=2)

        predicted_names = [r["name"] for r in results]

        # Check if top-1 matches any expected
        top1_match = any(
            exp_name.lower() in predicted_names[0].lower() or predicted_names[0].lower() in exp_name.lower()
            for exp_name in expected_names
        ) if predicted_names else False

        # Check if any of the top-2 match
        top2_match = any(
            any(
                exp_name.lower() in pred.lower() or pred.lower() in exp_name.lower()
                for exp_name in expected_names
            )
            for pred in predicted_names
        )

        if top1_match:
            correct_top1 += 1
        if top2_match:
            correct_top2 += 1

        emoji_str = " ".join([f"{r['emoji']}" for r in results])
        name_str = ", ".join([r['name'][:14] for r in results])
        match_str = "✓" if top2_match else "✗"
        conf_str = f"{results[0]['score']:.1%}" if results else "N/A"

        match_color = match_str
        print(f"  {text:<40s}  {emoji_str} ({name_str:20s})  {match_color:6s}  {conf_str:10s}")

    # ========================================
    # Section 3: Accuracy Summary
    # ========================================
    print(f"\n{'=' * 70}")
    print("  3. PREDICTION ACCURACY SUMMARY")
    print("=" * 70)

    top1_acc = correct_top1 / total
    top2_acc = correct_top2 / total

    print(f"\n  Human-Verified Test Cases: {total}")
    print()

    bar1 = "█" * int(top1_acc * 40) + "░" * (40 - int(top1_acc * 40))
    bar2 = "█" * int(top2_acc * 40) + "░" * (40 - int(top2_acc * 40))

    print(f"  Top-1 Accuracy:  {bar1} {top1_acc:.1%}  ({correct_top1}/{total})")
    print(f"  Top-2 Accuracy:  {bar2} {top2_acc:.1%}  ({correct_top2}/{total})")

    # Training metrics for comparison
    print(f"\n  Model Training Accuracy (on test set):")
    bar3 = "█" * int(ens.get('accuracy', 0) * 40) + "░" * (40 - int(ens.get('accuracy', 0) * 40))
    print(f"  Overall:         {bar3} {ens.get('accuracy', 0):.2%}")

    print(f"\n{'=' * 70}")
    print(f"  Evaluation Complete!")
    print(f"{'=' * 70}\n")

    # Save results
    results_path = os.path.join(ROOT_DIR, "accuracy_report.txt")
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("Context-Aware Emoji Prediction - Accuracy Report\n")
        f.write("=" * 50 + "\n\n")
        f.write("Training Metrics:\n")
        f.write(f"  Accuracy:      {ens.get('accuracy', 0):.2%}\n")
        f.write(f"  Precision:     {ens.get('precision', 0):.2%}\n")
        f.write(f"  Recall:        {ens.get('recall', 0):.2%}\n")
        f.write(f"  F1-Score:      {ens.get('f1_score', 0):.2%}\n")
        f.write(f"  Top-3 Acc:     {ens.get('top_3_accuracy', 0):.2%}\n")
        f.write(f"  Top-5 Acc:     {ens.get('top_5_accuracy', 0):.2%}\n\n")
        f.write(f"Human Test Accuracy:\n")
        f.write(f"  Top-1: {top1_acc:.1%} ({correct_top1}/{total})\n")
        f.write(f"  Top-2: {top2_acc:.1%} ({correct_top2}/{total})\n")

    print(f"  Report saved to: {results_path}")


if __name__ == "__main__":
    main()
