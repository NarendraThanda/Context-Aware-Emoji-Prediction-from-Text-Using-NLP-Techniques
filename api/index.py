"""
Vercel Serverless Function: GET /api
Health check / API info endpoint.
"""
from http.server import BaseHTTPRequestHandler
import json


class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        """Return API information."""
        data = {
            "message": "Context-Aware Emoji Prediction API v2.0",
            "status": "running",
            "platform": "Vercel Serverless",
            "features": [
                "Text Preprocessing (Tokenization, Stop-words, Lemmatization)",
                "Feature Extraction (TF-IDF, N-grams)",
                "Semantic Similarity Matching",
                "Sentiment & Emotion Analysis",
                "Multi-class Emoji Classification",
                "Top-K Prediction"
            ],
            "endpoints": {
                "GET /api": "This endpoint - API info",
                "POST /api/predict": "Predict emojis for given text",
                "POST /api/analyze": "Get detailed NLP analysis"
            }
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format, *args):
        pass
