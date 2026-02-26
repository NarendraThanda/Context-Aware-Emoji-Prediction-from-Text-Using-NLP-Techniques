"""
Vercel Serverless Function: POST /api/analyze
Text analysis endpoint (NLP analysis without emoji prediction).
"""
from http.server import BaseHTTPRequestHandler
import json


class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_POST(self):
        """Handle text analysis requests."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}

            text = data.get("text", "")
            if not text:
                self._send_json(400, {"error": "No text provided"})
                return

            from shared.nlp_engine import analyze_text

            result = analyze_text(text)
            self._send_json(200, result)

        except Exception as e:
            self._send_json(500, {"detail": f"Analysis error: {str(e)}"})

    def _send_json(self, status_code, data):
        """Send a JSON response with CORS headers."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        pass
