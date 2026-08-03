"""
Simple Flask demo: paste a URL, extract lightweight features, get a prediction
from the trained ANN model. This is what you actually demo live in interviews.

Run: python app/app.py  -> open http://localhost:5000
"""
import os
import re
from urllib.parse import urlparse

import numpy as np
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # allows the Chrome extension (different origin) to call this API

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ann_model.keras")
model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

PAGE = """
<!doctype html>
<html>
<head>
  <title>Phishing URL Detector</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 600px; margin: 60px auto; }
    input { width: 100%; padding: 10px; font-size: 16px; }
    button { padding: 10px 20px; margin-top: 10px; font-size: 16px; cursor: pointer; }
    .result { margin-top: 20px; padding: 15px; border-radius: 6px; font-weight: bold; }
    .safe { background: #d4edda; color: #155724; }
    .phishing { background: #f8d7da; color: #721c24; }
  </style>
</head>
<body>
  <h2>🔒 Phishing URL Detector</h2>
  <form method="POST">
    <input type="text" name="url" placeholder="Paste a URL, e.g. http://example.com" value="{{ url or '' }}">
    <button type="submit">Check</button>
  </form>
  {% if result %}
    <div class="result {{ 'phishing' if result == 'Phishing' else 'safe' }}">
      Prediction: {{ result }} (confidence: {{ confidence }}%)
    </div>
  {% endif %}
</body>
</html>
"""


def extract_quick_features(url: str) -> np.ndarray:
    """
    Lightweight heuristic feature extraction — a real subset of the UCI
    feature set, padded with zeros to match the 30-feature input shape the
    ANN was trained on. Predictions are approximate until this is replaced
    with the full UCI feature definitions (see README "Next steps").
    """
    parsed = urlparse(url if "://" in url else "http://" + url)
    base = [
        1 if len(url) > 54 else 0,                          # long URL
        1 if "@" in url else 0,                              # has @ symbol
        1 if url.count("-") > 0 else 0,                      # has hyphen in domain
        1 if re.match(r"^https?://\d+\.\d+\.\d+\.\d+", url) else 0,  # IP as domain
        0 if parsed.scheme == "https" else 1,                 # not https
        1 if url.count(".") > 3 else 0,                       # many subdomains
    ]
    padded = base + [0] * (30 - len(base))
    return np.array(padded, dtype=float)


def predict_url(url: str):
    if model is None:
        return None, None
    feats = extract_quick_features(url).reshape(1, -1)
    pred = model.predict(feats, verbose=0)[0][0]
    result = "Phishing" if pred > 0.5 else "Safe"
    confidence = round(float(pred if pred > 0.5 else 1 - pred) * 100, 2)
    return result, confidence


@app.route("/", methods=["GET", "POST"])
def index():
    result, confidence, url = None, None, None
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if url and model is not None:
            result, confidence = predict_url(url)
        elif model is None:
            result = "Model not found — run models/train.py first"
            confidence = "-"
    return render_template_string(PAGE, result=result, confidence=confidence, url=url)


@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON API for the Chrome extension.
    Request:  { "url": "https://example.com" }
    Response: { "result": "Safe" | "Phishing", "confidence": 92.5 }
    """
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "No url provided"}), 400
    if model is None:
        return jsonify({"error": "Model not found — run models/train.py first"}), 500

    result, confidence = predict_url(url)
    return jsonify({"result": result, "confidence": confidence})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
