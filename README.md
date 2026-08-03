# Phishing Website Detection using Deep Learning (Hybrid Fusion Model)

Mini project: detects phishing websites by combining two deep learning approaches:
1. **Feature-based ANN** — trained on 30 handcrafted URL/website features (UCI Phishing dataset)
2. **Character-sequence BiLSTM** — learns directly from raw URL text, no manual feature engineering
3. **Fusion model** — concatenates both branches before final classification (this is your "novelty")

## Why this approach (reference papers)
- Adebowale et al., "Deep learning with CNN and LSTM for phishing detection" (IEEE, SKIMA 2019)
- "Analysis of Phishing Website Detection Using CNN and Bidirectional LSTM" (IEEE, 2020)
- "Phishing Website Detection Using Deep Learning Models" (IEEE, 2024) — ensemble + feature selection, 99% acc on 11,055 sites
- "Design of a Hybrid AI-based Phishing Website Detection using LSTM, CNN, RF Ensemble" (IEEE, 2024)

Your contribution: most papers pick ONE approach (features OR raw URL). This project fuses both,
which is a legitimate, defensible novelty for a mini project report/viva.

## Folder structure
```
phishing-detector/
├── data/
│   └── download_data.py       # fetches UCI phishing dataset
├── models/
│   ├── preprocessing.py       # feature prep + URL tokenization
│   ├── model_ann.py           # baseline: ANN on 30 features
│   ├── model_bilstm.py        # BiLSTM on raw URL characters
│   ├── model_fusion.py        # combined fusion model
│   └── train.py               # trains all 3, saves comparison table
├── app/
│   └── app.py                 # Flask app: paste URL -> get prediction
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Steps
1. `python data/download_data.py` — downloads dataset to `data/phishing.csv`
2. `python models/train.py` — trains ANN, BiLSTM, and Fusion model, prints comparison table, saves best model
3. `python app/app.py` — starts local Flask server, open `http://localhost:5000`, paste any URL

## Dataset
UCI Phishing Websites Dataset (11,055 samples, 30 features, binary label).
Source: https://archive.ics.uci.edu/dataset/327/phishing+websites
(also mirrored on Kaggle as "Phishing Website Detector")

## ⚠️ Important before running the Flask demo
`app/app.py` uses a quick 6-feature heuristic extractor for live demo purposes,
but `models/train.py` trains the ANN on the full 30 UCI features. These won't
match in size. Before demoing live, either:
- Rewrite `extract_quick_features()` in `app.py` to compute the same 30 features
  the UCI dataset uses (the UCI page lists exact definitions for each), or
- Retrain a small ANN specifically on the 6 quick features for demo purposes
  (lower accuracy, but works end-to-end fast) — good enough for interview demos.

This mismatch is intentional to keep the starter code light — closing this gap
is a good first task once you get the skeleton running.

## Chrome Extension
`chrome-extension/` contains a Manifest V3 extension that checks whatever page
you're currently on, using your local Flask API (`app/app.py`) as the backend.

**Setup:**
1. Make sure `app/app.py` is running (`python app/app.py`) — the extension
   calls `http://localhost:5000/predict`, so this must stay running whenever
   you use the extension.
2. Open Chrome, go to `chrome://extensions`
3. Turn on **Developer mode** (toggle, top-right)
4. Click **Load unpacked**
5. Select the `chrome-extension` folder
6. Pin the extension (puzzle-piece icon in toolbar → pin the lock icon)
7. Visit any website, click the extension icon, click "Check this page"

The extension reads the current tab's URL, sends it to your local Flask
`/predict` endpoint, and shows Safe/Phishing with a confidence score.

⚠️ This only works while `app.py` is running on your machine — it's a local
demo setup, not a published/hosted extension. That's fine for a mini project
demo; deploying it publicly (hosting the Flask API on a server) is a good
"future scope" point for your report.

## Next steps / extension ideas (for viva "future scope")
- Add SHAP explainability to show which features triggered the phishing flag
- Deploy as a Chrome extension instead of standalone Flask app
- Add live WHOIS/SSL certificate age lookup as an extra real-time feature
- Try replacing BiLSTM with a small Transformer encoder for the URL branch
