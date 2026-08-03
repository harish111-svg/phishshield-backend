"""
Trains the ANN baseline (on tabular features) and the BiLSTM branch (on raw
URLs, if a URL-labeled dataset is available), then prints a comparison table.

Usage:
    python models/train.py
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocessing import load_tabular, load_url_sequences
from model_ann import build_ann
from model_bilstm import build_bilstm

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "phishing.csv")
URL_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "urls.csv")
RESULTS = {}


def evaluate(name, y_true, y_pred):
    RESULTS[name] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def train_ann():
    print("\n=== Training ANN (tabular features) ===")
    X_train, X_test, y_train, y_test, scaler = load_tabular(DATA_PATH)

    model = build_ann(input_dim=X_train.shape[1])
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=32,
        verbose=1,
    )

    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    evaluate("ANN (features)", y_test, y_pred)

    model.save(os.path.join(os.path.dirname(__file__), "ann_model.keras"))
    print("Saved ANN model -> models/ann_model.keras")


def train_bilstm(sample_size=50000):
    print("\n=== Training BiLSTM (raw URL sequences) ===")
    import pandas as pd
    # Full dataset is 400k+ rows -- subsample for faster training on a laptop CPU.
    # Increase sample_size (or remove sampling) if you have more time/GPU access.
    df = pd.read_csv(URL_DATA_PATH)
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    tmp_path = os.path.join(os.path.dirname(URL_DATA_PATH), "_urls_sample.csv")
    df.to_csv(tmp_path, index=False)

    X_train, X_test, y_train, y_test = load_url_sequences(tmp_path)

    model = build_bilstm()
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=64,
        verbose=1,
    )

    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    evaluate("BiLSTM (raw URL)", y_test, y_pred)

    model.save(os.path.join(os.path.dirname(__file__), "bilstm_model.keras"))
    print("Saved BiLSTM model -> models/bilstm_model.keras")
    os.remove(tmp_path)


def print_comparison_table():
    print("\n=== Model Comparison ===")
    header = f"{'Model':<25}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}"
    print(header)
    print("-" * len(header))
    for name, metrics in RESULTS.items():
        print(
            f"{name:<25}"
            f"{metrics['accuracy']*100:<12.2f}"
            f"{metrics['precision']*100:<12.2f}"
            f"{metrics['recall']*100:<12.2f}"
            f"{metrics['f1']*100:<12.2f}"
        )


if __name__ == "__main__":
    train_ann()
    train_bilstm()
    print_comparison_table()
