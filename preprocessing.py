"""
Two preprocessing paths:
1. Tabular features (30 cols from UCI dataset) -> scaled numpy array, for the ANN branch
2. Raw URL strings -> padded character-index sequences, for the BiLSTM branch

If you only have the UCI tabular dataset (no raw URL strings), use
`synthesize_urls_from_features()` as a placeholder, OR download a URL-only
dataset (e.g. Kaggle "Phishing Site URLs" by taruntiwarihp) for the BiLSTM branch
and train it separately before fusion. See README for dataset links.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_URL_LEN = 200  # characters
CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=% "
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHAR_VOCAB)}  # 0 reserved for padding
VOCAB_SIZE = len(CHAR_TO_IDX) + 1


def load_tabular(csv_path, label_col="Result", test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    y = df[label_col].values
    # UCI labels are typically {-1, 1} -> convert to {0, 1}
    y = np.where(y == -1, 0, y)
    X = df.drop(columns=[label_col]).values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler


def url_to_sequence(url: str, max_len=MAX_URL_LEN):
    url = url.lower().strip()
    seq = [CHAR_TO_IDX.get(c, 0) for c in url]
    return seq


def load_url_sequences(csv_path, url_col="url", label_col="label", test_size=0.2, random_state=42):
    """For a dataset with raw URL strings + label column (0=legit, 1=phishing)."""
    df = pd.read_csv(csv_path)
    sequences = [url_to_sequence(u) for u in df[url_col]]
    X = pad_sequences(sequences, maxlen=MAX_URL_LEN, padding="post", truncating="post")
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
