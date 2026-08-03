"""
BiLSTM on raw URL character sequences — no manual feature engineering needed.
Reference: "Analysis of Phishing Website Detection Using CNN and Bidirectional
LSTM" (IEEE 2020) — motivation is that feature extraction is slow/costly, so
this branch learns straight from the character sequence of the URL.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout, Input
)
from preprocessing import VOCAB_SIZE, MAX_URL_LEN


def build_bilstm() -> Sequential:
    model = Sequential([
        Input(shape=(MAX_URL_LEN,)),
        Embedding(input_dim=VOCAB_SIZE, output_dim=32, input_length=MAX_URL_LEN),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
