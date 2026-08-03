"""
Fusion model: concatenates the ANN feature branch + BiLSTM URL branch before
the final classification layer. This is the "novel" contribution of the
project — most IEEE papers pick one branch OR the other, not both.

NOTE: to train this end-to-end you need a dataset where each row has BOTH
the 30 handcrafted features AND the raw URL string for the same sample.
If your two datasets don't overlap 1:1, train ANN and BiLSTM separately
(see train.py) and report them as strong standalone baselines, then discuss
the fusion architecture in your report as the proposed extension with
expected improvement (cite the ensemble papers' accuracy gains as evidence).
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Embedding, Bidirectional, LSTM, Concatenate
)
from preprocessing import VOCAB_SIZE, MAX_URL_LEN


def build_fusion_model(feature_dim: int) -> Model:
    # Branch 1: tabular features
    feat_input = Input(shape=(feature_dim,), name="feature_input")
    f = Dense(64, activation="relu")(feat_input)
    f = Dropout(0.3)(f)
    f = Dense(32, activation="relu")(f)

    # Branch 2: raw URL character sequence
    url_input = Input(shape=(MAX_URL_LEN,), name="url_input")
    u = Embedding(input_dim=VOCAB_SIZE, output_dim=32, input_length=MAX_URL_LEN)(url_input)
    u = Bidirectional(LSTM(64, return_sequences=True))(u)
    u = Bidirectional(LSTM(32))(u)
    u = Dense(32, activation="relu")(u)

    # Fusion
    merged = Concatenate()([f, u])
    x = Dense(32, activation="relu")(merged)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=[feat_input, url_input], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
