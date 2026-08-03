"""
Baseline: simple ANN (MLP) on the 30 UCI handcrafted features.
Reference: Adebowale et al. 2019 (IEEE SKIMA), and the ensemble-feature-selection
paper (IEEE 2024) that used info gain / gain ratio / PCA on this same feature set.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input


def build_ann(input_dim: int) -> Sequential:
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
