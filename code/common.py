"""Shared definitions: data loading, model architectures and evaluation metrics.

Paths are relative to the repository root, so run the scripts from there, e.g.
    python code/evaluate_saved_models.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

# The 24 URL-only features of Baseline U, in the order the models expect them.
FEATURES_U = [l.strip() for l in open(DATA / "features_U.txt") if l.strip()]
MIN_U = np.load(DATA / "scaler_min_U.npy")
RNG_U = np.load(DATA / "scaler_rng_U.npy")

# Class weights for the weighted binary cross-entropy (1 minus each class's share).
# Label 0 = phishing, label 1 = legitimate.
W_PHISH, W_LEGIT = 0.5732, 0.4268


def load_split(name):
    """Load the train, val or test split and apply the min-max scaling fitted on training data."""
    df = pd.read_parquet(DATA / f"split_{name}.parquet")
    x = ((df[FEATURES_U].values.astype(np.float64) - MIN_U) / RNG_U).astype(np.float32)
    return df, torch.tensor(x), torch.tensor(df["label"].values.astype(np.float32))


class Teacher(nn.Module):
    """Re-implementation of ADUIN used as Baseline U (24 URL-only features)."""

    def __init__(self, n_features=24):
        super().__init__()
        self.input_proj = nn.Linear(n_features, 128)
        self.attn = nn.MultiheadAttention(128, 8, batch_first=True)
        self.norm = nn.LayerNorm(128)
        self.mlp = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        h = self.input_proj(x).unsqueeze(1)
        a, _ = self.attn(h, h, h)
        h = self.norm(h + a).squeeze(1)
        return self.out(self.mlp(h)).squeeze(-1)


class Student(nn.Module):
    """Plain feed-forward students with no attention block."""

    SIZES = {"S-192": [128, 64, 32], "S-64": [64, 32], "S-16": [16, 8], "S-LIN": []}

    def __init__(self, kind, n_features=24):
        super().__init__()
        layers, prev = [], n_features
        for width in self.SIZES[kind]:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_weights(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def evaluate(logits, y, df, train_domains):
    """Metrics as reported in the thesis. The model's output is the probability of 'legitimate'."""
    pred = (torch.sigmoid(logits).numpy() >= 0.5).astype(int)
    y = y.numpy().astype(int)
    phish, legit = y == 0, y == 1
    unseen = phish & ~df["Domain"].isin(train_domains).values
    return {
        "accuracy": float((pred == y).mean()),
        "precision": float(((pred == 0) & phish).sum() / max((pred == 0).sum(), 1)),
        "zeroday": float((pred[phish] == 0).mean()),        # recall on all test phishing
        "zd_unseen": float((pred[unseen] == 0).mean()),     # recall on phishing from domains never seen in training
        "fpr": float((pred[legit] == 0).mean()),
    }
