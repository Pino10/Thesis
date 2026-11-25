"""Revision analysis for supervisor comment 7: domain-grouped folds.

The main results use one fixed split and three seeds, so the seeds only vary the
initialisation. Here the whole dataset is split into five folds grouped by
registered domain (public suffix list, ICANN section), so no registered domain
appears in both training and test data of a fold. Gradient boosting and the S-16
network (trained on labels only) are trained on each fold with the 24 Baseline U
features, to see how much results move when the data split itself changes.

The full Baseline U network is not included: one fold takes roughly an hour on the
available CPU, so five folds were out of reach.

    python code/revision/grouped_folds.py
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import tldextract
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA, FEATURES_U, RESULTS, Student

torch.set_num_threads(2)
ex = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=False)
df = pd.concat([pd.read_parquet(DATA / f"split_{s}.parquet") for s in ("train", "val", "test")], ignore_index=True)
groups = df.Domain.str.lower().map(lambda h: ex(h).top_domain_under_public_suffix or h).values
X = df[FEATURES_U].values.astype(np.float32); y = df.label.values

def scale(xtr, *others):
    mn, mx = xtr.min(0), xtr.max(0); rng = np.where(mx - mn == 0, 1, mx - mn)
    return [(xtr - mn) / rng] + [np.clip((o - mn) / rng, 0, 1).astype(np.float32) for o in others]

def s16(xtr, ytr, xva, yva, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    frac = float((ytr == 0).mean())
    xt, yt = torch.tensor(xtr), torch.tensor(ytr, dtype=torch.float32)
    wt = torch.where(yt == 0, torch.tensor(1 - frac), torch.tensor(frac))
    xv, yv = torch.tensor(xva), torch.tensor(yva, dtype=torch.float32)
    m = Student("S-16"); opt = torch.optim.Adam(m.parameters(), lr=1e-3); best, state, wait = 1e9, None, 0
    for _ in range(30):
        m.train(); perm = torch.randperm(len(xt))
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]; opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(m(xt[idx]), yt[idx], weight=wt[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = nn.functional.binary_cross_entropy_with_logits(m(xv), yv).item()
        if vl < best - 1e-5: best, wait, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= 4: break
    m.load_state_dict(state); m.eval()
    return lambda a: torch.sigmoid(m(torch.tensor(a))).detach().numpy()

rows = []
for fold, (tr_idx, te_idx) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
    rng = np.random.default_rng(fold); tr_idx = rng.permutation(tr_idx)
    n_val = len(tr_idx) // 10; va_idx, tr_idx = tr_idx[:n_val], tr_idx[n_val:]
    xtr, xva, xte = scale(X[tr_idx], X[va_idx], X[te_idx])
    frac = float((y[tr_idx] == 0).mean())
    gb = HistGradientBoostingClassifier(random_state=42, max_iter=500, early_stopping=True, validation_fraction=None,
                                        class_weight={0: 1 - frac, 1: frac}).fit(xtr, y[tr_idx], X_val=xva, y_val=y[va_idx])
    for name, prob in (("Gradient boosting", gb.predict_proba(xte)[:, 1]), ("S-16 (labels only)", s16(xtr, y[tr_idx], xva, y[va_idx])(xte))):
        pred_phish = prob < 0.5; act = y[te_idx] == 0
        rows.append(dict(fold=fold, model=name, n_test=len(te_idx), accuracy=float((pred_phish == act).mean()),
                         phishing_recall=float(pred_phish[act].mean()), fpr=float(pred_phish[~act].mean())))
        print(json.dumps(rows[-1]), flush=True)
res = pd.DataFrame(rows); res.to_csv(RESULTS / "revision" / "grouped_folds.csv", index=False)
print(res.groupby("model")[["accuracy", "phishing_recall", "fpr"]].agg(["mean", "std"]).mul(100).round(2).to_string())
