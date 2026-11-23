"""Revision analyses for supervisor comments 1 and 4.

Comment 1 (possible leakage in Baseline U): retrain Baseline U without
TLDLegitimateProb and URLCharProb (22 features), three seeds, with the notebook's
Baseline U recipe, and compare with the 24-feature model.

Comment 4 (missing simple baselines): logistic regression, a decision tree and
gradient boosting on the same 24 URL-only features and the same split. These are
deterministic or seeded, so three seeds are run for the tree-based models.

    python code/revision/leak_and_baselines.py [leak | baselines | all]
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA, FEATURES_U, RESULTS, Teacher

torch.set_num_threads(2)
OUT = RESULTS / "revision"

train = pd.read_parquet(DATA / "split_train.parquet")
val = pd.read_parquet(DATA / "split_val.parquet")
test = pd.read_parquet(DATA / "split_test.parquet")
train_domains = set(train["Domain"].str.lower())
unseen = (test["label"].values == 0) & ~test["Domain"].str.lower().isin(train_domains).values


def scale(feats):
    xtr = train[feats].values.astype(np.float32)
    mn, mx = xtr.min(0), xtr.max(0)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    f = lambda d, clip: np.clip((d[feats].values.astype(np.float32) - mn) / rng, 0, 1) if clip else (d[feats].values.astype(np.float32) - mn) / rng
    return f(train, False), f(val, True), f(test, True)


def metrics(p_legit, name, seed, n_feats):
    y = test["label"].values
    p_phish = 1 - p_legit
    pred_phish = p_phish >= 0.5
    act = y == 0
    tp = (pred_phish & act).sum(); tn = (~pred_phish & ~act).sum(); fp = (pred_phish & ~act).sum(); fn = (~pred_phish & act).sum()
    return dict(model=name, seed=seed, n_features=n_feats,
                accuracy=(tp + tn) / len(y), precision=tp / max(tp + fp, 1), zeroday=tp / (tp + fn),
                f1=2 * tp / (2 * tp + fp + fn), fpr=fp / (fp + tn), auc=roc_auc_score(act, p_phish),
                pr_auc=average_precision_score(act, p_phish), zd_unseen=pred_phish[unseen].mean())


rows = []
y_tr = train["label"].values.astype(np.float32); y_va = val["label"].values.astype(np.float32)
frac_phish = float((y_tr == 0).mean())
w_tr = np.where(y_tr == 0, 1 - frac_phish, frac_phish).astype(np.float32)

PART = sys.argv[1] if len(sys.argv) > 1 else "all"
OUTFILE = OUT / f"leak_and_baselines_{PART}.csv"

# ---- comment 1: Baseline U without the two probability features (notebook Step 7 recipe) ----
FEATS_22 = [f for f in FEATURES_U if f not in ("TLDLegitimateProb", "URLCharProb")]
for feats, tag in (((FEATS_22, "Baseline U without TLDLegitimateProb and URLCharProb"),) if PART in ("leak", "all") else ()):
    xtr, xva, xte = (torch.tensor(a) for a in scale(feats))
    ytr, yva, wtr = torch.tensor(y_tr), torch.tensor(y_va), torch.tensor(w_tr)
    for seed in (42, 43, 44):
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        model = Teacher(len(feats)); opt = torch.optim.Adam(model.parameters(), lr=0.001)
        best, best_state, wait = float("inf"), None, 0
        for ep in range(50):
            model.train(); perm = torch.randperm(len(xtr))
            for i in range(0, len(perm), 128):
                idx = perm[i:i + 128]; opt.zero_grad()
                nn.functional.binary_cross_entropy_with_logits(model(xtr[idx]), ytr[idx], weight=wtr[idx]).backward(); opt.step()
            model.eval()
            with torch.no_grad(): vl = nn.functional.binary_cross_entropy_with_logits(model(xva), yva).item()
            if vl < best - 1e-5: best, wait, best_state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= 5: break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad(): p = torch.sigmoid(model(xte)).numpy()
        r = metrics(p, tag, seed, len(feats)); r["epochs"] = ep + 1; r["minutes"] = round((time.time() - t0) / 60, 1)
        rows.append(r); print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}, default=str), flush=True)
        pd.DataFrame(rows).to_csv(OUTFILE, index=False)

# ---- comment 4: simple baselines on the same 24 features ----
if PART not in ("baselines", "all"):
    pd.DataFrame(rows).to_csv(OUTFILE, index=False); print("DONE"); sys.exit()
xtr, xva, xte = scale(FEATURES_U)
lr = LogisticRegression(max_iter=2000, class_weight={0: 1 - frac_phish, 1: frac_phish})
lr.fit(xtr, y_tr); rows.append(metrics(lr.predict_proba(xte)[:, 1], "Logistic regression", 0, 24))
for seed in (42, 43, 44):
    dt = DecisionTreeClassifier(random_state=seed, class_weight={0: 1 - frac_phish, 1: frac_phish}, min_samples_leaf=5)
    dt.fit(xtr, y_tr); rows.append(metrics(dt.predict_proba(xte)[:, 1], "Decision tree", seed, 24))
    gb = HistGradientBoostingClassifier(random_state=seed, class_weight={0: 1 - frac_phish, 1: frac_phish},
                                        early_stopping=True, validation_fraction=None, max_iter=500)
    gb.fit(xtr, y_tr, X_val=xva, y_val=y_va)
    rows.append(metrics(gb.predict_proba(xte)[:, 1], "Gradient boosting", seed, 24))
    print("baselines seed", seed, "done", flush=True)
pd.DataFrame(rows).to_csv(OUTFILE, index=False)
print("DONE")
