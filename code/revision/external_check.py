"""Revision analysis for the supervisor's suggestion of a quick external check.

Fresh data (retrieved 23 November 2026):
  - phishing: the OpenPhish public feed (github.com/openphish/public_feed, feed.txt)
  - legitimate: a random sample of 1,000 domains from the Tranco top-1M list dated
    22 November 2026, turned into URLs in two ways, "https://www.<domain>" and
    "https://<domain>", because PhiUSIIL's legitimate URLs are www-style homepages
    and the choice affects several features.

Only features that can be recomputed exactly from a raw URL are used (13 of the 24;
check_url_features showed exact agreement with the dataset for these). Models are
trained on the PhiUSIIL training split with these 13 features and tested both on the
PhiUSIIL test split and on the fresh URLs.

    python code/revision/external_check.py <feed.txt> <tranco top-1m.csv> [--teacher]
"""
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, FEATURES_U, RESULTS, Student, Teacher
from url_features import extract

torch.set_num_threads(2)
OUT = RESULTS / "revision" / "external"; OUT.mkdir(parents=True, exist_ok=True)
EXACT = ["DomainLength", "IsDomainIP", "TLDLegitimateProb", "TLDLength", "NoOfSubDomain", "HasObfuscation",
         "NoOfObfuscatedChar", "ObfuscationRatio", "NoOfDegitsInURL", "NoOfEqualsInURL", "NoOfQMarkInURL",
         "NoOfAmpersandInURL", "IsHTTPS"]
IDX = [FEATURES_U.index(f) for f in EXACT]

train = pd.read_parquet(DATA / "split_train.parquet"); val = pd.read_parquet(DATA / "split_val.parquet")
test = pd.read_parquet(DATA / "split_test.parquet")
tld_table = train.groupby(train.TLD.str.lower())["TLDLegitimateProb"].first().to_dict()
char_table = {}  # URLCharProb is not used here

# ---------------- fresh data ----------------
feed = [l.strip() for l in open(sys.argv[1]) if l.strip().startswith("http")]
feed = list(dict.fromkeys(feed))
tr_hosts = set(train.Domain.str.lower())
from urllib.parse import urlsplit
host = lambda u: (urlsplit(u).hostname or "").lower()
feed = [u for u in feed if host(u) not in tr_hosts]
tranco = pd.read_csv(sys.argv[2], header=None, names=["rank", "domain"])
legit_domains = tranco.sample(1000, random_state=0)["domain"].tolist()
fresh = {
    "phishing (OpenPhish)": feed,
    "legitimate, https://www.<domain>": [f"https://www.{d}" for d in legit_domains],
    "legitimate, https://<domain>": [f"https://{d}" for d in legit_domains],
}
pd.DataFrame({"url": feed}).to_csv(OUT / "openphish_2026-11-23.csv", index=False)
pd.DataFrame({"domain": legit_domains}).to_csv(OUT / "tranco_sample_2026-11-22.csv", index=False)

def feats_from_urls(urls):
    return np.array([[extract(u, tld_table, char_table)[i] for i in IDX] for u in urls], dtype=np.float32)

xtr = train[EXACT].values.astype(np.float32); mn, mx = xtr.min(0), xtr.max(0); rng = np.where(mx - mn == 0, 1, mx - mn)
sc = lambda a: np.clip((a - mn) / rng, 0, 1).astype(np.float32)
Xtr, Xva, Xte = (xtr - mn) / rng, sc(val[EXACT].values.astype(np.float32)), sc(test[EXACT].values.astype(np.float32))
ytr, yva, yte = (d.label.values for d in (train, val, test))
Fresh = {k: sc(feats_from_urls(v)) for k, v in fresh.items()}
frac = float((ytr == 0).mean()); cw = {0: 1 - frac, 1: frac}

def torch_fit(model, seed, epochs=50, patience=5):
    torch.manual_seed(seed); np.random.seed(seed)
    xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    wt = torch.where(yt == 0, torch.tensor(1 - frac), torch.tensor(frac))
    xv, yv = torch.tensor(Xva), torch.tensor(yva, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); best, state, wait = 1e9, None, 0
    for _ in range(epochs):
        model.train(); perm = torch.randperm(len(xt))
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]; opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(model(xt[idx]), yt[idx], weight=wt[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = nn.functional.binary_cross_entropy_with_logits(model(xv), yv).item()
        if vl < best - 1e-5: best, wait, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(state); model.eval()
    return lambda a: torch.sigmoid(model(torch.tensor(a))).detach().numpy()

models = {}
models[("Logistic regression", 0)] = LogisticRegression(max_iter=2000, class_weight=cw).fit(Xtr, ytr).predict_proba
for seed in (42, 43, 44):
    gb = HistGradientBoostingClassifier(random_state=seed, class_weight=cw, max_iter=500, early_stopping=True,
                                        validation_fraction=None).fit(Xtr, ytr, X_val=Xva, y_val=yva)
    models[("Gradient boosting", seed)] = gb.predict_proba
    models[("S-16 (labels only)", seed)] = torch_fit(Student("S-16", len(EXACT)), seed)
    if "--teacher" in sys.argv:
        models[("Baseline U architecture", seed)] = torch_fit(Teacher(len(EXACT)), seed)
    print("seed", seed, "trained", flush=True)

rows = []
for (name, seed), f in models.items():
    p = lambda a: (f(a)[:, 1] if f(a).ndim == 2 else f(a))
    pte = p(Xte); pred_phish = pte < 0.5
    r = dict(model=name, seed=seed, phiusiil_accuracy=float((pred_phish == (yte == 0)).mean()),
             phiusiil_recall=float(pred_phish[yte == 0].mean()), phiusiil_fpr=float(pred_phish[yte == 1].mean()))
    for k, X in Fresh.items():
        flagged = float((p(X) < 0.5).mean())
        r[("fresh_recall " if k.startswith("phishing") else "fresh_fpr ") + k] = flagged
    rows.append(r)
res = pd.DataFrame(rows); res.to_csv(OUT / "external_check.csv", index=False)
summary = res.groupby("model", sort=False).mean(numeric_only=True).drop(columns="seed").mul(100).round(2)
print(summary.T.to_string())
json.dump({"n_phishing_fresh": len(feed), "n_legit_fresh": len(legit_domains), "features": EXACT}, open(OUT / "external_meta.json", "w"), indent=1)
