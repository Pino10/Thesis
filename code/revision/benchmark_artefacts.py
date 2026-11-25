"""Revision analysis for supervisor comment 10 (quantify the saturation) and comment 1.

Measures, on the PhiUSIIL splits:
  - how URLLength relates to the real length of the stored URL, by class;
  - the "leftover length": URLLength minus every character count the dataset reports
    (letters, digits, other special characters, '=', '?', '&'), by class, and how well
    a small decision tree on that single number separates the classes;
  - the share of bare homepages and of free-hosting platforms in each class;
  - whether Bank, Pay and Crypto can be derived from the URL text;
  - gradient boosting accuracy as groups of features are removed (seed 42).

    python code/revision/benchmark_artefacts.py
"""
import json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA, FEATURES_U, RESULTS

OUT = RESULTS / "revision"
tr, va, te = (pd.read_parquet(DATA / f"split_{s}.parquet") for s in ("train", "val", "test"))
allv = pd.concat([tr, va, te])
rep = {}

d = allv.URLLength - allv.URL.str.len()
rep["urllength_minus_true_length_by_class_pct"] = {
    cls: (d[allv.label == lab].value_counts(normalize=True) * 100).round(2).to_dict()
    for cls, lab in (("phishing", 0), ("legitimate", 1))}

def leftover(df):
    return df.URLLength - (df.NoOfLettersInURL + df.NoOfDegitsInURL + df.NoOfOtherSpecialCharsInURL
                           + df.NoOfEqualsInURL + df.NoOfQMarkInURL + df.NoOfAmpersandInURL)
for df in (tr, te):
    df["leftover"] = leftover(df)
rep["test_leftover_value_counts"] = {
    "legitimate": te[te.label == 1].leftover.value_counts().head(5).to_dict(),
    "phishing": te[te.label == 0].leftover.value_counts().head(8).to_dict()}
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(tr[["leftover"]], tr.label)
rep["accuracy_from_leftover_alone_pct"] = round(float((tree.predict(te[["leftover"]]) == te.label).mean() * 100), 2)

path = allv.URL.str.replace(r"^https?://", "", regex=True).str.split("/", n=1).str[1].fillna("")
rep["bare_homepage_pct"] = {"phishing": round(float((path[allv.label == 0].str.len() == 0).mean() * 100), 1),
                            "legitimate": round(float((path[allv.label == 1].str.len() == 0).mean() * 100), 1)}
hosts = ["000webhostapp", "weebly", "wixsite", "firebaseapp", "web.app", "blogspot", "github.io", "glitch.me",
         "netlify", "herokuapp", "vercel", "sites.google", "godaddysites", "ipfs", "r2.dev", "pages.dev",
         "square.site", "webflow", "duckdns", "ngrok", "workers.dev", "azurewebsites", "appspot", "myshopify",
         "wordpress.com", "yolasite", "jimdo", "repl.co"]
pat = "|".join(re.escape(h) for h in hosts)
rep["free_hosting_pct"] = {"phishing": round(float(allv[allv.label == 0].Domain.str.contains(pat, case=False).mean() * 100), 1),
                           "legitimate": round(float(allv[allv.label == 1].Domain.str.contains(pat, case=False).mean() * 100), 1)}

u = allv.URL.str.lower()
rep["keyword_flag_agreement_with_url_text_pct"] = {
    c: round(float((u.str.contains(k).astype(int) == allv[c]).mean() * 100), 2) for c, k in (("Bank", "bank"), ("Pay", "pay"), ("Crypto", "crypto"))}
rep["keyword_flag_set_but_word_absent_from_url"] = {
    c: int(((allv[c] == 1) & ~u.str.contains(k)).sum()) for c, k in (("Bank", "bank"), ("Pay", "pay"), ("Crypto", "crypto"))}

page = ["Bank", "Pay", "Crypto"]
lenf = ["URLLength", "LetterRatioInURL", "DegitRatioInURL", "SpacialCharRatioInURL", "ObfuscationRatio", "CharContinuationRate"]
counts = ["NoOfLettersInURL", "NoOfDegitsInURL", "NoOfOtherSpecialCharsInURL"]
lookup = ["TLDLegitimateProb", "URLCharProb"]
sets = [("All 24 Baseline U features", FEATURES_U),
        ("Without the three page-content flags", [f for f in FEATURES_U if f not in page]),
        ("Also without URLLength and the ratios built on it", [f for f in FEATURES_U if f not in page + lenf]),
        ("Also without the letter, digit and special-character counts", [f for f in FEATURES_U if f not in page + lenf + counts]),
        ("Also without the two lookup-table features", [f for f in FEATURES_U if f not in page + lenf + counts + lookup])]
rows = []
frac = float((tr.label == 0).mean())
for name, feats in sets:
    m = HistGradientBoostingClassifier(random_state=42, max_iter=500, early_stopping=True, validation_fraction=None,
                                       class_weight={0: 1 - frac, 1: frac}).fit(tr[feats], tr.label, X_val=va[feats], y_val=va.label)
    p = m.predict(te[feats]); ph = te.label.values == 0
    rows.append(dict(feature_set=name, n_features=len(feats), accuracy=float((p == te.label).mean()),
                     phishing_recall=float((p[ph] == 0).mean()), fpr=float((p[~ph] == 0).mean()), features=";".join(feats)))
pd.DataFrame(rows).to_csv(OUT / "feature_removal_gbm.csv", index=False)
json.dump(rep, open(OUT / "benchmark_artefacts.json", "w"), indent=1, default=str)
print(json.dumps(rep, indent=1, default=str))
print(pd.DataFrame(rows).drop(columns="features").round(4).to_string())
