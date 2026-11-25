"""Revision analyses for supervisor comments 3, 5 and 7, using the saved models only.

Comment 3: the zero-day proxy.
  - "Domain" in PhiUSIIL is the full hostname. Recompute unseen-domain recall with
    unseen defined at the registered-domain level (public suffix list, ICANN section),
    so subdomains of a domain already seen in training no longer count as unseen.
  - Count domains that appear in both classes.
  - Report the false positive rate on legitimate URLs from unseen domains.
  - Look at what separates "seen" from "unseen" phishing.
Comment 5: precision-recall AUC, FPR at fixed recall, precision at realistic prevalence.
Comment 7: bootstrap 95% confidence intervals, McNemar tests, and model selection on
  the validation set instead of the test set.

    python code/revision/zero_day_and_metrics.py
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import tldextract
import torch
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import MODELS, RESULTS, Student, Teacher, load_split, load_weights

OUT = RESULTS / "revision"
ex = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=False)
reg = lambda h: ex(h).top_domain_under_public_suffix or h

train_df, x_tr, y_tr = load_split("train")
val_df, x_va, y_va = load_split("val")
test_df, x_te, y_te = load_split("test")
y = test_df["label"].values
phish, legit = y == 0, y == 1

host_tr = set(train_df["Domain"].str.lower())
reg_tr = set(train_df["Domain"].str.lower().map(reg))
host_te = test_df["Domain"].str.lower()
reg_te = host_te.map(reg)
unseen_host = ~host_te.isin(host_tr).values
unseen_reg = ~reg_te.isin(reg_tr).values

report = {}
report["domain_definition"] = "PhiUSIIL 'Domain' is the full hostname, e.g. " + ", ".join(test_df["Domain"].head(3))
both = set(train_df[train_df.label == 0].Domain.str.lower()) & set(train_df[train_df.label == 1].Domain.str.lower())
report["train_hostnames_in_both_classes"] = len(both)
both_reg = set(train_df[train_df.label == 0].Domain.str.lower().map(reg)) & set(train_df[train_df.label == 1].Domain.str.lower().map(reg))
report["train_registered_domains_in_both_classes"] = len(both_reg)
report["test_phishing_rows"] = int(phish.sum())
report["unseen_phishing_rows_hostname_level"] = int((phish & unseen_host).sum())
report["unseen_phishing_rows_registered_level"] = int((phish & unseen_reg).sum())
report["test_legit_rows"] = int(legit.sum())
report["unseen_legit_rows_hostname_level"] = int((legit & unseen_host).sum())
report["unseen_legit_rows_registered_level"] = int((legit & unseen_reg).sum())

# what do 'seen' phishing hostnames look like? (share on platforms whose registered domain is shared)
seen_ph = test_df[phish & ~unseen_host]
unseen_ph = test_df[phish & unseen_host]
share_reg = lambda d: (d.Domain.str.lower() != d.Domain.str.lower().map(reg)).mean()
report["seen_phishing_share_with_subdomain"] = float(share_reg(seen_ph))
report["unseen_phishing_share_with_subdomain"] = float(share_reg(unseen_ph))
for name, d in (("seen", seen_ph), ("unseen", unseen_ph)):
    path = d.URL.str.replace(r"^https?://", "", regex=True).str.split("/", n=1).str[1].fillna("")
    report[f"{name}_phishing_share_bare_homepage"] = float((path.str.len() == 0).mean())
    report[f"{name}_phishing_mean_url_length"] = float(d.URLLength.mean())

# ---------------- models: probability of 'legitimate' on val and test ----------------
def probs(model, x):
    with torch.no_grad():
        return torch.sigmoid(model(x)).numpy()

models = {}
for seed in (42, 43, 44):
    t = load_weights(Teacher(), MODELS / "baselines" / f"baselineU_seed{seed}.pt")
    models[("Baseline U", seed)] = (probs(t, x_va), probs(t, x_te))
    for kind in ("S-192", "S-64", "S-16", "S-LIN"):
        s = load_weights(Student(kind), MODELS / "distillation" / f"student_{kind}_seed{seed}.pt")
        models[(kind, seed)] = (probs(s, x_va), probs(s, x_te))
    # pruned with the output layer protected (revision rerun), see prune_protected.py
    p = load_weights(Teacher(), RESULTS / "revision" / f"prune_protected_iterative_s90_seed{seed}.pt")
    models[("Pruned 90% (iterative)", seed)] = (probs(p, x_va), probs(p, x_te))
    frac = float((y_tr == 0).float().mean())
    gb = HistGradientBoostingClassifier(random_state=seed, class_weight={0: 1 - frac, 1: frac}, max_iter=500,
                                        early_stopping=True, validation_fraction=None)
    gb.fit(x_tr.numpy(), y_tr.numpy(), X_val=x_va.numpy(), y_val=y_va.numpy())
    models[("Gradient boosting", seed)] = (gb.predict_proba(x_va.numpy())[:, 1], gb.predict_proba(x_te.numpy())[:, 1])

def fpr_at_recall(p_phish, act, target):
    fpr, tpr, _ = roc_curve(act, p_phish)
    return float(fpr[np.searchsorted(tpr, target)])

def prec_at_prev(tpr, fpr, prev):
    return tpr * prev / (tpr * prev + fpr * (1 - prev)) if tpr * prev + fpr * (1 - prev) > 0 else float("nan")

rows = []
for (name, seed), (pva, pte) in models.items():
    pp = 1 - pte
    pred = pp >= 0.5
    tpr = pred[phish].mean(); fpr = pred[legit].mean()
    yv = val_df["label"].values
    rows.append(dict(
        model=name, seed=seed,
        val_accuracy=float(((1 - pva >= 0.5) == (yv == 0)).mean()),
        val_fpr=float((1 - pva >= 0.5)[yv == 1].mean()),
        accuracy=float((pred == phish).mean()), zeroday=float(tpr), fpr=float(fpr),
        zd_unseen_hostname=float(pred[phish & unseen_host].mean()),
        zd_unseen_registered=float(pred[phish & unseen_reg].mean()),
        zd_seen_hostname=float(pred[phish & ~unseen_host].mean()),
        fpr_unseen_legit_hostname=float(pred[legit & unseen_host].mean()),
        fpr_unseen_legit_registered=float(pred[legit & unseen_reg].mean()),
        pr_auc=float(average_precision_score(phish, pp)),
        fpr_at_recall_99=fpr_at_recall(pp, phish, 0.99),
        fpr_at_recall_995=fpr_at_recall(pp, phish, 0.995),
        precision_at_prev_5pct=prec_at_prev(tpr, fpr, 0.05),
        precision_at_prev_1pct=prec_at_prev(tpr, fpr, 0.01),
    ))
res = pd.DataFrame(rows)
res.to_csv(OUT / "zero_day_and_metrics.csv", index=False)

# ---------------- comment 7: bootstrap CIs and McNemar (seed 42 vs seed 42) ----------------
rng = np.random.default_rng(0)
def boot_diff(pa, pb, mask, B=2000):
    a = ((1 - pa) >= 0.5)[mask] == phish[mask]; b = ((1 - pb) >= 0.5)[mask] == phish[mask]
    idx = np.arange(mask.sum()); diffs = []
    for _ in range(B):
        s = rng.choice(idx, size=len(idx), replace=True); diffs.append(a[s].mean() - b[s].mean())
    return float(a.mean() - b.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

def mcnemar(pa, pb):
    a = ((1 - pa) >= 0.5) == phish; b = ((1 - pb) >= 0.5) == phish
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, float(p)

stat_rows = []
allmask = np.ones_like(phish, dtype=bool)
for other in ("S-16", "S-LIN", "Gradient boosting", "Pruned 90% (iterative)", "S-64"):
    for seed in (42, 43, 44):
        pa = models[(other, seed)][1]; pb = models[("Baseline U", seed)][1]
        d, lo, hi = boot_diff(pa, pb, allmask)
        du, ulo, uhi = boot_diff(pa, pb, phish & unseen_host)
        n01, n10, p = mcnemar(pa, pb)
        stat_rows.append(dict(model=other, seed=seed, acc_diff_vs_teacher=d, acc_ci_low=lo, acc_ci_high=hi,
                              zdU_diff_vs_teacher=du, zdU_ci_low=ulo, zdU_ci_high=uhi,
                              only_model_correct=n01, only_teacher_correct=n10, mcnemar_p=p))
pd.DataFrame(stat_rows).to_csv(OUT / "bootstrap_mcnemar.csv", index=False)
json.dump(report, open(OUT / "zero_day_report.json", "w"), indent=1)
print(json.dumps(report, indent=1))
print("DONE")
