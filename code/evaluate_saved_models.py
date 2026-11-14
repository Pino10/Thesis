"""Reload the saved Baseline U and distilled student models and check that they
reproduce the test results recorded in results/ (originally produced in Google Colab).

    python code/evaluate_saved_models.py
"""
import pandas as pd
import torch

from common import MODELS, RESULTS, Student, Teacher, evaluate, load_split, load_weights

train_df, _, _ = load_split("train")
test_df, x_test, y_test = load_split("test")
train_domains = set(train_df["Domain"].str.lower())

rows = []
for seed in (42, 43, 44):
    model = load_weights(Teacher(), MODELS / "baselines" / f"baselineU_seed{seed}.pt")
    with torch.no_grad():
        rows.append({"model": "Baseline U", "seed": seed, **evaluate(model(x_test), y_test, test_df, train_domains)})
    for kind in ("S-192", "S-64", "S-16", "S-LIN"):
        model = load_weights(Student(kind), MODELS / "distillation" / f"student_{kind}_seed{seed}.pt")
        with torch.no_grad():
            rows.append({"model": kind, "seed": seed, **evaluate(model(x_test), y_test, test_df, train_domains)})
now = pd.DataFrame(rows)

# Compare with the recorded results.
rec_u = pd.read_csv(RESULTS / "baselineU_test_results.csv").rename(columns={"recall_zeroday": "zeroday"})
rec_s = pd.read_csv(RESULTS / "distillation_results.csv").rename(columns={"student": "model"})
worst = 0.0
for _, r in now.iterrows():
    ref = rec_u[rec_u.seed == r.seed].iloc[0] if r.model == "Baseline U" else \
          rec_s[(rec_s.model == r.model) & (rec_s.seed == r.seed)].iloc[0]
    for m in ("accuracy", "zeroday", "fpr"):
        worst = max(worst, abs(r[m] - ref[m]))
    print(f"{r.model:10s} seed {r.seed}: accuracy {r.accuracy:.6f} (recorded {ref.accuracy:.6f})  "
          f"unseen-domain recall {r.zd_unseen:.4f}")
print(f"\nLargest difference from the recorded results: {worst:.2e}")
