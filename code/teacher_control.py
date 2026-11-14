"""Control experiment (thesis Section 4.5.5): is the teacher necessary?

Trains S-16 and S-LIN twice on identical data and settings, once on the true labels
alone and once with knowledge distillation from the saved Baseline U teacher, for
seeds 42, 43 and 44. Writes results/teacher_control_results.csv.

Settings follow Step 11C of the original notebook: Adam (lr 0.001), batch 128, up
to 30 epochs, early stopping with patience 4 when the unweighted validation BCE stops
improving by more than 1e-5, sample weights of 1 minus each class's training share,
and for distillation loss = 0.5 * weighted BCE + 0.5 * T^2 * BCE(student/T, sigmoid(teacher/T)), T = 2.

    python code/teacher_control.py
"""
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import MODELS, RESULTS, Student, Teacher, evaluate, load_split, load_weights

torch.set_num_threads(4)
T = 2.0

train_df, x_tr, y_tr = load_split("train")
_, x_va, y_va = load_split("val")
test_df, x_te, y_te = load_split("test")
train_domains = set(train_df["Domain"].str.lower())


frac_phish = float((y_tr == 0).float().mean())
sample_w = torch.where(y_tr == 0, torch.tensor(1.0 - frac_phish), torch.tensor(frac_phish))


def val_loss_fn(model):
    with torch.no_grad():
        return nn.functional.binary_cross_entropy_with_logits(model(x_va), y_va).item()


results = []
for seed in (42, 43, 44):
    teacher = load_weights(Teacher(), MODELS / "baselines" / f"baselineU_seed{seed}.pt")
    with torch.no_grad():
        soft_tr = torch.sigmoid(teacher(x_tr) / T)

    for kind in ("S-16", "S-LIN"):
        for mode in ("no_teacher", "distilled"):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = Student(kind)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            def loss_fn(logits, y, w, soft):
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y, weight=w)
                if mode == "distilled":
                    loss = 0.5 * loss + 0.5 * T * T * nn.functional.binary_cross_entropy_with_logits(logits / T, soft)
                return loss

            best, best_state, bad, start = float("inf"), None, 0, time.time()
            gen = torch.Generator().manual_seed(seed)
            for epoch in range(30):
                model.train()
                perm = torch.randperm(len(x_tr), generator=gen)
                for i in range(0, len(perm), 128):
                    idx = perm[i:i + 128]
                    opt.zero_grad()
                    loss_fn(model(x_tr[idx]), y_tr[idx], sample_w[idx], soft_tr[idx]).backward()
                    opt.step()
                model.eval()
                val_loss = val_loss_fn(model)
                if val_loss < best - 1e-5:
                    best, bad = val_loss, 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 4:
                        break

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                r = evaluate(model(x_te), y_te, test_df, train_domains)
            r.update(seed=seed, student=kind, mode=mode, epochs=epoch + 1, secs=round(time.time() - start, 1))
            results.append(r)
            print(json.dumps(r), flush=True)

pd.DataFrame(results).to_csv(RESULTS / "teacher_control_results.csv", index=False)
