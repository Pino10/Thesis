"""Revision analysis for supervisor comment 8: pruning with the output layer protected.

The original runs (notebook Steps 8 and 9) pruned all weight matrices globally,
including the 32-weight output layer. In every collapsed run the output layer was
zeroed completely. This script repeats the same procedure with the output layer
excluded from pruning, and adds a one-shot variant whose maximum fine-tuning budget
matches the iterative schedule.

Settings otherwise follow the notebook: global L1 magnitude pruning over weight
matrices (biases and LayerNorm untouched), fine-tuning with Adam lr 0.0005, batch
128, weighted BCE, early stopping on unweighted validation BCE (min delta 1e-5,
patience 3), masks re-applied after every optimiser step.

    python code/revision/prune_protected.py [mode]   # mode: oneshot | iterative | matched
"""
import copy, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import MODELS, RESULTS, Teacher, evaluate, load_split, load_weights

torch.set_num_threads(2)
MODE = sys.argv[1] if len(sys.argv) > 1 else "oneshot"

train_df, x_tr, y_tr = load_split("train")
_, x_va, y_va = load_split("val")
test_df, x_te, y_te = load_split("test")
train_domains = set(train_df["Domain"].str.lower())
frac_phish = float((y_tr == 0).float().mean())
w_tr = torch.where(y_tr == 0, torch.tensor(1.0 - frac_phish), torch.tensor(frac_phish))


def prunable(model):
    # Same as the notebook EXCEPT model.out.weight is left out.
    ts = [model.input_proj.weight, model.attn.out_proj.weight, model.attn.in_proj_weight]
    ts += [m.weight for m in model.mlp if isinstance(m, nn.Linear)]
    return ts


def prune_to(model, amount):
    ts = prunable(model)
    allw = torch.cat([t.detach().abs().flatten() for t in ts])
    k = max(1, int(amount * allw.numel()))
    thr = torch.kthvalue(allw, k).values
    masks = []
    with torch.no_grad():
        for t in ts:
            m = (t.abs() > thr).float()
            t.mul_(m)
            masks.append(m)
    return masks


def reapply(model, masks):
    with torch.no_grad():
        for t, m in zip(prunable(model), masks):
            t.mul_(m)


def finetune(model, masks, epochs, patience=3, lr=0.0005, batch=128):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, wait, n = float("inf"), None, 0, len(x_tr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(model(x_tr[idx]), y_tr[idx], weight=w_tr[idx]).backward()
            opt.step()
            reapply(model, masks)
        model.eval()
        with torch.no_grad():
            vl = nn.functional.binary_cross_entropy_with_logits(model(x_va), y_va).item()
        if vl < best - 1e-5:
            best, wait, best_state = vl, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    reapply(model, masks)
    return model


def stats(model):
    model.eval()
    with torch.no_grad():
        r = evaluate(model(x_te), y_te, test_df, train_domains)
        pred_legit = (torch.sigmoid(model(x_te)) >= 0.5)
    r["collapsed"] = int(pred_legit.all() or (~pred_legit).all())
    ts = prunable(model)
    r["sparsity_prunable"] = float(sum((t == 0).sum().item() for t in ts) / sum(t.numel() for t in ts))
    r["nonzero_params_total"] = int(sum((p != 0).sum().item() for p in model.parameters()))
    r["out_weight_nonzero"] = int((model.out.weight != 0).sum().item())
    return r


SCHEDULES = {0.70: [0.5, 0.7], 0.90: [0.5, 0.7, 0.85, 0.90], 0.95: [0.5, 0.7, 0.85, 0.90, 0.95]}
out_csv = RESULTS / "revision" / f"pruning_protected_{MODE}.csv"
rows = []
t0 = time.time()
for seed in (42, 43, 44):
    targets = [0.5, 0.7, 0.9, 0.95] if MODE == "oneshot" else ([0.7, 0.9, 0.95] if MODE == "iterative" else [0.9, 0.95])
    for target in targets:
        torch.manual_seed(seed); np.random.seed(seed)
        model = load_weights(Teacher(), MODELS / "baselines" / f"baselineU_seed{seed}.pt")
        if MODE == "oneshot":
            masks = prune_to(model, target); model = finetune(model, masks, epochs=10)
        elif MODE == "matched":
            masks = prune_to(model, target); model = finetune(model, masks, epochs=30)
        else:
            for s in SCHEDULES[target]:
                masks = prune_to(model, s); model = finetune(model, masks, epochs=5)
            model = finetune(model, masks, epochs=10)
        r = stats(model); r.update(seed=seed, target_sparsity=target, mode=MODE, minutes=round((time.time() - t0) / 60, 1))
        rows.append(r); print(json.dumps(r), flush=True)
        torch.save(model.state_dict(), RESULTS / "revision" / f"prune_protected_{MODE}_s{int(target*100)}_seed{seed}.pt")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
print("DONE", MODE)
