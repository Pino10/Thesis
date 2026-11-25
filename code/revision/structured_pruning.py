"""Revision analysis for supervisor comment 8: structured pruning.

Structured pruning removes whole units, so the result is a genuinely smaller dense
network (unlike unstructured pruning, whose zeros still take space).

Starting from each saved Baseline U model (seeds 42, 43, 44):
  - Attention: the block sees a sequence of length one, so every attention weight is 1
    and the block computes out_proj(v_proj(h)). The query and key projections have no
    effect on the output and are removed first (this changes no prediction; checked
    below). Heads are then ranked by the L2 norm of their value and output weights,
    and the weakest are removed.
  - MLP: hidden units in each of the three layers are ranked by the L1 norm of their
    incoming weights, and the weakest are removed together with their outgoing weights.
The pruned network is then fine-tuned with the same recipe as prune_protected.py
(Adam lr 0.0005, batch 128, weighted BCE, early stopping on unweighted validation BCE,
min delta 1e-5, patience 3, at most 10 epochs).

Levels (share of heads and hidden units removed): 0 (query/key only), 50, 75 and
87.5 per cent, and one level with the whole attention block removed (MLP kept whole).

    python code/revision/structured_pruning.py
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
HD = 16  # head dimension (128 / 8 heads)

train_df, x_tr, y_tr = load_split("train")
_, x_va, y_va = load_split("val")
test_df, x_te, y_te = load_split("test")
train_domains = set(train_df["Domain"].str.lower())
frac_phish = float((y_tr == 0).float().mean())
w_tr = torch.where(y_tr == 0, torch.tensor(1.0 - frac_phish), torch.tensor(frac_phish))


class Slim(nn.Module):
    """Baseline U with query/key removed, some heads kept and narrower MLP layers."""

    def __init__(self, heads, widths, n_features=24):
        super().__init__()
        self.heads = heads
        self.input_proj = nn.Linear(n_features, 128)
        if heads:
            self.v = nn.Linear(128, heads * HD)
            self.o = nn.Linear(heads * HD, 128)
        self.norm = nn.LayerNorm(128)
        a, b, c = widths
        self.mlp = nn.Sequential(nn.Linear(128, a), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(a, b), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(b, c), nn.ReLU(), nn.Dropout(0.2))
        self.out = nn.Linear(c, 1)

    def forward(self, x):
        h = self.input_proj(x)
        if self.heads:
            h = h + self.o(self.v(h))
        return self.out(self.mlp(self.norm(h))).squeeze(-1)


def slim_from(t, heads_keep, widths):
    """Copy the kept heads and units of a trained Teacher into a Slim network."""
    s = Slim(len(heads_keep), [len(w) for w in widths])
    with torch.no_grad():
        s.input_proj.load_state_dict(t.input_proj.state_dict())
        s.norm.load_state_dict(t.norm.state_dict())
        W, B = t.attn.in_proj_weight, t.attn.in_proj_bias
        Wv, Bv = W[256:384], B[256:384]
        Wo, Bo = t.attn.out_proj.weight, t.attn.out_proj.bias
        if heads_keep:
            idx = torch.cat([torch.arange(h * HD, (h + 1) * HD) for h in heads_keep])
            s.v.weight.copy_(Wv[idx]); s.v.bias.copy_(Bv[idx])
            s.o.weight.copy_(Wo[:, idx]); s.o.bias.copy_(Bo)
        lin_t = [m for m in t.mlp if isinstance(m, nn.Linear)]
        lin_s = [m for m in s.mlp if isinstance(m, nn.Linear)]
        prev = torch.arange(128)
        for lt, ls, keep in zip(lin_t, lin_s, widths):
            ls.weight.copy_(lt.weight[keep][:, prev]); ls.bias.copy_(lt.bias[keep]); prev = keep
        s.out.weight.copy_(t.out.weight[:, prev]); s.out.bias.copy_(t.out.bias)
    return s


def choose(t, head_frac, unit_frac, drop_attention=False):
    Wv, Wo = t.attn.in_proj_weight[256:384], t.attn.out_proj.weight
    score = [float(Wv[h * HD:(h + 1) * HD].norm() * Wo[:, h * HD:(h + 1) * HD].norm()) for h in range(8)]
    n_heads = 0 if drop_attention else 8 - int(round(8 * head_frac))
    heads = sorted(np.argsort(score)[::-1][:n_heads].tolist())
    widths = []
    for lt in [m for m in t.mlp if isinstance(m, nn.Linear)]:
        n = lt.out_features - int(round(lt.out_features * unit_frac))
        widths.append(torch.sort(torch.argsort(lt.weight.abs().sum(1), descending=True)[:n]).values)
    return heads, widths


def finetune(model, epochs=10, patience=3, lr=0.0005, batch=128):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, wait, n = float("inf"), None, 0, len(x_tr)
    for _ in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]; opt.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(model(x_tr[idx]), y_tr[idx], weight=w_tr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.binary_cross_entropy_with_logits(model(x_va), y_va).item()
        if vl < best - 1e-5:
            best, wait, best_state = vl, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state); model.eval()
    return model


LEVELS = [("Query and key removed only", 0.0, 0.0, False), ("50% of heads and units", 0.5, 0.5, False),
          ("75% of heads and units", 0.75, 0.75, False), ("87.5% of heads and units", 0.875, 0.875, False),
          ("Attention block removed", 0.0, 0.0, True)]
rows, t0 = [], time.time()
for seed in (42, 43, 44):
    teacher = load_weights(Teacher(), MODELS / "baselines" / f"baselineU_seed{seed}.pt")
    for name, hf, uf, drop in LEVELS:
        torch.manual_seed(seed); np.random.seed(seed)
        heads, widths = choose(teacher, hf, uf, drop)
        m = slim_from(teacher, heads, widths); m.eval()
        if hf == 0 and uf == 0 and not drop:   # exact: no fine-tuning needed, check equality
            with torch.no_grad():
                diff = float((m(x_te) - teacher(x_te)).abs().max())
            assert diff < 1e-4, diff
        else:
            m = finetune(m)
        with torch.no_grad():
            r = evaluate(m(x_te), y_te, test_df, train_domains)
        r.update(level=name, seed=seed, heads=len(heads), widths="/".join(str(len(w)) for w in widths),
                 parameters=int(sum(p.numel() for p in m.parameters())), minutes=round((time.time() - t0) / 60, 1))
        rows.append(r); print(json.dumps(r), flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / "revision" / "structured_pruning.csv", index=False)
print("DONE")
