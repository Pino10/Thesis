"""Revision analysis for supervisor comment 2: deployability end to end.

1. Raw weight bytes: parameters x 4 bytes (float32), compared with the PyTorch file size.
2. ONNX export of the Baseline U teacher and the S-16 student; ONNX file sizes.
3. Latency with ONNX Runtime (single thread) for: feature extraction from the raw URL
   string alone, model inference alone, and the full pipeline (URL string -> features
   -> scaling -> prediction). Warm-up then timed repetitions, as in the notebook.
4. The CPU model is recorded, since the supervisor noted it was unidentified.

    python code/revision/deployment_timing.py
"""
import json, os, platform, statistics, sys, time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collections import Counter

from common import DATA, MIN_U, MODELS, RESULTS, RNG_U, Student, Teacher, load_weights
from url_features import extract

OUT = RESULTS / "revision"
cpu = "unknown"
try:
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip(); break
except OSError:
    pass

train = pd.read_parquet(DATA / "split_train.parquet")
test = pd.read_parquet(DATA / "split_test.parquet")
tld_table = train.groupby(train.TLD.str.lower())["TLDLegitimateProb"].first().to_dict()
c = Counter()
for u in train.URL:
    c.update(u.lower())
tot = sum(c.values()); char_table = {k: v / tot for k, v in c.items()}
urls = test.URL.sample(2000, random_state=0).tolist()

so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
results = {"cpu": cpu, "python": platform.python_version(), "onnxruntime": ort.__version__,
           "tld_table_entries": len(tld_table), "char_table_entries": len(char_table)}

def timed(fn, items, warm=20):
    for it in items[:warm]:
        fn(it)
    ts = []
    for it in items:
        t = time.perf_counter(); fn(it); ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), float(np.percentile(ts, 95))

def scale(f):
    return np.clip((np.asarray(f, dtype=np.float32) - MIN_U) / RNG_U, 0, 1).astype(np.float32)[None, :]

for name, model, path in (("Baseline U", Teacher(), MODELS / "baselines" / "baselineU_seed42.pt"),
                          ("S-16", Student("S-16"), MODELS / "distillation" / "student_S-16_seed42.pt")):
    load_weights(model, path)
    n_params = sum(p.numel() for p in model.parameters())
    onnx_path = OUT / f"{name.replace(' ', '_')}_seed42.onnx"
    torch.onnx.export(model, torch.zeros(1, 24), onnx_path, input_names=["x"], output_names=["logit"],
                      dynamic_axes={"x": {0: "batch"}}, opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    run = lambda f: sess.run(None, {"x": f})
    feats = [scale(extract(u, tld_table, char_table)) for u in urls]
    inf_med, inf_p95 = timed(run, feats)
    full_med, full_p95 = timed(lambda u: run(scale(extract(u, tld_table, char_table))), urls)
    batch = np.concatenate(feats[:512])
    b_med, _ = timed(lambda b: run(b), [batch] * 50, warm=5)
    results[name] = dict(
        parameters=n_params, raw_weight_bytes=n_params * 4,
        pytorch_file_bytes=os.path.getsize(path), onnx_file_bytes=os.path.getsize(onnx_path),
        inference_ms_median=inf_med, inference_ms_p95=inf_p95,
        full_pipeline_ms_median=full_med, full_pipeline_ms_p95=full_p95,
        batch512_ms_per_url=b_med / 512)

fe_med, fe_p95 = timed(lambda u: extract(u, tld_table, char_table), urls)
results["feature_extraction_ms_median"] = fe_med
results["feature_extraction_ms_p95"] = fe_p95
json.dump(results, open(OUT / "deployment_timing.json", "w"), indent=1)
print(json.dumps(results, indent=1))
