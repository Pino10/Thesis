"""Revision analysis for supervisor comment 10: how URLs in each class begin.

Reports, for each class, the share of URLs that start with https:// and the share whose
hostname starts with www. This explains the "www." shortcut seen in the external check.

    python code/revision/url_prefix_share.py
"""
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA, RESULTS

df = pd.concat([pd.read_parquet(DATA / f"split_{s}.parquet") for s in ("train", "val", "test")], ignore_index=True)
u = df.URL.str.lower()
host = u.str.replace(r"^https?://", "", regex=True)
rep = {}
for cls, lab in (("phishing", 0), ("legitimate", 1)):
    m = df.label == lab
    rep[cls] = {"https_pct": round(float(u[m].str.startswith("https://").mean() * 100), 1),
                "www_pct": round(float(host[m].str.startswith("www.").mean() * 100), 1)}
json.dump(rep, open(RESULTS / "revision" / "url_prefix_share.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
