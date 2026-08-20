import json, numpy as np
for tag, f in (("旧标定", "results/fp_holdout_old.json"),
               ("新标定", "results/fp_holdout_new.json")):
    d = json.load(open(f))
    rc = d["jepa"]["clean"]
    fired = [r for r in d["jepa"]["perturbed"] if r["perturbed"]]
    lats = [r["trigger_latency"] for r in fired if r["trigger_latency"] is not None]
    n = len(fired)
    print(f"{tag}  留出clean n={len(rc)}  "
          f"FP_ep={np.mean([r['n_replans']>0 for r in rc]):.2f}  "
          f"浪费/条={np.mean([r['n_replans'] for r in rc]):.2f}  |  "
          f"≤10={sum(l<=10 for l in lats)/n:.2f}  "
          f"成功={np.mean([r['success'] for r in fired]):.2f}  "
          f"重规划={np.mean([r['n_replans'] for r in fired]):.2f}")