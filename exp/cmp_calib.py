import json, numpy as np
for tag, f in (("旧标定", "results/dev_old_calib.json"),
               ("新标定", "results/dev_new_calib.json")):
    d = json.load(open(f))
    rp, rc = d["jepa"]["perturbed"], d["jepa"]["clean"]
    fired = [r for r in rp if r["perturbed"]]
    lats = [r["trigger_latency"] for r in fired if r["trigger_latency"] is not None]
    n = len(fired)
    print(f"{tag}  ≤5={sum(l<=5 for l in lats)/n:.2f}  ≤10={sum(l<=10 for l in lats)/n:.2f}"
          f"  中位={np.median(lats) if lats else float('nan'):.0f}"
          f"  FP_ep={np.mean([r['n_replans']>0 for r in rc]):.2f}"
          f"  成功={np.mean([r['success'] for r in fired]):.2f}"
          f"  重规划={np.mean([r['n_replans'] for r in rp]):.2f}")