import json, numpy as np
d = json.load(open("results/closed_loop_s5000_k3.5_c1.json"))
for lo, hi in [(15,17),(18,20),(21,23),(24,27)]:
    g = [r for r in d["jepa"]["perturbed"]
         if r["perturbed"] and lo <= r["perturb_step"] <= hi]
    if not g: continue
    ok = [(r["trigger_latency"] is not None and r["trigger_latency"] <= 10) for r in g]
    peaks, shifts = [], []
    for r in g:
        p = r["perturb_step"]
        post = [v for t, v in r["z_log"] if p <= t <= p + 8]
        pre  = [v for t, v in r["z_log"] if t < p]
        peaks.append(max(post) if post else np.nan)
        shifts.append((np.mean(post) - np.mean(pre)) if pre and post else np.nan)
    print(f"{lo}-{hi}: n={len(g):3d}  ≤10检出={np.mean(ok):.2f}  "
          f"z峰中位={np.nanmedian(peaks):5.2f}  "
          f"post-pre均值差={np.nanmean(shifts):5.2f}  "
          f"失败率={1-np.mean([r['success'] for r in g]):.2f}")