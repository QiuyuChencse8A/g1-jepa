import json, numpy as np
d = json.load(open("results/final_run1.json"))
cal = json.load(open("calib_f16s1_wrist_grid_k1.json"))

by_b = {}
for r in d["jepa"]["clean"]:
    for t, b, fb, err, mu, sd, z in (r.get("dbg_log") or []):
        by_b.setdefault(int(b), []).append(float(err))

print(f"{'桶':>4}{'在线 n':>8}{'在线 μ':>10}{'在线 σ':>9}"
      f"{'标定 μ':>10}{'标定 σ':>9}{'Δμ/σ':>8}")
for b in sorted(by_b):
    v = np.array(by_b[b])
    if str(b) in cal["buckets"]:
        cmu, csd = cal["buckets"][str(b)]
    else:
        cmu, csd = cal["global_mu"], cal["global_sd"]
    print(f"{b:>4}{len(v):>8}{v.mean():>10.1f}{v.std():>9.2f}"
          f"{cmu:>10.1f}{csd:>9.2f}{(v.mean()-cmu)/csd:>8.2f}")

allv = np.concatenate(list(by_b.values()))
print(f"\n在线 clean 全部步数 n={len(allv)}，"
      f"超过各自桶阈值 (μ+3.5σ) 的步数: "
      f"{sum(1 for b in by_b for e in by_b[b] if e > (cal['buckets'].get(str(b), [cal['global_mu'], cal['global_sd']])[0] + 3.5 * cal['buckets'].get(str(b), [cal['global_mu'], cal['global_sd']])[1]))}")