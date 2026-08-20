import json, numpy as np
from collections import defaultdict

SRC = "results/observe_calib_dev.json"
OUT = "calib_deploy_matched.json"
MIN_N = 30

old = json.load(open("calib_f16s1_wrist_grid_k1.json"))
d = json.load(open(SRC))

acc = defaultdict(list)
for r in d["jepa"]["clean"]:
    if r["n_replans"] != 0:
        print(f"警告: seed {r['seed']} 有 {r['n_replans']} 次重规划，跳过")
        continue
    for t, b, fb, err, mu, sd, z in (r.get("dbg_log") or []):
        acc[int(b)].append(float(err))

new = dict(old)
new["buckets"] = {}
new["source"] = "observe-only dev clean, seed0=0"
print(f"{'桶':>4}{'n':>7}{'新 μ':>9}{'新 σ':>8}{'旧 μ':>9}{'旧 σ':>8}{'Δμ/σ旧':>9}")
for b in sorted(acc):
    v = np.array(acc[b])
    if len(v) < MIN_N:
        print(f"{b:>4}{len(v):>7}  样本不足，沿用旧值或回退 global")
        if str(b) in old["buckets"]:
            new["buckets"][str(b)] = old["buckets"][str(b)]
        continue
    m, s = float(v.mean()), float(v.std() + 1e-9)
    new["buckets"][str(b)] = [m, s]
    om, os_ = old["buckets"].get(str(b), [old["global_mu"], old["global_sd"]])
    print(f"{b:>4}{len(v):>7}{m:>9.1f}{s:>8.2f}{om:>9.1f}{os_:>8.2f}{(m-om)/os_:>9.2f}")

allv = np.concatenate([np.array(v) for v in acc.values()])
new["global_mu"], new["global_sd"] = float(allv.mean()), float(allv.std() + 1e-9)
json.dump(new, open(OUT, "w"), indent=1)
print(f"\n已写入 {OUT}: {len(new['buckets'])} 个桶")