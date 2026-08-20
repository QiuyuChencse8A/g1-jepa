import json, numpy as np
from collections import Counter
from analyze_error2 import bucket, errors, feat, load_split, valid_mask

NAME, K, METRIC, WARMUP = "grid", 1, "abs", 16
cal = json.load(open("calib_f16s1_wrist_grid_k1.json"))

def thr_of(b):
    mu, sd = cal["buckets"].get(str(b), [cal["global_mu"], cal["global_sd"]])
    return mu + 3.5 * sd

off = []
for ep in load_split("latents/clean_f16s1_wrist")[:100]:
    e = errors(feat(ep, NAME), K, METRIC)
    m = valid_mask(ep, WARMUP)
    for t, ei, ph, pst, ok in zip(ep["t"], e, ep["phase"], ep["phase_step"], m):
        if ok and not np.isnan(ei):
            off.append((int(t), int(bucket(ph, pst)), int(pst), float(ei)))

d = json.load(open("results/final_run1.json"))
on = [(int(t), int(b), None, float(err))
      for r in d["jepa"]["clean"] for t, b, fb, err, mu, sd, z in (r.get("dbg_log") or [])]

print("== 桶3 的时间支撑集 ==")
for lab, rows in (("离线标定", off), ("在线    ", on)):
    ts = [r[0] for r in rows if r[1] == 3]
    print(f"{lab} n={len(ts):4d}  t 分布: {sorted(Counter(ts).items())}")

print("\n== 桶3 的 phase_step（仅离线可得）==")
print(sorted(Counter(r[2] for r in off if r[1] == 3).items()))

print("\n== 在线 clean 越阈归因 ==")
tot, exc = Counter(), Counter()
for t, b, pst, ei in on:
    tot[b] += 1
    if ei > thr_of(b):
        exc[b] += 1
for b in sorted(tot):
    print(f"  桶 {b}: {exc[b]:3d} / {tot[b]:5d}")