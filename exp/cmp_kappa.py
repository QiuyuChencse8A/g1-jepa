
import json, numpy as np, itertools

TAGS = [f"k{k}_c{c}" for k, c in itertools.product([2.5, 3.5], [1, 2])]
rows, per_scen = {}, {}

for tag in TAGS:
    try:
        d = json.load(open(f"results/closed_loop_{tag}.json"))
    except FileNotFoundError:
        continue
    rp, rc = d["jepa"]["perturbed"], d["jepa"]["clean"]
    fired = [r for r in rp if r["perturbed"]]
    lats = [r["trigger_latency"] for r in fired if r["trigger_latency"] is not None]
    n = len(fired)
    rows[tag] = dict(
        n=n, triggered=len(lats),
        med=np.median(lats), p75=np.percentile(lats, 75),
        p90=np.percentile(lats, 90), mx=max(lats), mean=np.mean(lats),
        le5=sum(l <= 5 for l in lats) / n, le10=sum(l <= 10 for l in lats) / n,
        # episode 级：至少一次误触发的 clean episode 占比（与离线 FP 同口径）
        fp_ep=np.mean([r["n_replans"] > 0 for r in rc]),
        # 次数级：每条 clean 的平均浪费重规划
        fp_cnt=np.mean([r["n_replans"] for r in rc]),
        succ=np.mean([r["success"] for r in fired]),
        steps=np.mean([r["steps"] for r in fired if r["success"]]),
    )
    per_scen[tag] = {r["seed"]: r["trigger_latency"] for r in fired}

hdr = f"{'config':<12}{'med':>5}{'P75':>5}{'P90':>6}{'max':>6}{'mean':>7}" \
      f"{'≤5':>7}{'≤10':>7}{'FP_ep':>7}{'FP_cnt':>8}{'succ':>6}{'steps':>7}"
print(hdr); print("-" * len(hdr))
for tag, r in rows.items():
    print(f"{tag:<12}{r['med']:>5.0f}{r['p75']:>5.0f}{r['p90']:>6.0f}{r['mx']:>6.0f}"
          f"{r['mean']:>7.1f}{r['le5']:>7.2f}{r['le10']:>7.2f}"
          f"{r['fp_ep']:>7.2f}{r['fp_cnt']:>8.2f}{r['succ']:>6.2f}{r['steps']:>7.1f}")

# paired: 同场景逐条对比
a, b = "k2.5_c2", "k3.5_c1"
if a in per_scen and b in per_scen:
    common = sorted(set(per_scen[a]) & set(per_scen[b]))
    diffs = [(s, per_scen[a][s], per_scen[b][s]) for s in common
             if per_scen[a][s] is not None and per_scen[b][s] is not None]
    print(f"\n配对比较 {a} → {b}（同场景，共 {len(diffs)} 条）")
    print(f"  {b} 更快: {sum(x[2] < x[1] for x in diffs)} 条, "
          f"更慢: {sum(x[2] > x[1] for x in diffs)} 条, "
          f"相同: {sum(x[2] == x[1] for x in diffs)} 条")
    big = [x for x in diffs if abs(x[1] - x[2]) >= 10]
    print(f"  差异≥10步的场景: {[(s, la, lb) for s, la, lb in big] or '无'}")
