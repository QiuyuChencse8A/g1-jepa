import json, numpy as np
vB_run = [f"results/vB_run{i}.json" for i in range(1, 6)]

def stats(rp, rc):
    fired = [r for r in rp if r["perturbed"]]
    lats = [r["trigger_latency"] for r in fired if r["trigger_latency"] is not None]
    n = len(fired)
    succ = [r for r in fired if r["success"]]
    return dict(
        n_all=n, n_detected=len(lats),
        SR=np.mean([r["success"] for r in fired]),
        SRc=np.mean([r["success"] for r in rc]),
        med=np.median(lats), p90=np.percentile(lats, 90), mx=max(lats),
        mean=np.mean(lats),
        le5=sum(l <= 5 for l in lats)/n, le10=sum(l <= 10 for l in lats)/n,
        fp=np.mean([r["n_replans"] > 0 for r in rc]),
        steps=np.mean([r["steps"] for r in succ]),
    )

acc = {}
for f in vB_run:
    d = json.load(open(f))
    for cond in ["fixed", "oracle", "jepa"]:
        acc.setdefault(cond, []).append(stats(d[cond]["perturbed"], d[cond]["clean"]))

keys = ["SR","SRc","med","p90","mx","mean","le5","le10","fp","steps"]
print(f"{'cond':<8}" + "".join(f"{k:>14}" for k in keys))
for cond, rows in acc.items():
    line = f"{cond:<8}"
    for k in keys:
        v = [r[k] for r in rows]
        line += f"{np.mean(v):>8.2f}±{np.std(v, ddof=1):<5.2f}"
    print(line)