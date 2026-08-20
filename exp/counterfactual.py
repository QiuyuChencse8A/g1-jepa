import json, numpy as np
from collections import defaultdict

KAPPA = 3.5
d = json.load(open("results/final_run1.json"))
cal = json.load(open("calib_f16s1_wrist_grid_k1.json"))

def old_par(b):
    return tuple(cal["buckets"].get(str(b), [cal["global_mu"], cal["global_sd"]]))

# 在线标定：只取每条 clean 首次误触发之前的步，避开重规划污染
acc = defaultdict(list)
for r in d["jepa"]["clean"]:
    for t, b, fb, err, mu, sd, z in (r.get("dbg_log") or []):
        m0, s0 = old_par(b)
        if err > m0 + KAPPA * s0:
            break                      # 首次误触发，之后的数据已被污染
        acc[int(b)].append(float(err))
new = {b: (float(np.mean(v)), float(np.std(v))) for b, v in acc.items() if len(v) >= 30}
print("在线标定桶:", {b: (round(m,1), round(s,2), len(acc[b])) for b,(m,s) in new.items()})

def first_trig(log, par):
    for t, b, fb, err, mu, sd, z in log:
        m0, s0 = par(int(b))
        if err > m0 + KAPPA * s0:
            return int(t)
    return None

def par_new(b):
    return new.get(b, old_par(b))

for lab, par in (("旧标定", old_par), ("在线匹配标定", par_new)):
    fp = np.mean([first_trig(r.get("dbg_log") or [], par) is not None
                  for r in d["jepa"]["clean"]])
    le5 = le10 = 0
    for r in d["jepa"]["perturbed"]:
        if not r.get("perturbed"): continue
        p, ft = r["perturb_step"], first_trig(r.get("dbg_log") or [], par)
        if ft is not None and ft >= p:
            le5  += (ft - p) <= 5
            le10 += (ft - p) <= 10
    n = sum(1 for r in d["jepa"]["perturbed"] if r.get("perturbed"))
    print(f"{lab:<14} FP_ep={fp:.2f}  ≤5={le5/n:.2f}  ≤10={le10/n:.2f}")