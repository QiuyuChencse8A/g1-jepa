import json, numpy as np
d = json.load(open("results/final_run1.json"))
rows = []
for r in d["jepa"]["perturbed"]:
    if not r.get("perturbed"): continue
    p = r["perturb_step"]
    pre = [s for s in (r.get("replan_steps") or []) if s < p]
    L = r["trigger_latency"]
    rows.append((r["seed"], p, len(pre),
                 (L is not None and L <= 10), bool(r["success"])))

n = len(rows)
has = [x for x in rows if x[2] > 0]
non = [x for x in rows if x[2] == 0]
print(f"共 {n} 条，扰动前发生过重规划: {len(has)} 条\n")
for grp, lab in ((non, "扰动前无重规划"), (has, "扰动前有重规划")):
    if grp:
        print(f"{lab:<16} n={len(grp):3d}  ≤10检出={np.mean([x[3] for x in grp]):.2f}  "
              f"成功率={np.mean([x[4] for x in grp]):.2f}")
print("\n扰动前有重规划且漏检的 seed:",
      [x[0] for x in has if not x[3]])