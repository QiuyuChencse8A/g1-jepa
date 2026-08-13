import json, numpy as np
RUN, WIN = "results/final_run1.json", 8
d = json.load(open(RUN))

rows = []
for r in d["jepa"]["perturbed"]:
    if not r.get("perturbed") or not r.get("dbg_log"): continue
    p = r["perturb_step"]
    win = [x for x in r["dbg_log"] if p <= x[0] <= p + WIN]
    if not win: continue
    bz  = max(win, key=lambda x: x[6])   # (t, bucket, fb, err, mu, sd, z)
    be  = max(win, key=lambda x: x[3])
    L = r["trigger_latency"]
    rows.append((r["seed"], p, (L is not None and L <= 10),
                 bz[0], bz[1], bz[3], bz[6],
                 be[0], be[1], be[3], be[6],
                 len({x[1] for x in win})))

print(f"{'seed':>6}{'p':>4}{'检出':>5} | {'argmax-z: t':>12}{'桶':>3}{'err':>7}{'z':>7}"
      f" | {'argmax-err: t':>14}{'桶':>3}{'err':>7}{'z':>7}{'跨桶数':>7}")
for x in sorted(rows, key=lambda r: (r[2], r[1])):
    if x[2]: continue                     # 只看失效
    print(f"{x[0]:>6}{x[1]:>4}{'':>5} | {x[3]:>12}{x[4]:>3}{x[5]:>7.1f}{x[6]:>7.2f}"
          f" | {x[7]:>14}{x[8]:>3}{x[9]:>7.1f}{x[10]:>7.2f}{x[11]:>7}")

fail = [x for x in rows if not x[2]]
print(f"\n失效 {len(fail)} 条，其中两种取法落在不同步的: "
      f"{sum(1 for x in fail if x[3] != x[7])} 条")
print(f"按 argmax-err 的桶分布: "
      f"{dict(zip(*np.unique([x[8] for x in fail], return_counts=True)))}")
print(f"按 argmax-err 的原始误差中位: {np.median([x[9] for x in fail]):.1f}")