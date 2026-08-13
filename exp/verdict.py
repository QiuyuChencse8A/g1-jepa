import json, numpy as np
d = json.load(open("results/closed_loop_s5000_k3.5_c1.json"))
rows = []
for r in d["jepa"]["perturbed"]:
    if not r["perturbed"]: continue
    p = r["perturb_step"]
    win = [x for x in r["dbg_log"] if p <= x[0] <= p + 8]
    if not win: continue
    top = max(win, key=lambda x: x[6])          # z 最大的那一步
    t, b, fb, err, mu, sd, z = top
    L = r["trigger_latency"]
    grp = "正常" if (L is not None and L <= 10) else "失效"
    rows.append((grp, r["seed"], p, b, fb, err, mu, sd, z))

print(f"{'组':<5}{'seed':>6}{'扰动步':>7}{'桶':>4}{'回退':>6}"
      f"{'原始误差':>10}{'μ':>9}{'σ':>8}{'z':>7}")
for r in sorted(rows, key=lambda x: (x[0], x[2]))[:12] + sorted(rows, key=lambda x: (x[0], x[2]))[-8:]:
    print(f"{r[0]:<5}{r[1]:>6}{r[2]:>7}{r[3]:>4}{str(r[4]):>6}"
          f"{r[5]:>10.1f}{r[6]:>9.1f}{r[7]:>8.2f}{r[8]:>7.2f}")

print("\n=== 按组统计 ===")
for grp in ["正常", "失效"]:
    g = [r for r in rows if r[0] == grp]
    print(f"{grp}: n={len(g)}  原始误差中位={np.median([r[5] for r in g]):.1f}  "
          f"z中位={np.median([r[8] for r in g]):.2f}  "
          f"回退比例={np.mean([r[4] for r in g]):.2f}  "
          f"桶分布={dict(zip(*np.unique([r[3] for r in g], return_counts=True)))}")