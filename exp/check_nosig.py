import json
import numpy as np

d = json.load(open("results/closed_loop_s5000_k3.5_c1.json"))
NO_SIG = {5011, 5017, 5034, 5051, 5060, 5089}

rows = []
for r in d["jepa"]["perturbed"]:
    if not r["perturbed"]:
        continue
    dl = np.array(r["perturb_delta"][:2])
    mag = np.linalg.norm(dl) * 100
    ang = np.degrees(np.arctan2(dl[1], dl[0]))
    lat = r["trigger_latency"]
    if r["seed"] in NO_SIG:
        tag = "无信号"
    elif lat is None or lat > 10:
        tag = "弱信号"
    else:
        tag = "正常"
    rows.append((tag, r["seed"], r["perturb_step"], mag, ang))

for tag in ["无信号", "弱信号"]:
    print(f"\n=== {tag} ===")
    for t, s, p, m, a in rows:
        if t == tag:
            print(f"  seed={s} 扰动步={p:3d} |delta|={m:5.1f}cm 方向={a:7.0f}°")

print("\n=== 三组统计 ===")
for tag in ["正常", "弱信号", "无信号"]:
    g = [r for r in rows if r[0] == tag]
    if not g:
        continue
    mags = [r[3] for r in g]
    steps = [r[2] for r in g]
    print(f"{tag}: n={len(g)}  |delta| 均值={np.mean(mags):.1f}cm "
          f"范围=[{min(mags):.1f}, {max(mags):.1f}]  "
          f"扰动步均值={np.mean(steps):.0f} 范围=[{min(steps)}, {max(steps)}]")