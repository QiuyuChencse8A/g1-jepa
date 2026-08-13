import numpy as np
from analyze_error2 import bucket, errors, feat, load_split, valid_mask

NAME, K, METRIC, WARMUP = "grid", 1, "abs", 16
ROOT, TAG = "latents", "f16s1_wrist"

clean = load_split(f"{ROOT}/clean_{TAG}")
pert  = load_split(f"{ROOT}/perturbed_{TAG}")
calib, val = clean[:100], clean[100:200]

def steps(ep, every=1, ph0=0):
    e = errors(feat(ep, NAME), K, METRIC)
    m = valid_mask(ep, WARMUP)
    out = [(int(t), int(bucket(bph, pst)), float(ei))
           for t, ei, bph, pst, ok in zip(ep["t"], e, ep["phase"], ep["phase_step"], m)
           if ok and not np.isnan(ei)]
    return out if every == 1 else [x for x in out if x[0] % every == ph0]

by_b = {}
for ep in calib:
    for _, b, ei in steps(ep):
        by_b.setdefault(b, []).append(ei)
mu = {b: float(np.mean(v)) for b, v in by_b.items()}
sd = {b: float(np.std(v)) for b, v in by_b.items()}

def fp_ep(thr, every=1, ph0=0):
    return float(np.mean([any(ei > thr.get(b, np.inf)
                              for _, b, ei in steps(ep, every, ph0)) for ep in val]))

def detect(thr, win, every=1, ph0=0):
    ok = []
    for ep in pert:
        p = int(ep["perturb_step"])
        hit = [t for t, b, ei in steps(ep, every, ph0)
               if t >= p and ei > thr.get(b, np.inf)]
        ok.append(bool(hit) and (min(hit) - p) <= win)
    return float(np.mean(ok))

def row(lab, thr, every=1, ph0=0):
    print(f"{lab:<24}{fp_ep(thr,every,ph0):>8.2f}"
          f"{detect(thr,5,every,ph0):>8.2f}{detect(thr,10,every,ph0):>8.2f}")

base = {b: mu[b] + 3.5 * sd[b] for b in mu}
HDR = f"{'方案':<24}{'FP_ep':>8}{'≤5':>8}{'≤10':>8}"

print("== 采样节拍 (阈值固定为基线) ==\n" + HDR)
row("每步 (离线现状)", base)
row("每2步 偶数", base, 2, 0)
row("每2步 奇数", base, 2, 1)

print("\n== 按桶经验分位 ==\n" + HDR)
row("全局 κ=3.5 (基线)", base)
for q in (0.99, 0.995, 0.999):
    row(f"按桶分位 q={q}", {b: float(np.percentile(v, q*100)) for b, v in by_b.items()})

print("\n== 仅放松桶3 ==\n" + HDR)
for k3 in (3.5, 3.0, 2.5, 2.0, 1.6):
    row(f"仅桶3 κ={k3}", {b: mu[b] + (k3 if b == 3 else 3.5) * sd[b] for b in mu})