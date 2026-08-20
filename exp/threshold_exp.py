import numpy as np
from analyze_error2 import bucket, errors, feat, load_split, valid_mask

NAME, K, METRIC, WARMUP = "grid", 1, "abs", 16
ROOT, TAG = "latents", "f16s1_wrist"

clean = load_split(f"{ROOT}/clean_{TAG}")
pert  = load_split(f"{ROOT}/perturbed_{TAG}")
calib, val = clean[:100], clean[100:200]

def steps(ep, every=1, ph0=0, tmin=0):
    e = errors(feat(ep, NAME), K, METRIC)
    m = valid_mask(ep, WARMUP)
    out = [(int(t), int(bucket(bph, pst)), float(ei))
           for t, ei, bph, pst, ok in zip(ep["t"], e, ep["phase"], ep["phase_step"], m)
           if ok and not np.isnan(ei) and int(t) >= tmin]
    return out if every == 1 else [x for x in out if x[0] % every == ph0]

by_b = {}
for ep in calib:
    for _, b, ei in steps(ep):
        by_b.setdefault(b, []).append(ei)
mu = {b: float(np.mean(v)) for b, v in by_b.items()}
sd = {b: float(np.std(v)) for b, v in by_b.items()}

def fp_ep(thr, every=1, ph0=0, tmin=0):
    return float(np.mean([any(ei > thr.get(b, np.inf)
                              for _, b, ei in steps(ep, every, ph0, tmin))
                          for ep in val]))

def detect(thr, win, every=1, ph0=0, tmin=0):
    ok = []
    for ep in pert:
        p = int(ep["perturb_step"])
        hit = [t for t, b, ei in steps(ep, every, ph0, tmin)
               if t >= p and ei > thr.get(b, np.inf)]
        ok.append(bool(hit) and (min(hit) - p) <= win)
    return float(np.mean(ok))

def row(lab, thr, every=1, ph0=0, tmin=0):
    print(f"{lab:<24}{fp_ep(thr,every,ph0,tmin):>8.2f}"
          f"{detect(thr,5,every,ph0,tmin):>8.2f}{detect(thr,10,every,ph0,tmin):>8.2f}")

base = {b: mu[b] + 3.5 * sd[b] for b in mu}
HDR = f"{'方案':<24}{'FP_ep':>8}{'≤5':>8}{'≤10':>8}"

print("== 检测器起始时刻 ==\n" + HDR)
for tmin in (0, 16, 18, 20, 22):
    row(f"首个 z 不早于 t={tmin}", base, tmin=tmin)

print("\n== 仅放松桶3 (对照) ==\n" + HDR)
for k3 in (3.5, 2.5, 2.0):
    row(f"仅桶3 κ={k3}", {b: mu[b] + (k3 if b == 3 else 3.5) * sd[b] for b in mu})