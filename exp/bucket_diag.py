#!/usr/bin/env python3
"""桶内 clean vs perturbed 原始误差可分性——不做归一化。"""
import json, numpy as np
from analyze_error2 import bucket, errors, feat, load_split, valid_mask

NAME, K, METRIC, WARMUP = "grid", 1, "abs", 16
ROOT, TAG, WIN = "latents", "f16s1_wrist", 8

def per_step(ep):
    e = errors(feat(ep, NAME), K, METRIC)
    m = valid_mask(ep, WARMUP)
    return [(int(t), int(bucket(ph, pst)), float(ei))
            for t, ei, ph, pst, ok in zip(ep["t"], e, ep["phase"], ep["phase_step"], m)
            if ok and not np.isnan(ei)]

def pstep(ep):
    for key in ("perturb_step", "fire_step", "perturb_t", "p_step"):
        if key in ep:
            v = ep[key]
            return int(v if np.ndim(v) == 0 else v[0])
    return None

def q(v):
    v = np.asarray(v)
    return (f"n={len(v):5d}  mean={v.mean():7.1f}  sd={v.std():6.2f}  "
            f"P10={np.percentile(v,10):6.1f}  P50={np.percentile(v,50):6.1f}  "
            f"P90={np.percentile(v,90):6.1f}  P95={np.percentile(v,95):6.1f}")

def auc(neg, pos):                       # P(pos > neg)，0.5 = 完全不可分
    neg = np.sort(np.asarray(neg)); pos = np.asarray(pos)
    lo = np.searchsorted(neg, pos, "left"); hi = np.searchsorted(neg, pos, "right")
    return float(np.mean((lo + hi) / 2) / len(neg))

clean = load_split(f"{ROOT}/clean_{TAG}")
pert  = load_split(f"{ROOT}/perturbed_{TAG}")
print("perturbed episode 的键:", sorted(pert[0].keys()), "\n")

calib, holdout = clean[:100], clean[100:200]
cl_by_b, ho_by_b = {}, {}
for eps, tgt in ((calib, cl_by_b), (holdout, ho_by_b)):
    for ep in eps:
        for _, b, ei in per_step(ep):
            tgt.setdefault(b, []).append(ei)

pt_by_b, skipped = {}, 0
for ep in pert:
    p = pstep(ep)
    if p is None:
        skipped += 1; continue
    win = [(b, ei) for t, b, ei in per_step(ep) if p <= t <= p + WIN]
    if win:
        b, ei = max(win, key=lambda x: x[1])
        pt_by_b.setdefault(b, []).append(ei)
if skipped:
    print(f"警告: {skipped} 条 perturbed episode 找不到扰动时刻\n")

for b in sorted(set(cl_by_b) | set(pt_by_b)):
    print(f"===== bucket {b} =====")
    if b in cl_by_b:  print(f"  clean(标定)  {q(cl_by_b[b])}")
    if b in ho_by_b:  print(f"  clean(留出)  {q(ho_by_b[b])}")
    if b in pt_by_b:  print(f"  perturbed    {q(pt_by_b[b])}")
    if b in cl_by_b and b in pt_by_b:
        thr = np.percentile(ho_by_b.get(b, cl_by_b[b]), 95)
        det = np.mean(np.asarray(pt_by_b[b]) > thr)
        print(f"  >>> AUC={auc(cl_by_b[b], pt_by_b[b]):.3f}   "
              f"留出FP=0.05 的原始误差阈值={thr:.1f} → 检出={det:.2f}")
    print()

for b in (3, 4):                          # 看 clean 是否多峰
    if b not in cl_by_b: continue
    v = np.asarray(cl_by_b[b])
    cnt, edge = np.histogram(v, bins=12)
    print(f"bucket {b} clean 直方图:")
    for c, lo in zip(cnt, edge):
        print(f"  {lo:7.1f} | {'#' * int(40 * c / max(cnt))} {c}")
    print()