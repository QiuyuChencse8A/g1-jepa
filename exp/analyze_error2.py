"""
analyze_error2.py —— Day 7 修订版

相对 v1 的四个改动:

  1. 预热屏蔽 (--warmup)
     前 frames*stride 步的 clip 是用重复首帧填充的, 会产生虚假运动。
     这段既不参与标定, 也不参与检测。

  2. 连续触发去抖 (--consec)
     要求连续 m 个时刻都越过阈值才算触发。
     每时刻误报率 p -> episode 级误报约 p^m, 延迟只增加 (m-1) 个编码步。
     这是压误报性价比最高的手段。

  3. 更细的相位分桶
     除了策略状态, 再按 phase_step 分档 —— DESCEND 开头和中段的误差水平不同。

  4. 相对误差度量 (--metric rel)
     e_rel = ||ẑ - z|| / ||z_t - z_{t-k}||
     "变化中有多大比例没被预测到", 天然对运动快慢免疫。

用法:
    python analyze_error2.py --sweep
    python analyze_error2.py --feature mean --k 1 --metric rel --consec 2
"""

import os
import glob
import argparse

import numpy as np

PHASES = ["APPROACH", "DESCEND", "GRASP", "LIFT", "DONE"]


def load_split(d):
    return [dict(np.load(p, allow_pickle=True))
            for p in sorted(glob.glob(os.path.join(d, "ep_*.npz")))]


def feat(ep, name):
    f = ep[name].astype(np.float32)
    return f.reshape(len(f), -1)


def errors(f, k=1, metric="abs"):
    """ẑ[i] = 2*f[i-k] - f[i-2k];  返回误差数组 (前 2k 个为 nan)。"""
    n = len(f)
    e = np.full(n, np.nan)
    for i in range(2 * k, n):
        pred = 2 * f[i - k] - f[i - 2 * k]
        resid = np.linalg.norm(pred - f[i])
        if metric == "rel":
            motion = np.linalg.norm(f[i] - f[i - k]) + 1e-6
            e[i] = resid / motion
        else:
            e[i] = resid
    return e


def bucket(phase, phase_step, n_sub=3, sub_size=6):
    """(相位, phase_step 档) -> 桶 id"""
    sub = min(int(phase_step) // sub_size, n_sub - 1)
    return int(phase) * n_sub + sub


def valid_mask(ep, warmup):
    return ep["t"] >= warmup


def calibrate(clean_eps, name, k, metric, warmup, fine=True):
    buckets = {}
    allv = []
    for ep in clean_eps:
        e = errors(feat(ep, name), k, metric)
        m = valid_mask(ep, warmup)
        for ei, ph, pst, ok in zip(e, ep["phase"], ep["phase_step"], m):
            if np.isnan(ei) or not ok:
                continue
            b = bucket(ph, pst) if fine else int(ph) * 3
            buckets.setdefault(b, []).append(ei)
            allv.append(ei)
    allv = np.array(allv)
    gmu, gsd = float(allv.mean()), float(allv.std() + 1e-9)
    mu, sd = {}, {}
    for b, v in buckets.items():
        if len(v) >= 15:
            mu[b], sd[b] = float(np.mean(v)), float(np.std(v) + 1e-9)
        else:
            mu[b], sd[b] = gmu, gsd
    return mu, sd, gmu, gsd


def zscores(ep, name, k, metric, mu, sd, gmu, gsd, warmup, fine=True):
    e = errors(feat(ep, name), k, metric)
    m = valid_mask(ep, warmup)
    z = np.full(len(e), np.nan)
    for i, (ei, ph, pst, ok) in enumerate(
            zip(e, ep["phase"], ep["phase_step"], m)):
        if np.isnan(ei) or not ok:
            continue
        b = bucket(ph, pst) if fine else int(ph) * 3
        z[i] = (ei - mu.get(b, gmu)) / sd.get(b, gsd)
    return z


def first_run(z, kappa, consec, start_idx=0):
    """返回连续 consec 个 > kappa 的那一段的最后一个下标 (触发时刻)。"""
    run = 0
    for i in range(start_idx, len(z)):
        if not np.isnan(z[i]) and z[i] > kappa:
            run += 1
            if run >= consec:
                return i
        else:
            run = 0
    return None


def evaluate(pert, clean, name, k, metric, kappas, consec, warmup, fine=True):
    mu, sd, gmu, gsd = calibrate(clean, name, k, metric, warmup, fine)

    fp = {kp: 0 for kp in kappas}
    for ep in clean:
        z = zscores(ep, name, k, metric, mu, sd, gmu, gsd, warmup, fine)
        for kp in kappas:
            if first_run(z, kp, consec) is not None:
                fp[kp] += 1

    det = {kp: [] for kp in kappas}
    for ep in pert:
        ts, ps = ep["t"], int(ep["perturb_step"])
        z = zscores(ep, name, k, metric, mu, sd, gmu, gsd, warmup, fine)
        after = int(np.searchsorted(ts, ps))
        for kp in kappas:
            hit = first_run(z, kp, consec, start_idx=after)
            det[kp].append(ts[hit] - ps if hit is not None else np.nan)

    rows = []
    for kp in kappas:
        lat = np.array(det[kp], dtype=float)
        ok = (~np.isnan(lat)) & (lat <= 10)   # 10 步之后触发, 任务已经失败了
        rows.append({
            "kappa": kp,
            "detect": float(ok.mean()),
            "fp": fp[kp] / max(1, len(clean)),
            "lat": float(np.median(lat[ok])) if ok.any() else np.nan,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="latents")
    ap.add_argument("--tag", default="f16s1_agentview")
    ap.add_argument("--feature", default="mean")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--metric", default="abs", choices=["abs", "rel"])
    ap.add_argument("--consec", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--coarse", action="store_true", help="用粗相位分桶")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    pert = load_split(os.path.join(args.root, f"perturbed_{args.tag}"))
    clean = load_split(os.path.join(args.root, f"clean_{args.tag}"))
    print(f"perturbed {len(pert)} 条, clean {len(clean)} 条, "
          f"warmup={args.warmup}, tag={args.tag}\n")

    kappas = [1.5, 2, 2.5, 3, 4, 5, 6]
    fine = not args.coarse

    if args.sweep:
        print(f"{'特征':<9}{'度量':>5}{'k':>3}{'连续':>5}{'κ':>6}"
              f"{'检出':>8}{'误报':>8}{'延迟':>8}")
        print("-" * 52)
        best = []
        for name in ["mean", "temporal", "grid"]:
            for metric in ["abs", "rel"]:
                for k in [1, 2]:
                    for consec in [1, 2, 3]:
                        rows = evaluate(pert, clean, name, k, metric,
                                        kappas, consec, args.warmup, fine)
                        for r in rows:
                            if r["detect"] >= 0.75 and r["fp"] <= 0.15:
                                print(f"{name:<9}{metric:>5}{k:>3}{consec:>5}"
                                      f"{r['kappa']:>6.1f}{r['detect']:>8.2f}"
                                      f"{r['fp']:>8.2f}{r['lat']:>8.1f}")
                                best.append((r["fp"] - r["detect"] + r["lat"] / 50,
                                             name, metric, k, consec, r))
        print()
        if best:
            best.sort(key=lambda x: x[0])
            s = best[0]
            r = s[5]
            print(f"综合最优: 特征={s[1]} 度量={s[2]} k={s[3]} 连续={s[4]} "
                  f"κ={r['kappa']}")
            print(f"          检出={r['detect']:.2f} 误报={r['fp']:.2f} "
                  f"延迟={r['lat']:.1f} 步")
        else:
            print("仍无组合满足 检出>=0.75 且 误报<=0.15")
            print("下一步: 试 --tag f8s1_agentview (更短窗口), 或上 Level-1 predictor")
        return

    rows = evaluate(pert, clean, args.feature, args.k, args.metric,
                    kappas, args.consec, args.warmup, fine)
    print(f"特征={args.feature} 度量={args.metric} k={args.k} "
          f"连续={args.consec} 细分桶={fine}")
    print(f"{'κ':>6}{'检出':>8}{'误报':>8}{'延迟':>8}")
    for r in rows:
        print(f"{r['kappa']:>6.1f}{r['detect']:>8.2f}{r['fp']:>8.2f}{r['lat']:>8.1f}")


if __name__ == "__main__":
    main()
