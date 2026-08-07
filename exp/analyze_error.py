"""
analyze_error.py —— Day 7 第二步: 误差曲线 + 检测 ROC

这是整个项目的胜负手。它回答一个问题:
    latent 预测误差在扰动后是否出现可分离的尖峰?

方法 (Level-0, 零训练):
    线性外推   ẑ[i] = f[i-k] + (f[i-k] - f[i-2k])
    误差       e[i] = ||ẑ[i] - f[i]||

    latent 沿平滑轨迹移动时外推很准, e 很小;
    扰动打断轨迹时 e 跳起来。不需要任何训练。

相位归一化 (关键):
    e 在快速运动、夹爪闭合时天然偏高。用固定阈值会让误报集中在这些时刻。
    所以用 clean 数据算出每个策略状态的 μ(φ)、σ(φ), 再判断
        z[i] = (e[i] - μ(φ)) / σ(φ) > κ
    φ 用策略状态机的状态, 在线时就能拿到。

用法:
    python analyze_error.py --feature grid --k 2
    python analyze_error.py --sweep          # 扫所有特征和 k
"""

import os
import glob
import argparse

import numpy as np

PHASES = ["APPROACH", "DESCEND", "GRASP", "LIFT", "DONE"]


def load_split(d):
    eps = []
    for p in sorted(glob.glob(os.path.join(d, "ep_*.npz"))):
        eps.append(dict(np.load(p, allow_pickle=True)))
    return eps


def feat(ep, name):
    f = ep[name].astype(np.float32)
    return f.reshape(len(f), -1)


def extrap_error(f, k=2):
    """
    ẑ[i] = f[i-k] + (f[i-k] - f[i-2k]);  e[i] = ||ẑ[i] - f[i]||
    前 2k 个时刻无定义, 填 nan。
    """
    n = len(f)
    e = np.full(n, np.nan)
    for i in range(2 * k, n):
        pred = 2 * f[i - k] - f[i - 2 * k]
        e[i] = np.linalg.norm(pred - f[i])
    return e


def phase_stats(clean_eps, name, k):
    """从 clean 数据算每个相位的 μ, σ。"""
    buckets = {p: [] for p in range(len(PHASES))}
    for ep in clean_eps:
        e = extrap_error(feat(ep, name), k)
        for ei, ph in zip(e, ep["phase"]):
            if not np.isnan(ei):
                buckets[int(ph)].append(ei)
    mu, sd = {}, {}
    allv = np.concatenate([np.array(v) for v in buckets.values() if v])
    for p, v in buckets.items():
        if len(v) >= 20:
            mu[p], sd[p] = float(np.mean(v)), float(np.std(v) + 1e-6)
        else:
            mu[p], sd[p] = float(allv.mean()), float(allv.std() + 1e-6)
    return mu, sd


def zscore(e, phase, mu, sd):
    return np.array([(ei - mu[int(p)]) / sd[int(p)] if not np.isnan(ei) else np.nan
                     for ei, p in zip(e, phase)])


def evaluate(pert_eps, clean_eps, name, k, kappas):
    mu, sd = phase_stats(clean_eps, name, k)

    # 误报: clean episode 里是否有任何时刻越过阈值
    clean_max = []
    for ep in clean_eps:
        z = zscore(extrap_error(feat(ep, name), k), ep["phase"], mu, sd)
        clean_max.append(np.nanmax(z) if np.any(~np.isnan(z)) else -np.inf)
    clean_max = np.array(clean_max)

    # 检测: 扰动之后首次越过阈值的延迟 (以控制步计)
    det = {kp: [] for kp in kappas}
    for ep in pert_eps:
        ts = ep["t"]
        ps = int(ep["perturb_step"])
        z = zscore(extrap_error(feat(ep, name), k), ep["phase"], mu, sd)
        after = np.where(ts >= ps)[0]
        for kp in kappas:
            hit = [i for i in after if not np.isnan(z[i]) and z[i] > kp]
            det[kp].append(ts[hit[0]] - ps if hit else np.nan)

    rows = []
    for kp in kappas:
        lat = np.array(det[kp], dtype=float)
        ok = (~np.isnan(lat)) & (lat <= 10)   # 10 步之后触发, 任务已经失败了
        rows.append({
            "kappa": kp,
            "detect_rate": float(np.mean(~np.isnan(lat))),
            "fp_rate": float(np.mean(clean_max > kp)),
            "median_latency": float(np.nanmedian(lat)) if np.any(~np.isnan(lat)) else np.nan,
        })
    return rows, mu, sd


def curve(pert_eps, clean_eps, name, k, window=(-20, 60)):
    """误差曲线, 以扰动时刻对齐。"""
    mu, sd = phase_stats(clean_eps, name, k)
    lo, hi = window
    acc = {d: [] for d in range(lo, hi + 1)}
    for ep in pert_eps:
        ts, ps = ep["t"], int(ep["perturb_step"])
        z = zscore(extrap_error(feat(ep, name), k), ep["phase"], mu, sd)
        for t, zi in zip(ts, z):
            d = int(t - ps)
            if lo <= d <= hi and not np.isnan(zi):
                acc[d].append(zi)
    ds = sorted(d for d in acc if acc[d])
    return np.array(ds), np.array([np.mean(acc[d]) for d in ds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="latents")
    ap.add_argument("--tag", default="f16s1_agentview")
    ap.add_argument("--feature", default="grid",
                    choices=["mean", "temporal", "grid"])
    ap.add_argument("--k", type=int, default=2, help="外推跨度 (特征索引单位)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--plot", default="error_curve.png")
    args = ap.parse_args()

    pert = load_split(os.path.join(args.root, f"perturbed_{args.tag}"))
    clean = load_split(os.path.join(args.root, f"clean_{args.tag}"))
    print(f"perturbed {len(pert)} 条, clean {len(clean)} 条\n")

    kappas = [1, 2, 3, 4, 6, 8]

    if args.sweep:
        print(f"{'特征':<10}{'k':>3}{'κ':>4}{'检出率':>9}{'误报率':>9}{'中位延迟':>10}")
        print("-" * 48)
        best = None
        for name in ["mean", "temporal", "grid"]:
            for k in [1, 2, 4]:
                rows, _, _ = evaluate(pert, clean, name, k, kappas)
                for r in rows:
                    print(f"{name:<10}{k:>3}{r['kappa']:>4}"
                          f"{r['detect_rate']:>9.2f}{r['fp_rate']:>9.2f}"
                          f"{r['median_latency']:>10.1f}")
                    # 选标准: 误报<=0.1 前提下, 延迟最小
                    if r["fp_rate"] <= 0.10 and r["detect_rate"] >= 0.8:
                        cand = (r["median_latency"], name, k, r["kappa"])
                        if best is None or cand[0] < best[0]:
                            best = cand
        print()
        if best:
            print(f"最佳组合: 特征={best[1]} k={best[2]} κ={best[3]} "
                  f"中位延迟={best[0]:.1f} 步")
        else:
            print("没有组合能在误报<=0.1 下达到检出率>=0.8 —— 见下方诊断")
        return

    rows, mu, sd = evaluate(pert, clean, args.feature, args.k, kappas)
    print(f"特征={args.feature} k={args.k}")
    print(f"{'κ':>4}{'检出率':>9}{'误报率':>9}{'中位延迟':>10}")
    for r in rows:
        print(f"{r['kappa']:>4}{r['detect_rate']:>9.2f}"
              f"{r['fp_rate']:>9.2f}{r['median_latency']:>10.1f}")

    print(f"\n各相位基线 (clean):")
    for p in sorted(mu):
        print(f"  {PHASES[p]:<10} μ={mu[p]:>9.1f}  σ={sd[p]:>8.1f}")

    ds, zs = curve(pert, clean, args.feature, args.k)
    print(f"\n扰动对齐的平均 z 曲线 (负数=扰动前):")
    for d, z in zip(ds, zs):
        if -10 <= d <= 40 and d % 2 == 0:
            bar = "#" * max(0, min(50, int(z * 5)))
            print(f"  t{d:+4d}  z={z:>7.2f}  {bar}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(ds, zs, lw=2)
        plt.axvline(0, color="r", ls="--", label="perturbation")
        plt.axhline(3, color="g", ls=":", label="kappa=3")
        plt.xlabel("steps from perturbation")
        plt.ylabel("phase-normalized error (z)")
        plt.title(f"{args.feature}, k={args.k}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot, dpi=120)
        print(f"\n已保存 {args.plot}")
    except ImportError:
        print("\n(matplotlib 未安装, 跳过绘图: pip install matplotlib)")


if __name__ == "__main__":
    main()
