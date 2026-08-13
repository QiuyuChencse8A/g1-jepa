#!/usr/bin/env python3
"""
make_figures.py — generate report figures for the JEPA-triggered replanning project.

Usage:
    python make_figures.py --outdir figs

All labels are English on purpose: the server may not have a CJK font installed,
and matplotlib silently renders missing glyphs as tofu boxes.

Each figure has a REAL_DATA hook at the top of its function. Where you already
have the numbers, they are hard-coded below. Where the figure needs a per-step
log you have not dumped yet, the function falls back to DEMO data so you can see
the layout immediately — replace the loader and the figure is done.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless server: no display
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- global style
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})
COL = {"no_replan": "#9e9e9e", "fixed": "#e08214", "oracle": "#4d4d4d", "jepa": "#2166ac"}
ONE_COL = 3.5    # inches, single-column width
TWO_COL = 7.0

def save(fig, outdir, name):
    for ext in ("pdf", "png"):           # pdf for papers, png for Word
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.pdf / {name}.png")


# =============================================================== FIGURE 1
def fig_z_trace(outdir, log_path=None):
    """Single-episode z-score trace. The most persuasive single figure you have.

    REAL_DATA: dump this from the closed-loop runner, one row per step:
        np.savez(f"logs/closed_loop/ep{seed}.npz",
                 z=z_per_step, triggered=trigger_flags,
                 perturb_step=perturb_t, phase=phase_idx)
    """
    KAPPA, REARM = 2.5, 1.0
    if log_path and os.path.exists(log_path):
        d = np.load(log_path)
        z, perturb_t = d["z"], int(d["perturb_step"])
        trig = np.flatnonzero(d["triggered"])
        trig_t = int(trig[0]) if len(trig) else None
    else:
        # DEMO reproducing the smoke-test episode described in the notes
        rng = np.random.default_rng(0)
        T, perturb_t, trig_t = 90, 23, 26
        z = rng.normal(-0.8, 0.35, T)
        z[24], z[25], z[26] = 6.29, 4.10, 7.28
        z[40], z[56] = 2.04, 2.95          # spikes suppressed by debounce/hysteresis
        print("  [fig1] using DEMO trace — pass --zlog to use a real episode")

    t = np.arange(len(z))
    fig, ax = plt.subplots(figsize=(TWO_COL, 2.4))
    ax.plot(t, z, lw=1.2, color=COL["jepa"], zorder=3)
    ax.axhline(KAPPA, ls="--", lw=1, color="#c0392b", zorder=2)
    ax.axhline(REARM, ls=":", lw=1, color="#7f7f7f", zorder=2)
    ax.text(len(z) * 0.985, KAPPA, r"$\kappa=2.5$", color="#c0392b",
            va="bottom", ha="right", fontsize=8)
    ax.text(len(z) * 0.985, REARM, "re-arm", color="#7f7f7f",
            va="bottom", ha="right", fontsize=8)

    ax.axvline(perturb_t, color="#333", lw=1)
    ax.annotate("perturbation", (perturb_t, z.max() * 0.98),
                xytext=(perturb_t - 16, z.max() * 0.98), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    if trig_t is not None:
        ax.plot(trig_t, z[trig_t], "v", ms=8, color="#c0392b", zorder=4)
        ax.annotate(f"trigger (latency {trig_t - perturb_t} steps)",
                    (trig_t, z[trig_t]), xytext=(trig_t + 6, z[trig_t] * 0.92),
                    fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    # call out the suppressed spikes — this is what sells the debounce
    for s in (40, 56):
        if s < len(z) and z[s] > REARM:
            ax.annotate("suppressed\n(debounce)", (s, z[s]), xytext=(s - 3, z[s] + 1.6),
                        fontsize=7, color="#7f7f7f", ha="center",
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="#aaa"))
    ax.set_xlabel("timestep"); ax.set_ylabel("z-score of latent error")
    ax.set_xlim(0, len(z) - 1)
    save(fig, outdir, "fig1_z_trace")


# =============================================================== FIGURE 2
def fig_latency(outdir, lat_path=None):
    """Latency histogram + CDF. Makes the bimodal story visible in one glance.

    REAL_DATA: one latency value per episode (np.nan for the 2 that never fired).
    """
    if lat_path and os.path.exists(lat_path):
        lat = np.asarray(json.load(open(lat_path)), dtype=float)
    else:
        # DEMO consistent with: 48/50 fired, median 3, P75 4, P90 34, max 253,
        # six episodes clustered at 34-39 (abort-retry detections)
        lat = np.array([2]*8 + [3]*19 + [4]*10 + [5]*2 + [8, 11] +
                       [34, 35, 36, 37, 38, 39] + [253] + [np.nan, np.nan], float)
        print("  [fig2] using DEMO latencies — pass --latencies to use real ones")

    fired = lat[~np.isnan(lat)]
    n_all, med, p90 = len(lat), np.median(fired), np.percentile(fired, 90)
    within5 = np.sum(fired <= 5) / n_all
    within10 = np.sum(fired <= 10) / n_all

    fig, (a, b) = plt.subplots(1, 2, figsize=(TWO_COL, 2.4))
    a.hist(np.clip(fired, 0, 60), bins=np.arange(0, 62, 2), color=COL["jepa"])
    a.axvline(med, color="#c0392b", lw=1.2, label=f"median = {med:.0f}")
    a.axvline(p90, color="#e08214", lw=1.2, ls="--", label=f"P90 = {p90:.0f}")
    a.set_xlabel("trigger latency (steps, clipped at 60)")
    a.set_ylabel("episodes"); a.legend(frameon=False)

    xs = np.sort(fired)
    a2 = b
    a2.step(np.r_[0, xs], np.r_[0, np.arange(1, len(xs) + 1) / n_all], where="post",
            color=COL["jepa"], lw=1.4)
    a2.axhline(1.0, color="#ddd", lw=0.8)
    for x, y, lab, dy in ((5, within5, f"{within5:.0%} within 5", -0.16),
                          (10, within10, f"{within10:.0%} within 10", -0.30)):
        a2.plot([x, x], [0, y], ls=":", lw=0.9, color="#7f7f7f")
        a2.plot(x, y, "o", ms=4, color="#c0392b")
        a2.annotate(lab, (x, y), xytext=(30, y + dy), fontsize=8,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#bbb"))
    a2.set_xscale("symlog"); a2.set_xlim(0, 300); a2.set_ylim(0, 1.05)
    a2.set_xlabel("trigger latency (steps, log scale)")
    a2.set_ylabel(f"fraction of all {n_all} episodes")
    save(fig, outdir, "fig2_latency")


# =============================================================== FIGURE 3
def fig_replan_cost(outdir):
    """Cost comparison. Success saturates at 1.00, so the message lives here."""
    # REAL_DATA — closed-loop table, n=50 per condition
    rows = [
        ("no replan", 0.00, np.nan, np.nan, np.nan),   # success only; never replans
        ("fixed",     1.00, 3.9, 2.0, 88.7),
        ("oracle",    1.00, 1.0, 0.0, 71.2),
        ("JEPA",      0.94, 0.9, 0.0, 99.7),
    ]
    names = [r[0] for r in rows]
    keys = ["no_replan", "fixed", "oracle", "jepa"]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 4, figsize=(TWO_COL, 2.4), constrained_layout=True)
    panels = [
        ("perturbed\nsuccess", [r[1] for r in rows], (0, 1.08), True),
        ("replans /\nperturbed ep.", [r[2] for r in rows], None, False),
        ("wasted replans /\nclean ep.", [r[3] for r in rows], None, False),
        ("completion\nsteps", [r[4] for r in rows], None, False),
    ]
    for ax, (title, vals, ylim, pct) in zip(axes, panels):
        vals = np.array(vals, float)
        ax.bar(x, np.nan_to_num(vals), color=[COL[k] for k in keys], width=0.68)
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.5)
        if ylim: ax.set_ylim(*ylim)
        for xi, v in zip(x, vals):
            if np.isnan(v):
                ax.text(xi, 0.02 * (ax.get_ylim()[1] or 1), "n/a", ha="center", fontsize=7, color="#888")
            else:
                ax.text(xi, v, f"{v:.2f}" if pct else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7.5)
    save(fig, outdir, "fig3_replan_cost")


# =============================================================== FIGURE 4
def fig_detection_fp(outdir, sweep_path=None):
    """Detection vs false-positive, swept over kappa. Justifies lowering kappa.

    REAL_DATA: your offline sweep already produces these tuples. Dump as
        json.dump([{"camera":..,"feature":..,"kappa":..,"detect":..,"fp":..}, ...])
    with FP measured on the HELD-OUT clean split (clean[100:200]).
    """
    if sweep_path:
        recs = []
        for path in sweep_path.split(","):
            for r in json.load(open(path.strip())):
                r["camera"] = "wrist" if "wrist" in r.get("tag", "") else "agentview"
                recs.append(r)
    else:
        print("  [fig4] using DEMO sweep — pass --sweep with your held-out results")
        recs = []
        for cam, feat, base in (("wrist", "grid", 0.94), ("wrist", "temporal", 0.88),
                                ("agentview", "temporal", 0.90), ("agentview", "mean", 0.80)):
            for k in np.arange(1.5, 5.01, 0.25):
                fp = float(np.clip(0.35 * np.exp(-1.15 * (k - 1.5)), 0, 1))
                det = float(np.clip(base * np.exp(-0.14 * max(0, k - 2.0)), 0, 1))
                recs.append(dict(camera=cam, feature=feat, kappa=float(k), detect=det, fp=fp))

    fig, ax = plt.subplots(figsize=(ONE_COL + 0.9, 3.0))
    style = {("wrist", "grid"): ("-", "o", COL["jepa"]),
             ("wrist", "temporal"): ("--", "s", "#6baed6"),
             ("agentview", "temporal"): ("-", "^", "#e08214"),
             ("agentview", "mean"): ("--", "v", "#fdae61")}
    for (cam, feat), (ls, mk, c) in style.items():
        pts = sorted([r for r in recs if r["camera"] == cam and r["feature"] == feat
                      and r["metric"] == "abs" and r["k"] == 1 and r["consec"] == 1],
                     key=lambda r: r["fp"])
        if not pts: continue
        ax.plot([p["fp"] for p in pts], [p["detect"] for p in pts], ls, marker=mk,
                ms=3, lw=1.2, color=c, label=f"{cam} / {feat}")
        if (cam, feat) == ("wrist", "grid"):
            for p in pts:
                if p["kappa"] in (3.0, 3.2, 3.5, 3.8, 4.0):
                    sel = " (selected)" if p["kappa"] == 3.5 else ""
                    ax.annotate(f"$\\kappa$={p['kappa']:.1f}{sel}",
                                (p["fp"], p["detect"]), xytext=(6, -3),
                                textcoords="offset points", fontsize=7,
                                color=c, fontweight="bold" if sel else "normal")
        if (cam, feat) == ("wrist", "grid"):
            for p in pts:
                if p["kappa"] in (3.0, 3.2, 3.5, 3.8, 4.0):
                    sel = " (selected)" if p["kappa"] == 3.5 else ""
                    ax.annotate(f"$\\kappa$={p['kappa']:.1f}{sel}",
                                (p["fp"], p["detect"]), xytext=(6, -3),
                                textcoords="offset points", fontsize=7,
                                color=c, fontweight="bold" if sel else "normal")
    ax.axvline(0.05, ls=":", lw=1, color="#c0392b")
    ax.text(0.052, 0.02, "FP = 0.05", fontsize=7.5, color="#c0392b")
    ax.set_xlabel("false-positive rate (held-out clean)")
    ax.set_ylabel("detection rate (within 10 steps)")
    ax.set_title("validation set (clean[100:200])", fontsize=8.5, color="#666")
    ax.set_title("validation set (clean[100:200])", fontsize=8.5, color="#666")
    ax.set_xlim(0, 0.4); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    save(fig, outdir, "fig4_detection_fp")


# =============================================================== FIGURE 5
def fig_cam_feature_heatmap(outdir, table_path=None):
    """camera x feature detection heatmap. Replaces two paragraphs of section 5."""
    features = ["mean", "temporal", "grid"]
    cameras = ["agentview", "wrist"]
    if table_path and os.path.exists(table_path):
        M = np.array(json.load(open(table_path)))
    else:
        print("  [fig5] fill in your held-out sweep best-per-cell numbers")
         M = np.array([[0.68, 0.83, 0.73],     # agentview: mean, temporal, grid
                  [0.4, 0.91, 0.94]])    # wrist:     mean, temporal, grid   # wrist
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.0))
    im = ax.imshow(M, cmap="Blues", vmin=0.6, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(features)), features)
    ax.set_yticks(range(len(cameras)), cameras)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] > 0.88 else "#222")
    ax.set_title("detection @ FP $\\leq$ 0.05 (validation)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)
    save(fig, outdir, "fig5_cam_feature")


# =============================================================== main
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="figs")
    p.add_argument("--zlog", default=None, help="npz of one closed-loop episode")
    p.add_argument("--latencies", default=None, help="json list of per-episode latencies")
    p.add_argument("--sweep", default=None, help="json of held-out sweep records")
    p.add_argument("--camtable", default=None, help="json 2x3 detection matrix")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    fig_z_trace(a.outdir, a.zlog)
    fig_latency(a.outdir, a.latencies)
    fig_replan_cost(a.outdir)
    fig_detection_fp(a.outdir, a.sweep)
    fig_cam_feature_heatmap(a.outdir, a.camtable)
    print(f"\ndone -> {os.path.abspath(a.outdir)}")
