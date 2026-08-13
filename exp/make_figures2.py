#!/usr/bin/env python3
"""
make_figures2.py — diagnostic figures for the failure-mode analysis.

Usage (on the server, in exp/):
    python make_figures2.py --runs results/final_run1.json results/final_run2.json \
                            results/final_run3.json results/final_run4.json \
                            results/final_run5.json \
                            --calib calib_f16s1_wrist_grid_k1.json --outdir figs

Reads the per-episode dbg_log written by jepa_trigger:
    (t, bucket, fallback, raw_error, mu, sd, z)

Figure A  bucket3_vs_bucket4  — the normalization failure mode, in one picture
Figure B  bucket4_timing      — the geometry effect with normalization held fixed
Figure C  latency_bimodal     — the two-population latency distribution

Labels are English on purpose: the server may not have a CJK font and
matplotlib renders missing glyphs as tofu boxes.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight",
})
C_OK, C_BAD, C_GREY = "#2166ac", "#c0392b", "#9e9e9e"
KAPPA = 3.5
TWO_COL = 7.0


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.pdf / {name}.png")


# ------------------------------------------------------------------ loading
def load(run_paths):
    """One record per perturbed episode: the peak-z step within 8 steps of the
    perturbation, which is the step the detector had its best chance on."""
    recs = []
    for ri, path in enumerate(run_paths):
        d = json.load(open(path))
        for r in d["jepa"]["perturbed"]:
            if not r.get("perturbed") or not r.get("dbg_log"):
                continue
            p = r["perturb_step"]
            win = [x for x in r["dbg_log"] if p <= x[0] <= p + 8]
            if not win:
                continue
            t, b, fb, err, mu, sd, z = max(win, key=lambda x: x[6])
            L = r["trigger_latency"]
            recs.append(dict(run=ri, seed=r["seed"], perturb_step=p, bucket=int(b),
                             fallback=bool(fb), err=float(err), mu=float(mu),
                             sd=float(sd), z=float(z),
                             detected=(L is not None and L <= 10),
                             latency=L, success=bool(r["success"])))
    return recs


def all_latencies(run_paths):
    out = []
    for path in run_paths:
        d = json.load(open(path))
        for r in d["jepa"]["perturbed"]:
            if r.get("perturbed"):
                out.append(r["trigger_latency"])
    return out


# ============================================================== FIGURE A
def fig_bucket_contrast(recs, calib, outdir):
    """Left: raw latent error. Right: the same episodes after normalization.

    The point of the figure is that the left panel shows no separation for the
    bucket-3 failures while the right panel pushes them under the threshold.
    """
    groups = [(3, False, "bucket 3\nmissed"), (3, True, "bucket 3\ndetected"),
              (4, False, "bucket 4\nmissed"), (4, True, "bucket 4\ndetected")]
    rng = np.random.default_rng(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TWO_COL, 3.3), gridspec_kw={"wspace": .28})

    # the overlap band is the argument: bucket-3 misses sit at raw errors that
    # bucket 4 detects without trouble
    b3m = [r["err"] for r in recs if r["bucket"] == 3 and not r["detected"]]
    if b3m:
        ax1.axhspan(min(b3m), max(b3m), color="#fdf0d5", zorder=0)
        ax1.annotate("raw-error range of the\nbucket-3 misses",
                     (3.5, (min(b3m) + max(b3m)) / 2), ha="right", va="center",
                     fontsize=7.5, color="#8a6d1f", zorder=5)

    for xi, (b, det, lab) in enumerate(groups):
        g = [r for r in recs if r["bucket"] == b and r["detected"] == det]
        if not g:
            continue
        jitter = rng.uniform(-0.16, 0.16, len(g))
        col = C_OK if det else C_BAD
        ax1.scatter(xi + jitter, [r["err"] for r in g], s=11, alpha=.55,
                    color=col, linewidths=0, zorder=3)
        ax2.scatter(xi + jitter, [r["z"] for r in g], s=11, alpha=.55,
                    color=col, linewidths=0, zorder=3)

    # per-bucket mu and trigger threshold in RAW units — this is the whole story
    for b, xs in ((3, (0, 1)), (4, (2, 3))):
        if str(b) not in calib.get("buckets", {}):
            continue
        mu, sd = calib["buckets"][str(b)]
        thr = mu + KAPPA * sd
        lo, hi = min(xs) - .42, max(xs) + .42
        ax1.plot([lo, hi], [mu, mu], color=C_GREY, lw=1.1, zorder=2)
        ax1.plot([lo, hi], [thr, thr], color=C_BAD, lw=1.3, ls="--", zorder=2)
        ax1.annotate(f"trigger @ {thr:.0f}", (lo, thr), xytext=(1, 3),
                     textcoords="offset points", fontsize=7.5, color=C_BAD)
        ax1.annotate(f"$\\mu$={mu:.0f}", (lo, mu), xytext=(1, -9),
                     textcoords="offset points", fontsize=7.5, color="#666")

    ax2.axhline(KAPPA, color=C_BAD, lw=1.3, ls="--", zorder=2)
    ax2.annotate(f"$\\kappa$={KAPPA}", (-0.5, KAPPA), xytext=(1, 4),
                 textcoords="offset points", fontsize=7.5, color=C_BAD)

    for ax, ylab, ttl in ((ax1, "raw latent error", "before normalization"),
                          (ax2, "z-score", "after normalization")):
        ax.set_xticks(range(4), [g[2] for g in groups], fontsize=7.5)
        ax.set_xlim(-.55, 3.55)
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=9, color="#444")
        ax.axvline(1.5, color="#ddd", lw=.9, zorder=1)
        ax.margins(y=.14)
    save(fig, outdir, "figA_bucket_contrast")


# ============================================================== FIGURE B
def fig_bucket4_timing(recs, calib, outdir):
    """Within bucket 4 only: raw error against perturbation timing.

    Same bucket means the same mu and sd, so any trend here is a property of
    the signal itself rather than of the normalization.
    """
    g = [r for r in recs if r["bucket"] == 4]
    if not g:
        print("  [figB] no bucket-4 records"); return
    x = np.array([r["perturb_step"] for r in g], float)
    y = np.array([r["err"] for r in g], float)
    det = np.array([r["detected"] for r in g])

    fig, ax = plt.subplots(figsize=(TWO_COL * .62, 3.0))
    rng = np.random.default_rng(1)
    jx = x + rng.uniform(-.22, .22, len(x))
    ax.scatter(jx[det], y[det], s=13, alpha=.55, color=C_OK,
               linewidths=0, label="detected $\\leq$10 steps", zorder=3)
    ax.scatter(jx[~det], y[~det], s=22, alpha=.85, color=C_BAD, marker="x",
               linewidths=1.1, label="missed", zorder=4)

    if len(set(x)) > 2:                       # least-squares trend
        k, c = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, k * xs + c, color="#444", lw=1.1, ls="-", zorder=2,
                label=f"trend ({k:+.1f} / step)")

    if "4" in calib.get("buckets", {}):
        mu, sd = calib["buckets"]["4"]
        ax.axhline(mu + KAPPA * sd, color=C_BAD, lw=1.2, ls="--", zorder=2)
        ax.annotate(f"trigger threshold ({mu + KAPPA*sd:.0f})",
                    (x.max(), mu + KAPPA * sd), xytext=(0, 4),
                    textcoords="offset points", ha="right", fontsize=7.5, color=C_BAD)
        ax.axhline(mu, color=C_GREY, lw=1.0, zorder=2)
        ax.annotate(f"bucket mean ({mu:.0f})", (x.min(), mu), xytext=(0, 3),
                    textcoords="offset points", fontsize=7.5, color="#666")
        ax.margins(y=.10)

    ax.set_xlabel("perturbation timestep")
    ax.set_ylabel("raw latent error")
    ax.set_title("bucket 4 only — normalization held constant", fontsize=9, color="#444")
    ax.legend(frameon=False, loc="upper left")
    save(fig, outdir, "figB_bucket4_timing")


# ============================================================== FIGURE C
def fig_latency_bimodal(lats, outdir):
    """Histogram with a broken axis: the gap between the two populations is the
    finding, so the plot has to make the empty region visible."""
    fast = [l for l in lats if l is not None and l <= 30]
    slow = [l for l in lats if l is not None and l > 30]
    never = sum(1 for l in lats if l is None)
    n = len(lats)

    fig, (a, b) = plt.subplots(1, 2, figsize=(TWO_COL, 2.6),
                               gridspec_kw={"width_ratios": [2, 1.15], "wspace": .08})
    a.hist(fast, bins=np.arange(0, 31, 1), color=C_OK)
    a.set_xlim(0, 30)
    a.set_ylabel(f"episodes (100 scenarios × {len(...)} runs)")
    a.set_title("fast population", fontsize=9, color="#444")

    GAP_LO, GAP_HI = 6, 30
    in_gap = [l for l in lats if l is not None and GAP_LO <= l <= GAP_HI]
    a.axvspan(GAP_LO - .5, GAP_HI, color="#f2f2f2", zorder=0)
    a.annotate(f"only {len(in_gap)} of {n} episodes ({len(in_gap)/n:.1%})\n"
               f"land between {GAP_LO} and {GAP_HI} steps",
               ((GAP_LO + GAP_HI) / 2, a.get_ylim()[1] * .55),
               ha="center", fontsize=8, color="#888")

    edges = np.arange(30, max(slow + [300]) + 20, 20)
    b.hist(slow, bins=edges, color=C_BAD)
    if never:
        b.bar([edges[-1] + 25], [never], width=18, color="#7b241c")
        b.annotate(f"never\ntriggered\n({never})", (edges[-1] + 25, never),
                   xytext=(0, 4), textcoords="offset points",
                   ha="center", fontsize=7.5, color="#7b241c")
    b.set_title("tail", fontsize=9, color="#444")
    b.spines["left"].set_visible(False)
    b.tick_params(left=False, labelleft=False)
    b.set_ylim(a.get_ylim())
    fig.supxlabel("trigger latency (steps)  —  note the different scales",
                  fontsize=9, y=-0.02)
    save(fig, outdir, "figC_latency_bimodal")


# ================================================================== main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--calib", default="calib_f16s1_wrist_grid_k1.json")
    ap.add_argument("--outdir", default="figs")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    calib = json.load(open(a.calib))
    recs = load(a.runs)
    print(f"loaded {len(recs)} perturbed episodes from {len(a.runs)} run(s)")
    nfb = sum(r["fallback"] for r in recs)
    if nfb:
        print(f"  note: {nfb} episodes fell back to the global bucket")

    fig_bucket_contrast(recs, calib, a.outdir)
    fig_bucket4_timing(recs, calib, a.outdir)
    fig_latency_bimodal(all_latencies(a.runs), a.outdir)
    print(f"\ndone -> {os.path.abspath(a.outdir)}")
