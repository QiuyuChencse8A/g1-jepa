import json, numpy as np

d = json.load(open("results/closed_loop_s5000_k3.5_c1.json"))

hdr = (f"{'cond':<11}{'SR_p':>6}{'SR_c':>6}{'med':>6}{'P75':>6}{'P90':>6}{'max':>6}"
       f"{'mean':>7}{'≤5':>7}{'≤10':>7}{'FP_ep':>7}{'FP_cnt':>8}{'steps':>7}")
print(hdr); print("-" * len(hdr))

for cond in ["no_replan", "fixed", "oracle", "jepa"]:
    rp, rc = d[cond]["perturbed"], d[cond]["clean"]
    fired = [r for r in rp if r["perturbed"]]
    lats = [r["trigger_latency"] for r in fired if r["trigger_latency"] is not None]
    n = len(fired)
    succ = [r for r in fired if r["success"]]
    f = lambda v: f"{v:>6.0f}" if lats else "     —"
    print(f"{cond:<11}{np.mean([r['success'] for r in fired]):>6.2f}"
          f"{np.mean([r['success'] for r in rc]):>6.2f}"
          f"{f(np.median(lats)) if lats else '     —'}"
          f"{f(np.percentile(lats,75)) if lats else '     —'}"
          f"{f(np.percentile(lats,90)) if lats else '     —'}"
          f"{f(max(lats)) if lats else '     —'}"
          f"{np.mean(lats):>7.1f}" if lats else f"{'—':>7}",
          end="")
    print(f"{sum(l<=5 for l in lats)/n:>7.2f}{sum(l<=10 for l in lats)/n:>7.2f}"
          f"{np.mean([r['n_replans']>0 for r in rc]):>7.2f}"
          f"{np.mean([r['n_replans'] for r in rc]):>8.2f}"
          f"{np.mean([r['steps'] for r in succ]) if succ else float('nan'):>7.1f}"
          if lats else "")

# JEPA 长尾逐条
print("\n=== JEPA 延迟 > 10 步的 episode ===")
for r in d["jepa"]["perturbed"]:
    L = r["trigger_latency"]
    if r["perturbed"] and (L is None or L > 10):
        z = [v for _, v in r["z_log"]]
        p = r["perturb_step"]
        peak = max([v for t, v in r["z_log"] if p <= t <= p + 8], default=None)
        print(f"  seed={r['seed']} latency={L} success={r['success']} "
              f"扰动后8步内z峰={peak:.2f}" if peak else f"  seed={r['seed']} latency={L}")

# 失败 episode 是否都来自长尾
fails = [r for r in d["jepa"]["perturbed"] if r["perturbed"] and not r["success"]]
print(f"\n失败 {len(fails)} 条，其 latency: {[r['trigger_latency'] for r in fails]}")