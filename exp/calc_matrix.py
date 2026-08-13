"""从 sweep json 算 camera x feature 矩阵：FP<=0.05 约束下的最好 detection。"""
import json, sys

FP_MAX = 0.05
features = ["mean", "temporal", "grid"]
cameras = {"agentview": "results/sweep_f16s1_agentview.json",
           "wrist":     "results/sweep_f16s1_wrist.json"}

matrix, detail = [], []
for cam, path in cameras.items():
    recs = json.load(open(path))
    row = []
    for feat in features:
        cand = [r for r in recs if r["feature"] == feat and r["fp"] <= FP_MAX]
        if not cand:
            row.append(float("nan"))
            detail.append(f"{cam:10s} {feat:9s}  无配置满足 FP<={FP_MAX}")
            continue
        best = max(cand, key=lambda r: (r["detect"], -r["lat"]))
        row.append(round(best["detect"], 2))
        detail.append(f"{cam:10s} {feat:9s}  detect={best['detect']:.2f} "
                      f"fp={best['fp']:.2f} lat={best['lat']:.0f}  "
                      f"[{best['metric']} k={best['k']} consec={best['consec']} "
                      f"kappa={best['kappa']}]")
    matrix.append(row)

print(f"约束: FP <= {FP_MAX}（validation set）\n")
print("\n".join(detail))
print("\n粘进 make_figures.py 的 fig_cam_feature_heatmap：\n")
print(f"    M = np.array([{matrix[0]},     # agentview: {', '.join(features)}")
print(f"                  {matrix[1]}])    # wrist:     {', '.join(features)}")
