import json, numpy as np
c = json.load(open("calib_f16s1_wrist_grid_k1.json"))
for k, v in c.items():
    if isinstance(v, dict):
        sub = list(v)[:6]
        print(f"{k}: dict, {len(v)} 项, 键示例 {sub}")
        first = v[list(v)[0]]
        print(f"   值示例: {first}")
    elif isinstance(v, list):
        print(f"{k}: list, len={len(v)}, 前3={v[:3]}")
    else:
        print(f"{k} = {v}")