import json
a = json.load(open("/tmp/r1.json"))["jepa"]["perturbed"]
b = json.load(open("/tmp/r2.json"))["jepa"]["perturbed"]
for x, y in zip(a, b):
    za = [round(v, 4) for _, v in x["z_log"]]
    zb = [round(v, 4) for _, v in y["z_log"]]
    same = za == zb
    print(f"seed={x['seed']} z相同={same} replans={x['n_replans']}/{y['n_replans']}")
    if not same:
        for i, (u, v) in enumerate(zip(za, zb)):
            if u != v:
                print(f"   首个差异 idx={i}: {u} vs {v}")
                break