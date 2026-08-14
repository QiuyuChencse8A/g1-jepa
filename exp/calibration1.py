import json

c = json.load(open("calib_f16s1_wrist_grid_k1.json"))

print("global:",
      round(c["global_mu"], 2),
      round(c["global_sd"], 2))

for k in sorted(c["buckets"], key=lambda x: int(x)):
    mu, sd = c["buckets"][k]
    print(
        f"bucket {k:>3}: "
        f"mu={mu:7.2f}  sd={sd:6.2f}"
    )