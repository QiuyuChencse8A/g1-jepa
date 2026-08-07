"""
compute_latents.py —— Day 7 第一步: 编码全部 rollout, 缓存池化后的 latent

为什么要缓存池化特征而不是完整 latent:
    完整 latent 是 (2048, 1024) fp16 = 4MB/时刻。100 条 x 50 时刻 = 20GB, 不现实。
    池化到 (1+8+16) x 1024 = 51KB/时刻, 总共约 250MB, 而且保留了三种粒度:

      mean     (1024,)     全局池化 —— 最粗, 小物体位移会被稀释
      temporal (8, 1024)   每个时间片的空间平均 —— 保留时间结构
      grid     (16, 1024)  最后一个时间片的 4x4 空间池化 —— 保留位置信息
                           (立方体只占画面一小块, 空间信息很可能是关键)

编码步长 step=2 对应 20Hz 控制下的 10Hz, 正是 JEPA 触发器的实际运行频率。

用法:
    python compute_latents.py --split perturbed --n 100 --device cuda:5
    python compute_latents.py --split clean     --n 100 --device cuda:5
"""

import os
import argparse
import time

import numpy as np

from jepa_encoder import JepaEncoder, load_episode, build_clip


def pool(z, n_temporal=8, grid=16):
    """(N_tokens, D) -> (mean, temporal, grid4x4)"""
    D = z.shape[-1]
    S = z.shape[0] // n_temporal
    g = int(round(S ** 0.5))
    zz = z.reshape(n_temporal, g, g, D)

    mean = z.mean(axis=0)                       # (D,)
    temporal = zz.mean(axis=(1, 2))             # (n_temporal, D)

    last = zz[-1]                               # (g, g, D)
    b = g // 4
    coarse = last.reshape(4, b, 4, b, D).mean(axis=(1, 3))   # (4, 4, D)
    return mean, temporal, coarse.reshape(16, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["clean", "perturbed"], required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="latents")
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--model", default="./vjepa2-vitl")
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--step", type=int, default=2, help="编码步长, 2 = 10Hz")
    ap.add_argument("--tail", type=int, default=80,
                    help="扰动后再编码多少步就停 (省时间)")
    args = ap.parse_args()

    src = os.path.join(args.root, args.split)
    dst = os.path.join(args.out, f"{args.split}_f{args.frames}s{args.stride}_{args.camera}")
    os.makedirs(dst, exist_ok=True)

    enc = JepaEncoder(args.model, device=args.device)
    t0 = time.time()

    for i in range(args.n):
        out_path = os.path.join(dst, f"ep_{i:05d}.npz")
        if os.path.exists(out_path):
            continue

        frames, meta = load_episode(src, i, args.camera)
        T = len(frames)
        ps = int(meta["perturb_step"])
        T_max = min(T, ps + args.tail) if ps >= 0 else T

        ts, means, temps, grids = [], [], [], []
        for t in range(0, T_max, args.step):
            z = enc.encode(build_clip(frames, t, args.frames, args.stride))
            m, tp, gd = pool(z)
            ts.append(t)
            means.append(m.astype(np.float16))
            temps.append(tp.astype(np.float16))
            grids.append(gd.astype(np.float16))

        ts = np.array(ts)
        np.savez_compressed(
            out_path,
            t=ts,
            mean=np.stack(means),
            temporal=np.stack(temps),
            grid=np.stack(grids),
            phase=meta["phase"][ts],
            phase_step=meta["phase_step"][ts],
            action=meta["action"][ts],
            obj=meta["obj"][ts],
            eef=meta["eef"][ts],
            perturb_step=ps,
            success=meta["success"],
        )

        if (i + 1) % 10 == 0 or i == args.n - 1:
            el = time.time() - t0
            print(f"[{i+1}/{args.n}] {len(ts)} 时刻/条  用时 {el:.0f}s "
                  f"({el/(i+1):.1f}s/条)")

    sz = sum(os.path.getsize(os.path.join(dst, f))
             for f in os.listdir(dst)) / 1e6
    print(f"\n完成: {dst}  占用 {sz:.0f} MB")


if __name__ == "__main__":
    main()
