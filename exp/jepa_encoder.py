"""
jepa_encoder.py —— Day 6: 读回 rollout、构造 clip、跑 V-JEPA encoder

分两段, 权重没到也能先跑前半段:

    python jepa_encoder.py --check-data          # 不需要权重
    python jepa_encoder.py --model ./vjepa2-vitl # 需要权重

设计要点:
  - clip 是"以 t 结尾、向前取 n 帧"。所以 latent[t] 只依赖 t 时刻及之前的信息,
    在线运行时完全可用, 不会偷看未来。
  - stride 控制时间窗口长度: n=16, stride=1, 20Hz -> 窗口 0.8s。
    窗口越长, 扰动进入窗口后被"稀释"得越厉害, 检测越慢。
    这是触发延迟的物理下界, 第二周要扫这个参数。
  - 不足 n 帧时重复首帧填充, 保证 episode 开头也能出 latent。
"""

import os
import argparse
import time
import glob

import numpy as np

CAMERAS = ["agentview", "wrist"]


# --------------------------------------------------------------------------
# 数据读取
# --------------------------------------------------------------------------

def load_episode(root, idx, camera="agentview"):
    """返回 (frames[T,H,W,3] uint8, lowdim dict)。"""
    base = os.path.join(root, f"ep_{idx:05d}")
    data = dict(np.load(base + ".npz", allow_pickle=True))

    mp4 = f"{base}_{camera}.mp4"
    npz = f"{base}_{camera}.npz"
    if os.path.exists(mp4):
        import imageio.v2 as imageio
        rdr = imageio.get_reader(mp4)
        frames = np.stack([f for f in rdr])
        rdr.close()
    elif os.path.exists(npz):
        frames = np.load(npz)["frames"]
    else:
        raise FileNotFoundError(f"找不到 {mp4} 或 {npz}")
    return frames, data


def build_clip(frames, t, n_frames=16, stride=1):
    """
    以第 t 帧结尾、向前取 n_frames 帧 (间隔 stride)。
    越界部分用首帧填充。返回 (n_frames, H, W, 3) uint8。
    """
    idx = [max(0, t - stride * (n_frames - 1 - i)) for i in range(n_frames)]
    return frames[idx]


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------

class JepaEncoder:
    def __init__(self, model_path, device="cuda:0", dtype="fp16"):
        import torch
        from transformers import AutoVideoProcessor, AutoModel

        self.torch = torch
        self.device = device
        self.dtype = torch.float16 if dtype == "fp16" else torch.float32

        print(f"加载 {model_path} -> {device} ({dtype})")
        self.processor = AutoVideoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        if dtype == "fp16":
            self.model = self.model.half()

        n = sum(p.numel() for p in self.model.parameters())
        print(f"参数量: {n/1e6:.0f}M")

    @property
    def _no_grad(self):
        return self.torch.no_grad()

    def encode(self, clip):
        """
        clip: (T, H, W, 3) uint8
        返回 (N_tokens, D) float32 numpy
        """
        torch = self.torch
        v = torch.from_numpy(np.ascontiguousarray(clip)).permute(0, 3, 1, 2)  # T,C,H,W
        inputs = self.processor(v, return_tensors="pt")
        inputs = {k: (val.to(self.device).to(self.dtype)
                      if val.dtype.is_floating_point else val.to(self.device))
                  for k, val in inputs.items()}
        with torch.no_grad():
            if hasattr(self.model, "get_vision_features"):
                out = self.model.get_vision_features(**inputs)
            else:
                out = self.model(**inputs).last_hidden_state
        return out[0].float().cpu().numpy()


# --------------------------------------------------------------------------

def check_data(root):
    print(f"=== 检查 {root} ===")
    eps = sorted(glob.glob(os.path.join(root, "ep_*.npz")))
    eps = [e for e in eps if "_agentview" not in e and "_wrist" not in e]
    print(f"episode 数: {len(eps)}")

    frames, data = load_episode(root, 0, "agentview")
    print(f"\nep_00000:")
    print(f"  帧: {frames.shape} {frames.dtype}")
    for k in ["action", "phase", "phase_step", "eef", "obj"]:
        if k in data:
            print(f"  {k:11s} {data[k].shape}")
    print(f"  steps={data['steps']} success={data['success']} "
          f"perturbed={data['perturbed']} perturb_step={data['perturb_step']}")

    T = len(frames)
    assert T == len(data["action"]), \
        f"帧数 {T} != 动作数 {len(data['action'])}, 时间对齐坏了!"
    print(f"  OK 帧数与动作数一致 ({T})")

    ps = int(data["perturb_step"])
    if ps >= 0:
        d = np.linalg.norm(data["obj"][ps + 1][:2] - data["obj"][ps - 1][:2])
        print(f"  扰动步 {ps} 前后物体位移 {d:.4f} m (应为 0.05-0.10)")

    clip = build_clip(frames, t=min(20, T - 1), n_frames=16, stride=1)
    print(f"\nclip: {clip.shape} (以第 20 帧结尾, 向前 16 帧)")
    clip0 = build_clip(frames, t=0, n_frames=16, stride=1)
    print(f"t=0 的 clip: {clip0.shape}, 全部相同={np.all(clip0 == clip0[0])} (应为 True)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/perturbed")
    ap.add_argument("--model", default=None, help="权重目录, 不给则只检查数据")
    ap.add_argument("--check-data", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    check_data(args.root)
    if args.check_data or args.model is None:
        print("\n(未指定 --model, 跳过 encoder 测试)")
        return

    print("\n=== Encoder ===")
    enc = JepaEncoder(args.model, device=args.device)
    frames, data = load_episode(args.root, 0, "agentview")

    clip = build_clip(frames, t=30, n_frames=args.frames, stride=args.stride)
    z = enc.encode(clip)
    print(f"latent: {z.shape}  (N_tokens, D)")
    print(f"  范数 {np.linalg.norm(z):.2f}  均值 {z.mean():.4f}  标准差 {z.std():.4f}")

    # 时延: 决定在线能不能跑 10Hz
    for _ in range(3):
        enc.encode(clip)
    t0 = time.time()
    N = 10
    for _ in range(N):
        enc.encode(clip)
    ms = (time.time() - t0) / N * 1000
    print(f"\n单次编码 {ms:.1f} ms -> 最高 {1000/ms:.1f} Hz "
          f"({'10Hz 没问题' if ms < 100 else '需要减帧数或降分辨率'})")

    # 相邻时刻的 latent 距离, 看看信号有没有区分度
    z1 = enc.encode(build_clip(frames, 30, args.frames, args.stride))
    z2 = enc.encode(build_clip(frames, 31, args.frames, args.stride))
    z3 = enc.encode(build_clip(frames, 45, args.frames, args.stride))
    print(f"\n||z30 - z31|| = {np.linalg.norm(z1-z2):.3f}  (相邻帧, 应较小)")
    print(f"||z30 - z45|| = {np.linalg.norm(z1-z3):.3f}  (相隔 15 帧, 应明显更大)")


if __name__ == "__main__":
    main()
