"""
collect_data.py —— Day 5: 收 rollout 数据

关键设计 (决定第二周能不能顺利往下走):
  1. 图像在 env.step() 之前采集 -> frame[t] 与 action[t] 严格配对
     ("在 t 时刻看到 frame[t], 然后施加 action[t]")
     这是 action-conditioned predictor 的输入格式, 错开一位就全错。
  2. 相位标签用策略状态机 (APPROACH/DESCEND/GRASP/LIFT), 不用 t/T。
     t/T 需要事后才知道的 episode 总长, 在线触发时拿不到; 状态机运行时就有。
  3. clean (无扰动) 与 perturbed (有扰动) 分开存:
     predictor 只在 clean 上训练, 扰动必须是分布外的, 误差才会尖峰。
  4. 图像 384x384, 匹配 V-JEPA 2.1 ViT-B/16 的原生输入分辨率。

用法:
    conda activate g1jepa && cd ~/qiuyu/g1_jepa/exp
    export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=4

    python collect_data.py --split clean     --n 3   --verify   # 先小批验证
    python collect_data.py --split clean     --n 300
    python collect_data.py --split perturbed --n 100
"""

import os
import json
import argparse
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import logging
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

import numpy as np

from task_env import (
    make_env, get_object_pos, get_eef_pos, PerturbationScheduler, CONTROL_FREQ,
)
from run_episode import ScriptedPickPlace

CAMERAS = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}
PHASES = ["APPROACH", "DESCEND", "GRASP", "LIFT", "DONE"]


def render_frame(env, cam_name, size):
    """
    在当前物理状态下即时渲染一帧。不依赖 env.step() 返回的 obs,
    所以可以在施加动作之前采集。返回 (H, W, 3) uint8, 已修正上下翻转。
    """
    try:
        img = env.sim.render(width=size, height=size, camera_name=cam_name)
    except TypeError:
        img = env.sim.render(size, size, camera_name=cam_name)
    return np.asarray(img)[::-1].copy()


def collect_episode(env, seed, perturb, img_size=384, max_steps=300,
                    policy_kwargs=None):
    rng = np.random.default_rng(seed)
    env.reset()

    policy = ScriptedPickPlace(**(policy_kwargs or {}))
    policy.replan(env, count=False)
    sched = PerturbationScheduler(enabled=perturb, rng=rng)

    obj_init_z = get_object_pos(env)[2]
    frames = {k: [] for k in CAMERAS}
    rec = {"eef": [], "obj": [], "action": [], "phase": [],
           "phase_step": [], "gripper": []}
    phase_step = 0
    prev_phase = None
    success = False
    t = 0

    for t in range(max_steps):
        eef, obj = get_eef_pos(env), get_object_pos(env)
        fired = sched.maybe_fire(env, t, eef, obj, policy.state)
        if fired:
            obj = get_object_pos(env)

        phase = policy.state
        phase_step = phase_step + 1 if phase == prev_phase else 0
        prev_phase = phase

        # ---- 先采图, 再决定动作, 再 step ----
        for key, cam in CAMERAS.items():
            frames[key].append(render_frame(env, cam, img_size))

        action = policy.act(env)

        rec["eef"].append(eef)
        rec["obj"].append(obj)
        rec["action"].append(action.copy())
        rec["phase"].append(PHASES.index(phase) if phase in PHASES else -1)
        rec["phase_step"].append(phase_step)
        rec["gripper"].append(
            env.sim.data.get_joint_qpos("gripper0_right_finger_joint1").copy())

        env.step(action)

        if get_object_pos(env)[2] > obj_init_z + 0.04:
            success = True
            break

    out = {k: np.asarray(v) for k, v in rec.items()}
    out["success"] = success
    out["steps"] = t + 1
    out["seed"] = seed
    out["perturbed"] = bool(sched.fired)
    out["perturb_step"] = -1 if sched.fire_step is None else int(sched.fire_step)
    out["perturb_delta"] = (np.zeros(2) if sched.delta is None
                            else np.asarray(sched.delta))
    return out, frames


def save_episode(outdir, idx, data, frames, fps, use_mp4=True):
    base = os.path.join(outdir, f"ep_{idx:05d}")
    np.savez_compressed(base + ".npz", **data)

    if use_mp4:
        try:
            import imageio.v2 as imageio
            for key, arr in frames.items():
                w = imageio.get_writer(f"{base}_{key}.mp4", fps=fps,
                                       macro_block_size=None, quality=9)
                for f in arr:
                    w.append_data(f)
                w.close()
            return "mp4"
        except Exception as e:
            print(f"  mp4 写入失败 ({e}), 回退到 npz")

    for key, arr in frames.items():
        np.savez_compressed(f"{base}_{key}.npz", frames=np.stack(arr))
    return "npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["clean", "perturbed"], required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--img", type=int, default=384)
    ap.add_argument("--root", default="data")
    ap.add_argument("--seed0", type=int, default=None,
                    help="默认 clean 从 0 起, perturbed 从 100000 起, 保证不重叠")
    ap.add_argument("--npz", action="store_true", help="强制用 npz 而非 mp4")
    ap.add_argument("--verify", action="store_true", help="小批验证并存样例图")
    args = ap.parse_args()

    perturb = (args.split == "perturbed")
    seed0 = args.seed0 if args.seed0 is not None else (100000 if perturb else 0)
    outdir = os.path.join(args.root, args.split)
    os.makedirs(outdir, exist_ok=True)

    env = make_env(img_size=args.img, use_camera_obs=False)
    # sim.render 需要离屏渲染器; use_camera_obs=False 避免 step 时重复渲染
    env.has_offscreen_renderer = True

    t0 = time.time()
    n_ok = n_fired = 0
    fmt = None

    for i in range(args.n):
        data, frames = collect_episode(env, seed=seed0 + i, perturb=perturb,
                                       img_size=args.img)
        fmt = save_episode(outdir, i, data, frames, fps=CONTROL_FREQ,
                           use_mp4=not args.npz)
        n_ok += int(data["success"])
        n_fired += int(data["perturbed"])

        if args.verify and i == 0:
            import imageio.v2 as imageio
            ps = int(data["perturb_step"])
            marks = [0, max(0, ps - 1), ps, min(len(frames["agentview"]) - 1, ps + 5)] \
                if ps >= 0 else [0, 5, 10, 20]
            for m in marks:
                imageio.imwrite(f"verify_t{m:03d}.png", frames["agentview"][m])
            print(f"  已存样例图 verify_t*.png (步号 {marks})")
            print(f"  帧数={len(frames['agentview'])} 形状={frames['agentview'][0].shape}")
            print(f"  action={data['action'].shape} phase={data['phase'].shape}")
            print(f"  扰动步={ps} 位移={np.round(data['perturb_delta'], 4)}")

        if (i + 1) % 25 == 0 or i == args.n - 1:
            el = time.time() - t0
            print(f"[{i+1}/{args.n}] 成功 {n_ok} 触发扰动 {n_fired} "
                  f"用时 {el:.0f}s ({el/(i+1):.2f}s/条)")

    meta = {
        "split": args.split, "n": args.n, "img_size": args.img,
        "control_freq": CONTROL_FREQ, "cameras": list(CAMERAS),
        "phases": PHASES, "format": fmt, "seed0": seed0,
        "n_success": n_ok, "n_perturbed": n_fired,
        "note": "frame[t] 在 action[t] 之前采集; phase 是策略状态机的下标",
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    env.close()
    sz = sum(os.path.getsize(os.path.join(outdir, f))
             for f in os.listdir(outdir)) / 1e9
    print(f"\n完成: {outdir}  成功率 {n_ok/args.n:.2f}  占用 {sz:.2f} GB")


if __name__ == "__main__":
    main()
