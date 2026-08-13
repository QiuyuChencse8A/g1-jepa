"""
jepa_trigger.py —— 在线 JEPA 触发器 + 标定导出

两个模式:

  1) 导出标定 (只做一次, 用 clean 的前 100 条):
     python jepa_trigger.py --export --tag f16s1_wrist \
            --feature grid --k 1 --metric abs --n-calib 100

     产出 calib_<tag>_<feature>_k<k>.json, 里面是冻结的 μ/σ。
     之后闭环只读这个文件, 绝不再看数据 —— 这样闭环实验就是干净的
     held-out 测试: 新种子、新 episode、不复用任何标定信息。

  2) 在线触发 (被 run_episode.py 调用):
     trig = OnlineJepaTrigger("calib_....json", model="./vjepa2-vitl",
                              device="cuda:5")
     trig.reset()
     fired = trig.step(t, env, policy)

关键: 在线特征必须和离线完全一致 —— 同一相机、同样 384 渲染、
同样 build_clip、同样 pool、同样分桶。任何一处不同, 冻结的 μ/σ 就失效。
"""

import os
import json
import glob
import argparse
from collections import deque

import numpy as np

from jepa_encoder import JepaEncoder, build_clip
from compute_latents import pool
from analyze_error2 import bucket, errors, feat, load_split

WRIST_CAM = "robot0_eye_in_hand"
AGENT_CAM = "agentview"


# --------------------------------------------------------------------------
# 标定导出
# --------------------------------------------------------------------------

def export_calib(tag, feature, k, metric, warmup, n_calib, root="latents",
                 out=None):
    eps = load_split(os.path.join(root, f"clean_{tag}"))[:n_calib]
    print(f"标定用 {len(eps)} 条 clean episode")

    buckets, allv = {}, []
    for ep in eps:
        e = errors(feat(ep, feature), k, metric)
        for ei, t, ph, pst in zip(e, ep["t"], ep["phase"], ep["phase_step"]):
            if np.isnan(ei) or t < warmup:
                continue
            buckets.setdefault(bucket(ph, pst), []).append(float(ei))
            allv.append(float(ei))

    allv = np.array(allv)
    cal = {
        "tag": tag, "feature": feature, "k": int(k), "metric": metric,
        "warmup": int(warmup), "n_calib": len(eps),
        "camera": WRIST_CAM if "wrist" in tag else AGENT_CAM,
        "frames": int(tag.split("f")[1].split("s")[0]),
        "stride": int(tag.split("s")[1].split("_")[0]),
        "global_mu": float(allv.mean()), "global_sd": float(allv.std() + 1e-9),
        "buckets": {},
    }
    for b, v in buckets.items():
        if len(v) >= 15:
            cal["buckets"][str(b)] = [float(np.mean(v)), float(np.std(v) + 1e-9)]

    out = out or f"calib_{tag}_{feature}_k{k}.json"
    with open(out, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"已保存 {out}: {len(cal['buckets'])} 个桶, "
          f"全局 μ={cal['global_mu']:.1f} σ={cal['global_sd']:.1f}")
    return out


# --------------------------------------------------------------------------
# 在线触发器
# --------------------------------------------------------------------------

class OnlineJepaTrigger:
    """
    每 step 个控制步编码一次 (step=2 -> 20Hz 控制下的 10Hz)。
    连续 consec 次 z > kappa_high 才触发;
    触发后进入 disarmed, 直到 z 回落到 kappa_low 以下才重新 armed
    (迟滞, 防止同一次扰动引发连续多次重规划)。
    """

    def __init__(self, calib_path, model="./vjepa2-vitl", device="cuda:5",
                 kappa_high=2.5, kappa_low=1.0, consec=2, step=2, encoder=None):
        with open(calib_path) as f:
            self.cal = json.load(f)
        self.enc = encoder or JepaEncoder(model, device=device)
        self.kappa_high = kappa_high
        self.kappa_low = kappa_low
        self.consec = consec
        self.step = step

        self.frames_n = self.cal["frames"]
        self.stride = self.cal["stride"]
        self.k = self.cal["k"]
        self.feature = self.cal["feature"]
        self.metric = self.cal["metric"]
        self.warmup = self.cal["warmup"]
        self.camera = self.cal["camera"]
        self.buckets = {int(b): tuple(v) for b, v in self.cal["buckets"].items()}
        self.gmu, self.gsd = self.cal["global_mu"], self.cal["global_sd"]
        self.reset()

    def reset(self):
        self.buf = deque(maxlen=self.frames_n * self.stride)
        self.hist = deque(maxlen=2 * self.k + 1)
        self.run = 0
        self.armed = True
        self.n_encodes = 0
        self.z_log = []
        self.dbg_log = []
        self.last_z = None


    def _render(self, env):
        img = env.sim.render(width=384, height=384, camera_name=self.camera)
        return np.asarray(img)[::-1].copy()

    def _pooled(self, clip):
        z = self.enc.encode(clip)
        m, tp, gd = pool(z)
        f = {"mean": m, "temporal": tp, "grid": gd}[self.feature]
        return np.asarray(f, dtype=np.float32).reshape(-1)

    def step_(self, t, env, policy):
        """返回 True 表示触发重规划。每个控制步都要调用一次(要维护帧缓冲)。"""
        self.buf.append(self._render(env))
        if t % self.step != 0 or t < self.warmup:
            return False

        frames = np.stack(self.buf)
        clip = build_clip(frames, len(frames) - 1, self.frames_n, self.stride)
        self.hist.append(self._pooled(clip))
        self.n_encodes += 1

        if len(self.hist) < 2 * self.k + 1:
            return False

        h = list(self.hist)
        pred = 2 * h[-1 - self.k] - h[-1 - 2 * self.k]
        resid = float(np.linalg.norm(pred - h[-1]))
        if self.metric == "rel":
            denom = max(float(np.linalg.norm(h[-1 - self.k] - h[-1 - 2 * self.k])), 1e-3)
            e = resid / denom
        else:
            e = resid

        b = bucket(PHASE_ID.get(policy.state, 0), policy_phase_step(policy))
        mu, sd = self.buckets.get(b, (self.gmu, self.gsd))
        z = (e - mu) / sd
        self.z_log.append((t, float(z)))
        self.dbg_log.append((t, b, b not in self.buckets,
                             float(e), float(mu), float(sd), float(z)))

        if not self.armed:
            if z < self.kappa_low:
                self.armed = True
                self.run = 0
            return False

        if z > self.kappa_high:
            self.run += 1
            if self.run >= self.consec:
                self.run = 0
                self.armed = False
                return True
        else:
            self.run = 0
        return False

    __call__ = step_


PHASE_ID = {"APPROACH": 0, "DESCEND": 1, "GRASP": 2, "LIFT": 3, "DONE": 4}


def policy_phase_step(policy):
    """策略在当前状态里已经待了多少步。由 run_episode 维护。"""
    return getattr(policy, "phase_step", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--tag", default="f16s1_wrist")
    ap.add_argument("--feature", default="grid")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--metric", default="abs")
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--n-calib", type=int, default=100)
    args = ap.parse_args()

    if args.export:
        export_calib(args.tag, args.feature, args.k, args.metric,
                     args.warmup, args.n_calib)
    else:
        print("用 --export 导出标定; 在线触发由 run_episode.py 调用")


if __name__ == "__main__":
    main()
