"""
run_episode.py —— 脚本策略 + 四组条件的统一 runner

设计核心:
    策略在"规划时刻"把物体位置缓存进 self.target, 之后照着缓存执行。
    四组条件唯一的区别就是什么时候调用 policy.replan():

        no_replan : 永不调用            -> 物体移动后毫无察觉, 抓空
        fixed     : 每 T 步一次         -> 不管有没有扰动都重规划
        oracle    : 扰动发生的那一步    -> 理想触发器, 性能上界
        jepa      : 预测误差超阈值      -> 第二三周接进来, 只需替换一个函数

用法:
    # 单条 episode 详细输出, Day 3 调参用这个
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=4 python run_episode.py --debug

    # 三组条件各 20 条, Day 4 用这个
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=4 python run_episode.py --n 20
"""

import os
import argparse
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import logging
logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

from task_env import (
    make_env, get_object_pos, get_eef_pos, world_to_base,
    PerturbationScheduler, ACTION_SCALE, is_grasping
)


class ScriptedPickPlace:
    """
    OSC delta 控制下的路点状态机。

    动作语义: action[:3] 是基座系下的位移指令, 1.0 对应 ACTION_SCALE(=5cm)。
    所以 kp 的物理含义是"每步消掉剩余误差的 kp*ACTION_SCALE 倍"。
    kp=10 -> 每步走剩余距离的一半, 单步最多 5cm。这是个稳妥的起点。
    """

    HOVER_H = 0.10       # 悬停在物体上方 10cm
    GRASP_DZ = 0.002     # 下降到物体中心略上方
    LIFT_H = 0.18        # 抓起后抬升

    def __init__(self, kp=10.0, hover_thresh=0.02, descend_thresh=0.008,
                 grasp_steps=15, lift_thresh=0.05, retreat_thresh=0.02):
        self.kp = kp
        self.hover_thresh = hover_thresh
        self.descend_thresh = descend_thresh
        self.grasp_steps = grasp_steps
        self.lift_thresh = lift_thresh
        self.retreat_thresh = retreat_thresh
        self.reset()

    def reset(self):
        self.state = "APPROACH"
        self.target = None         # 缓存的物体位置 —— 只有 replan 会更新
        self.grasp_counter = 0
        self.n_replans = 0
        self.replan_steps = []

    def replan(self, env, step=None, count=True):
        new_target = get_object_pos(env).copy()
        moved = (self.target is None) or \
                (np.linalg.norm(new_target[:2] - self.target[:2]) > self.retreat_thresh)
        self.target = new_target
        if moved and self.state in ("APPROACH", "DESCEND"):
            self.state = "APPROACH"       # 只有目标真的移动了才退回重新对准
            self.grasp_counter = 0
        if count:
            self.n_replans += 1
            if step is not None:
                self.replan_steps.append(step)

    def _goto(self, env, eef, tgt, grip):
        err_world = np.asarray(tgt) - np.asarray(eef)
        cmd_world = np.clip(self.kp * err_world, -1.0, 1.0)
        cmd_base = world_to_base(env, cmd_world)
        cmd_base = np.clip(cmd_base, -1.0, 1.0)
        return np.array([cmd_base[0], cmd_base[1], cmd_base[2], 0., 0., 0., grip])

    def act(self, env):
        # 抓空了就中止重试。注意: 不刷新 self.target,
        # 否则 no_replan 相当于免费重规划, 对照组就失效了。
        if self.state in ("LIFT", "DONE") and not is_grasping(env):
            self.state = "APPROACH"
            self.grasp_counter = 0
        eef = get_eef_pos(env)
        if self.target is None:
            self.replan(env, count=False)

        if self.state == "APPROACH":
            tgt = self.target + np.array([0, 0, self.HOVER_H])
            a = self._goto(env, eef, tgt, -1.0)
            if np.linalg.norm(eef - tgt) < self.hover_thresh:
                self.state = "DESCEND"

        elif self.state == "DESCEND":
            tgt = self.target + np.array([0, 0, self.GRASP_DZ])
            a = self._goto(env, eef, tgt, -1.0)
            if np.linalg.norm(eef - tgt) < self.descend_thresh:
                self.state = "GRASP"

        elif self.state == "GRASP":
            a = np.zeros(7)
            a[-1] = 1.0
            self.grasp_counter += 1
            if self.grasp_counter >= self.grasp_steps:
                self.state = "LIFT"
                self.lift_from = eef[2]

        elif self.state == "LIFT":
            tgt = np.array([self.target[0], self.target[1],
                            self.target[2] + self.LIFT_H])
            a = self._goto(env, eef, tgt, 1.0)
            if eef[2] > self.lift_from + self.lift_thresh:
                self.state = "DONE"

        else:  # DONE —— 保持夹持并悬停
            tgt = np.array([self.target[0], self.target[1],
                            self.target[2] + self.LIFT_H])
            a = self._goto(env, eef, tgt, 1.0)

        return a


def run_episode(env, condition, seed=0, perturb=True, max_steps=300,
                fixed_period=20, collect_images=False, jepa_trigger=None,
                debug=False, policy_kwargs=None):
    rng = np.random.default_rng(seed)
    env.deterministic_reset = False
    obs = env.reset()

    policy = ScriptedPickPlace(**(policy_kwargs or {}))
    policy.replan(env, count=False)

    sched = PerturbationScheduler(enabled=perturb, rng=rng)

    obj_init_z = get_object_pos(env)[2]
    frames = {"agentview": [], "wrist": []} if collect_images else None
    log = {"eef": [], "obj": [], "state": [], "action": [], "replan": []}

    trigger_step = None
    success = False
    t = 0
    for t in range(max_steps):
        eef = get_eef_pos(env)
        obj = get_object_pos(env)

        fired = sched.maybe_fire(env, t, eef, obj, policy.state)
        if fired:
            obj = get_object_pos(env)
            if debug:
                print(f"  [t={t:3d}] *** 扰动 delta={np.round(sched.delta, 3)} ***")

        do_replan = False
        if condition == "no_replan":
            pass
        elif condition == "fixed":
            do_replan = (t > 0 and t % fixed_period == 0)
        elif condition == "oracle":
            do_replan = fired
        elif condition == "jepa":
            do_replan = bool(jepa_trigger(t, env, obs)) if jepa_trigger else False
        else:
            raise ValueError(condition)

        if do_replan:
            policy.replan(env, step=t)
            if trigger_step is None and sched.fired:
                trigger_step = t

        state_before = policy.state
        action = policy.act(env)
        obs, _, _, _ = env.step(action)

        log["eef"].append(eef)
        log["obj"].append(obj)
        log["state"].append(state_before)
        log["action"].append(action.copy())
        log["replan"].append(bool(do_replan))
        if collect_images:
            frames["agentview"].append(obs["agentview_image"][::-1].copy())
            frames["wrist"].append(obs["robot0_eye_in_hand_image"][::-1].copy())

        if debug and (t % 10 == 0 or state_before != policy.state):
            d = np.linalg.norm(eef[:2] - obj[:2])
            print(f"  t={t:3d} {state_before:9s} eef={np.round(eef,3)} "
                  f"cube={np.round(obj,3)} xy={d:.3f}")

        if get_object_pos(env)[2] > obj_init_z + 0.04:
            success = True
            if debug:
                print(f"  [t={t}] 成功: 物体已抬起")
            break

    return {
        "condition": condition, "seed": seed, "success": success, "steps": t + 1,
        "perturbed": sched.fired, "perturb_step": sched.fire_step,
        "perturb_delta": sched.delta, "trigger_step": trigger_step,
        "trigger_latency": (trigger_step - sched.fire_step)
                           if (trigger_step is not None and sched.fire_step is not None) else None,
        "n_replans": policy.n_replans, "replan_steps": policy.replan_steps,
        "final_state": policy.state,
        "log": {k: (v if k == "state" else np.array(v)) for k, v in log.items()},
        "frames": frames,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="单条 episode 详细输出")
    ap.add_argument("--n", type=int, default=20, help="每组条件的 episode 数")
    ap.add_argument("--kp", type=float, default=10.0)
    args = ap.parse_args()

    env = make_env(use_camera_obs=False)
    pk = {"kp": args.kp}

    if args.debug:
        print("=== 无扰动 (先确认脚本策略本身能抓起来) ===")
        r = run_episode(env, "no_replan", seed=0, perturb=False,
                        debug=True, policy_kwargs=pk)
        print(f"结果: success={r['success']}  steps={r['steps']}  "
              f"最终状态={r['final_state']}\n")

        print("=== 有扰动 + oracle 重规划 ===")
        r = run_episode(env, "oracle", seed=0, perturb=True,
                        debug=True, policy_kwargs=pk)
        print(f"结果: success={r['success']}  steps={r['steps']}  "
              f"扰动步={r['perturb_step']}  重规划={r['n_replans']} 次")
        env.close()
        return

    print(f"{'cond':<12s}{'SR_pert':>10s}{'SR_clean':>12s}"
          f"{'latency':>10s}{'replans':>12s}{'replans_clean':>14s}{'steps':>12s}")
    print("-" * 80)
    for cond in ["no_replan", "fixed", "oracle"]:
        rp = [run_episode(env, cond, seed=s, perturb=True, policy_kwargs=pk)
              for s in range(args.n)]
        rc = [run_episode(env, cond, seed=1000 + s, perturb=False, policy_kwargs=pk)
              for s in range(args.n)]
        fired = [r for r in rp if r["perturbed"]]
        sr_p = np.mean([r["success"] for r in fired]) if fired else float("nan")
        sr_c = np.mean([r["success"] for r in rc])
        lats = [r["trigger_latency"] for r in fired if r["trigger_latency"] is not None]
        if len(fired) < len(rp):
            print(f"  (注意: {len(rp)} 条里只有 {len(fired)} 条真正触发了扰动)")
        lat = np.mean(lats) if lats else float("nan")
        nrp = np.mean([r["n_replans"] for r in rp])
        succ = [r for r in fired if r["success"]]
        tsec = np.mean([r["steps"] for r in succ]) if succ else float("nan")
        nrp_clean = np.mean([r["n_replans"] for r in rc])   # 无扰动时的重规划次数 = 纯浪费
        print(f"{cond:<12s}{sr_p:>10.2f}{sr_c:>12.2f}{lat:>10.1f}"
              f"{nrp:>12.1f}{nrp_clean:>14.1f}{tsec:>12.1f}")

    env.close()
    print("\n期望: no_replan 扰动后接近 0 而无扰动很高; oracle 扰动后最高且延迟=0;")
    print("      fixed 居中但重规划次数远多于 oracle。")


if __name__ == "__main__":
    main()
