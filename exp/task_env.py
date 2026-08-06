"""
task_env.py —— 环境工厂 + 扰动 hook
针对实测环境: robosuite 1.5.2 + mujoco 3.2.7 + Panda + Lift

要点:
  - 物体位置一律从 sim 直接读 (get_object_pose), 不读 obs。
    obs 是 env.step() 时刷新的, 瞬移之后 obs 里的 cube_pos 是旧值, 读它会出错。
  - OSC 的 input_ref_frame 是 "base", 所以动作要从世界系转到基座系。
    base_rot() 做这件事; 如果基座和世界系本来就对齐, 变换退化成恒等, 无副作用。
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import robosuite as suite

OBJ_JOINT = "cube_joint0"
OBJ_BODY = "cube_main"
BASE_BODY = "robot0_base"
CONTROL_FREQ = 20          # Hz。JEPA 跑 10Hz = 每 2 个控制步一次
ACTION_SCALE = 0.05        # OSC output_max, 动作值 1.0 = 5cm 位移指令


def load_ctrl_cfg():
    from robosuite.controllers import load_composite_controller_config
    return load_composite_controller_config(controller="BASIC", robot="Panda")


def make_env(img_size=128, cameras=("agentview", "robot0_eye_in_hand"),
             horizon=1000, use_camera_obs=False):
    """
    use_camera_obs=False 时不开离屏渲染, 速度快很多。
    Day 3 调策略参数用 False, Day 5 收数据用 True。
    """
    return suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=load_ctrl_cfg(),
        has_renderer=False,
        has_offscreen_renderer=use_camera_obs,
        use_camera_obs=use_camera_obs,
        use_object_obs=True,
        camera_names=list(cameras),
        camera_heights=img_size,
        camera_widths=img_size,
        control_freq=CONTROL_FREQ,
        horizon=horizon,
        reward_shaping=False,
        ignore_done=True,
    )


# --------------------------------------------------------------------------
# 坐标系
# --------------------------------------------------------------------------

def base_rot(env):
    """基座坐标系 -> 世界坐标系的旋转矩阵 R。世界向量转基座系用 R.T @ v。"""
    try:
        return np.array(env.sim.data.get_body_xmat(BASE_BODY)).reshape(3, 3).copy()
    except Exception:
        return np.eye(3)


def world_to_base(env, v_world):
    return base_rot(env).T @ np.asarray(v_world, dtype=float)


# --------------------------------------------------------------------------
# 物体状态 / 扰动
# --------------------------------------------------------------------------

def get_object_pose(env):
    """直接从 sim 读, 永远是最新的。返回 (pos[3], quat[4])。"""
    q = env.sim.data.get_joint_qpos(OBJ_JOINT).copy()
    return q[:3].copy(), q[3:7].copy()


def get_object_pos(env):
    return get_object_pose(env)[0]


def get_eef_pos(env):
    return np.array(env.sim.data.get_site_xpos(
        env.robots[0].gripper["right"].important_sites["grip_site"])).copy()


def teleport_object(env, delta_xy):
    """水平瞬移物体, 速度清零。只能在物体尚未被夹住时调用。"""
    q = env.sim.data.get_joint_qpos(OBJ_JOINT).copy()
    q[0] += float(delta_xy[0])
    q[1] += float(delta_xy[1])
    env.sim.data.set_joint_qpos(OBJ_JOINT, q)
    env.sim.data.set_joint_qvel(OBJ_JOINT, np.zeros(6))
    env.sim.forward()
    return q[:3].copy()


def sample_perturbation(rng, mag_range=(0.05, 0.10)):
    """水平面上随机方向、5-10cm 的位移。"""
    theta = rng.uniform(0, 2 * np.pi)
    mag = rng.uniform(*mag_range)
    return np.array([mag * np.cos(theta), mag * np.sin(theta)])


class PerturbationScheduler:
    """
    在 DESCEND 阶段（已经对准、正在下落抓取时）扰动一次。

    用状态而不是距离作为触发条件的三个理由:
      - 保证扰动前已经积累了足够的历史帧, JEPA 才有可能检测到
      - 保证扰动发生在"预测已经形成"之后, 语义上才是预测被违背
      - 不同 episode 的扰动都落在同一个任务相位, 第二周算相位归一化阈值时省事
    """

    def __init__(self, fire_state="DESCEND", delay_range=(2, 12), min_step=10,
                 enabled=True, rng=None, mag_range=(0.05, 0.10)):
        self.fire_state = fire_state
        self.min_step = min_step
        self.enabled = enabled
        self.rng = rng if rng is not None else np.random.default_rng()
        self.delay_steps = int(self.rng.integers(delay_range[0], delay_range[1]))
        self.mag_range = mag_range
        self.fired = False
        self.fire_step = None
        self.delta = None
        self.new_pos = None
        self._in_state = 0

    def maybe_fire(self, env, step, eef_pos, obj_pos, policy_state):
        if not self.enabled or self.fired:
            return False
        if policy_state != self.fire_state:
            self._in_state = 0
            return False
        self._in_state += 1
        if self._in_state <= self.delay_steps or step < self.min_step:
            return False
        self.delta = sample_perturbation(self.rng, self.mag_range)
        self.new_pos = teleport_object(env, self.delta)
        self.fired = True
        self.fire_step = step
        return True

def is_grasping(env):
    """夹爪是否真的夹住了物体。"""
    try:
        return bool(env._check_grasp(
            gripper=env.robots[0].gripper["right"],
            object_geoms=env.cube,
        ))
    except Exception:
        w = abs(env.sim.data.get_joint_qpos("gripper0_right_finger_joint1")) \
            + abs(env.sim.data.get_joint_qpos("gripper0_right_finger_joint2"))
        return 0.008 < w < 0.038


if __name__ == "__main__":
    env = make_env(use_camera_obs=False)
    env.reset()

    R = base_rot(env)
    print("robot0_base 旋转矩阵:")
    print(np.round(R, 4))
    print(f"是否为单位阵: {np.allclose(R, np.eye(3), atol=1e-6)}")
    print("(若为 True, 基座系与世界系对齐, 坐标变换退化成恒等)")

    print(f"\neef  = {np.round(get_eef_pos(env), 4)}")
    print(f"cube = {np.round(get_object_pos(env), 4)}")

    print("\n--- 扰动自检: 张开夹爪下落 60 步, 途中触发一次 ---")
    rng = np.random.default_rng(0)
    sched = PerturbationScheduler(enabled=True, rng=rng)
    for t in range(60):
        a = np.zeros(7)
        a[2] = -0.3      # 缓慢下降
        a[-1] = -1.0     # 夹爪张开
        env.step(a)
        eef, obj = get_eef_pos(env), get_object_pos(env)
        if sched.maybe_fire(env, t, eef, obj, "DESCEND"):
            print(f"[t={t}] 扰动! delta={np.round(sched.delta, 4)} "
                  f"新位置={np.round(sched.new_pos, 4)}")
        if t % 10 == 0:
            d = np.linalg.norm(eef[:2] - obj[:2])
            print(f"  t={t:3d}  eef={np.round(eef, 3)}  cube={np.round(obj, 3)}  xy距离={d:.3f}")

    env.close()
    print("\n若看到 cube 的 xy 出现 5-10cm 跳变, 扰动 hook 就成了。")



