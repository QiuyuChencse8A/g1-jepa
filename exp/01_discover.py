"""
Step 1 —— 探明你那边 robosuite 的实际 API 表面。

我需要这个脚本的完整输出, 才能把后面的扰动 hook 和脚本策略写准。
robosuite 1.4 / 1.5 之间的关节命名、控制器配置、动作维度都可能不同。

用法:
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=4 python 01_discover.py > discover.log 2>&1
    然后把 discover.log 贴给我
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import robosuite as suite

print("=" * 70)
print("版本信息")
print("=" * 70)
import mujoco

print(f"robosuite : {suite.__version__}")
print(f"mujoco    : {mujoco.__version__}")
import sys

print(f"python    : {sys.version.split()[0]}")

print("\n" + "=" * 70)
print("控制器配置")
print("=" * 70)
try:
    from robosuite.controllers import load_composite_controller_config

    ctrl_cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
    print("API: 1.5+  load_composite_controller_config")
except Exception as e:
    print(f"1.5 API 不可用 ({e})")
    from robosuite.controllers import load_controller_config

    ctrl_cfg = load_controller_config(default_controller="OSC_POSE")
    print("API: 1.4   load_controller_config(default_controller='OSC_POSE')")

import json

print(json.dumps(ctrl_cfg, indent=2, default=str)[:2000])

env = suite.make(
    env_name="Lift",
    robots="Panda",
    controller_configs=ctrl_cfg,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    use_object_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_heights=128,
    camera_widths=128,
    control_freq=20,
    horizon=200,
)
obs = env.reset()

print("\n" + "=" * 70)
print("动作空间")
print("=" * 70)
low, high = env.action_spec
print(f"action dim : {len(low)}")
print(f"low        : {np.round(low, 3)}")
print(f"high       : {np.round(high, 3)}")
print("(OSC_POSE 通常是 [dx,dy,dz,drx,dry,drz,gripper], 7 维)")

print("\n" + "=" * 70)
print("观测 key")
print("=" * 70)
for k, v in sorted(obs.items()):
    print(f"  {k:34s} {str(getattr(v, 'shape', type(v)))}")

print("\n" + "=" * 70)
print("MuJoCo 关节名 (找物体的 free joint —— 扰动 hook 靠它)")
print("=" * 70)
m = env.sim.model
for i in range(m.njnt):
    name = m.joint_id2name(i)
    jtype = m.jnt_type[i]  # 0=free 1=ball 2=slide 3=hinge
    print(f"  [{i:3d}] type={jtype}  {name}")

print("\n" + "=" * 70)
print("MuJoCo body 名")
print("=" * 70)
for i in range(m.nbody):
    print(f"  [{i:3d}] {m.body_id2name(i)}")

print("\n" + "=" * 70)
print("相机名")
print("=" * 70)
for i in range(m.ncam):
    print(f"  [{i:3d}] {m.camera_id2name(i)}")

print("\n" + "=" * 70)
print("物体位姿读写测试 (扰动 hook 的核心)")
print("=" * 70)
# Lift 任务里物体对象通常是 env.cube, free joint 叫 cube_joint0
candidates = ["cube_joint0", "cube_main_joint0", "cube_joint"]
obj_joint = None
for c in candidates:
    try:
        q = env.sim.data.get_joint_qpos(c)
        obj_joint = c
        print(f"OK  找到物体 free joint: '{c}'  qpos={np.round(q, 4)}")
        break
    except Exception:
        continue

if obj_joint is None:
    print("没找到, 请从上面的关节列表里手动挑一个 type=0 (free) 的关节名")
else:
    q0 = env.sim.data.get_joint_qpos(obj_joint).copy()
    q1 = q0.copy()
    q1[1] += 0.08  # 沿 y 横移 8cm
    env.sim.data.set_joint_qpos(obj_joint, q1)
    env.sim.data.set_joint_qvel(obj_joint, np.zeros(6))
    env.sim.forward()
    q_after = env.sim.data.get_joint_qpos(obj_joint)
    print(f"    瞬移前 xyz = {np.round(q0[:3], 4)}")
    print(f"    瞬移后 xyz = {np.round(q_after[:3], 4)}")
    print(f"    位移       = {np.round(q_after[:3] - q0[:3], 4)}  (期望 [0, 0.08, 0])")

print("\n" + "=" * 70)
print("末端执行器观测")
print("=" * 70)
for k in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "cube_pos"]:
    if k in obs:
        print(f"  {k:24s} = {np.round(obs[k], 4)}")
    else:
        print(f"  {k:24s} = <不存在>")

env.close()
print("\n完成。把整个输出贴给我。")
