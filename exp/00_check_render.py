"""
Step 0 —— 验证服务器上的离屏渲染能力。
这一步跑不通，后面全都不用做。跑通了，平台就选定了。

用法:
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=4 python 00_check_render.py

注意:
  - MuJoCo 走 EGL/OpenGL，和 Isaac Sim 的 Vulkan/RTX 是完全不同的路径，
    Isaac 崩溃不代表这里也会崩。
  - CUDA_VISIBLE_DEVICES 控制不了 EGL 选哪张卡，要用 MUJOCO_EGL_DEVICE_ID。
  - 如果 egl 失败，改成 MUJOCO_GL=osmesa 再试一次（CPU 软件渲染，慢但几乎不会挂）。
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
print(f"[cfg] MUJOCO_GL          = {os.environ.get('MUJOCO_GL')}")
print(f"[cfg] MUJOCO_EGL_DEVICE_ID = {os.environ.get('MUJOCO_EGL_DEVICE_ID', '(unset)')}")

# ---------------------------------------------------------------- 1. 裸 MuJoCo
print("\n=== [1/2] 裸 MuJoCo 离屏渲染 ===")
try:
    import mujoco
    import numpy as np

    print(f"mujoco version: {mujoco.__version__}")

    XML = """
    <mujoco>
      <visual><global offwidth="640" offheight="480"/></visual>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="2 2 .1" rgba=".8 .8 .8 1"/>
        <body pos="0 0 .5">
          <freejoint/>
          <geom name="ball" type="sphere" size=".1" rgba="1 .3 .3 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data)
    img = renderer.render()
    print(f"OK  渲染成功, shape={img.shape}, dtype={img.dtype}, mean={img.mean():.1f}")
    if img.std() < 1.0:
        print("警告: 图像几乎是纯色, 可能渲染管线有问题, 请打开 PNG 目视确认")

    try:
        import imageio.v2 as imageio
        imageio.imwrite("check_mujoco.png", img)
        print("已保存 check_mujoco.png —— 用 scp 拉回本地肉眼看一下")
    except ImportError:
        np.save("check_mujoco.npy", img)
        print("imageio 未安装, 已保存 check_mujoco.npy")

except Exception as e:
    print(f"FAIL  裸 MuJoCo 渲染失败: {type(e).__name__}: {e}")
    print("→ 先试 MUJOCO_GL=osmesa; 再不行说明这台机器的图形栈整体有问题, 需要换机器")
    sys.exit(1)

# ------------------------------------------------------------- 2. robosuite
print("\n=== [2/2] robosuite 离屏渲染 ===")
try:
    import robosuite as suite

    print(f"robosuite version: {suite.__version__}")
except ImportError:
    print("robosuite 未安装。先执行:  pip install robosuite")
    print("(裸 MuJoCo 已通过, 这一步只是还没装包, 不是环境问题)")
    sys.exit(0)

try:
    # 控制器配置的加载方式在 robosuite 1.4 / 1.5 之间变过, 两条路都试
    ctrl_cfg, ctrl_api = None, None
    try:
        from robosuite.controllers import load_composite_controller_config

        ctrl_cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
        ctrl_api = "1.5+ (load_composite_controller_config)"
    except Exception:
        from robosuite.controllers import load_controller_config

        ctrl_cfg = load_controller_config(default_controller="OSC_POSE")
        ctrl_api = "1.4  (load_controller_config)"
    print(f"controller API: {ctrl_api}")

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=ctrl_cfg,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
        horizon=200,
        reward_shaping=True,
    )
    obs = env.reset()
    print("OK  env.reset() 成功")
    for k, v in sorted(obs.items()):
        shape = getattr(v, "shape", None)
        print(f"    {k:32s} {str(shape)}")

    # robosuite 的相机图像默认是上下翻转的, 这是个经典坑
    img = obs["agentview_image"][::-1]
    try:
        import imageio.v2 as imageio

        imageio.imwrite("check_agentview.png", img)
        imageio.imwrite("check_wrist.png", obs["robot0_eye_in_hand_image"][::-1])
        print("已保存 check_agentview.png / check_wrist.png")
    except ImportError:
        pass

    env.close()
    print("\n全部通过。平台确定为 robosuite + MuJoCo。")

except Exception as e:
    import traceback

    print(f"FAIL  robosuite 失败: {type(e).__name__}: {e}")
    traceback.print_exc()
