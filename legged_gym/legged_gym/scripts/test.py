from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.task_registry import task_registry
import torch, os

env_cfg, train_cfg = task_registry.get_cfgs(name="go2_distill_awbc")
# 关闭 viewer，避免 GLFW 报错
try:
    env_cfg.viewer.headless = True
except Exception:
    try:
        env_cfg.viewer.enable_viewer = False
    except Exception:
        pass

env, _ = task_registry.make_env(name="go2_distill_awbc", args=None, env_cfg=env_cfg)

print("has root_states:", hasattr(env, "root_states"))
if hasattr(env, "root_states"):
    print("root_states shape:", tuple(env.root_states.shape))

obs = env.get_observations(); critic_obs = env.get_privileged_observations()

printed = False
for i in range(2048):
    obs, priv, rew, dones, infos = env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
    if i < 3:
        print(f"[step {i}] infos keys:", list(infos.keys()))
        if "time_outs" in infos:
            print("  time_outs:", infos["time_outs"].shape, infos["time_outs"].dtype)
    # episode 仅在有 env 重置的步出现
    if (not printed) and ("episode" in infos) and bool(infos["episode"]):
        print("  episode keys:", list(infos["episode"].keys()))
        for k,v in infos["episode"].items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: tensor {tuple(v.shape)} {v.dtype}")
            else:
                print(f"    {k}: {type(v)} {v}")
        printed = True
        break
print("done.")

# 可放到你的测试脚本里
def quat_wxyz_to_euler_rpy(q):  # q: (...,4) [w,x,y,z]
    w,x,y,z = q.unbind(-1)
    # roll
    sinr_cosp = 2*(w*x + y*z); cosr_cosp = 1 - 2*(x*x + y*y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    # pitch
    sinp = 2*(w*y - z*x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp)*torch.tensor(3.14159265/2, device=q.device), torch.asin(sinp))
    # yaw
    siny_cosp = 2*(w*z + x*y); cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

from isaacgym.torch_utils import get_euler_xyz

# 使用示例
rs = env.root_states  # [num_envs, 13]
pos = rs[:, :3]
quat_xyzw = rs[:, 3:7]
r, p, y = get_euler_xyz(quat_xyzw)
print("pos[0]:", pos[0].tolist(), "rpy[0]:", [r[0].item(), p[0].item(), y[0].item()])