""" The script to collect demonstrations for the legged robot """
# 1. Embedded with the network file transfer feature
# 2. Monitor the trajectories, pack N trajectories into one file, and upload to a bucket
# 3. 


import isaacgym
from collections import OrderedDict
import torch
from datetime import datetime
import numpy as np
np.float = float
import os
import json
import os.path as osp

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import update_cfg_from_args, class_to_dict, update_class_from_dict
from legged_gym.debugger import break_into_debugger

from rsl_rl.modules import build_actor_critic
from rsl_rl.runners.dagger_saver import DemonstrationSaver, DaggerSaver
from typing import Optional, Any
# os.environ['MESA_VK_DEVICE_SELECT'] = '10de:24b0'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# torch.cuda.set_device(1)


def _as_bool(v, default=False):
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "on")


def deep_remap_paths(obj, old_root: str, new_root: str):
    """递归替换任意字符串值中的路径前缀 old_root -> new_root。
    支持 dict / list / tuple / str，其它类型原样返回。
    """
    if isinstance(obj, str):
        return obj.replace(old_root, new_root)
    if isinstance(obj, dict):
        return {k: deep_remap_paths(v, old_root, new_root) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        mapped = [deep_remap_paths(v, old_root, new_root) for v in obj]
        return type(obj)(mapped)
    return obj


def maybe_patch_summarywriter(remap_old: Optional[str], remap_new: Optional[str], enable: Optional[bool] = None):
    """（可选）猴子补丁 tensorboardX SummaryWriter 以及 dagger_saver 内部引用。
    说明：DaggerSaver 模块中是 `from tensorboardX import SummaryWriter`，因此仅替换
    tensorboardX.SummaryWriter 不一定生效；这里也尝试替换 rsl_rl.runners.dagger_saver.SummaryWriter。
    """
    active = enable if enable is not None else bool(remap_old and remap_new)
    if not active or not (remap_old and remap_new):
        return
    try:
        from tensorboardX import SummaryWriter as OriginalSummaryWriter  # type: ignore
        import tensorboardX  # type: ignore
    except Exception:
        return

    class PatchedSummaryWriter(OriginalSummaryWriter):
        def __init__(self, log_dir=None, *args, **kwargs):
            if isinstance(log_dir, str) and log_dir.startswith(remap_old):
                new_log_dir = log_dir.replace(remap_old, remap_new)
                print(f"[MonkeyPatch] SummaryWriter log_dir remap:\n  from: {log_dir}\n    to: {new_log_dir}")
                log_dir = new_log_dir
            super().__init__(log_dir=log_dir, *args, **kwargs)

    # 覆盖 tensorboardX 模块中的引用
    try:
        tensorboardX.SummaryWriter = PatchedSummaryWriter  # type: ignore
    except Exception:
        pass
    # 覆盖 DaggerSaver 模块中的引用（防止其内部独立导入）
    try:
        import rsl_rl.runners.dagger_saver as _ds  # type: ignore
        _ds.SummaryWriter = PatchedSummaryWriter  # type: ignore
    except Exception:
        pass

# -----------------------------
# Pretty printing utilities
# -----------------------------
_USE_COLOR = os.getenv("NO_COLOR", "0").lower() not in ("1", "true", "yes")
class _C:
    RESET="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"; ITALIC="\033[3m"; UNDERLINE="\033[4m"
    BLACK="\033[30m"; RED="\033[31m"; GREEN="\033[32m"; YELLOW="\033[33m"; BLUE="\033[34m"; MAGENTA="\033[35m"; CYAN="\033[36m"; WHITE="\033[37m"
    BRIGHT_BLACK="\033[90m"; BRIGHT_RED="\033[91m"; BRIGHT_GREEN="\033[92m"; BRIGHT_YELLOW="\033[93m"; BRIGHT_BLUE="\033[94m"; BRIGHT_MAGENTA="\033[95m"; BRIGHT_CYAN="\033[96m"; BRIGHT_WHITE="\033[97m"

def _color(text: str, *styles: str):
    if not _USE_COLOR:
        return text
    seq = ''.join(styles)
    return f"{seq}{text}{_C.RESET}"

def _section(title: str):
    return _color(f"{title}", _C.BOLD, _C.BRIGHT_CYAN)

def _kv(label: str, value: Any, color=_C.BRIGHT_WHITE):
    return f"{_color(label + ':', _C.BOLD, color)} {value}" if isinstance(value, str) else f"{_color(label + ':', _C.BOLD, color)} {value}"

def main(args):

    # print("sim device:", args.sim_device)
    # print("rl device:", args.rl_device)
    # print("graphics device id", args.graphics_device_id)

    RunnerCls = DaggerSaver if args.load_run else DemonstrationSaver
    success_traj_only = False
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)


    # 统一从配置读取根路径（Go2DistillCfg.custom 已根据 shared_path 开关选择本地或 NFS）
    custom = getattr(env_cfg, 'custom', None)
    logs_root = getattr(custom, 'logs_root', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs'))
    print(_kv("[COLLECT] logs_root", logs_root, _C.CYAN))
    
    # 可选：跨节点路径 remap（环境变量优先，其次 CLI）
    remap_old_env = os.getenv("COLLECT_REMAP_OLD_BASE")
    remap_new_env = os.getenv("COLLECT_REMAP_NEW_BASE")
    remap_old = args.remap_old or remap_old_env
    remap_new = args.remap_new or remap_new_env
    summary_patch_env = os.getenv("COLLECT_SUMMARY_REMAP_ENABLE")
    maybe_patch_summarywriter(remap_old, remap_new, enable=_as_bool(summary_patch_env, default=bool(remap_old and remap_new)))

    training_policy_logdir = osp.join(logs_root, train_cfg.runner.experiment_name, args.load_run) if args.load_run else osp.join(logs_root, train_cfg.runner.experiment_name)
    if args.load_run and remap_old and remap_new and training_policy_logdir.startswith(remap_old):
        new_dir = training_policy_logdir.replace(remap_old, remap_new, 1)
        print(f"[COLLECT] remap training_policy_logdir:\n  from: {training_policy_logdir}\n    to: {new_dir}")
        training_policy_logdir = new_dir
        
    # config path used only in DAgger branch
    training_policy_log_cfg_path = osp.join(training_policy_logdir, "config.json") if args.load_run else None

    ### DEBUGGING
    print(_kv("[COLLECT] task", args.task, _C.MAGENTA))

    # args.log = True
    # args.load_run = "Jun27_14-58-44_Go2_10skills_fromMay26_20-05-28"

    if RunnerCls == DaggerSaver:
        if not args.load_run:
            raise ValueError("--load_run is required for DAgger collection")
    if training_policy_log_cfg_path:
        print(_kv("[COLLECT] loading config", training_policy_log_cfg_path, _C.YELLOW))
        with open(training_policy_log_cfg_path, "r") as f:
            d = json.load(f, object_pairs_hook=OrderedDict)
        if remap_old and remap_new:
            d = deep_remap_paths(d, remap_old, remap_new)
        update_class_from_dict(env_cfg, d, strict=True)
        update_class_from_dict(train_cfg, d, strict=True)
    else:
        print(_color("[COLLECT] no training policy config (Demonstration mode)", _C.BOLD, _C.BRIGHT_YELLOW))
            
    
    ####### customized option to increase data distribution #######
    env_cfg.env.num_envs = 128      # << modify here for quick run
    # env_cfg.terrain.curriculum = True
    # env_cfg.terrain.max_init_terrain_level = 0
    # env_cfg.terrain.border_size = 1.
    ############# some predefined options #############

    env_cfg.terrain.num_rows = 8 ; env_cfg.terrain.num_cols = 20

    print(_kv("[COLLECT] num_envs", str(env_cfg.env.num_envs), _C.GREEN))
    # Done custom settings

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)

    config = class_to_dict(train_cfg)
    config.update(class_to_dict(env_cfg))
    teacher_act_prob = config["algorithm"]["teacher_act_prob"] if args.teacher_prob is None else args.teacher_prob
    action_std = config["policy"]["init_noise_std"] if args.action_std is None else args.action_std

    # create teacher policy
    policy = build_actor_critic(
        env,
        config["algorithm"]["teacher_policy_class_name"],
        config["algorithm"]["teacher_policy"],
    ).to(env.device)

    # load the teacher policy
    teacher_path = None
    if config["algorithm"].get("teacher_ac_path"):
        teacher_path = config["algorithm"]["teacher_ac_path"]
        if "{LEGGED_GYM_ROOT_DIR}" in teacher_path:
            teacher_path = teacher_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        if remap_old and remap_new:
            teacher_path = teacher_path.replace(remap_old, remap_new)
        print(_kv("[COLLECT] teacher_ac_path", teacher_path, _C.BRIGHT_BLUE))
        try:
            state_dict = torch.load(teacher_path, map_location="cpu")
            teacher_actor_critic_state_dict = state_dict["model_state_dict"]
            policy.load_state_dict(teacher_actor_critic_state_dict)
            print(_color("[COLLECT] teacher weights loaded", _C.GREEN, _C.BOLD))
        except Exception as e:
            print(_color(f"[COLLECT][WARN] Failed to load teacher weights: {e}", _C.BOLD, _C.BRIGHT_RED))
    else:
        print(_color("[COLLECT] no teacher_ac_path provided", _C.BRIGHT_YELLOW))

    # build runner
    track_header = "".join(env_cfg.terrain.BarrierTrack_kwargs["options"])
    # 统一数据根目录（用于保存采集的数据）
    data_root = getattr(custom, 'data_root', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data'))

    # 根据任务名将数据按子目录区分：
    # - 纯 Demonstrationr: data/<task>/...
    # - DAgge（--load_run 存在）: data/<experiment_name>_dagger/...
    task_subdir = args.task  # 例如 go2_distill_awbc / go2_distill_no_awbc
    base_dataset_root = (
        os.path.join(data_root, f"{task_subdir}_dagger")
        if RunnerCls == DaggerSaver
        else os.path.join(data_root, task_subdir)
    )

    runner_kwargs = dict(
        env= env,
        policy= policy,
        save_dir= osp.join(
            base_dataset_root,
            datetime.now().strftime('%b%d_%H-%M-%S') + "_" + "".join([
                track_header,
                "_lowBorder" if env_cfg.terrain.BarrierTrack_kwargs["border_height"] < 0 else "",
                "_trackWidth{:.1f}".format(env_cfg.terrain.BarrierTrack_kwargs["track_width"]) if env_cfg.terrain.BarrierTrack_kwargs["track_width"] < 1.8 else "",
                "_blockLength{:.1f}".format(env_cfg.terrain.BarrierTrack_kwargs["track_block_length"]) if env_cfg.terrain.BarrierTrack_kwargs["track_block_length"] > 1.6 else "",
                "_addMassMin{:.1f}".format(env_cfg.domain_rand.added_mass_range[0]) if env_cfg.domain_rand.added_mass_range[0] > 1. else "",
                "_teacherProb{:.1f}".format(teacher_act_prob),
                "_randOrder" if env_cfg.terrain.BarrierTrack_kwargs.get("randomize_obstacle_order", False) else "",
                ("_noPerlinRate{:.1f}".format(
                    (env_cfg.terrain.BarrierTrack_kwargs["no_perlin_threshold"] - env_cfg.terrain.TerrainPerlin_kwargs["zScale"][0]) / \
                    (env_cfg.terrain.TerrainPerlin_kwargs["zScale"][1] - env_cfg.terrain.TerrainPerlin_kwargs["zScale"][0])
                ) if isinstance(env_cfg.terrain.TerrainPerlin_kwargs["zScale"], (tuple, list)) else ""),
                ("_fric{:.1f}-{:.1f}".format(*env_cfg.domain_rand.friction_range)),
                "_successOnly" if success_traj_only else "",
                "_aStd{:.2f}".format(action_std) if (action_std > 0. and RunnerCls == DaggerSaver) else "",
            ] + ([] if RunnerCls == DemonstrationSaver else ["_" + "_".join(args.load_run.split("_")[:2])])
            ),
        ),
        rollout_storage_length= 256,
        min_timesteps= 1e9, # 1e6,
        min_episodes= 1e6 if RunnerCls == DaggerSaver else 2e5,
        use_critic_obs= True,
        success_traj_only= success_traj_only,
        obs_disassemble_mapping= dict(
            forward_depth= "normalized_image",
        ),
        demo_by_sample= config["algorithm"].get("action_labels_from_sample", False),

        # compute_advantages= True,
        # gae_lambda= config.get("gae_lambda", 0.95),
        # gamma= config.get("gamma", 0.99),
    )
    if RunnerCls == DaggerSaver:
        # kwargs for dagger saver
        print(_section("[DAGGER_SAVER]"), _color("init", _C.BRIGHT_YELLOW))
        print(_kv("  training_policy_logdir", training_policy_logdir, _C.BRIGHT_YELLOW))
        print(_kv("  teacher_act_prob", teacher_act_prob, _C.BRIGHT_YELLOW))
        print(_kv("  update_times_scale", config['algorithm'].get('update_times_scale', 1e5), _C.BRIGHT_YELLOW))
        print(_kv("  action_sample_std", action_std, _C.BRIGHT_YELLOW))
        runner_kwargs.update(dict(
            training_policy_logdir=training_policy_logdir,
            teacher_act_prob=teacher_act_prob,
            update_times_scale=config["algorithm"].get("update_times_scale", 1e5),
            action_sample_std=action_std,
            log_to_tensorboard=args.log,
        ))
    print(_section("[COLLECT] save_dir"), _color(runner_kwargs['save_dir'], _C.BRIGHT_GREEN))
    print(_color("[COLLECT] Advantage fields will be included when available.", _C.BOLD, _C.BRIGHT_WHITE))
    runner = RunnerCls(**runner_kwargs)
    runner.collect_and_save(config= config) 

if __name__ == "__main__":
    args = get_args(
        custom_args= [
            {"name": "--teacher_prob", "type": float, "default": None, "help": "probability of using teacher's action"},
            {"name": "--action_std", "type": float, "default": None, "help": "override the action sample std during rollout. None for using model's std"},
            {"name": "--log", "action": "store_true", "help": "log the data to tensorboard"},
            {"name": "--shared", "type": bool, "default": False, "help": "use shared path for NFS"},
            {"name": "--remap-old", "type": str, "default": None, "help": "old base path to replace (e.g., /mnt/rpl_project)"},
            {"name": "--remap-new", "type": str, "default": None, "help": "new base path (e.g., /cs/.../network_test)"},
        ],
    )
    main(args)
