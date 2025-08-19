"""Collect demonstrations/DAgger with cross-node path remap support.

Features:
- Optional path remap (OLD->NEW) for any string in loaded config.json
- Optional SummaryWriter monkey-patch to remap log_dir on the fly
- Uses env_cfg.custom.logs_root/data_root when available; otherwise falls back to repo/logs & repo/data
- Backward compatible with existing collect.py signatures

Env vars (optional):
- COLLECT_REMAP_OLD_BASE: old path prefix to replace
- COLLECT_REMAP_NEW_BASE: new path prefix to use
- COLLECT_SUMMARY_REMAP_ENABLE: '1/true/yes' to enable SummaryWriter patch (default: enabled if both OLD/NEW provided)
- LEGGED_GYM_LOGS_ROOT, LEGGED_GYM_DATA_ROOT: explicit overrides for roots (optional)
"""

from __future__ import annotations

import os
import os.path as osp
import json
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict

try:
    import isaacgym  # noqa: F401
except Exception:
    # Allow import in environments without isaacgym; actual run will fail earlier with clear error
    isaacgym = None  # type: ignore

import numpy as np
np.float = float  # compatibility
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import update_cfg_from_args, class_to_dict, update_class_from_dict

from rsl_rl.modules import build_actor_critic
from rsl_rl.runners.dagger_saver import DemonstrationSaver, DaggerSaver


def _as_bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


def deep_remap_paths(obj: Any, old_root: str, new_root: str) -> Any:
    """Recursively replace old_root prefix with new_root in any string values.

    Handles dicts, lists/tuples, and strings. Returns a new object with changes applied.
    """
    if isinstance(obj, str):
        return obj.replace(old_root, new_root)
    if isinstance(obj, dict):
        return {k: deep_remap_paths(v, old_root, new_root) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        mapped = [deep_remap_paths(v, old_root, new_root) for v in obj]
        return type(obj)(mapped)
    return obj


def get_repo_root() -> str:
    return osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))


def resolve_roots(env_cfg) -> tuple[str, str]:
    """Resolve logs_root and data_root from env_cfg.custom, with env and repo fallbacks."""
    repo_root = get_repo_root()
    # Highest priority: explicit env overrides
    env_logs = os.getenv("LEGGED_GYM_LOGS_ROOT")
    env_data = os.getenv("LEGGED_GYM_DATA_ROOT")
    if env_logs and env_data:
        return env_logs, env_data

    # Next: custom on env_cfg if available
    custom = getattr(env_cfg, "custom", None)
    logs_root = getattr(custom, "logs_root", None)
    data_root = getattr(custom, "data_root", None)
    if isinstance(logs_root, str) and isinstance(data_root, str):
        return logs_root, data_root

    # Fallback: repo defaults
    return osp.join(repo_root, "logs"), osp.join(repo_root, "data")


def maybe_patch_summarywriter(remap_old: str | None, remap_new: str | None, enable: bool | None = None) -> None:
    """Optionally patch tensorboardX.SummaryWriter to remap log_dir at runtime."""
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
                print(f"[MonkeyPatch] Redirect SummaryWriter log_dir:\n  from: {log_dir}\n    to: {new_log_dir}")
                log_dir = new_log_dir
            super().__init__(log_dir=log_dir, *args, **kwargs)

    tensorboardX.SummaryWriter = PatchedSummaryWriter  # type: ignore


def main(args):
    RunnerCls = DaggerSaver if args.load_run else DemonstrationSaver
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Roots resolution
    logs_root, data_root = resolve_roots(env_cfg)
    print(f"[COLLECT] logs_root={logs_root}\n[COLLECT] data_root={data_root}")

    # Remap settings (env > CLI defaults)
    remap_old_env = os.getenv("COLLECT_REMAP_OLD_BASE")
    remap_new_env = os.getenv("COLLECT_REMAP_NEW_BASE")
    remap_old = args.remap_old or remap_old_env
    remap_new = args.remap_new or remap_new_env
    summary_patch_env = os.getenv("COLLECT_SUMMARY_REMAP_ENABLE")
    summary_patch = _as_bool(summary_patch_env, default=(bool(remap_old and remap_new)))

    maybe_patch_summarywriter(remap_old, remap_new, enable=summary_patch)

    # Compute training policy paths (for DAgger)
    training_policy_logdir = osp.join(logs_root, train_cfg.runner.experiment_name, args.load_run or "")
    training_policy_log_cfg_path = osp.join(training_policy_logdir, "config.json")

    print(f"[COLLECT] task={args.task}")
    if RunnerCls == DaggerSaver and not args.load_run:
        raise ValueError("--load_run is required for DAgger collection")

    # For DAgger, load training config and optionally remap its paths
    if RunnerCls == DaggerSaver:
        print(f"[COLLECT] Loading config: {training_policy_log_cfg_path}")
        with open(training_policy_log_cfg_path, "r") as f:
            d: Dict[str, Any] = json.load(f, object_pairs_hook=OrderedDict)
        if remap_old and remap_new:
            d = deep_remap_paths(d, remap_old, remap_new)
        update_class_from_dict(env_cfg, d, strict=True)
        update_class_from_dict(train_cfg, d, strict=True)
        # Ensure runner logs_root aligns to resolved roots after potential remap
        try:
            train_cfg.runner.logs_root = logs_root
        except Exception:
            pass

    # Custom sampling tweaks (kept minimal, consistent with original collect.py options)
    # You can edit below to change the number of parallel envs for data diversity.
    env_cfg.env.num_envs = getattr(env_cfg.env, "num_envs", 6)
    env_cfg.terrain.num_rows = getattr(env_cfg.terrain, "num_rows", 2)
    env_cfg.terrain.num_cols = getattr(env_cfg.terrain, "num_cols", 4)

    # Build env and apply CLI overrides to train_cfg
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)

    # Merge config dicts and extract a few knobs
    config = class_to_dict(train_cfg)
    config.update(class_to_dict(env_cfg))
    teacher_act_prob = config["algorithm"].get("teacher_act_prob", 0.0) if args.teacher_prob is None else args.teacher_prob
    action_std = config["policy"].get("init_noise_std", 0.0) if args.action_std is None else args.action_std

    # Build teacher policy
    policy = build_actor_critic(
        env,
        config["algorithm"]["teacher_policy_class_name"],
        config["algorithm"]["teacher_policy"],
        ).to(env.device)

    # Load teacher checkpoint with optional path remap
    teacher_ac_path = config["algorithm"].get("teacher_ac_path")
    if teacher_ac_path:
        if "{LEGGED_GYM_ROOT_DIR}" in teacher_ac_path:
            teacher_ac_path = teacher_ac_path.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        if remap_old and remap_new:
            teacher_ac_path = teacher_ac_path.replace(remap_old, remap_new)
        print(f"[COLLECT] teacher_ac_path: {teacher_ac_path}")
        state = torch.load(teacher_ac_path, map_location="cpu")
        policy.load_state_dict(state["model_state_dict"])

    # Build runner kwargs
    track_header = "".join(env_cfg.terrain.BarrierTrack_kwargs["options"]) if hasattr(env_cfg.terrain, "BarrierTrack_kwargs") else ""

    # Save directory: use data_root and include task-specific suffixes
    save_dir = osp.join(
        data_root,
        datetime.now().strftime("%b%d_%H-%M-%S") + "_" + "".join([
            track_header,
        ] + ([] if RunnerCls == DemonstrationSaver else ["_" + "_".join((args.load_run or "").split("_")[:2])]))
    )

    runner_kwargs = dict(
        env=env,
        policy=policy,
        save_dir=save_dir,
        rollout_storage_length=256,
        min_timesteps=1e9,
        min_episodes=(1e6 if RunnerCls == DaggerSaver else 2e5),
        use_critic_obs=True,
        success_traj_only=False,
        obs_disassemble_mapping=dict(forward_depth="normalized_image"),
        demo_by_sample=config["algorithm"].get("action_labels_from_sample", False),
    )

    if RunnerCls == DaggerSaver:
        runner_kwargs.update(dict(
            training_policy_logdir=training_policy_logdir,
            teacher_act_prob=teacher_act_prob,
            update_times_scale=config["algorithm"].get("update_times_scale", 1e5),
            action_sample_std=action_std,
            log_to_tensorboard=args.log,
        ))

    runner = RunnerCls(**runner_kwargs)
    print("[COLLECT] starting collect_and_save …")
    runner.collect_and_save(config=config)


if __name__ == "__main__":
    args = get_args(
        custom_args=[
            {"name": "--teacher_prob", "type": float, "default": None, "help": "probability of using teacher's action"},
            {"name": "--action_std", "type": float, "default": None, "help": "override action sample std during rollout"},
            {"name": "--log", "action": "store_true", "help": "log the data to tensorboard"},
            {"name": "--remap-old", "type": str, "default": None, "help": "old base path to replace (e.g., /mnt/rpl_project)"},
            {"name": "--remap-new", "type": str, "default": None, "help": "new base path (e.g., /cs/.../network_test)"},
        ],
    )
    main(args)
