# context_go2_distill

This file summarizes the minimal context to run and debug `--task go2_distill` across Local (A) and Shared/NFS (C) setups, plus where paths are resolved in code.

## Quick start (copy-paste)

### A: Local training (no NFS)
```bash

tmux new -s train bash
tmux attach -t train

conda activate isaac_gym_parkour
cd ~/hongbo_li/robot_parkour_learning/legged_gym


echo $LEGGED_GYM_USE_SHARED_PATH
echo $LEGGED_GYM_SHARED_PATH
unset LEGGED_GYM_USE_SHARED_PATH LEGGED_GYM_SHARED_PATH

# Option 1: multi-node training 
export LEGGED_GYM_USE_SHARED_PATH=1
export LEGGED_GYM_SHARED_PATH=/home/data/datasets/robot_parkour_learning # /mnt/rpl_project or /home/data/datasets/robot_parkour_learning

python legged_gym/scripts/train.py --task go2_distill_awbc --headless
python legged_gym/scripts/train.py --task go2_distill_no_awbc --headless

# Option 2: local training (using in-project log dir)
export LEGGED_GYM_USE_SHARED_PATH=0
python legged_gym/scripts/train.py --task go2_distill --headless
```

### Push logs to C (run on A) && Pull data back to A
```bash

tmux new -s sync_ucl bash

# export REMOTE_DIR="user@{server}:/cs/student/projects2/rai/2024/hongboli/network_test/logs"
# export REMOTE_DIR="user@{server}:/cs/student/projects2/rai/2024/hongboli/network_test/data"
# echo a lot of things to test:
echo $DIR_NAME $LOCAL_LOG_DIR  $LOCAL_DATA_DIR  $REMOTE_LOG_DIR  $REMOTE_DATA_DIR
unset DIR_NAME LOCAL_LOG_DIR LOCAL_DATA_DIR REMOTE_LOG_DIR REMOTE_DATA_DIR

export RSYNC_DRYRUN="1"
unset RSYNC_DRYRUN
echo $RSYNC_DRYRUN

cd /home/data/projects/robot_parkour_learning/legged_gym

export DIR_NAME="Aug22_21-46-35_Go2_4skills_fromAug19_18-16-38"  # <<< modify here every time before training !!!
export LEGGED_GYM_SHARED_PATH=/mnt/rpl_project                   #  /home/data/datasets/robot_parkour_learning or /mnt/rpl_project

export TASK_NAME="go2_distill_awbc"                           # go2_distill_no_awbc or go2_distill_awbc

export LOCAL_DATA_DIR="$LEGGED_GYM_SHARED_PATH/data/$TASK_NAME/"  
export LOCAL_LOG_DIR="$LEGGED_GYM_SHARED_PATH/logs/$TASK_NAME/"  

export REMOTE_LOG_DIR="hongboli@ucl-kup:/cs/student/projects2/rai/2024/hongboli/network_test/logs/$TASK_NAME/"   
export REMOTE_DATA_DIR="hongboli@ucl-kup:/cs/student/projects2/rai/2024/hongboli/network_test/data/$TASK_NAME_dagger/"

cd ../network_test
./sync_loop.sh
./sync_log_single.sh
./sync_data_single.sh
```


### C: Collect with remap (paths differ across nodes)
```bash

## Step 1

tmux new -s collect bash
or
tmux attach -t collect 

nvidia-smi
cd /cs/student/projects2/rai/2024/hongboli
mamba activate /cs/student/projects2/rai/2024/hongboli/mamba_envs/isaacgym_parkour

// python test_gui.py

## Step 2

### Option 1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export LD_LIBRARY_PATH=/cs/student/projects2/rai/2024/hongboli/mamba_envs/isaacgym_parkour/lib

### Option 2
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export LD_LIBRARY_PATH=/cs/student/projects2/rai/2024/hongboli/mamba_envs/isaacgym_parkour/lib


## Step 3
cd my_projects/robot_parkour_learning
git ...

cd my_projects/robot_parkour_learning/legged_gym/

# Modify Here !!!
export DIR_NAME="Aug22_21-46-35_Go2_4skills_fromAug19_18-16-38" 
export COLLECT_REMAP_OLD_BASE="/mnt/rpl_project"    # "/mnt/rpl_project" or "/home/data/datasets/robot_parkour_learning" 
export COLLECT_REMAP_NEW_BASE="/cs/student/projects2/rai/2024/hongboli/network_test"
export LEGGED_GYM_LOGS_ROOT="$COLLECT_REMAP_NEW_BASE/logs"
export LEGGED_GYM_DATA_ROOT="$COLLECT_REMAP_NEW_BASE/data"

# Collect Trajectory
python legged_gym/scripts/collect.py --headless --task go2_distill_awbc --log --load_run $DIR_NAME --log

# 
python legged_gym/scripts/train.py --headless --task go2_field 

```




### Shared/NFS on both nodes (no remap)
```bash
export LEGGED_GYM_USE_SHARED_PATH=1
export LEGGED_GYM_SHARED_PATH=/mnt/rpl_project/$USER
python3 legged_gym/legged_gym/scripts/train.py --task go2_distill --headless
python3 legged_gym/legged_gym/scripts/collect.py --task go2_distill --load_run <run_dir> --log
```

## Branch
- Current branch: `ac-net-test`

## Path strategy (shared vs local)
- Toggle shared path via env vars:
  - `LEGGED_GYM_USE_SHARED_PATH`: set to `1/true/yes` to enable shared mode; unset/`0` for local.
  - `LEGGED_GYM_NFS_PATH`: NFS root when shared is enabled. Default: `/mnt/rpl_project`.
- Resolution per module (current code):
  - `legged_gym/envs/go2/go2_config.py`
    - `Go2RoughCfg.custom`: computes
      - if shared: `logs_root = NFS_path/logs`, `data_root = NFS_path/data`
      - else: `logs_root = <repo>/logs`, `data_root = <repo>/data`
    - Module-level `logs_root` is set to `Go2RoughCfg.custom.logs_root` for PPO runner usage.
  - `legged_gym/envs/go2/go2_field_config.py`
    - `Go2FieldCfg.custom` inherits the above behavior; only `name = "field_go2"`.
    - Module-level `logs_root = Go2RoughCfg.custom.logs_root`.
  - `legged_gym/envs/go2/go2_distill_config.py`
    - `Go2DistillCfg.custom` uses the same shared toggles (`LEGGED_GYM_USE_SHARED_PATH`, `LEGGED_GYM_NFS_PATH`) and sets `logs_root/data_root` accordingly.
    - Note: Some references (e.g., `teacher_ac_path`, `load_run`) may still use the module-level `logs_root` defined at top of file. Ensure they point to the desired root for your run.

## Environment variables (quick reference)

Core path controls (training, collect, play):
- LEGGED_GYM_USE_SHARED_PATH (bool): enable shared/NFS mode. Default: off.
- LEGGED_GYM_SHARED_PATH (path): Shared/NFS root when shared is on. Example: /mnt/rpl_project/$USER
- LEGGED_GYM_LOGS_ROOT / LEGGED_GYM_DATA_ROOT: explicit overrides (highest priority).

Compatibility:
- LEGGED_GYM_NFS_PATH is deprecated but still accepted as a fallback if LEGGED_GYM_SHARED_PATH is unset.

Collect cross-node remap (only if paths differ between nodes):
- `COLLECT_REMAP_OLD_BASE` (path): old base prefix to replace, e.g. `/mnt/rpl_project`
- `COLLECT_REMAP_NEW_BASE` (path): new base prefix, e.g. `/cs/.../network_test`
- `COLLECT_SUMMARY_REMAP_ENABLE` (bool): `1/true/yes` to also remap SummaryWriter outputs. Default: auto-on when both bases set.

Network sync scripts (`network_test/sync_*.sh`):
- `REMOTE_DIR` (rsync target/src): e.g. `user@server:/abs/path/on_remote/data/` (note trailing slash)
- `LOCAL_DIR` (local path): e.g. `$(pwd)/legged_gym/data/`
- `PROXY_JUMP` (optional): SSH jump host, e.g. `user@jump-host`; empty for direct SSH
- `RSYNC_DRYRUN` (bool/int): `1` to preview without sending data
- `DIR_NAME` (logs subdir): e.g. `distill_go2` used by `sync_log_single.sh`

## Scripts behavior
- `legged_gym/scripts/train.py`
  - After env creation, if `env_cfg.custom.shared_path == True`, it sets `log_root = os.path.join(env_cfg.custom.logs_root, env_cfg.custom.name)` and passes it to `make_alg_runner`; otherwise uses default local logs.
- `legged_gym/scripts/collect.py`
  - Uses `env_cfg.custom.logs_root` for reading training checkpoints/configs, and `env_cfg.custom.data_root` for saving datasets (Dagger/Demonstration).

## How to run
- Local (no NFS):
  ```bash
  # do not set LEGGED_GYM_USE_SHARED_PATH
  python3 legged_gym/legged_gym/scripts/train.py --task go2_distill --headless
  ```
  - Logs/data go to `<repo>/logs` and `<repo>/data`.

- Shared/NFS:
  ```bash
  export LEGGED_GYM_USE_SHARED_PATH=1
  export LEGGED_GYM_NFS_PATH=/mnt/rpl_project  # change to a writable root, e.g., /mnt/rpl_project/$USER
  python3 legged_gym/legged_gym/scripts/train.py --task go2_distill --headless
  ```
  - Logs/data go to `$LEGGED_GYM_NFS_PATH/logs` and `$LEGGED_GYM_NFS_PATH/data`.

## Common issue: Permission denied on /mnt
If you see `PermissionError: [Errno 13] Permission denied: '/mnt/rpl_project/logs'`:
- Use a writable subdir:
  ```bash
  export LEGGED_GYM_USE_SHARED_PATH=1
  export LEGGED_GYM_NFS_PATH=/mnt/rpl_project/$USER
  ```
- Or have an admin create and chown `/mnt/rpl_project/logs` and `/mnt/rpl_project/data` to your user.
- Alternatively, run in Local mode.

## Network sync (A↔C) snippets
- Scripts: `network_test/sync_data_single.sh`, `sync_log_single.sh`, `sync_loop.sh`
- Overridable env vars:
  - `REMOTE_DIR`, `LOCAL_DIR`, `PROXY_JUMP` (empty for direct ssh), `RSYNC_DRYRUN=1` (dry-run), `DIR_NAME` (log subdir for sync_log_single.sh)
- Example (pull data dry-run):
  ```bash
  export REMOTE_DIR="user@server:/abs/path/on_C/data/"
  export LOCAL_DIR="/abs/path/on_A/data/"
  export PROXY_JUMP=""  # or user@jump-host
  export RSYNC_DRYRUN=1
  ./network_test/sync_data_single.sh
  ```

## Notes / Caveats
- `go2_distill_config.py` may still reference module-level `logs_root` for teacher checkpoints (`teacher_ac_path`) and `load_run`. Adjust those paths (or environment) to ensure they resolve to the intended location.
- If you need the same `LEGGED_GYM_LOGS_ROOT`/`LEGGED_GYM_DATA_ROOT` overrides across all tasks, they can be added consistently later.

## Use cases

### 1) Minimal A↔C loop (Local on A, remap on C)
- Train on A (local paths):
  ```bash
  # On A
  unset LEGGED_GYM_USE_SHARED_PATH
  python3 legged_gym/legged_gym/scripts/train.py --task go2_distill --headless
  # After some steps, note the run dir name printed in logs (e.g., Jul20_16-15-23_...)
  ```
- Push logs to C:
  ```bash
  # On A
  export LOCAL_DIR="$(pwd)/legged_gym/logs/"
  export REMOTE_DIR="user@server:/abs/path/on_C/logs/"
  export DIR_NAME="distill_go2"   # or field_go2, depending on your run
  ./network_test/sync_log_single.sh
  ```
- Collect on C with path remap:
  ```bash
  # On C
  export COLLECT_REMAP_OLD_BASE="/mnt/rpl_project"
  export COLLECT_REMAP_NEW_BASE="/cs/student/projects2/rai/2024/hongboli/network_test"
  # Optionally set explicit roots on C
  # export LEGGED_GYM_LOGS_ROOT=/cs/.../network_test/logs
  # export LEGGED_GYM_DATA_ROOT=/cs/.../network_test/data

  python3 legged_gym/legged_gym/scripts/collect.py \
    --task go2_distill \
    --load_run <run_dir_from_A> \
    --log
  ```
- Pull collected data back to A:
  ```bash
  # On A
  export REMOTE_DIR="user@server:/abs/path/on_C/data/"
  export LOCAL_DIR="$(pwd)/legged_gym/data/"
  ./network_test/sync_data_single.sh
  ```

### 2) Shared/NFS on both nodes (no remap)
- Configure both nodes to the same NFS root and ensure write permission:
  ```bash
  export LEGGED_GYM_USE_SHARED_PATH=1
  export LEGGED_GYM_NFS_PATH=/mnt/rpl_project/$USER  # or another writable NFS subdir
  ```
- Train on A (writes to $NFS/logs and $NFS/data):
  ```bash
  python3 legged_gym/legged_gym/scripts/train.py --task go2_distill --headless
  ```
- Collect on C (reads same logs, writes to same data):
  ```bash
  python3 legged_gym/legged_gym/scripts/collect.py --task go2_distill --load_run <run_dir> --log
  ```
  No remap needed; network sync scripts optional.

### 3) Explicit root overrides (quick hotfix per-node)
```bash
export LEGGED_GYM_LOGS_ROOT=/abs/custom/logs
export LEGGED_GYM_DATA_ROOT=/abs/custom/data
python3 legged_gym/legged_gym/scripts/collect.py --task go2_distill --load_run <run_dir>
```
Works even if shared_path is off; takes highest priority.

### 4) Using the alternative collector (collect_test.py)
- 如果希望与历史 Node C 的“路径重映射”方案完全隔离，可使用：
  ```bash
  python3 legged_gym/legged_gym/scripts/collect_test.py \
    --task go2_distill \
    --load_run <run_dir> \
    --remap-old /mnt/rpl_project \
    --remap-new /cs/student/projects2/rai/2024/hongboli/network_test \
    --log
  ```
- 功能与 collect.py 中的 remap 开关等价；长期建议直接使用 collect.py。



## Tensorboard:
tensorboard --logdir /mnt/rpl_project/logs/go2_distill_awbc/Aug22_00-56-05_Go2_4skills_fromAug19_18-16-38/

/mnt/rpl_project/logs/go2_distill_awbc/Aug22_00-56-05_Go2_4skills_fromAug19_18-16-38