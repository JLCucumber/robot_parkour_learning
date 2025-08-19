# context_go2_distill

This file summarizes the minimal context to run and debug `--task go2_distill` across Local (A) and Shared/NFS (C) setups, plus where paths are resolved in code.

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
