#!/usr/bin/env bash
# rpl.sh - One-liner wrapper for training, collecting, and syncing using profiles
# Usage examples:
#   ./network_test/rpl.sh use A.local        # load Node A local profile
#   ./network_test/rpl.sh use C.remap        # load Node C remap profile
#   ./network_test/rpl.sh train go2_distill  # run training
#   ./network_test/rpl.sh collect go2_distill <run_dir> [--log]
#   ./network_test/rpl.sh push-logs distill_go2
#   ./network_test/rpl.sh pull-data

set -euo pipefail
SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SELF_DIR/.." && pwd)

color() { case "$1" in
  green) echo -e "\033[32m$2\033[0m";;
  yellow) echo -e "\033[33m$2\033[0m";;
  red) echo -e "\033[31m$2\033[0m";;
  *) echo "$2";; esac }

cmd=${1:-help}
shift || true

case "$cmd" in
  use)
    profile=${1:-}
    if [[ -z "$profile" ]]; then color red "Profile name required"; exit 1; fi
    file="$ROOT_DIR/profiles/$profile.env"
    if [[ ! -f "$file" ]]; then color red "Profile not found: $file"; exit 1; fi
    # shellcheck disable=SC1090
    source "$file"
    color green "Loaded profile: $profile"
    ;;

  train)
    task=${1:-go2_distill}
    color yellow "Training: $task"
    python3 "$ROOT_DIR/legged_gym/legged_gym/scripts/train.py" --task "$task" --headless
    ;;

  collect)
    task=${1:-go2_distill}; shift || true
    run_dir=${1:-}; shift || true
    if [[ -z "$run_dir" ]]; then color red "run_dir required"; exit 1; fi
    color yellow "Collect: $task, run=$run_dir"
    python3 "$ROOT_DIR/legged_gym/legged_gym/scripts/collect.py" --task "$task" --load_run "$run_dir" "$@"
    ;;

  push-logs)
    dir=${1:-distill_go2}
    color yellow "Push logs dir: $dir"
    REMOTE_DIR=${SYNC_REMOTE_LOGS_ROOT:+${SYNC_REMOTE}:${SYNC_REMOTE_LOGS_ROOT}/}
    LOCAL_DIR=${SYNC_LOCAL_LOGS_ROOT:-$ROOT_DIR/legged_gym/logs/}
    DIR_NAME="$dir" RSYNC_DRYRUN=${RSYNC_DRYRUN:-0} "$SELF_DIR/sync_log_single.sh"
    ;;

  pull-data)
    color yellow "Pull data"
    REMOTE_DIR=${SYNC_REMOTE_DATA_ROOT:+${SYNC_REMOTE}:${SYNC_REMOTE_DATA_ROOT}/}
    LOCAL_DIR=${SYNC_LOCAL_DATA_ROOT:-$ROOT_DIR/legged_gym/data/}
    RSYNC_DRYRUN=${RSYNC_DRYRUN:-0} "$SELF_DIR/sync_data_single.sh"
    ;;

  *)
    cat <<EOF
$(color green "rpl.sh - profiles + one-liners")
Commands:
  use <Profile>            Load env profile (files in profiles/*.env)
  train [task]             Train task (default: go2_distill)
  collect [task] <run>     Collect DAgger data for run dir
  push-logs [dir]          Push logs (default dir: distill_go2)
  pull-data                Pull data

Profiles:
  A.local     - Local only
  C.remap     - Node C path remap /mnt->/cs

Tips:
  - Set SYNC_REMOTE/REMOTE_ROOT vars in the profile before push/pull
  - RSYNC_DRYRUN=1 to preview syncing
EOF
    ;;
 esac
