#!/usr/bin/env bash
set -euo pipefail

# ==== 用户自定义部分（可被环境变量覆盖）====

# 期望的日志 run 目录名（例如：Aug19_21-47-53_Go2_...）。若未设置，则允许直接用 *_LOG_DIR/REMOTE_DIR。
DIR_NAME="Aug19_22-11-00_Go2_9skills_fromJul20_16-15-23"

# 兼容你在 MD 里用的变量名（可指向 logs 根，或直接指到具体 run 目录）
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-}"

# 仍保持对 LOCAL_DIR/REMOTE_DIR 的支持（若已显式给出则优先）
LOCAL_DIR_DEFAULT=""
REMOTE_DIR_DEFAULT=""

# 本机（A）日志目录计算
if [[ -n "${LOCAL_DIR:-}" ]]; then
  LOCAL_DIR_DEFAULT="$LOCAL_DIR"
elif [[ -n "$LOCAL_LOG_DIR" && -n "$DIR_NAME" ]]; then
  LOCAL_DIR_DEFAULT="${LOCAL_LOG_DIR%/}/$DIR_NAME/"
elif [[ -n "$LOCAL_LOG_DIR" ]]; then
  LOCAL_DIR_DEFAULT="${LOCAL_LOG_DIR%/}/"
else
  # 最后兜底（不推荐，便于保留旧行为）
  LOCAL_DIR_DEFAULT="/mnt/rpl_project/logs/distill_go2/${DIR_NAME}/"
fi

# 远端（C）日志目录计算
if [[ -n "${REMOTE_DIR:-}" ]]; then
  REMOTE_DIR_DEFAULT="$REMOTE_DIR"
elif [[ -n "$REMOTE_LOG_DIR" && -n "$DIR_NAME" ]]; then
  REMOTE_DIR_DEFAULT="${REMOTE_LOG_DIR%/}/$DIR_NAME/"
elif [[ -n "$REMOTE_LOG_DIR" ]]; then
  REMOTE_DIR_DEFAULT="${REMOTE_LOG_DIR%/}/"
else
  REMOTE_DIR_DEFAULT="hongboli@beachcomber.cs.ucl.ac.uk:/cs/student/projects2/rai/2024/hongboli/network_test/logs/distill_go2/${DIR_NAME}/"
fi

# 跳板（留空表示使用 ~/.ssh/config 自动跳板）
PROXY_JUMP="${PROXY_JUMP:-}"

# Dry-run 开关（非空启用）
RSYNC_DRYRUN="${RSYNC_DRYRUN:-}"

# ==== 不建议修改的部分 ====

echo "[$(date)] 开始同步日志..."
echo "[DEBUG] LOCAL_DIR=${LOCAL_DIR_DEFAULT}"
echo "[DEBUG] REMOTE_DIR=${REMOTE_DIR_DEFAULT}"

SSH_OPT="ssh"
if [[ -n "$PROXY_JUMP" ]]; then
  SSH_OPT="ssh -J $PROXY_JUMP"
fi

DRYRUN_OPT=""
if [[ -n "$RSYNC_DRYRUN" ]]; then
  DRYRUN_OPT="--dry-run"
fi

# 校验本地源目录是否存在
if [[ ! -d "$LOCAL_DIR_DEFAULT" ]]; then
  echo "[ERROR] 本地日志目录不存在: $LOCAL_DIR_DEFAULT"
  echo "        请确认 DIR_NAME（应为具体 run 目录名）或 LOCAL_LOG_DIR/LOCAL_DIR 设置是否正确。"
  exit 1
fi

# 推送日志（本机 A -> 远端 C）
rsync -avP $DRYRUN_OPT -e "$SSH_OPT" "${LOCAL_DIR_DEFAULT%/}/" "$REMOTE_DIR_DEFAULT"

if [[ $? -eq 0 ]]; then
  echo "[$(date)] 同步完成"
else
  echo "[$(date)] 同步失败"
  exit 1
fi