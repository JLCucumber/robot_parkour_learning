#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
# ==== 用户自定义部分（可被环境变量覆盖）====

# 期望的日志 run 目录名（例如：Aug19_21-47-53_Go2_...）。若未设置，则允许直接用 *_LOG_DIR/REMOTE_DIR。

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
  LOCAL_DIR_DEFAULT="${LOCAL_LOG_DIR%/}"
else
  # raise error
  echo -e "${RED}[ERROR] Local log directory is not set${NC}"
  exit 1
fi

# 远端（C）日志目录计算
if [[ -n "${REMOTE_DIR:-}" ]]; then
  REMOTE_DIR_DEFAULT="$REMOTE_DIR"
elif [[ -n "$REMOTE_LOG_DIR" && -n "$DIR_NAME" ]]; then
  REMOTE_DIR_DEFAULT="${REMOTE_LOG_DIR%/}/$DIR_NAME/"
elif [[ -n "$REMOTE_LOG_DIR" ]]; then
  REMOTE_DIR_DEFAULT="${REMOTE_LOG_DIR%/}/"
else
  echo -e "${RED}[ERROR] Remote log directory is not set${NC}"
  exit 1
fi

# 跳板（留空表示使用 ~/.ssh/config 自动跳板）
PROXY_JUMP="${PROXY_JUMP:-}"

# Dry-run 开关（非空启用）
RSYNC_DRYRUN="${RSYNC_DRYRUN:-}"

# ==== 不建议修改的部分 ====

echo -e "${BLUE}[$(date)] Start Syncing Logs...${NC}"
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
  echo -e "${RED}[ERROR] 本地日志目录不存在: $LOCAL_DIR_DEFAULT${NC}"
  echo -e "${YELLOW}        请确认 DIR_NAME（应为具体 run 目录名）或 LOCAL_LOG_DIR/LOCAL_DIR 设置是否正确。${NC}"
  exit 1
fi

# 推送日志（本机 A -> 远端 C）
echo -e "${BLUE}[SYNC] Pushing logs to remote ...${NC}"
rsync -av --stats $DRYRUN_OPT -e "$SSH_OPT" "${LOCAL_DIR_DEFAULT%/}/" "$REMOTE_DIR_DEFAULT" | \
  grep -E "(Number of files|Number of regular files transferred|Total file size|sent.*received|speedup)" || true

if [[ $? -eq 0 ]]; then
  echo -e "${GREEN}[$(date)] ✅ Log Sync Completed${NC}"
else
  echo -e "${RED}[$(date)] ❌ Log Sync Failed${NC}"
  exit 1
fi