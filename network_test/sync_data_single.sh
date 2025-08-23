#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
# ==== 用户自定义部分（可被环境变量覆盖）====

# 兼容你在 MD 里用的变量名（可指向 data 根，或直接指到具体目录）
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-}"

# 如果你直接给了 LOCAL_DIR/REMOTE_DIR，也支持
LOCAL_DIR_DEFAULT=""
REMOTE_DIR_DEFAULT=""

# 远端（C）数据目录作为源
# if [[ -n "${REMOTE_DIR:-}" ]]; then
#   REMOTE_DIR_DEFAULT="$REMOTE_DIR"
# el

if [[ -n "$REMOTE_DATA_DIR" ]]; then
  REMOTE_DIR_DEFAULT="${REMOTE_DATA_DIR%/}"
else
  echo -e "${RED}[ERROR] Remote data directory is not set${NC}"
  exit 1
fi

# 本机（A）保存路径作为目标
if [[ -n "${LOCAL_DIR:-}" ]]; then
  LOCAL_DIR_DEFAULT="$LOCAL_DIR"
elif [[ -n "$LOCAL_DATA_DIR" ]]; then
  LOCAL_DIR_DEFAULT="${LOCAL_DATA_DIR%/}/"
else
  # raise error
  echo -e "${RED}[ERROR] Local data directory is not set${NC}"
  exit 1
fi

PROXY_JUMP="${PROXY_JUMP:-}"
RSYNC_DRYRUN="${RSYNC_DRYRUN:-}"

# ==== 不建议修改的部分 ====

echo -e "${BLUE}[$(date)] Start Syncing Data...${NC}"
echo "[DEBUG] REMOTE_DIR=${REMOTE_DIR_DEFAULT}"
echo "[DEBUG] LOCAL_DIR=${LOCAL_DIR_DEFAULT}"

SSH_OPT="ssh"
if [[ -n "$PROXY_JUMP" ]]; then
  SSH_OPT="ssh -J $PROXY_JUMP"
fi

DRYRUN_OPT=""
if [[ -n "$RSYNC_DRYRUN" ]]; then
  DRYRUN_OPT="--dry-run"
fi

# 确保本地目标存在
mkdir -p "$LOCAL_DIR_DEFAULT"

echo -e "${BLUE}[SYNC] Pulling data from C → A...${NC}"
rsync -av --stats $DRYRUN_OPT \
  --inplace --whole-file --no-compress --timeout=60 \
  --include="*/" --include="*.pkl" --exclude="*.tmp" --include="*" \
  -e "$SSH_OPT" \
  "${REMOTE_DIR_DEFAULT}" "$LOCAL_DIR_DEFAULT" | \
  grep -E "(Number of files|Number of regular files transferred|Total file size|sent.*received|speedup)" || true


if [[ $? -eq 0 ]]; then
  echo -e "${GREEN}[$(date)] ✅ Data Sync Completed${NC}"
else
  echo -e "${RED}[$(date)] ❌ Data Sync Failed${NC}"
  exit 1
fi


# rsync -avz --inplace --whole-file --no-compress --timeout=60 \
#   --include="*/" --include="*.pkl" --exclude="*.tmp" --include="*" \
#   -e "ssh -J hongboli@knuckles.cs.ucl.ac.uk" \
#   hongboli@trailbreaker.cs.ucl.ac.uk:/cs/student/projects2/rai/2024/hongboli/network_test/data/Jul13_03-56-49_jumphurdledowntilted_rampstairsupstairsdownslopewave_blockLength2.4_teacherProb0.0_randOrder_fric0.0-2.0_aStd0.10_Jul12_00-34-50 \
#   /mnt/rpl_project/data/Jul13_03-56-49_jumphurdledowntilted_rampstairsupstairsdownslopewave_blockLength2.4_teacherProb0.0_randOrder_fric0.0-2.0_aStd0.10_Jul12_00-34-50
#   >> "$LOG_FILE" 2>&1
# echo "[$(date)] 同步完成" #>> "$LOG_FILE"
# ssh -L 8081:trailbreaker.cs.ucl.ac.uk:8443 hongboli@knuckles.cs.ucl.ac.uk
