#!/bin/bash

##########################################
# UCL Sync Script via ProxyJump (template)
# Pull data from trailbreaker.cs.ucl.ac.uk
# through knuckles.cs.ucl.ac.uk via SSH
#
# Fill in the source and destination below
##########################################

# ==== 用户自定义部分（可被环境变量覆盖）====
# 远端（通常是 C 端）数据目录
REMOTE_DIR=${REMOTE_DIR:-"hongboli@trailbreaker.cs.ucl.ac.uk:/cs/student/projects2/rai/2024/hongboli/network_test/data/"}
# 本机（通常是 A 端）保存路径
LOCAL_DIR=${LOCAL_DIR:-"/mnt/rpl_project/data/"}

# 日志文件路径（可选）
# LOG_FILE="${HOME}/sync_trailbreaker.log"

# 跳板设置（留空表示直连）
PROXY_JUMP=${PROXY_JUMP:-"hongboli@knuckles.cs.ucl.ac.uk"}

# Dry-run 开关（非空启用）例如： export RSYNC_DRYRUN=1
RSYNC_DRYRUN=${RSYNC_DRYRUN:-""}

# ==== 不建议修改的部分 ====

echo "[$(date)] 开始同步..." #>> "$LOG_FILE"

# 执行同步
SSH_OPT="ssh"
if [ -n "$PROXY_JUMP" ]; then
  SSH_OPT="ssh -J $PROXY_JUMP"
fi

DRYRUN_OPT=""
if [ -n "$RSYNC_DRYRUN" ]; then
  DRYRUN_OPT="--dry-run"
fi

rsync -avP --stats --info=progress2 $DRYRUN_OPT \
  --inplace --whole-file --no-compress --timeout=60 \
  --include="*/" --include="*.pkl" --exclude="*.tmp" --include="*" \
  -e "$SSH_OPT" \
  "${REMOTE_DIR}" "$LOCAL_DIR" \
  #>> "$LOG_FILE" 2>&1


# 同步结果
if [ $? -eq 0 ]; then
    echo "[$(date)] 同步成功" #>> "$LOG_FILE"
else
    echo "[$(date)] 同步失败" #>> "$LOG_FILE"
fi

# rsync -avz --inplace --whole-file --no-compress --timeout=60 \
#   --include="*/" --include="*.pkl" --exclude="*.tmp" --include="*" \
#   -e "ssh -J hongboli@knuckles.cs.ucl.ac.uk" \
#   hongboli@trailbreaker.cs.ucl.ac.uk:/cs/student/projects2/rai/2024/hongboli/network_test/data/Jul13_03-56-49_jumphurdledowntilted_rampstairsupstairsdownslopewave_blockLength2.4_teacherProb0.0_randOrder_fric0.0-2.0_aStd0.10_Jul12_00-34-50 \
#   /mnt/rpl_project/data/Jul13_03-56-49_jumphurdledowntilted_rampstairsupstairsdownslopewave_blockLength2.4_teacherProb0.0_randOrder_fric0.0-2.0_aStd0.10_Jul12_00-34-50
#   >> "$LOG_FILE" 2>&1
# echo "[$(date)] 同步完成" #>> "$LOG_FILE"
# ssh -L 8081:trailbreaker.cs.ucl.ac.uk:8443 hongboli@knuckles.cs.ucl.ac.uk
