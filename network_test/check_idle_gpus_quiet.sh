#!/usr/bin/env bash
# check_idle_gpus_quiet.sh - 静默版GPU检查，只显示结果
set -euo pipefail

THRESHOLD=500
THRESHOLD_3090=7000
PARALLEL=8
SSH_CONFIG="${SSH_CONFIG:-$HOME/.ssh/config}"
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=6 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
CMD_TIMEOUT=10
MODE="any"

# 解析参数
while (( $# )); do
  case "$1" in
    -t) THRESHOLD="$2"; shift 2;;
    -m) MODE="$2"; shift 2;;
    *) shift;;
  esac
done

# 抽取主机
mapfile -t HOSTS < <(
  awk '/^Host[ \t]+ucl-lab-[a-z0-9-]+-(3090|4070)([ \t]|$)/{
    for(i=2;i<=NF;i++){
      if($i~/^ucl-lab-[a-z0-9-]+-(3090|4070)$/) print $i
    }
  }' "$SSH_CONFIG" | sort -u
)

[[ ${#HOSTS[@]} -eq 0 ]] && { echo "未匹配到主机"; exit 0; }

# 并发控制
tmpfifo=$(mktemp -u)
mkfifo "$tmpfifo"
exec 9<>"$tmpfifo"
rm -f "$tmpfifo"
for ((i=0;i<PARALLEL;i++)); do echo >&9; done

# 临时目录
WORKDIR=$(mktemp -d -t check_idle_gpu_XXXX)
trap 'rm -rf "$WORKDIR"' EXIT

MATCHED_FILE="$WORKDIR/matched.list"
BUSY_FILE="$WORKDIR/busy.list"
WARN_FILE="$WORKDIR/warn.list"
FAIL_FILE="$WORKDIR/fail.list"
RESULTS_FILE="$WORKDIR/results.txt"

check_host() {
  local host="$1"
  local raw
  raw=$(timeout "$CMD_TIMEOUT" ssh $SSH_OPTS "$host" \
    "nvidia-smi --query-gpu=index,memory.used,memory.total,name --format=csv,noheader,nounits" 2>&1 || true)

  # 错误处理（静默）
  if grep -qiE "not found|No such file|command not found" <<<"$raw"; then
    echo "$host" >> "$WARN_FILE"
    return
  fi
  if [[ -z "$raw" ]] || grep -qiE "Permission denied|Connection timed out|Could not resolve|No route to host|Connection closed" <<<"$raw"; then
    echo "$host" >> "$FAIL_FILE"
    return
  fi

  local idle_lines=()
  local all_idle=1
  while IFS=',' read -r idx used total name_rest; do
    idx="${idx//[[:space:]]/}"
    used="${used//[[:space:]]/}"
    total="${total//[[:space:]]/}"
    local name
    name=$(echo "$name_rest" | sed 's/^ *//')
    [[ "$used" =~ ^[0-9]+$ ]] || continue
    
    local per_gpu_threshold="$THRESHOLD"
    if grep -qi '3090' <<<"$name" || grep -q -- '-3090' <<<"$host"; then
      per_gpu_threshold="$THRESHOLD_3090"
    fi

    if (( used < per_gpu_threshold )); then
      idle_lines+=("$idx,$used,$total,$name")
    else
      all_idle=0
    fi
  done <<<"$raw"

  if [[ "$MODE" == "any" ]]; then
    if (( ${#idle_lines[@]} > 0 )); then
      for l in "${idle_lines[@]}"; do
        IFS=',' read -r gi gu gt gn <<<"$l"
        local per_gpu_threshold_disp="$THRESHOLD"
        if grep -qi '3090' <<<"$gn" || grep -q -- '-3090' <<<"$host"; then
          per_gpu_threshold_disp="$THRESHOLD_3090"
        fi
        echo "$host GPU$gi ${gu}MB/${gt}MB <${per_gpu_threshold_disp} (${gn})" >> "$RESULTS_FILE"
      done
      echo "$host" >> "$MATCHED_FILE"
    else
      echo "$host" >> "$BUSY_FILE"
    fi
  else
    if (( all_idle == 1 )) && (( ${#idle_lines[@]} > 0 )); then
      echo "$host (全部GPU空闲)" >> "$RESULTS_FILE"
      echo "$host" >> "$MATCHED_FILE"
    else
      echo "$host" >> "$BUSY_FILE"
    fi
  fi
}

# 执行检查
for h in "${HOSTS[@]}"; do
  read -u 9
  {
    check_host "$h"
    echo >&9
  } &
done
wait
exec 9>&-

# 显示结果
if [[ -s "$RESULTS_FILE" ]]; then
  echo "空闲GPU:"
  sort "$RESULTS_FILE" | while read line; do
    echo "  ✓ $line"
  done
else
  echo "无空闲GPU"
fi

# 简洁摘要
total=${#HOSTS[@]}
matched=$( [[ -s "$MATCHED_FILE" ]] && wc -l < "$MATCHED_FILE" || echo 0 )
busy=$( [[ -s "$BUSY_FILE" ]] && wc -l < "$BUSY_FILE" || echo 0 )
warn=$( [[ -s "$WARN_FILE" ]] && wc -l < "$WARN_FILE" || echo 0 )
fail=$( [[ -s "$FAIL_FILE" ]] && wc -l < "$FAIL_FILE" || echo 0 )

echo "总计: $total 台 | 空闲: $matched | 忙碌: $busy | 警告: $warn | 失败: $fail"
