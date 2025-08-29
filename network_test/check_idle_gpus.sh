#!/usr/bin/env bash
# check_idle_gpus.sh
set -euo pipefail

THRESHOLD=500
# 3090 专用空闲判断阈值（固定 3000MB，不随 -t 改变）
THRESHOLD_3090=7000
PARALLEL=8
SSH_CONFIG="${SSH_CONFIG:-$HOME/.ssh/config}"
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=6 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
CMD_TIMEOUT=10
MODE="any"
OUTPUT_TSV=""
QUIET=0
SHOW_BUSY=0
SHOW_HOSTS=0
LIST_HOSTS_ONLY=0

usage() {
  cat <<EOF
Usage: $0 [options]
  -t <MB>       阈值 (默认 500)
  -p <N>        并发数 (默认 8)
  -m any|all    any=含至少一块 <阈值 GPU; all=全部 GPU <阈值
  -o <file>     输出 TSV
  -q            静默(仅最终匹配摘要)
  --show-busy   最终再输出一遍忙碌主机列表
  --show-hosts  开始检测前列出本次要检查的主机列表
  --list-hosts  仅列出主机列表后退出（不做检测）
  -h            帮助

说明:
  * 对于 3090 显卡，会自动使用固定阈值 ${THRESHOLD_3090}MB 来判定是否空闲；
    也就是说：只要 3090 的已用显存 < ${THRESHOLD_3090}MB 即视为空闲（忽略 -t 的设置）。
  * 其它型号仍使用 -t 指定的 THRESHOLD。
EOF
}

# 解析参数（支持长参数）
while (( $# )); do
  case "$1" in
    -t) THRESHOLD="$2"; shift 2;;
    -p) PARALLEL="$2"; shift 2;;
    -m) MODE="$2"; shift 2;;
    -o) OUTPUT_TSV="$2"; shift 2;;
    -q) QUIET=1; shift;;
    --show-busy) SHOW_BUSY=1; shift;;
  --show-hosts) SHOW_HOSTS=1; shift;;
  --list-hosts) LIST_HOSTS_ONLY=1; shift;;
    -h) usage; exit 0;;
    --) shift; break;;
    -*) echo "未知选项: $1"; usage; exit 1;;
    *) break;;
  esac
done

if [[ ! -f "$SSH_CONFIG" ]]; then
  echo "找不到 SSH config: $SSH_CONFIG" >&2
  exit 1
fi
if [[ "$MODE" != "any" && "$MODE" != "all" ]]; then
  echo "MODE 必须为 any 或 all" >&2
  exit 1
fi

# 颜色
GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"; RESET="\033[0m"

# 抽取主机
mapfile -t HOSTS < <(
  awk '/^Host[ \t]+ucl-lab-[a-z0-9-]+-(3090|4070)([ \t]|$)/{
    for(i=2;i<=NF;i++){
      if($i~/^ucl-lab-[a-z0-9-]+-(3090|4070)$/) print $i
    }
  }' "$SSH_CONFIG" | sort -u
)

[[ ${#HOSTS[@]} -eq 0 ]] && { echo "未匹配到主机"; exit 0; }

if (( LIST_HOSTS_ONLY == 1 )); then
  echo "匹配到 ${#HOSTS[@]} 个主机:"
  printf '%s\n' "${HOSTS[@]}"
  exit 0
fi

if (( QUIET == 0 )); then
  echo -e "${CYAN}共发现 ${#HOSTS[@]} 个主机，阈值 < ${THRESHOLD}MB，模式=${MODE}，开始检测...${RESET}"
  echo -e "${CYAN}即将连接 ${#HOSTS[@]} 台主机 (最大并发 ${PARALLEL})${RESET}"
  if (( SHOW_HOSTS == 1 )); then
    echo -e "${CYAN}主机列表:${RESET}"
    printf '%s\n' "${HOSTS[@]}"
    echo
  fi
fi

# 并发控制 FIFO
tmpfifo=$(mktemp -u)
mkfifo "$tmpfifo"
exec 9<>"$tmpfifo"
rm -f "$tmpfifo"
for ((i=0;i<PARALLEL;i++)); do echo >&9; done

# 临时目录存储结果
WORKDIR=$(mktemp -d -t check_idle_gpu_XXXX)
trap 'rm -rf "$WORKDIR"' EXIT

MATCHED_FILE="$WORKDIR/matched.list"
BUSY_FILE="$WORKDIR/busy.list"
WARN_FILE="$WORKDIR/warn.list"
FAIL_FILE="$WORKDIR/fail.list"
TSV_BUFFER_FILE="$WORKDIR/results.tsv"
echo -e "host\tmode\tgpu_index\tmem_used_MB\tmem_total_MB\tname" > "$TSV_BUFFER_FILE"

check_host() {
  local host="$1"
  local raw
  raw=$(timeout "$CMD_TIMEOUT" ssh $SSH_OPTS "$host" \
    "nvidia-smi --query-gpu=index,memory.used,memory.total,name --format=csv,noheader,nounits" 2>&1 || true)

  # 分类错误
  if grep -qiE "not found|No such file|command not found" <<<"$raw"; then
    (( QUIET == 0 )) && echo -e "${YELLOW}[WARN] $host: nvidia-smi 不存在${RESET}"
    echo "$host" >> "$WARN_FILE"
    return
  fi
  if [[ -z "$raw" ]]; then
    (( QUIET == 0 )) && echo -e "${RED}[FAIL] $host: 无输出(超时/无法连接)${RESET}"
    echo "$host" >> "$FAIL_FILE"
    return
  fi
  if grep -qiE "Permission denied|Connection timed out|Could not resolve|No route to host|Connection closed" <<<"$raw"; then
    (( QUIET == 0 )) && echo -e "${RED}[FAIL] $host: 连接错误${RESET}"
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
    # 针对 3090 使用单独阈值
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
        (( QUIET == 0 )) && echo -e "${GREEN}[OK] $host GPU$gi ${gu}MB/${gt}MB <${per_gpu_threshold_disp} (${gn})${RESET}"
        printf "%s\tany\t%s\t%s\t%s\t%s\n" "$host" "$gi" "$gu" "$gt" "$gn" >> "$TSV_BUFFER_FILE"
      done
      echo "$host" >> "$MATCHED_FILE"
    else
  (( QUIET == 0 )) && echo -e "${RED}[BUSY] $host: 无符合空闲阈值 GPU (普通 <${THRESHOLD}MB, 3090 <${THRESHOLD_3090}MB)${RESET}"
      echo "$host" >> "$BUSY_FILE"
    fi
  else
    if (( all_idle == 1 )) && (( ${#idle_lines[@]} > 0 )); then
      IFS=',' read -r gi gu gt gn <<<"${idle_lines[0]}"
      (( QUIET == 0 )) && echo -e "${GREEN}[ALL-IDLE] $host 全部 GPU 低于其各自阈值 (示例 GPU$gi ${gu}MB/${gt}MB)${RESET}"
      for l in "${idle_lines[@]}"; do
        IFS=',' read -r gi gu gt gn <<<"$l"
        local per_gpu_threshold_disp="$THRESHOLD"
        if grep -qi '3090' <<<"$gn" || grep -q -- '-3090' <<<"$host"; then
          per_gpu_threshold_disp="$THRESHOLD_3090"
        fi
        printf "%s\tall\t%s\t%s\t%s\t%s\n" "$host" "$gi" "$gu" "$gt" "$gn" >> "$TSV_BUFFER_FILE"
      done
      echo "$host" >> "$MATCHED_FILE"
    else
  (( QUIET == 0 )) && echo -e "${RED}[NOT-IDLE] $host: 至少一块 GPU 超过其空闲阈值 (普通阈值 ${THRESHOLD}MB, 3090 阈值 ${THRESHOLD_3090}MB)${RESET}"
      echo "$host" >> "$BUSY_FILE"
    fi
  fi
}

for h in "${HOSTS[@]}"; do
  read -u 9
  {
    check_host "$h"
    echo >&9
  } &
done
wait
exec 9>&-

echo
if [[ -s "$MATCHED_FILE" ]]; then
  echo -e "${CYAN}符合条件主机 (${MODE}, 阈值 ${THRESHOLD}MB):$(wc -l < "$MATCHED_FILE") 个${RESET}"
  sort "$MATCHED_FILE"
else
  echo -e "${RED}没有匹配到符合条件主机 (${MODE}, 阈值 ${THRESHOLD}MB).${RESET}"
fi

# 摘要统计
total=${#HOSTS[@]}
matched=$( [[ -s "$MATCHED_FILE" ]] && wc -l < "$MATCHED_FILE" || echo 0 )
busy=$( [[ -s "$BUSY_FILE" ]] && wc -l < "$BUSY_FILE" || echo 0 )
warn=$( [[ -s "$WARN_FILE" ]] && wc -l < "$WARN_FILE" || echo 0 )
fail=$( [[ -s "$FAIL_FILE" ]] && wc -l < "$FAIL_FILE" || echo 0 )

echo -e "${CYAN}摘要: total=$total matched=$matched busy=$busy warn=$warn fail=$fail${RESET}"

if (( SHOW_BUSY == 1 )) && [[ -s "$BUSY_FILE" ]]; then
  echo -e "${YELLOW}忙碌主机列表:${RESET}"
  sort "$BUSY_FILE"
fi

if [[ -n "$OUTPUT_TSV" ]]; then
  cp "$TSV_BUFFER_FILE" "$OUTPUT_TSV"
  echo -e "${CYAN}已写 TSV 到 $OUTPUT_TSV${RESET}"
fi