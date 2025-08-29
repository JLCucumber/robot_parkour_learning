#!/usr/bin/env bash
# monitor_idle_gpus.sh - 定期监控空闲GPU
set -euo pipefail

INTERVAL=60  # 默认60秒间隔
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_SCRIPT="$SCRIPT_DIR/check_idle_gpus.sh"

usage() {
  cat <<EOF
Usage: $0 [options]
  -i <seconds>  检查间隔 (默认 60 秒)
  -h            帮助
EOF
}

while (( $# )); do
  case "$1" in
    -i) INTERVAL="$2"; shift 2;;
    -h) usage; exit 0;;
    -*) echo "未知选项: $1"; usage; exit 1;;
    *) break;;
  esac
done

if [[ ! -f "$CHECK_SCRIPT" ]]; then
  echo "错误: 找不到检查脚本 $CHECK_SCRIPT" >&2
  exit 1
fi

echo "开始监控空闲GPU，每 ${INTERVAL} 秒检查一次..."
echo "按 Ctrl+C 停止监控"
echo "=" * 50

while true; do
  echo -e "\n$(date '+%Y-%m-%d %H:%M:%S') - GPU空闲检查:"
  
  # 运行检查脚本，只显示结果部分
  "$CHECK_SCRIPT" -q 2>/dev/null || true
  
  echo "下次检查: $(date -d "+${INTERVAL} seconds" '+%H:%M:%S')"
  sleep "$INTERVAL"
done
