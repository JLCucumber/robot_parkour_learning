#!/usr/bin/env bash
# list_active_machines.sh - 列出当前连接的机器
set -euo pipefail

echo "=== 当前活跃的远程机器连接 ==="
echo

# 从SSH进程中提取连接的机器名
echo "已连接的机器:"
ps aux | grep 'ssh -v -T -D' | grep -v grep | awk '{
    for(i=1;i<=NF;i++) {
        if($i ~ /ucl-lab-.*-(3090|4070)$/) {
            split($i, parts, "-")
            machine = $i
            break
        }
        if($i ~ /G[0-9]+-4090$/) {
            machine = $i
            break
        }
    }
    if(machine) {
        for(j=1;j<=NF;j++) {
            if($j == "-D" && $(j+1) ~ /^[0-9]+$/) {
                port = $(j+1)
                break
            }
        }
        printf "  ✓ %s (隧道端口: %s)\n", machine, port
        machine = ""
        port = ""
    }
}' | sort

echo

# 统计连接数
total_machines=$(ps aux | grep -E 'ssh.*-D.*(ucl-lab-|G[0-9]+-4090)' | grep -v grep | wc -l)
ucl_lab_count=$(ps aux | grep -E 'ssh.*-D.*ucl-lab-' | grep -v grep | wc -l)
g_series_count=$(ps aux | grep -E 'ssh.*-D.*G[0-9]+-4090' | grep -v grep | wc -l)

echo "连接摘要:"
echo "  总连接数: $total_machines"
echo "  UCL Lab 机器: $ucl_lab_count"
echo "  G系列机器: $g_series_count"
echo

# 显示网络连接统计
ssh_connections=$(ss -tn | grep ':22' | grep 'ESTAB' | wc -l)
echo "网络连接:"
echo "  活跃SSH连接数: $ssh_connections"
