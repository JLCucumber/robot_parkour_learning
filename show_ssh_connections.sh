#!/usr/bin/env bash
# show_ssh_connections.sh
set -euo pipefail

echo "=== 活跃的 SSH 连接信息 ==="
echo

# 1. 显示已建立的SSH连接
echo "1. 已建立的SSH连接 (端口22):"
if command -v ss >/dev/null 2>&1; then
    ssh_lines=$(ss -tn | grep ':22' | grep 'ESTAB')
    if [[ -n "$ssh_lines" ]]; then
        echo "$ssh_lines" | while read line; do
            echo "   $line"
        done
    else
        echo "   无活跃SSH连接"
    fi
else
    ssh_lines=$(netstat -tn | grep ':22' | grep 'ESTABLISHED')
    if [[ -n "$ssh_lines" ]]; then
        echo "$ssh_lines" | while read line; do
            echo "   $line"
        done
    else
        echo "   无活跃SSH连接"
    fi
fi

# 计算连接数
ssh_count=$(ss -tn 2>/dev/null | grep ':22' | grep 'ESTAB' | wc -l || netstat -tn 2>/dev/null | grep ':22' | grep 'ESTABLISHED' | wc -l)
echo "   总计: $ssh_count 个SSH连接"
echo

# 2. 显示SSH客户端进程
echo "2. SSH客户端进程:"
ps aux | grep -E 'ssh\s+.*@|ssh\s+-' | grep -v grep | while read line; do
    echo "   $line"
done
echo

# 3. 显示SSH ControlMaster连接（如果有）
echo "3. SSH ControlMaster 套接字:"
if [[ -d ~/.ssh ]]; then
    find ~/.ssh -name "*control*" -o -name "*master*" 2>/dev/null | while read socket; do
        if [[ -S "$socket" ]]; then
            echo "   $socket (活跃)"
        fi
    done
fi

# 4. 显示当前登录会话
echo "4. 当前登录会话:"
who | while read line; do
    echo "   $line"
done
echo

# 5. 如果有SSH config，分析可能的目标主机
echo "5. SSH Config 中的主机 (前10个):"
if [[ -f ~/.ssh/config ]]; then
    awk '/^Host\s+/ && !/\*/ {print "   " $2}' ~/.ssh/config | head -10
else
    echo "   ~/.ssh/config 不存在"
fi
echo

# 6. 显示最近的SSH连接日志
echo "6. 最近的SSH连接记录 (last 5):"
if command -v journalctl >/dev/null 2>&1; then
    journalctl -u ssh --since "1 hour ago" --no-pager -n 5 2>/dev/null | grep -E "Accepted|Connection" | tail -5 | while read line; do
        echo "   $line"
    done
elif [[ -f /var/log/auth.log ]]; then
    tail -20 /var/log/auth.log 2>/dev/null | grep -E "ssh.*Accepted|ssh.*Connection" | tail -5 | while read line; do
        echo "   $line"
    done
else
    echo "   无法访问SSH日志"
fi
