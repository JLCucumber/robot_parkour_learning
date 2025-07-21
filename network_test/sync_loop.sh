#!/bin/bash
while true; do
    ./sync_data_single.sh
    ./sync_log_single.sh
    sleep 300  # 每 5 分钟执行一次
done
