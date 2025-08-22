#!/bin/bash
while true; do
    ./sync_data_single.sh
    ./sync_log_single.sh
    sleep 60  # 每 1 分钟执行一次
done
