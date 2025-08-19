# network_test A↔C 最小测试说明

本目录包含 A(本机) 与 C(采集机) 之间的最小同步脚本。脚本已参数化，支持通过环境变量覆盖默认配置，并可选跳板。

## 核心脚本
- sync_data_single.sh: 从远端(C) 拉取数据到本机(A)
- sync_log_single.sh: 将本机(A) 的日志推送到远端(C)
- sync_loop.sh: 每 5 分钟轮询执行一次上述两脚本

## 环境变量
- REMOTE_DIR: 远端(C) 的目录（含 user@host:abs/path/）
- LOCAL_DIR: 本机(A) 的目录（绝对路径）
- PROXY_JUMP: 跳板机（留空表示直连）
- DIR_NAME: 日志子目录（仅 sync_log_single.sh 用）
- RSYNC_DRYRUN: 非空启用 dry-run（先验证后正式执行）

## 快速使用
1) 配置 SSH（推荐在 ~/.ssh/config 中设置 Host 别名，或在脚本中使用 PROXY_JUMP）
2) 在 A 上执行（先 dry-run 验证）：

```
export REMOTE_DIR="user_on_C@server-c:/abs/path/to/data/"
export LOCAL_DIR="/abs/path/on_A/data/"
export PROXY_JUMP=""      # 直连留空；如需跳板则填写 user@jump-host
export RSYNC_DRYRUN=1
./sync_data_single.sh
```

验证通过后，去掉 RSYNC_DRYRUN 再执行一次。

3) 推送日志示例：
```
export LOCAL_DIR="/abs/path/on_A/logs/exp123/"
export REMOTE_DIR="user_on_C@server-c:/abs/path/on_C/logs/exp123/"
export RSYNC_DRYRUN=1
./sync_log_single.sh
```

4) 连续轮询：
```
./sync_loop.sh
```

## 分支协作建议
- 使用临时分支 ac-net-test，在 A 修改、提交并推送；C 端检出同名分支以联调。
