import os
import re
import torch
import numpy as np
import pickle
import json
import time
import traceback
from collections import OrderedDict, defaultdict

from rsl_rl.utils.collections import namedarraytuple
import rsl_rl.utils.data_compresser as compresser
from rsl_rl.storage.rollout_files.base import RolloutFileBase

class RolloutDataset(RolloutFileBase):
    Transition = namedarraytuple("Transition", [
        "observation",
        "privileged_observation",
        "action",
        "reward",
        "done",
        "timeout",
        "next_observation",
        "next_privileged_observation",
        # 新增优势相关字段
        "teacher_advantages",
        "positive_advantages", 
        "difficulty_scores",
    ])
    def __init__(self, data_dir, num_envs,
            dataset_loops: int = 1,
            random_shuffle_traj_order= False,
            keep_latest_n_trajs= 0, # If > 0 and more than n_trajectories, ignores keep_latest_ratio and keeps the latest n trajectories.
            starting_frame_range= [0, 1], # if set, the starting timestep will be uniformly chose from this, when each new trajectory is loaded.
                # if sampled starting frame is bigger than the trajectory length, starting frame will be 0
            device= "cuda",
        ):
        super().__init__(data_dir, num_envs, device= device)
        self.dataset_loops = dataset_loops
        self.random_shuffle_traj_order = random_shuffle_traj_order
        self.keep_latest_n_trajs = keep_latest_n_trajs
        self.starting_frame_range = starting_frame_range
        
        self.num_dataset_looped = 0
        # 统计字段
        self._cumulative_transitions = 0
        self._sampled_traj_identifier_set = set()
        self._last_stat_report_iter = 0
        # incremental scan stats
        self._last_dir_set = set()
        self._scan_index = 0
        self._last_total_dirs = 0
        
        # 🚀 性能优化配置 - 安全开关
        self.enable_batch_tensor_copy = True   # 可以设为False回退到原始方法
        self.enable_async_gpu_ops = False      # 默认关闭，更安全
        self._optimization_fallback_count = 0  # 记录优化失败次数
        
        # ⏱️ 时间统计 - 性能分析工具
        self._timing_stats = {
            'directory_scanning': 0.0,      # 目录扫描时间
            'file_sorting': 0.0,            # 文件排序时间
            'data_loading': 0.0,            # 数据加载时间 - 总体
            'pickle_loading': 0.0,          # 🆕 pickle文件加载时间
            'tensor_conversion': 0.0,       # 🆕 numpy到torch转换时间
            'buffer_allocation': 0.0,       # 🆕 缓冲区分配时间
            'tensor_copying': 0.0,          # 🆕 tensor复制时间
            'metadata_processing': 0.0,     # 元数据处理时间
            'trajectory_refresh': 0.0,      # 轨迹刷新时间
            'total_calls': {
                'directory_scanning': 0,
                'file_sorting': 0,
                'data_loading': 0,
                'pickle_loading': 0,
                'tensor_conversion': 0,
                'buffer_allocation': 0,
                'tensor_copying': 0,
                'metadata_processing': 0,
                'trajectory_refresh': 0,
            }
        }
        self._last_timing_report = time.time()

    @staticmethod
    def get_frame_range(filename: str) -> tuple:
        """ Get the frame range from the filename. Return a tuple [start, end). (end is exclusive)
        """
        # print(f"RolloutData set: get frame range from {filename}")
        # print(f"RolloutData set: get frame range from {filename}")
        return (
            int(filename.split(".")[0].split("_")[1]),
            int(filename.split(".")[0].split("_")[2]),
        )

    @staticmethod
    def _is_main_traj_file(filename: str) -> bool:
        """Identify a main trajectory pickle (exclude sidecar or npz)."""
        if not filename.startswith("traj_"):
            return False
        if "sidecar" in filename:
            return False
        return filename.endswith((".pickle", ".pkl"))

    def _time_operation(self, operation_name: str):
        """Context manager for timing operations"""
        class TimingContext:
            def __init__(self, parent, op_name):
                self.parent = parent
                self.op_name = op_name
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.perf_counter()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.perf_counter() - self.start_time
                self.parent._timing_stats[self.op_name] += duration
                self.parent._timing_stats['total_calls'][self.op_name] += 1
                
        return TimingContext(self, operation_name)
    
    def _print_timing_report(self, force=False, show_in_terminal=False, save_to_log=True, log_file_path=None):
        """
        打印和记录性能统计信息
        
        Args:
            force (bool): 强制打印，忽略时间间隔限制
            show_in_terminal (bool): 是否在终端显示详细报告，默认False
            save_to_log (bool): 是否保存到日志文件，默认True
            log_file_path (str): 自定义日志文件路径，默认None
        """
        import logging
        import os
        from datetime import datetime
        
        current_time = time.time()
        # 每60秒或强制打印一次报告
        if not force and (current_time - self._last_timing_report) < 60:
            return
            
        self._last_timing_report = current_time
        
        # 设置日志
        log_file = None
        perf_logger = None
        if save_to_log:
            if log_file_path:
                # 使用自定义路径
                log_file = log_file_path
            else:
                # 使用默认路径
                log_dir = os.path.join(os.path.dirname(self.data_dir[0] if isinstance(self.data_dir, (list, tuple)) else self.data_dir), 'logs') if hasattr(self, 'data_dir') else './logs'
                os.makedirs(log_dir, exist_ok=True)
                
                # 🏷️ 生成任务特定的日志文件名后缀
                task_suffix = ""
                if hasattr(self, 'data_dir') and self.data_dir:
                    try:
                        # 从数据目录路径中提取任务名称
                        data_path = self.data_dir[0] if isinstance(self.data_dir, (list, tuple)) else self.data_dir
                        # 查找包含任务名的部分，通常在 "data/" 之后
                        if "data/" in data_path:
                            path_after_data = data_path.split("data/")[-1]
                            # 提取任务名（第一个下划线前的部分，或整个路径段）
                            task_name = path_after_data.split("/")[0].split("_")[0]
                            if task_name and len(task_name) > 0:
                                task_suffix = f"_{task_name}"
                        else:
                            # 备用方案：使用目录名的最后一段
                            dir_name = os.path.basename(data_path.rstrip('/'))
                            if dir_name and dir_name != 'data':
                                task_suffix = f"_{dir_name}"
                    except (IndexError, AttributeError):
                        # 如果提取失败，使用时间戳作为后缀
                        task_suffix = f"_{int(time.time())}"
                
                log_file = os.path.join(log_dir, f'rollout_performance{task_suffix}.log')
            
            # 配置logger
            perf_logger = logging.getLogger('rollout_performance')
            perf_logger.setLevel(logging.INFO)
            
            # 避免重复添加handler
            if not perf_logger.handlers:
                handler = logging.FileHandler(log_file, encoding='utf-8')
                formatter = logging.Formatter('%(asctime)s - %(message)s')
                handler.setFormatter(formatter)
                perf_logger.addHandler(handler)
        
        def log_and_print(message, print_to_terminal=show_in_terminal):
            """同时记录和可选打印消息"""
            if print_to_terminal:
                print(message)
            if save_to_log and perf_logger:
                # 移除ANSI颜色代码和emoji字符用于日志
                import re
                clean_message = message.replace('\033[1;96m', '').replace('\033[1;93m', '').replace('\033[1;92m', '').replace('\033[1;91m', '').replace('\033[0m', '')
                # 移除emoji字符，只保留ASCII字符
                clean_message = re.sub(r'[^\x00-\x7F]+', '', clean_message)
                perf_logger.info(clean_message)
        
        # 性能报告头部
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_and_print("="*80)
        log_and_print(f"🔍 RolloutDataset Performance Analysis Report - {timestamp}")
        log_and_print("="*80)
        
        total_time = sum(self._timing_stats[key] for key in self._timing_stats if key != 'total_calls')
        
        if total_time == 0:
            log_and_print("📊 No timing data collected yet.")
            return
            
        for operation in ['directory_scanning', 'file_sorting', 'data_loading', 'pickle_loading', 'tensor_conversion', 'buffer_allocation', 'tensor_copying', 'metadata_processing', 'trajectory_refresh']:
            op_time = self._timing_stats[operation]
            op_calls = self._timing_stats['total_calls'][operation]
            percentage = (op_time / total_time) * 100 if total_time > 0 else 0
            avg_time = (op_time / op_calls) if op_calls > 0 else 0
            
            # 跳过没有数据的操作
            if op_time == 0 and op_calls == 0:
                continue
            
            # 使用不同图标表示不同操作
            icons = {
                'directory_scanning': '📁',
                'file_sorting': '🔄', 
                'data_loading': '📥',
                'pickle_loading': '🍉',      # pickle文件加载
                'tensor_conversion': '🔄',   # numpy到torch转换
                'buffer_allocation': '📦',   # 缓冲区分配
                'tensor_copying': '📋',      # tensor复制
                'metadata_processing': '📋',
                'trajectory_refresh': '🔄'
            }
            
            icon = icons.get(operation, '⚙️')
            operation_display = operation.replace('_', ' ').title()
            
            message = (f"{icon} {operation_display:20} | "
                      f"Total: \033[1;96m{op_time:8.3f}s\033[0m | "
                      f"Calls: \033[1;93m{op_calls:5d}\033[0m | "
                      f"Avg: \033[1;92m{avg_time:7.4f}s\033[0m | "
                      f"Share: \033[1;91m{percentage:5.1f}%\033[0m")
            log_and_print(message)
        
        log_and_print("-" * 80)
        log_and_print(f"🕒 Total Processing Time: \033[1;96m{total_time:.3f}s\033[0m")
        
        # 性能建议
        max_time_op = max(self._timing_stats.items(), key=lambda x: x[1] if x[0] != 'total_calls' else 0)
        if max_time_op[0] != 'total_calls' and max_time_op[1] > 0:
            hotspot_message = (f"⚡ Performance Hotspot: {max_time_op[0].replace('_', ' ').title()} "
                              f"({(max_time_op[1]/total_time)*100:.1f}% of total time)")
            log_and_print(hotspot_message)
        
        # 优化状态报告
        optimization_status = []
        if hasattr(self, 'enable_directory_caching') and self.enable_directory_caching:
            optimization_status.append("✅ Directory Caching")
        if hasattr(self, 'enable_batch_tensor_copy') and self.enable_batch_tensor_copy:
            optimization_status.append("✅ Batch Tensor Copy")
        if hasattr(self, '_optimization_fallback_count') and self._optimization_fallback_count > 0:
            optimization_status.append(f"⚠️ Fallback Count: {self._optimization_fallback_count}")
            
        if optimization_status:
            log_and_print(f"🎯 Optimizations Active: {', '.join(optimization_status)}")
        
        log_and_print("="*80)
        
        # 简短的终端总结
        if not show_in_terminal and save_to_log:
            # 只在日志中记录，终端显示简短信息
            print(f"📊 Performance report logged to: {log_file}")
            print(f"   Total time: {total_time:.3f}s | Top bottleneck: {max_time_op[0].replace('_', ' ').title()} ({(max_time_op[1]/total_time)*100:.1f}%)")
        elif show_in_terminal and save_to_log:
            print(f"\n💾 Performance report also saved to: {log_file}")
    
    def print_timing_stats(self, print_to_console=False, log_file_path=None):
        """
        打印性能统计
        
        Args:
            print_to_console (bool): 是否打印到控制台，默认False
            log_file_path (str): 日志文件路径，默认None使用任务特定的日志文件名
        """
        # 设置默认日志文件路径，包含任务后缀
        if log_file_path is None:
            # 🏷️ 生成任务特定的默认日志文件名
            task_suffix = ""
            if hasattr(self, 'data_dir') and self.data_dir:
                try:
                    # 从数据目录路径中提取任务名称
                    data_path = self.data_dir[0] if isinstance(self.data_dir, (list, tuple)) else self.data_dir
                    # 查找包含任务名的部分，通常在 "data/" 之后
                    if "data/" in data_path:
                        path_after_data = data_path.split("data/")[-1]
                        # 提取任务名（第一个下划线前的部分，或整个路径段）
                        task_name = path_after_data.split("/")[0].split("_")[0]
                        if task_name and len(task_name) > 0:
                            task_suffix = f"_{task_name}"
                    else:
                        # 备用方案：使用目录名的最后一段
                        dir_name = os.path.basename(data_path.rstrip('/'))
                        if dir_name and dir_name != 'data':
                            task_suffix = f"_{dir_name}"
                except (IndexError, AttributeError):
                    # 如果提取失败，使用时间戳作为后缀
                    import time
                    task_suffix = f"_{int(time.time())}"
            
            log_file_path = f"performance_optimized{task_suffix}.log"
        
        # 默认保存到日志，可选打印到控制台
        self._print_timing_report(
            force=True, 
            show_in_terminal=print_to_console, 
            save_to_log=True,
            log_file_path=log_file_path
        )

    def read_dataset_directory(self):
        """ Refresh file-related information by scanning the directory. All traj_handlers must be
        updated from attributes here.
        """
        print("RolloutDataset: reading dataset directory...")

        # ⏱️ 开始时间测量
        with self._time_operation('directory_scanning'):
            # 创建扫描锁文件，通知清理脚本正在扫描
            scan_lock_path = None
            if isinstance(self.data_dir, str):
                scan_lock_path = os.path.join(os.path.dirname(self.data_dir), ".rollout_scan_lock")
            elif isinstance(self.data_dir, (tuple, list)) and self.data_dir:
                scan_lock_path = os.path.join(os.path.dirname(self.data_dir[0]), ".rollout_scan_lock")
            
            if scan_lock_path:
                try:
                    with open(scan_lock_path, 'w') as f:
                        f.write(f"scan_started:{time.time()}\n")
                        f.write(f"pid:{os.getpid()}\n")
                except OSError:
                    scan_lock_path = None  # 如果无法创建锁文件，继续执行

            if isinstance(self.data_dir, str):
                self.data_dirs = [self.data_dir]
            elif isinstance(self.data_dir, (tuple, list)):
                self.data_dirs = self.data_dir
            else:
                raise ValueError("data_dir should be a string or a list of strings.")
                
            # print("Data Directories: {}".format(self.data_dirs))
            self.all_available_trajectory_dirs = []
            # reset metadata; keep all raw metadata objects for consistency checking
            self.metadata = None
            self._all_metadatas = []
            metadata_repeated = False
            total_timesteps = 0
            for data_dir in self.data_dirs:
                if not os.path.exists(data_dir):
                    os.mkdir(data_dir)
                    print("RolloutDataset: {} not found, created...".format(data_dir))
                    continue
                
                # 🚀 优化1: 使用 os.scandir() 代替 os.walk() 获取更好性能
                try:
                    with os.scandir(data_dir) as entries:
                        subdir_paths = [entry.path for entry in entries if entry.is_dir()]
                except OSError:
                    print(f"RolloutDataset: Warning - Failed to scan {data_dir}")
                    continue
                
                # 🚀 优化2: 只处理trajectory目录，避免深度遍历
                for subdir_path in subdir_paths:
                    try:
                        with os.scandir(subdir_path) as sub_entries:
                            traj_dirs = [entry.path for entry in sub_entries 
                                       if entry.is_dir() and entry.name.startswith("trajectory_")]
                    except OSError:
                        continue
                    
                    for dir_path in traj_dirs:
                        # 🚀 优化3: 智能缓存 - 检查目录修改时间
                        try:
                            dir_mtime = os.path.getmtime(dir_path)
                            
                            # 如果缓存中有且未修改，直接使用缓存
                            if (hasattr(self, '_dir_cache') and dir_path in self._dir_cache and 
                                self._dir_cache[dir_path][0] >= dir_mtime):
                                cached_info = self._dir_cache[dir_path][1]
                                if cached_info['valid']:
                                    self.all_available_trajectory_dirs.append(dir_path)
                                    total_timesteps += cached_info['timesteps']
                                continue
                            
                            # 初始化缓存
                            if not hasattr(self, '_dir_cache'):
                                self._dir_cache = {}
                            
                            # 🚀 优化4: 快速文件检查 - 避免完整目录列表
                            has_pickle = False
                            has_tmp = False
                            max_timesteps = 0
                            
                            try:
                                with os.scandir(dir_path) as files:
                                    for file_entry in files:
                                        if file_entry.is_file():
                                            fname = file_entry.name
                                            if fname.endswith('.tmp'):
                                                has_tmp = True
                                                break
                                            if self._is_main_traj_file(fname):
                                                has_pickle = True
                                                # 🚀 优化5: 解析最大timestep，避免排序
                                                try:
                                                    _, end_frame = self.get_frame_range(fname)
                                                    max_timesteps = max(max_timesteps, end_frame)
                                                except:
                                                    pass
                            except OSError:
                                # 缓存无效条目
                                self._dir_cache[dir_path] = (dir_mtime, {'valid': False, 'timesteps': 0})
                                continue
                            
                            # 更新缓存
                            if has_pickle and not has_tmp:
                                self.all_available_trajectory_dirs.append(dir_path)
                                total_timesteps += max_timesteps
                                self._dir_cache[dir_path] = (dir_mtime, {'valid': True, 'timesteps': max_timesteps})
                            else:
                                if has_tmp:
                                    print(f"RolloutDataset: skipping {dir_path} due to a .tmp file being present.")
                                self._dir_cache[dir_path] = (dir_mtime, {'valid': False, 'timesteps': 0})
                                
                        except OSError:
                            # 目录可能已被删除，从缓存中移除
                            if hasattr(self, '_dir_cache') and dir_path in self._dir_cache:
                                del self._dir_cache[dir_path]
                            continue
                    
                    # ⏱️ 元数据处理时间测量 - 在每个子目录中查找metadata.json
                    with self._time_operation('metadata_processing'):
                        try:
                            meta_path = os.path.join(subdir_path, "metadata.json")
                            if os.path.exists(meta_path):
                                with open(meta_path, "r") as f:
                                    md = json.load(f, object_pairs_hook= OrderedDict)
                                    self._all_metadatas.append((meta_path, md))
                                    if self.metadata is None:
                                        self.metadata = md
                                    elif not metadata_repeated:
                                        # only print once; we still keep first as authoritative
                                        # print("RolloutDataset: multiple metadata files found, using the first one ({}).".format(self._all_metadatas[0][0]))
                                        metadata_repeated = True
                        except FileNotFoundError:
                            pass # skip

        # ⏱️ 文件排序时间测量
        with self._time_operation('file_sorting'):
            def get_mtime_safe(path):
                try:
                    return os.path.getmtime(path)
                except FileNotFoundError:
                    return 0  # Treat deleted files as very old

            self.all_available_trajectory_dirs.sort(key=get_mtime_safe)
            # Filter out any directories that might have been deleted before or during the sort
            self.all_available_trajectory_dirs = [d for d in self.all_available_trajectory_dirs if os.path.exists(d)]
            
            # 🚀 优化6: 清理缓存中不存在的目录
            if hasattr(self, '_dir_cache'):
                existing_dirs = set(self.all_available_trajectory_dirs)
                stale_dirs = [path for path in self._dir_cache.keys() if path not in existing_dirs]
                for stale_dir in stale_dirs:
                    if not os.path.exists(stale_dir):
                        del self._dir_cache[stale_dir]
            self.all_available_trajectory_dirs = [d for d in self.all_available_trajectory_dirs if os.path.exists(d)]
            
        current_dir_set = set(self.all_available_trajectory_dirs)
        new_dirs = current_dir_set - self._last_dir_set
        reused_dirs = len(self.all_available_trajectory_dirs) - len(new_dirs)
        self._scan_index += 1
        
        # 🚀 缓存性能统计
        cache_stats = ""
        if hasattr(self, '_dir_cache'):
            cache_hits = len(self.all_available_trajectory_dirs) - len(new_dirs)
            cache_stats = f" [Cache: {len(self._dir_cache)} entries, {cache_hits} hits]"
            
        # print("RolloutDataset: {} trajectories found. {} timesteps in total.{}".format(
        #     len(self.all_available_trajectory_dirs),
        #     total_timesteps,
        #     cache_stats
        # ))
        # print(f"RolloutDataset: scan#{self._scan_index} new_dirs={len(new_dirs)} reused_dirs={reused_dirs} (prev_total={self._last_total_dirs} -> curr_total={len(self.all_available_trajectory_dirs)})")
        # update last snapshots BEFORE trimming to latest N
        self._last_dir_set = current_dir_set
        self._last_total_dirs = len(self.all_available_trajectory_dirs)

        # === Metadata consistency check ===
        if len(self._all_metadatas) > 1:
            def _fingerprint(md):
                # focus on observation layout & compression mapping which must match
                return json.dumps({
                    'obs_segments': md.get('obs_segments', {}),
                    'obs_disassemble_mapping': md.get('obs_disassemble_mapping', {}),
                    'num_obs': md.get('num_obs'),
                    'num_privileged_obs': md.get('num_privileged_obs'),
                }, sort_keys=True)
            fp0 = _fingerprint(self._all_metadatas[0][1])
            inconsistent = []
            for p, md in self._all_metadatas[1:]:
                if _fingerprint(md) != fp0:
                    inconsistent.append(p)
            if inconsistent:
                print("[WARNING] RolloutDataset: Found metadata.json inconsistencies in: {}".format(inconsistent))
                print("[WARNING] Using the first metadata ONLY. Consider cleaning or unifying metadata files.")
            else:
                # 🔍 更清晰的metadata检查消息
                unique_runs = set()
                for meta_path, _ in self._all_metadatas:
                    run_root = meta_path.split("data/")[-1].split("/")[0] if "data/" in meta_path else "unknown"
                    unique_runs.add(run_root)
                # print("RolloutDataset: All metadata files consistent ({} files from {} data collection runs).".format(
                #     len(self._all_metadatas), len(unique_runs)))

        if len(self.all_available_trajectory_dirs) < self.keep_latest_n_trajs:
            return False
        else:
            self.all_available_trajectory_dirs = self.all_available_trajectory_dirs[-self.keep_latest_n_trajs:]
            self.suffixs = [d.split("trajectory_")[-1] for d in self.all_available_trajectory_dirs]
            # print(f"RolloutDataset: trajectory suffixes range: {min(self.suffixs)} - {max(self.suffixs)}, total {len(self.suffixs)} trajectories.")
            # distribution over run roots (e.g., task/run_timestamp) to show multi-machine mixing
            run_root_counts = defaultdict(int)
            for d in self.all_available_trajectory_dirs:
                rel = d.split("data/")[-1]
                run_root = rel.split("trajectory_")[0].rstrip('/')  # e.g., go2_distill_awbc/.../Aug22_01-07-45
                run_root_counts[run_root] += 1
            if len(run_root_counts) > 1:
                def _extract_ts(s: str):
                    # Match pattern like Aug22_01-07-45
                    m = re.search(r'[A-Z][a-z]{2}\d{2}_\d{2}-\d{2}-\d{2}', s)
                    if m:
                        return m.group(0)
                    # Fallback: last path component before optional underscore details
                    base = s.rstrip('/').split('/')[-1]
                    return base.split('_')[0][:14]
                dist_str = ", ".join([f"{_extract_ts(k)}:{v}" for k, v in sorted(run_root_counts.items())])
                print(f"RolloutDataset: Training window using trajectories from: {dist_str} (latest {self.keep_latest_n_trajs} trajectories)")

            # === New Debug: per data collector (run root) suffix range & new trajectory ratios ===
            # Identify newly introduced directories in THIS trimmed window compared to previous window.
            current_window_set = set(self.all_available_trajectory_dirs)
            prev_window_set = getattr(self, '_prev_window_set', set())
            new_in_window = current_window_set - prev_window_set
            reused_in_window = current_window_set & prev_window_set
            window_new_ratio = (len(new_in_window) / len(current_window_set)) if current_window_set else 0.0
            print(f"RolloutDataset: window incremental new trajectories: {len(new_in_window)}/{len(current_window_set)} ({window_new_ratio:.2%})  reused={len(reused_in_window)}")

            # Group suffix ranges per run_root and compute new/reused stats
            run_root_suffix_info = defaultdict(list)
            for d in self.all_available_trajectory_dirs:
                rel = d.split("data/")[-1]
                run_root = rel.split("trajectory_")[0].rstrip('/')
                try:
                    suffix_int = int(d.split("trajectory_")[-1])
                except Exception:
                    suffix_int = -1
                run_root_suffix_info[run_root].append((d, suffix_int))

            # print(f"RolloutDataset: detected data collector dirs: {len(run_root_suffix_info)}")
            def _extract_ts(s: str):
                m = re.search(r'[A-Z][a-z]{2}\d{2}_\d{2}-\d{2}-\d{2}', s)
                if m:
                    return m.group(0)
                base = s.rstrip('/').split('/')[-1]
                return base.split('_')[0][:14]
            for run_root, items in sorted(run_root_suffix_info.items()):
                suffix_vals = [s for _, s in items if s >= 0]
                if suffix_vals:
                    min_s, max_s = min(suffix_vals), max(suffix_vals)
                else:
                    min_s = max_s = -1
                dirs_in_root = {p for p, _ in items}
                new_dirs_root = dirs_in_root & new_in_window
                new_ratio_root = (len(new_dirs_root) / len(dirs_in_root)) if dirs_in_root else 0.0
                print(f"  - {_extract_ts(run_root)}: suffix_range={min_s}-{max_s} count={len(dirs_in_root)} new={len(new_dirs_root)} ({new_ratio_root:.2%})")

            # Persist current window set for next incremental diff
            self._prev_window_set = current_window_set
        if len(self.all_available_trajectory_dirs) > 0:
            task_names = [d.split("data/")[-1].split("_jump")[0] for d in self.all_available_trajectory_dirs]
            unique_task_names = set(task_names)
            # print("RolloutDataset: unique task names found in all available trajectories: {}".format(unique_task_names))
        self.unused_trajectory_idxs = list(range(len(self.all_available_trajectory_dirs)))
        if self.random_shuffle_traj_order:
            self.unused_trajectory_idxs = np.random.permutation(self.unused_trajectory_idxs)
        self._reload_notice_printed = False
        
        # 删除扫描锁文件
        if scan_lock_path and os.path.exists(scan_lock_path):
            try:
                os.remove(scan_lock_path)
            except OSError:
                pass  # 忽略删除失败
        
        # ⏱️ 打印时间统计报告
        self._print_timing_report()
        
        return True

    def assemble_obs_components(self, traj_data):
        assert "obs_segments" in self.metadata, "Corrupted metadata, obs_segments not found in metadata"
        observations = []
        for component_name in self.metadata["obs_segments"].keys():
            obs_component = traj_data.pop("obs_" + component_name)
            if component_name in self.metadata["obs_disassemble_mapping"]:
                obs_component = getattr(
                    compresser,
                    "decompress_" + self.metadata["obs_disassemble_mapping"][component_name],
                )(obs_component)
            observations.append(obs_component)
        traj_data["observations"] = np.concatenate(observations, axis= -1) # (n_steps, d_obs)
        return traj_data

    def reset_all(self):
        """ Reset and defines the handlers. Usually called in reset() to initialize the handlers.
        All handlers that identify which trajectory(file) is currently loaded for each env appear
        here.
        """
        while not self.read_dataset_directory():
            print("RolloutDataset: trajectory not enough, need {} at least, waiting for 15 minutes...".format(self.keep_latest_n_trajs))
            time.sleep(60 * 15)
        # use trajectory index to identify the trajectory in all_available_trajectory_dirs
        self.traj_identifiers = [None] * self.num_envs
        self.unused_trajectory_idxs = [i for i in self.unused_trajectory_idxs if i not in self.traj_identifiers]
        self.traj_file_names = [[] for _ in range(self.num_envs)]
        self.traj_lengths = [None for _ in range(self.num_envs)]
        self.traj_file_idxs = [None for _ in range(self.num_envs)]
        self.traj_datas = [None for _ in range(self.num_envs)]
        self.traj_cursors = np.zeros(self.num_envs, dtype= int)

        self.refresh_handlers()

    def _refresh_traj_data(self, env_idx, retries=5, delay=1.0):
        """ refresh `self.traj_data` based on current traj_file_idxs[env_idx]. usually called
        after refreshing traj_handler or updated traj_file_idxs[env_idx]
        with retry mechanism to handle potential file read errors.
        Returns True if successful, False if failed.
        """
        # ⏱️ 数据加载时间测量
        with self._time_operation('data_loading'):
            traj_dir = self.all_available_trajectory_dirs[self.traj_identifiers[env_idx]]
            traj_filename = self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]]
            traj_path = os.path.join(traj_dir, traj_filename)

            ### safety check
            attempt = 0
            while attempt < retries:
                attempt += 1
                try:
                    # === Check if file exists and is non-empty ===
                    if not os.path.exists(traj_path):
                        print(f"[Attempt {attempt}] File does not exist: {traj_path}")
                    elif os.path.getsize(traj_path) == 0:
                        print(f"[Attempt {attempt}] File size is 0, possibly still being written: {traj_path}")
                    else:
                        # === Try to load pickle file ===
                        with self._time_operation('pickle_loading'):
                            with open(traj_path, "rb") as f:
                                traj_data = pickle.load(f)
                        break  # Successfully loaded, exit loop
                except EOFError:
                    print(f"[Attempt {attempt}] Caught EOFError, file might still be writing: {traj_path}")
                except Exception as e:
                    print(f"[Attempt {attempt}] Error loading {traj_path}: {type(e).__name__}: {e}")
                    # 对于 pickle 截断错误，不要无限重试
                    if "truncated" in str(e).lower() or "UnpicklingError" in str(type(e).__name__):
                        print(f"[RolloutDataset] Skipping corrupted file (pickle truncated): {traj_path}")
                        # 标记这个文件为损坏，跳过到下一个文件
                        self._skip_corrupted_file(env_idx)
                        return False  # 返回失败
                    
                time.sleep(delay)
            else:
                print(f"[RolloutDataset] Failed to load trajectory file after {retries} attempts: {traj_path}")
                # 尝试跳过损坏的文件而不是崩溃
                self._skip_corrupted_file(env_idx)
                return False

            # 到这里说明成功加载了文件
            if "obs_disassemble_mapping" in self.metadata.keys():
                traj_data = self.assemble_obs_components(traj_data)

        # 新增：确保包含优势字段（向后兼容旧数据）
        if 'advantages' not in traj_data:
            traj_data['advantages'] = np.zeros_like(traj_data['rewards'])
            traj_data['positive_advantages'] = np.zeros_like(traj_data['rewards']) 
            traj_data['difficulty_scores'] = np.zeros_like(traj_data['rewards'])

        # ⏱️ 测量tensor转换时间
        with self._time_operation('tensor_conversion'):
            for k, v in traj_data.items():
                if isinstance(v, np.ndarray):
                    traj_data[k] = torch.from_numpy(v).to(self.device)
                # 保留非numpy数组的元数据（如标量值）
        self.traj_datas[env_idx] = traj_data
        return True  # 成功加载

    def _skip_corrupted_file(self, env_idx, max_attempts=10):
        """跳过损坏的文件，尝试加载下一个文件或下一个轨迹
        增加了max_attempts参数来防止无限递归
        """
        if not hasattr(self, '_skip_attempt_count'):
            self._skip_attempt_count = {}
        
        if env_idx not in self._skip_attempt_count:
            self._skip_attempt_count[env_idx] = 0
        
        self._skip_attempt_count[env_idx] += 1
        
        if self._skip_attempt_count[env_idx] > max_attempts:
            print(f"[RolloutDataset] Exceeded max skip attempts ({max_attempts}) for env {env_idx}, triggering dataset reload...")
            # 重置计数并触发数据集重新加载
            self._skip_attempt_count[env_idx] = 0
            raise StopIteration  # 这将触发dataset reload
        
        print(f"[RolloutDataset] Attempting to skip corrupted file for env {env_idx} (attempt {self._skip_attempt_count[env_idx]}/{max_attempts})")
        
        # 尝试移动到下一个文件
        if self.traj_file_idxs[env_idx] + 1 < len(self.traj_file_names[env_idx]):
            self.traj_file_idxs[env_idx] += 1
            print(f"[RolloutDataset] Moving to next file: {self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]]}")
            if self._refresh_traj_data(env_idx):  # 使用返回值检查是否成功
                # 成功加载，重置计数
                self._skip_attempt_count[env_idx] = 0
                return  # 成功加载下一个文件
            else:
                print(f"[RolloutDataset] Next file also corrupted, trying to refresh trajectory...")
        
        # 如果当前轨迹的所有文件都有问题，尝试切换到新轨迹
        print(f"[RolloutDataset] Switching to new trajectory for env {env_idx}")
        if len(self.unused_trajectory_idxs) > 0:
            # 分配新轨迹
            self.traj_identifiers[env_idx] = self.unused_trajectory_idxs.pop(0)
            self._refresh_traj_handler(env_idx)
        else:
            # 如果没有未使用的轨迹，随机选择一个
            import random
            self.traj_identifiers[env_idx] = random.randint(0, len(self.all_available_trajectory_dirs) - 1)
            self._refresh_traj_handler(env_idx)

    def _refresh_traj_handler(self, env_idx):
        """ update traj_handler for the given env and load the first traj_data. It does not update
        the traj_identifiers.
        """
        # ⏱️ 轨迹刷新时间测量
        with self._time_operation('trajectory_refresh'):
            traj_dir = self.all_available_trajectory_dirs[self.traj_identifiers[env_idx]]
            trajectory_files = [f for f in os.listdir(traj_dir) if self._is_main_traj_file(f)]
            if not trajectory_files:
                # No valid pickle files found - likely cleaned by trajectory cleaner
                print(f"[RolloutDataset] No valid pickle files in {traj_dir}, trying another trajectory...")
                self._skip_corrupted_file(env_idx)
                return
            trajectory_files.sort(key= lambda x: self.get_frame_range(x)[1])
            self.traj_cursors[env_idx] = np.random.randint(
                min(self.starting_frame_range[0], self.get_frame_range(trajectory_files[-1])[0]),
                min(self.starting_frame_range[1], self.get_frame_range(trajectory_files[-1])[1]),
            )
            self.traj_file_names[env_idx] = trajectory_files
        self.traj_lengths[env_idx] = self.get_frame_range(trajectory_files[-1])[1]
        self.traj_file_idxs[env_idx] = 0
        while (self.get_frame_range(self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]])[0] > self.traj_cursors[env_idx] \
            or self.get_frame_range(self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]])[1] <= self.traj_cursors[env_idx]):
            self.traj_file_idxs[env_idx] += 1 \
                if self.get_frame_range(self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]])[1] <= self.traj_cursors[env_idx] else -1
        if not self._refresh_traj_data(env_idx):
            # 如果加载失败，_refresh_traj_data 内部已经处理了跳过逻辑
            print(f"[RolloutDataset] Failed to load trajectory data for env {env_idx}, but handled gracefully")
            return
        self.traj_datas[env_idx]["dones"][0] = True # set the first frame as done

    def refresh_handlers(self, env_ids= None):
        # this method is called ONLY at the start of a new bunch of trajectories are init.
        # print()
        with self._time_operation('trajectory_refresh'):
            if env_ids is None: env_ids = self.all_env_ids
            self.count = 0
            for env_idx in env_ids:
                self.traj_identifiers[env_idx] = self.unused_trajectory_idxs.pop(0)
                self._refresh_traj_handler(env_idx)

    def _maintain_handler(self, env_idx):
        """ Maintain traj_handler and update traj_data if needed. Return whether a new trajectory is loaded.
        """
        try:
            if self.traj_cursors[env_idx] >= self.traj_lengths[env_idx]:
                # load a new trajectory
                # NOTE: self.unused_trajectory_idxs should be shuffled during read_dataset_directory if needed

                if len(self.unused_trajectory_idxs) == 0:
                    if not getattr(self, "_reload_notice_printed", False):
                        total = len(self.all_available_trajectory_dirs)
                        keep_n = self.keep_latest_n_trajs if hasattr(self, "keep_latest_n_trajs") else None
                        suffix = f", keep_latest_n_trajs={keep_n}" if keep_n else ""
                        print(f"\033[1;36m\033[1mRolloutDataset: ran out of trajectories (used {total}{suffix}). Reloading dataset...\033[0m")
                        self._reload_notice_printed = True
                    raise StopIteration
                
                self.traj_identifiers[env_idx] = self.unused_trajectory_idxs.pop(0)

                # ###### Debug #######
                # if 0 <= env_idx <= 10:
                #     print(f"traj id of first ten envs: {self.traj_identifiers[:10]}")
                #     print(f"traj file index of first ten envs: {self.traj_file_idxs[:10]}")
                #     # print(f"remaining trajectories: {len(self.unused_trajectory_idxs)}")
                # ###################

                self._refresh_traj_handler(env_idx)
                return True
            
            traj_cursor_range = self.get_frame_range(self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]])
            if self.traj_cursors[env_idx] < traj_cursor_range[0] or self.traj_cursors[env_idx] >= traj_cursor_range[1]:
                
                ###### Debug #######
                # if env_idx==0:
                #     print(f"current traj cursor for env {env_idx} : {self.traj_cursors[env_idx]}")
                #     print(f"next file index for env {env_idx} : {self.traj_file_idxs[env_idx] + 1}")
                ####################

                # load new traj_data from the same trajectory
                self.traj_file_idxs[env_idx] += 1
                if not self._refresh_traj_data(env_idx):
                    # 如果加载失败，_refresh_traj_data 内部已经处理了跳过逻辑
                    print(f"[RolloutDataset] Failed to load next trajectory file for env {env_idx}, but handled gracefully")
                return False  
            
        except StopIteration:
            if self.dataset_loops < 1 or self.num_dataset_looped < self.dataset_loops:
                # loop the dataset
                self.reset()
                return True
            else:
                raise StopIteration
        return False

    def get_buffer(self, num_transitions_per_env= None):
        with self._time_operation('data_loading'):
            leading_dims = ([] if num_transitions_per_env is None else [num_transitions_per_env]) + [self.num_envs]
            if not hasattr(self, "_output_transition_buffer") or self._output_transition_buffer_leading_dims != leading_dims:
                # ⏱️ 测量缓冲区分配时间
                with self._time_operation('buffer_allocation'):
                    observations = torch.empty(
                        leading_dims + list(self.traj_datas[0]["observations"].shape[1:]),
                        dtype= self.traj_datas[0]["observations"].dtype,
                        device= self.device,
                    )
                    privileged_observations = torch.empty(
                        leading_dims + list(self.traj_datas[0]["privileged_observations"].shape[1:]),
                        dtype= self.traj_datas[0]["privileged_observations"].dtype,
                        device= self.device,
                    )
                    actions = torch.empty(
                        leading_dims + list(self.traj_datas[0]["actions"].shape[1:]),
                        dtype= self.traj_datas[0]["actions"].dtype,
                        device= self.device,
                    )
                    rewards = torch.empty(
                        leading_dims,
                        dtype= self.traj_datas[0]["rewards"].dtype,
                        device= self.device,
                    )
                    dones = torch.empty(
                        leading_dims,
                        dtype= torch.bool,
                        device= self.device,
                    )
                    timeouts = torch.empty(
                        leading_dims,
                        dtype= self.traj_datas[0]["timeouts"].dtype,
                        device= self.device,
                    )
                    next_observations = torch.empty(
                        leading_dims + list(self.traj_datas[0]["observations"].shape[1:]),
                        dtype= self.traj_datas[0]["observations"].dtype,
                        device= self.device,
                    )
                    next_privileged_observations = torch.empty(
                        leading_dims + list(self.traj_datas[0]["privileged_observations"].shape[1:]),
                        dtype= self.traj_datas[0]["privileged_observations"].dtype,
                        device= self.device,
                    )

                    # 新增：优势相关字段的缓冲区
                    teacher_advantages = torch.empty(
                        leading_dims,
                        dtype= self.traj_datas[0]["advantages"].dtype,
                        device= self.device,
                    )
                    positive_advantages = torch.empty(
                        leading_dims,
                        dtype= self.traj_datas[0]["positive_advantages"].dtype,
                        device= self.device,
                    )
                    difficulty_scores = torch.empty(
                        leading_dims,
                        dtype= self.traj_datas[0]["difficulty_scores"].dtype,
                        device= self.device,
                    )
                    self._output_transition_buffer_leading_dims = leading_dims
                    self._output_transition_buffer = self.Transition(
                        observation= observations,
                        privileged_observation= privileged_observations,
                        action= actions,
                        reward= rewards,
                        done= dones,
                        timeout= timeouts,
                        next_observation= next_observations,
                        next_privileged_observation= next_privileged_observations,
                        # 新增字段
                        teacher_advantages= teacher_advantages,
                        positive_advantages= positive_advantages,
                        difficulty_scores= difficulty_scores,
                    ) # type: ignore
            return self._output_transition_buffer
    
    def _fill_transition_batch_optimized(self, buffer, env_ids):
        """
        🚀 优化版本：批量tensor复制，带安全回退
        相比原版本，这个方法将多个环境的数据一次性复制，减少GPU同步次数
        """
        try:
            # 转换为Python list以避免tensor索引问题
            if hasattr(env_ids, 'cpu'):
                env_list = env_ids.cpu().tolist()
            else:
                env_list = list(env_ids)
            
            batch_size = len(env_list)
            
            # 🛡️ 安全检查：确保所有环境都有有效数据
            for env_idx in env_list:
                if (self.traj_datas[env_idx] is None or 
                    env_idx >= len(self.traj_cursors) or 
                    env_idx >= len(self.traj_file_names) or
                    not self.traj_file_names[env_idx]):
                    raise ValueError(f"Invalid data for env {env_idx}")
            
            # 批量收集所有需要的数据
            observations_list = []
            privileged_obs_list = []
            actions_list = []
            rewards_list = []
            dones_list = []
            advantages_list = []
            positive_advantages_list = []
            difficulty_scores_list = []
            timeouts_list = []
            next_observations_list = []
            next_privileged_obs_list = []
            
            # 收集当前step的数据
            for env_idx in env_list:
                traj_cursor_in_file = self.traj_cursors[env_idx] - self.get_frame_range(
                    self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]])[0]
                
                observations_list.append(self.traj_datas[env_idx]["observations"][traj_cursor_in_file])
                privileged_obs_list.append(self.traj_datas[env_idx]["privileged_observations"][traj_cursor_in_file])
                actions_list.append(self.traj_datas[env_idx]["actions"][traj_cursor_in_file])
                rewards_list.append(self.traj_datas[env_idx]["rewards"][traj_cursor_in_file].squeeze())
                dones_list.append(self.traj_datas[env_idx]["dones"][traj_cursor_in_file].squeeze())
                advantages_list.append(self.traj_datas[env_idx]["advantages"][traj_cursor_in_file].squeeze())
                positive_advantages_list.append(self.traj_datas[env_idx]["positive_advantages"][traj_cursor_in_file].squeeze())
                difficulty_scores_list.append(self.traj_datas[env_idx]["difficulty_scores"][traj_cursor_in_file].squeeze())
                
                if "timeout" in self.traj_datas[env_idx].keys():
                    timeouts_list.append(self.traj_datas[env_idx]["timeouts"][traj_cursor_in_file].squeeze())
                else:
                    timeouts_list.append(torch.tensor([False], device=self.device).squeeze())
            
            # 更新cursors和处理trajectory维护
            for i, env_idx in enumerate(env_list):
                self.traj_cursors[env_idx] += 1
                if self._maintain_handler(env_idx):
                    if not dones_list[i].any():
                        timeouts_list[i] = torch.tensor([True], device=self.device).squeeze()
                    dones_list[i] = torch.tensor([True], device=self.device).squeeze()
            
            # 收集next step的数据
            for env_idx in env_list:
                traj_cursor_in_file = self.traj_cursors[env_idx] - self.get_frame_range(
                    self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]])[0]
                next_observations_list.append(self.traj_datas[env_idx]["observations"][traj_cursor_in_file])
                next_privileged_obs_list.append(self.traj_datas[env_idx]["privileged_observations"][traj_cursor_in_file])
            
            # 🚀 批量复制 - 一次性操作，减少GPU同步
            # 使用stack进行批量操作
            buffer.observation.copy_(torch.stack(observations_list))
            buffer.privileged_observation.copy_(torch.stack(privileged_obs_list))
            buffer.action.copy_(torch.stack(actions_list))
            buffer.reward.copy_(torch.stack(rewards_list))
            buffer.done.copy_(torch.stack(dones_list))
            buffer.teacher_advantages.copy_(torch.stack(advantages_list))
            buffer.positive_advantages.copy_(torch.stack(positive_advantages_list))
            buffer.difficulty_scores.copy_(torch.stack(difficulty_scores_list))
            buffer.timeout.copy_(torch.stack(timeouts_list))
            buffer.next_observation.copy_(torch.stack(next_observations_list))
            buffer.next_privileged_observation.copy_(torch.stack(next_privileged_obs_list))
            
            # 更新统计信息
            for env_idx in env_list:
                self._cumulative_transitions += 1
                try:
                    self._sampled_traj_identifier_set.add(int(self.traj_identifiers[env_idx]))
                except Exception:
                    pass
                    
            return True  # 成功
            
        except Exception as e:
            # 🛡️ 安全回退：如果批量操作失败，记录错误并返回False
            self._optimization_fallback_count += 1
            print(f"[WARNING] Batch tensor copy failed (attempt #{self._optimization_fallback_count}): {e}")
            print(f"[WARNING] Falling back to original method for safety...")
            return False
    
    def _fill_transition_per_env(self, buffer, env_idx: int):
        traj_cursor_in_file = self.traj_cursors[env_idx] - self.get_frame_range(self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]])[0]
        buffer.observation.copy_(self.traj_datas[env_idx]["observations"][traj_cursor_in_file])
        buffer.privileged_observation.copy_(self.traj_datas[env_idx]["privileged_observations"][traj_cursor_in_file])
        buffer.action.copy_(self.traj_datas[env_idx]["actions"][traj_cursor_in_file])
        buffer.reward.copy_(self.traj_datas[env_idx]["rewards"][traj_cursor_in_file].squeeze())
        buffer.done.copy_(self.traj_datas[env_idx]["dones"][traj_cursor_in_file].squeeze())

        # 新增：填充优势相关字段
        buffer.teacher_advantages.copy_(self.traj_datas[env_idx]["advantages"][traj_cursor_in_file].squeeze())
        buffer.positive_advantages.copy_(self.traj_datas[env_idx]["positive_advantages"][traj_cursor_in_file].squeeze())
        buffer.difficulty_scores.copy_(self.traj_datas[env_idx]["difficulty_scores"][traj_cursor_in_file].squeeze())
        
        if "timeout" in self.traj_datas[env_idx].keys():
            buffer.timeout.copy_(self.traj_datas[env_idx]["timeouts"][traj_cursor_in_file].squeeze())
        self.traj_cursors[env_idx] += 1
        if self._maintain_handler(env_idx):
            if not buffer.done.any():
                buffer.timeout.copy_(torch.tensor([True], device= self.device).squeeze())
            buffer.done.copy_(torch.tensor([True], device= self.device).squeeze())
        traj_cursor_in_file = self.traj_cursors[env_idx] - self.get_frame_range(self.traj_file_names[env_idx][self.traj_file_idxs[env_idx]])[0]
        buffer.next_observation.copy_(self.traj_datas[env_idx]["observations"][traj_cursor_in_file])
        buffer.next_privileged_observation.copy_(self.traj_datas[env_idx]["privileged_observations"][traj_cursor_in_file])
        # 统计：累计 transition & 轨迹覆盖
        self._cumulative_transitions += 1
        try:
            self._sampled_traj_identifier_set.add(int(self.traj_identifiers[env_idx]))
        except Exception:
            pass

    def fill_transition(self, buffer, env_ids= None):
        with self._time_operation('data_loading'):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device= self.device)
            # ⏱️ 测量tensor复制时间
            with self._time_operation('tensor_copying'):
                # 🚀 尝试使用优化的批量方法
                if self.enable_batch_tensor_copy and len(env_ids) > 1:
                    success = self._fill_transition_batch_optimized(buffer, env_ids)
                    if success:
                        # 批量优化成功，直接返回
                        pass
                    else:
                        # 批量失败，回退到原始方法
                        print(f"[INFO] Using original method as fallback...")
                        for env_idx in env_ids:
                            env_idx_int = int(env_idx.item() if hasattr(env_idx, 'item') else env_idx)
                            self._fill_transition_per_env(buffer[env_idx_int], env_idx_int)
                else:
                    # 使用原始方法（单环境或禁用优化时）
                    for env_idx in env_ids:
                        env_idx_int = int(env_idx.item() if hasattr(env_idx, 'item') else env_idx)
                        self._fill_transition_per_env(buffer[env_idx_int], env_idx_int)
            # 返回一个 info-like dict 可供 runner 进一步写 TensorBoard
        return {
            'cumulative_transitions': self._cumulative_transitions,
            'unique_traj_covered': len(self._sampled_traj_identifier_set),
            'total_traj_pool': len(getattr(self, 'all_available_trajectory_dirs', [])),
        }

    def print_performance_report(self):
        """打印性能分析报告 - 显示各个操作的耗时统计"""
        self._print_timing_report()
