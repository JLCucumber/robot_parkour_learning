import os
import re
import torch
import numpy as np
import pickle
import json
import time
import traceback
from collections import OrderedDict, defaultdict
from contextlib import contextmanager

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
        
        # ⏱️ 时间测量系统 (Baseline版本)
        self._timing_stats = {
            'directory_scanning': [],
            'file_sorting': [],
            'data_loading': [],
            'pickle_loading': [],
            'tensor_conversion': [],
            'buffer_allocation': [],
            'tensor_copying': [],
            'metadata_processing': [],
            'trajectory_refresh': []
        }

    @contextmanager
    def _time_operation(self, operation_name):
        """时间测量context manager (Baseline版本)"""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            if operation_name in self._timing_stats:
                self._timing_stats[operation_name].append(duration)

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

    def read_dataset_directory(self):
        """ Refresh file-related information by scanning the directory. All traj_handlers must be
        updated from attributes here.
        """
        print("RolloutDataset: reading dataset directory...")

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
        
        # ⏱️ 目录扫描时间测量
        with self._time_operation('directory_scanning'):
            for data_dir in self.data_dirs:
                if not os.path.exists(data_dir):
                    os.mkdir(data_dir)
                    print("RolloutDataset: {} not found, created...".format(data_dir))
                for root, dirs, _ in os.walk(data_dir):
                    for d in dirs:
                        if d.startswith("trajectory_"):
                            dir_path = os.path.join(root, d)
                            file_list = os.listdir(dir_path)
                            if not file_list:
                                continue
                            traj_pickles = [f for f in file_list if self._is_main_traj_file(f)]
                            if not traj_pickles:
                                continue
                            if any(f.endswith('.tmp') for f in traj_pickles):
                                print(f"RolloutDataset: skipping {dir_path} due to a .tmp file being present.")
                                continue
                            self.all_available_trajectory_dirs.append(dir_path)
                            traj_pickles.sort(key=lambda x: self.get_frame_range(x)[1])
                            try:
                                total_timesteps += self.get_frame_range(traj_pickles[-1])[1]
                            except Exception:
                                pass
                    
                    # ⏱️ 元数据处理时间测量
                    with self._time_operation('metadata_processing'):
                        try:
                            meta_path = os.path.join(root, "metadata.json")
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
            self.all_available_trajectory_dirs.sort(key= lambda x: os.path.getmtime(x))
            
        current_dir_set = set(self.all_available_trajectory_dirs)
        new_dirs = current_dir_set - self._last_dir_set
        reused_dirs = len(self.all_available_trajectory_dirs) - len(new_dirs)
        self._scan_index += 1
        print("RolloutDataset: {} trajectories found. {} timesteps in total.".format(
            len(self.all_available_trajectory_dirs),
            total_timesteps,
        ))
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
                # optional verbose once
                print("RolloutDataset: All metadata files consistent ({} files).".format(len(self._all_metadatas)))

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
                print(f"RolloutDataset: per-run trajectory counts (timestamps only): {dist_str}")

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

            print(f"RolloutDataset: detected data collector dirs: {len(run_root_suffix_info)}")
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
        """
        # ⏱️ 轨迹刷新时间测量
        with self._time_operation('trajectory_refresh'):
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
                        # ⏱️ pickle加载时间测量
                        with self._time_operation('pickle_loading'):
                            with open(traj_path, "rb") as f:
                                traj_data = pickle.load(f)
                        break  # Successfully loaded, exit loop
                except EOFError:
                    print(f"[Attempt {attempt}] Caught EOFError, file might still be writing: {traj_path}")
                except Exception:
                    traceback.print_exc()
                    print(f"[Attempt {attempt}] Other error occurred, retrying: {traj_path}")

                time.sleep(delay)
            else:
                raise RuntimeError(f"[RolloutDataset] Failed to load trajectory file after multiple attempts: {traj_path}")

            if "obs_disassemble_mapping" in self.metadata.keys():
                traj_data = self.assemble_obs_components(traj_data)

            # 新增：确保包含优势字段（向后兼容旧数据）
            if 'advantages' not in traj_data:
                traj_data['advantages'] = np.zeros_like(traj_data['rewards'])
                traj_data['positive_advantages'] = np.zeros_like(traj_data['rewards']) 
                traj_data['difficulty_scores'] = np.zeros_like(traj_data['rewards'])

            # ⏱️ tensor转换时间测量
            with self._time_operation('tensor_conversion'):
                for k, v in traj_data.items():
                    traj_data[k] = torch.from_numpy(v).to(self.device)
            self.traj_datas[env_idx] = traj_data

    def _refresh_traj_handler(self, env_idx):
        """ update traj_handler for the given env and load the first traj_data. It does not update
        the traj_identifiers.
        """
        traj_dir = self.all_available_trajectory_dirs[self.traj_identifiers[env_idx]]
        trajectory_files = [f for f in os.listdir(traj_dir) if self._is_main_traj_file(f)]
        if not trajectory_files:
            raise RuntimeError(f"RolloutDataset: no valid main trajectory pickle files in {traj_dir}")
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
        self._refresh_traj_data(env_idx)
        self.traj_datas[env_idx]["dones"][0] = True # set the first frame as done

    def refresh_handlers(self, env_ids= None):
        # this method is called ONLY at the start of a new bunch of trajectories are init.
        # print()
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
                        print(f"RolloutDataset: ran out of trajectories (used {total}{suffix}). Reloading dataset...")
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
                self._refresh_traj_data(env_idx)
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
        leading_dims = ([] if num_transitions_per_env is None else [num_transitions_per_env]) + [self.num_envs]
        if not hasattr(self, "_output_transition_buffer") or self._output_transition_buffer_leading_dims != leading_dims:
            # ⏱️ 缓冲区分配时间测量
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
                    dtype= bool,
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
                )
        return self._output_transition_buffer
    
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
        # ⏱️ 数据加载时间测量
        with self._time_operation('data_loading'):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device= self.device)
            # ⏱️ tensor复制时间测量
            with self._time_operation('tensor_copying'):
                for env_idx in env_ids:
                    self._fill_transition_per_env(buffer[env_idx], env_idx)
            # 返回一个 info-like dict 可供 runner 进一步写 TensorBoard
        return {
            'cumulative_transitions': self._cumulative_transitions,
            'unique_traj_covered': len(self._sampled_traj_identifier_set),
            'total_traj_pool': len(getattr(self, 'all_available_trajectory_dirs', [])),
        }
    
    def print_timing_stats(self, print_to_console=False, log_file_path=None):
        """打印性能统计 (Baseline版本)"""
        if not any(self._timing_stats.values()):
            msg = "📊 [BASELINE] No timing data available yet."
            if print_to_console:
                print(msg)
            if log_file_path:
                try:
                    with open(log_file_path, 'a') as f:
                        f.write(msg + '\n')
                except Exception as e:
                    print(f"Failed to write to log file {log_file_path}: {e}")
            return
        
        lines = []
        lines.append("=" * 60)
        lines.append("📊 RolloutDataset Performance Report (BASELINE)")
        lines.append("=" * 60)
        
        total_time = 0
        category_times = {}
        
        for category, times in self._timing_stats.items():
            if times:
                total_cat_time = sum(times)
                avg_time = total_cat_time / len(times)
                category_times[category] = total_cat_time
                total_time += total_cat_time
                
                lines.append(f"🔹 {category.replace('_', ' ').title()}:")
                lines.append(f"   Total: {total_cat_time:.4f}s | Calls: {len(times)} | Avg: {avg_time:.4f}s")
        
        lines.append("-" * 60)
        lines.append(f"⏱️  Total Measured Time: {total_time:.4f}s")
        
        if total_time > 0:
            lines.append("📈 Time Distribution:")
            for category, cat_time in sorted(category_times.items(), key=lambda x: x[1], reverse=True):
                percentage = (cat_time / total_time) * 100
                lines.append(f"   {category.replace('_', ' ').title()}: {percentage:.1f}%")
        
        lines.append("=" * 60)
        
        # 输出处理
        report = '\n'.join(lines)
        
        if print_to_console:
            print(report)
            
        if log_file_path:
            try:
                with open(log_file_path, 'a') as f:
                    f.write(f"\n{report}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                if not print_to_console:
                    print(f"📝 Performance report saved to: {log_file_path}")
            except Exception as e:
                print(f"Failed to write to log file {log_file_path}: {e}")
                if not print_to_console:
                    print(report)
        elif not print_to_console:
            # 默认保存位置
            try:
                default_log = "performance_baseline.log"
                with open(default_log, 'a') as f:
                    f.write(f"\n{report}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                print(f"📝 Performance report saved to: {default_log}")
            except Exception as e:
                print(f"Failed to write to default log: {e}")
                print(report)
