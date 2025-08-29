# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics

from tensorboardX import SummaryWriter
import torch

import rsl_rl.algorithms as algorithms
import rsl_rl.modules as modules
from rsl_rl.env import VecEnv
from rsl_rl.utils import ckpt_manipulator


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        actor_critic = modules.build_actor_critic(
            self.env,
            self.cfg["policy_class_name"],
            self.policy_cfg,
        ).to(self.device)

        alg_class = getattr(algorithms, self.cfg["algorithm_class_name"])  # PPO
        self.alg: algorithms.PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions]
        )

        # Log & counters
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.log_interval = self.cfg.get("log_interval", 1)

        # Emergency cleanup related state (cooldown to avoid repeated heavy cleanups)
        self._last_emergency_cleanup_ts = 0.0
        self._emergency_cooldown = self.cfg.get("emergency_cleanup_cooldown_sec", 3600)  # 1h default

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            # forward writer to algorithm for custom logs (e.g., AW-BC)
            if hasattr(self.alg, 'writer'):
                self.alg.writer = self.writer
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rframebuffer = deque(maxlen=2000)
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        print("Initialization done, start learning.")
        # print("NOTE: you may see a bunch of `NaN or Inf found in input tensor` once and appears in the log. Just ignore it if it does not affect the performance.")
        
        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        tot_start_time = time.time()
        start = time.time()
        while self.current_learning_iteration < tot_iter:
            # Rollout
            with torch.inference_mode(self.cfg.get("inference_mode_rollout", True)):
                for i in range(self.num_steps_per_env):
                    obs, critic_obs, rewards, dones, infos = self.rollout_step(obs, critic_obs)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rframebuffer.extend(rewards[dones < 1].cpu().numpy().tolist())
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            losses, stats = self.alg.update(self.current_learning_iteration)
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None and self.current_learning_iteration % self.log_interval == 0:
                self.log(locals())
            if self.current_learning_iteration % self.save_interval == 0 and self.current_learning_iteration > start_iter:
                if self.log_dir is not None:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            ep_infos.clear()
            self.current_learning_iteration = self.current_learning_iteration + 1
            start = time.time()
        
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def rollout_step(self, obs, critic_obs):
        actions = self.alg.act(obs, critic_obs)
        obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
        self.alg.process_env_step(rewards, dones, infos, obs, critic_obs)
        return obs, critic_obs, rewards, dones, infos

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time = time.time() - locs['tot_start_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                if "_max" in key:
                    infotensor = infotensor[~infotensor.isnan()]
                    value = torch.max(infotensor) if len(infotensor) > 0 else torch.tensor(float("nan"))
                elif "_min" in key:
                    infotensor = infotensor[~infotensor.isnan()]
                    value = torch.min(infotensor) if len(infotensor) > 0 else torch.tensor(float("nan"))
                else:
                    value = torch.nanmean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, self.current_learning_iteration)
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.action_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        for k, v in locs["losses"].items():
            self.writer.add_scalar("Loss/" + k, v.item(), self.current_learning_iteration)
        for k, v in locs["stats"].items():
            self.writer.add_scalar("Train/" + k, v.item(), self.current_learning_iteration)
        
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, self.current_learning_iteration)
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), self.current_learning_iteration)
        self.writer.add_scalar('Perf/total_fps', fps, self.current_learning_iteration)
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], self.current_learning_iteration)
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], self.current_learning_iteration)
        self.writer.add_scalar('Perf/gpu_allocated', torch.cuda.memory_allocated(self.device) / 1024 ** 3, self.current_learning_iteration)
        self.writer.add_scalar('Perf/gpu_global_free_mem', torch.cuda.mem_get_info(self.device)[0] / 1024 ** 3, self.current_learning_iteration)
        self.writer.add_scalar('Perf/gpu_total', torch.cuda.mem_get_info(self.device)[1] / 1024 ** 3, self.current_learning_iteration)
        self.writer.add_scalar('Train/mean_reward_each_timestep', statistics.mean(locs['rframebuffer']), self.current_learning_iteration)
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), self.current_learning_iteration)
            self.writer.add_scalar('Train/ratio_above_mean_reward', statistics.mean([(1. if rew > statistics.mean(locs['rewbuffer']) else 0) for rew in locs['rewbuffer']]), self.current_learning_iteration)
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), self.current_learning_iteration)
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {self.current_learning_iteration}/{locs['tot_iter']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            )
            for k, v in locs["losses"].items():
                log_string += f"""{k:>{pad}} {v.item():.4f}\n"""
            # Advantage / AW-BC related stats (if provided by algorithm)
            adv_stats_keys = [
                'advantage_weights_mean', 'advantage_weights_std', 'advantage_weights_max',
                'high_weight_ratio', 'unweighted_dist_loss'
            ]
            if any(k in locs["stats"] for k in adv_stats_keys):
                log_string += f"""{'Advantage weights:':>{pad}}\n"""
                for k in adv_stats_keys:
                    if k in locs["stats"]:
                        try:
                            log_string += f"""  {k:>{pad-2}} {locs['stats'][k].item():.4f}\n"""
                        except Exception:
                            # fallback for non-tensor values
                            log_string += f"""  {k:>{pad-2}} {locs['stats'][k]}\n"""
            log_string += (
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                # f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                # f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            )
            for k, v in locs["losses"].items():
                log_string += f"""{k:>{pad}} {v.item():.4f}\n"""
            # Advantage / AW-BC related stats (if provided by algorithm)
            adv_stats_keys = [
                'advantage_weights_mean', 'advantage_weights_std', 'advantage_weights_max',
                'high_weight_ratio', 'unweighted_dist_loss'
            ]
            if any(k in locs["stats"] for k in adv_stats_keys):
                log_string += f"""{'Advantage weights:':>{pad}}\n"""
                for k in adv_stats_keys:
                    if k in locs["stats"]:
                        try:
                            log_string += f"""  {k:>{pad-2}} {locs['stats'][k].item():.4f}\n"""
                        except Exception:
                            # fallback for non-tensor values
                            log_string += f"""  {k:>{pad-2}} {locs['stats'][k]}\n"""
            log_string += (
                f"""{'Value function loss:':>{pad}} {locs["losses"]['value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs["losses"]['surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                # f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                # f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (self.current_learning_iteration + 1 - locs["start_iter"]) * (
                               locs['tot_iter'] - self.current_learning_iteration):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        """Attempt to save model.

        If disk quota / no-space error occurs:
          1) Run emergency cleanup (respect a cooldown to avoid spamming).
          2) Sleep 10 minutes (configurable via self.cfg.get('emergency_wait_minutes', 10)).
          3) Retry up to max_retries.
        If still failing due to disk space, skip this checkpoint gracefully.
        """
        import tempfile, shutil, subprocess, sys

        run_state_dict = self.alg.state_dict()
        run_state_dict.update({'iter': self.current_learning_iteration, 'infos': infos})

        os.makedirs(os.path.dirname(path), exist_ok=True)

        max_retries = 3
        attempt = 0
        disk_error_happened = False
        wait_minutes = self.cfg.get('emergency_wait_minutes', 10)

        while attempt < max_retries:
            temp_path = None
            try:
                temp_dir = os.path.dirname(path)
                with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix='.pt.tmp') as temp_file:
                    temp_path = temp_file.name
                torch.save(run_state_dict, temp_path)
                shutil.move(temp_path, path)
                print(f"✅ Model successfully saved to {path}")
                return
            except Exception as e:
                error_msg = str(e)
                print(f"⚠️ Save attempt {attempt + 1}/{max_retries} failed: {error_msg}")
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

                is_disk_quota_error = any(tok in error_msg for tok in [
                    'Disk quota exceeded', 'No space left on device', '[Errno 122]', '[Errno 28]'
                ])

                if is_disk_quota_error:
                    disk_error_happened = True
                    now = time.time()
                    if now - self._last_emergency_cleanup_ts > self._emergency_cooldown:
                        print("🚨 \033[1;91mDisk quota / no space detected. Triggering emergency cleanup.\033[0m")
                        self._last_emergency_cleanup_ts = now
                        # Run external cleaner if present; fallback to inline.
                        try:
                            current_dir = os.path.dirname(os.path.abspath(__file__))
                            cleaner_script = os.path.join(current_dir, '..', 'utils', 'emergency_cleaner.py')
                            if os.path.exists(cleaner_script):
                                result = subprocess.run([sys.executable, cleaner_script], capture_output=True, text=True, timeout=600)
                                print("---- emergency_cleaner stdout (truncated 500 chars) ----")
                                if result.stdout:
                                    print(result.stdout[:500])
                                if result.returncode != 0:
                                    print(f"⚠️ emergency_cleaner exit code {result.returncode}; stderr: {result.stderr[:300]}")
                            else:
                                print(f"⚠️ Cleaner script missing at {cleaner_script}, using inline cleanup.")
                                self._inline_emergency_cleanup()
                        except Exception as ce:
                            print(f"❌ External emergency cleanup failed: {ce}; using inline fallback.")
                            self._inline_emergency_cleanup()
                    else:
                        remaining = int(self._emergency_cooldown - (now - self._last_emergency_cleanup_ts))
                        print(f"⏳ Skipping emergency cleanup (cooldown active, {remaining}s remaining).")

                    # Always wait after a disk error (single long sleep)
                    print(f"⏸️  Pausing training for {wait_minutes} minutes to allow space reclamation...")
                    for m in range(wait_minutes):
                        time.sleep(60)
                        if (m + 1) % 2 == 0 or m == wait_minutes - 1:
                            print(f"  ... waited {m + 1}/{wait_minutes} minutes")
                    print("🔄 Resuming and retrying save...")
                else:
                    # Non-disk errors: small backoff
                    time.sleep(2)

            attempt += 1

        # Exited loop without success
        print(f"❌ Failed to save model after {max_retries} attempts")
        if disk_error_happened:
            print("⚠️ \033[1;93mSkipping this checkpoint due to disk space issues; training continues.\033[0m")
            return
        # Non-disk error: propagate
        raise RuntimeError(f"Failed to save checkpoint to {path}")
    
    def _inline_emergency_cleanup(self):
        """
        Inline emergency cleanup when external script is not available.
        """
        import re
        from pathlib import Path
        
        try:
            # 基础数据目录 - 根据你的设置调整
            base_data_dir = "/cs/student/projects2/rai/2024/hongboli/network_test/data"
            
            if not os.path.isdir(base_data_dir):
                print(f"❌ Data directory not found: {base_data_dir}")
                return
            
            trajectory_pattern = re.compile(r"trajectory_(\d+)")
            emergency_delete_count = 1000  # 紧急情况下删除更多
            total_deleted = 0
            
            print(f"🔍 Scanning {base_data_dir} for cleanup...")
            
            # 遍历任务文件夹
            for task_entry in os.scandir(base_data_dir):
                if not task_entry.is_dir():
                    continue
                
                task_name = task_entry.name
                print(f"  📁 Checking task: {task_name}")
                
                # 遍历子目录
                for sub_dir_entry in os.scandir(task_entry.path):
                    if not sub_dir_entry.is_dir():
                        continue
                    
                    # 获取轨迹文件夹
                    trajectories = []
                    try:
                        for traj_entry in os.scandir(sub_dir_entry.path):
                            if traj_entry.is_dir():
                                match = trajectory_pattern.match(traj_entry.name)
                                if match:
                                    traj_num = int(match.group(1))
                                    # 检查是否有pickle文件
                                    try:
                                        files = os.listdir(traj_entry.path)
                                        has_pickle = any(f.endswith('.pickle') or f.endswith('.pkl') for f in files)
                                        if has_pickle:
                                            trajectories.append((traj_num, traj_entry.path))
                                    except OSError:
                                        continue
                    except OSError:
                        continue
                    
                    if len(trajectories) > 500:  # 如果有很多轨迹文件
                        trajectories.sort(key=lambda x: x[0])  # 按编号排序
                        to_delete = trajectories[:emergency_delete_count]
                        
                        print(f"    🗑️  Deleting {len(to_delete)} old trajectories from {sub_dir_entry.name}")
                        
                        for traj_num, traj_path in to_delete:
                            try:
                                if os.path.exists(traj_path):
                                    files = os.listdir(traj_path)
                                    pickle_files = [f for f in files if f.endswith('.pickle') or f.endswith('.pkl')]
                                    
                                    for pickle_file in pickle_files:
                                        pickle_path = os.path.join(traj_path, pickle_file)
                                        try:
                                            os.remove(pickle_path)
                                            total_deleted += 1
                                        except OSError:
                                            pass
                                    
                                    # 删除空目录
                                    remaining = os.listdir(traj_path)
                                    if not remaining:
                                        os.rmdir(traj_path)
                            except OSError:
                                pass
            
            print(f"✅ Inline emergency cleanup completed. Deleted {total_deleted} pickle files.")
            
        except Exception as e:
            print(f"❌ Inline emergency cleanup failed: {e}")

    def load(self, path, load_optimizer=True):
        print(f"\033[1;36m Loaded model from {path} at iteration {self.current_learning_iteration} \033[0m")
        loaded_dict = torch.load(path)
        if self.cfg.get("ckpt_manipulator", False):
            # suppose to be a string specifying which function to use
            print("\033[1;36m Warning: using a hacky way to load the model. \033[0m")
            loaded_dict = getattr(ckpt_manipulator, self.cfg["ckpt_manipulator"])(
                loaded_dict,
                self.alg.state_dict(),
            )
            print("\033[1;36m Done: using a hacky way to load the model. \033[0m")
        self.alg.load_state_dict(loaded_dict)
        self.current_learning_iteration = loaded_dict['iter']

        # print log info
        
        if self.cfg.get("ckpt_manipulator", False):
            try:
                if self.log_dir is not None:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            except:
                print("\033[1;36m Save manipulated checkpoint failed, ignored... \033[0m")
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
