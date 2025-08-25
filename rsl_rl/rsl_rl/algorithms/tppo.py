""" PPO with teacher network """

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import rsl_rl.modules as modules
from rsl_rl.utils import unpad_trajectories
from rsl_rl.storage.rollout_storage import ActionLabelRollout
from rsl_rl.algorithms.ppo import PPO
from legged_gym import LEGGED_GYM_ROOT_DIR


# assuming learning iteration is at an assumable iteration scale
def GET_PROB_FUNC(option, iteration_scale):
    PROB_options = {
        "linear": (lambda x: max(0.0, 1 - 1 / iteration_scale * x)),
        "exp": (lambda x: max(0.0, (1 - 1 / iteration_scale) ** x)),
        "tanh": (lambda x: max(0.0, 0.5 * (1 - math.tanh(1 / iteration_scale * (x - iteration_scale))))),
    }
    return PROB_options[option]


class TPPO(PPO):
    def __init__(self,
            *args,
            teacher_ac_path= None, # running device will be handled
            teacher_policy_class_name= "ActorCritic",
            teacher_policy= dict(),
            label_action_with_critic_obs= True, # else, use actor obs
            teacher_act_prob= "exp", # a number or a callable to (0 ~ 1) to the selection of act using teacher policy
            update_times_scale= 100, # a rough estimation of how many times the update will be called
            using_ppo= True, # If False, compute_losses will skip ppo loss computation and returns to DAGGR
            distillation_loss_coef= 1., # can also be string to select a prob function to scale the distillation loss
            distill_target= "real",
            distill_latent_coef= 1.,
            distill_latent_target= "real",
            distill_latent_obs_component_mapping= None,
            buffer_dilation_ratio= 1.,
            lr_scheduler_class_name= None,
            lr_scheduler= dict(),
            hidden_state_resample_prob= 0.0, # if > 0, Some hidden state in the minibatch will be resampled
            action_labels_from_sample= False, # if True, the action labels from teacher policy will be from policy.act instead of policy.act_inference
            # AW-BC options
            awbc_weighting: bool = True,
            awbc_weight_type: str = "percentile",  # 'percentile' | 'softmax'
            awbc_percentile: int = 95,              # used when percentile
            awbc_weight_clip: float = 1.0,          # clamp max weight
            # Audit config (Plan A)
            awbc_audit: dict = None,
            **kwargs,
        ):
        """
        Args:
        - distill_latent_obs_component_mapping: a dict of
            {student_obs_component_name: teacher_obs_component_name}
            only when both policy are the instance of EncoderActorCriticMixin
        """
        super().__init__(*args, **kwargs)
        self.label_action_with_critic_obs = label_action_with_critic_obs
        self.teacher_act_prob = teacher_act_prob
        self.update_times_scale = update_times_scale
        if isinstance(self.teacher_act_prob, str):
            self.teacher_act_prob = GET_PROB_FUNC(self.teacher_act_prob, update_times_scale)
        else:
            self.__teacher_act_prob = self.teacher_act_prob
            self.teacher_act_prob = lambda x: self.__teacher_act_prob
        self.using_ppo = using_ppo
        self.__distillation_loss_coef = distillation_loss_coef
        if isinstance(self.__distillation_loss_coef, str):
            self.distillation_loss_coef_func = GET_PROB_FUNC(self.__distillation_loss_coef, update_times_scale)
        self.distill_target = distill_target
        self.distill_latent_coef = distill_latent_coef
        self.distill_latent_target = distill_latent_target
        self.distill_latent_obs_component_mapping = distill_latent_obs_component_mapping
        self.buffer_dilation_ratio = buffer_dilation_ratio
        self.lr_scheduler_class_name = lr_scheduler_class_name
        self.lr_scheduler_kwargs = lr_scheduler
        self.hidden_state_resample_prob = hidden_state_resample_prob
        self.action_labels_from_sample = action_labels_from_sample
        self.transition = ActionLabelRollout.Transition()
        # AW-BC hyperparameters
        self.awbc_weighting = awbc_weighting
        self.awbc_weight_type = awbc_weight_type
        self.awbc_percentile = awbc_percentile
        self.awbc_weight_clip = awbc_weight_clip
        # optional writer for custom logs (set by runner)
        self.writer = None

        # Plan A: AW-BC audit configuration
        self._awbc_audit_enable = False
        self._awbc_audit_every = 1000
        self._awbc_audit_per_iter = 64
        self._awbc_audit_save_dir = None
        self._awbc_audit_position_key = None
        self._awbc_audit_file_count = 0
        if isinstance(awbc_audit, dict):
            self._awbc_audit_enable = bool(awbc_audit.get('enable', False))
            self._awbc_audit_every = int(awbc_audit.get('every', 1000))
            self._awbc_audit_per_iter = int(awbc_audit.get('per_iter', 64))
            self._awbc_audit_save_dir = awbc_audit.get('save_dir', None)
            self._awbc_audit_position_key = awbc_audit.get('position_obs_key', None)

        # Visibility: print whether AW-BC is enabled for this run
        try:
            print(f"\033[1;92m[DEBUG] - [TPPO] AW-BC enabled: {bool(self.awbc_weighting)} | type={self.awbc_weight_type} | pctl={self.awbc_percentile} | clip={self.awbc_weight_clip}\033[0m")
        except Exception:
            pass

        # build and load teacher network
        teacher_actor_critic = getattr(modules, teacher_policy_class_name)(**teacher_policy)
        if not teacher_ac_path is None:
            if "{LEGGED_GYM_ROOT_DIR}" in teacher_ac_path:
                teacher_ac_path = teacher_ac_path.format(LEGGED_GYM_ROOT_DIR= LEGGED_GYM_ROOT_DIR)
            state_dict = torch.load(teacher_ac_path, map_location= "cpu")
            teacher_actor_critic_state_dict = state_dict["model_state_dict"]
            teacher_actor_critic.load_state_dict(teacher_actor_critic_state_dict)
        else:
            print("TPPO Warning: No snapshot loaded for teacher policy. Make sure you have a pretrained teacher network")
        teacher_actor_critic.to(self.device)
        self.teacher_actor_critic = teacher_actor_critic
        self.teacher_actor_critic.eval()

        # initialize lr scheduler if needed
        if not self.lr_scheduler_class_name is None:
            self.lr_scheduler = getattr(optim.lr_scheduler, self.lr_scheduler_class_name)(
                self.optimizer,
                **self.lr_scheduler_kwargs,
            )

    def init_storage(self, *args, **kwargs):
        self.storage = ActionLabelRollout(
            *args,
            **kwargs,
            buffer_dilation_ratio= self.buffer_dilation_ratio,
            device= self.device,
        )

    def act(self, obs, critic_obs):
        # get actions via base (fills self.transition.*)
        super().act(obs, critic_obs)
        if self.label_action_with_critic_obs and self.action_labels_from_sample:
            self.transition.action_labels = self.teacher_actor_critic.act(critic_obs).detach()
        elif self.label_action_with_critic_obs:
            self.transition.action_labels = self.teacher_actor_critic.act_inference(critic_obs).detach()
        elif self.action_labels_from_sample:
            self.transition.action_labels = self.teacher_actor_critic.act(obs).detach()
        else:
            self.transition.action_labels = self.teacher_actor_critic.act_inference(obs).detach()

        # decide whose action to use
        if not hasattr(self, "use_teacher_act_mask"):
            self.use_teacher_act_mask = torch.ones(obs.shape[0], device= self.device, dtype= torch.bool)
        # replace selected actions with teacher labels
        if isinstance(self.transition.actions, torch.Tensor) and isinstance(self.transition.action_labels, torch.Tensor):
            mask = self.use_teacher_act_mask
            if mask.dim() == 1:
                mask = mask.unsqueeze(-1)
            # in-place blend
            self.transition.actions.copy_(torch.where(mask, self.transition.action_labels, self.transition.actions))
            return self.transition.actions
        else:
            # fallback: return teacher labels
            return self.transition.action_labels
    
    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        return_ = super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)
        self.teacher_actor_critic.reset(dones)
        # resample teacher action mask for those dones env
        prob = self.teacher_act_prob(self.current_learning_iteration) if callable(self.teacher_act_prob) else float(self.teacher_act_prob)
        self.use_teacher_act_mask[dones] = torch.rand(dones.sum(), device= self.device) < prob
        return return_

    def collect_transition_from_dataset(self, transition, infos):
        """ The interface to collect transition from dataset rather than env """
        super().act(transition.observation, transition.privileged_observation)
        self.transition.action_labels = transition.action

        # newly added
        if hasattr(transition, 'teacher_advantages') and transition.teacher_advantages is not None:
            # Normalize shapes to [B, 1] and move to correct device
            def _to_col(x):
                if x is None:
                    return None
                if isinstance(x, (tuple, list)):
                    x = torch.as_tensor(x)
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=torch.float32)
                x = x.to(self.device)
                if x.dim() == 1:
                    x = x.unsqueeze(-1)
                return x

            self.transition.teacher_advantages = _to_col(transition.teacher_advantages)
            self.transition.positive_advantages = _to_col(transition.positive_advantages)
            self.transition.difficulty_scores = _to_col(transition.difficulty_scores)
        
        super().process_env_step(
            transition.reward, 
            transition.done, 
            infos, 
            transition.next_observation, 
            transition.next_privileged_observation
        )

    def compute_returns(self, last_critic_obs):
        if not self.using_ppo:
            return
        return super().compute_returns(last_critic_obs)
    
    def update(self, *args, **kwargs):
        return_ = super().update(*args, **kwargs)
        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.step()
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
        return return_
    
    def compute_losses(self, minibatch):
        if self.hidden_state_resample_prob > 0.0:
            # assuming the hidden states are from LSTM or GRU, which are always betwein -1 and 1
            hidden_state_example = minibatch.hidden_states[0][0] if isinstance(minibatch.hidden_states[0], tuple) else minibatch.hidden_states[0]
            resample_mask = torch.rand(hidden_state_example.shape[1], device= self.device) < self.hidden_state_resample_prob
            # for each hidden state, resample from -1 to 1
            if isinstance(minibatch.hidden_states[0], tuple):
                # for LSTM not tested
                # iterate through actor and critic hidden state
                minibatch = minibatch._replace(hidden_states= tuple(
                    tuple(
                        torch.where(
                            resample_mask.unsqueeze(-1).unsqueeze(-1),
                            torch.rand_like(minibatch.hidden_states[i][j], device= self.device) * 2 - 1,
                            minibatch.hidden_states[i][j],
                        ) for j in range(len(minibatch.hidden_states[i]))
                    ) for i in range(len(minibatch.hidden_states))
                ))
            else:
                # for GRU
                # iterate through actor and critic hidden state
                minibatch = minibatch._replace(hidden_states= tuple(
                    torch.where(
                        resample_mask.unsqueeze(-1),
                        torch.rand_like(minibatch.hidden_states[i], device= self.device) * 2 - 1,
                        minibatch.hidden_states[i],
                    ) for i in range(len(minibatch.hidden_states))
                ))

        if self.using_ppo:
            losses, inter_vars, stats = super().compute_losses(minibatch)
        else:
            losses = dict()
            inter_vars = dict()
            stats = dict()
            self.actor_critic.act(minibatch.obs, masks=minibatch.masks, hidden_states=minibatch.hidden_states.actor)

        # distillation loss (with teacher actor)
        if self.distill_target == "real":
            dist_loss = torch.norm(
                self.actor_critic.action_mean - minibatch.action_labels,
                dim= -1
            )
        elif self.distill_target == "mse_sum":
            dist_loss = F.mse_loss(
                self.actor_critic.action_mean,
                minibatch.action_labels,
                reduction= "none",
            ).sum(-1)
        elif self.distill_target == "l1":
            dist_loss = torch.norm(
                self.actor_critic.action_mean - minibatch.action_labels,
                dim= -1,
                p= 1,
            )
        elif self.distill_target == "tanh":
            # for tanh, similar to loss function for sigmoid, refer to https://stats.stackexchange.com/questions/12754/matching-loss-function-for-tanh-units-in-a-neural-net
            dist_loss = F.binary_cross_entropy(
                (self.actor_critic.action_mean + 1) * 0.5,
                (minibatch.action_labels + 1) * 0.5,
            ) * 2
        elif self.distill_target == "scaled_tanh":
            l1 = torch.norm(
                self.actor_critic.action_mean - minibatch.action_labels,
                dim= -1,
                p= 1,
            )
            dist_loss = F.binary_cross_entropy(
                (self.actor_critic.action_mean + 1) * 0.5,
                (minibatch.action_labels + 1) * 0.5, # (n, t, d)
                reduction= "none",
            ).mean(-1) * 2 * l1 / self.actor_critic.action_mean.shape[-1] # (n, t)
        elif self.distill_target == "max_log_prob":
            action_labels_log_prob = self.actor_critic.get_actions_log_prob(minibatch.action_labels)
            dist_loss = -action_labels_log_prob
        elif self.distill_target == "kl":
            raise NotImplementedError()
        else:
            # fallback to L2 distance
            dist_loss = torch.norm(
                self.actor_critic.action_mean - minibatch.action_labels,
                dim=-1,
            )
        
        # Advantage-weighted BC (optional)
        if getattr(self, 'awbc_weighting', True) and hasattr(minibatch, 'positive_advantages') and minibatch.positive_advantages is not None:
            
            # DEBUG print
            advantage_weights = self.compute_advantage_weights(minibatch.positive_advantages)
            final_dist_loss = (dist_loss * advantage_weights).mean()
            if hasattr(self, 'writer') and self.writer is not None:
                self.log_advantage_weights(advantage_weights, dist_loss)
                # Optional histograms for distribution overview
                try:
                    self.writer.add_histogram('AW-BC/weights', advantage_weights, self.current_learning_iteration)
                    self.writer.add_histogram('AW-BC/positive_advantages', minibatch.positive_advantages.squeeze(-1), self.current_learning_iteration)
                except Exception:
                    pass
            stats.update({
                'advantage_weights_mean': advantage_weights.mean(),
                'advantage_weights_std': advantage_weights.std(),
                'advantage_weights_max': advantage_weights.max(),
                'high_weight_ratio': (advantage_weights > 0.7).float().mean(),
                'unweighted_dist_loss': dist_loss.mean(),
            })

            # print(f"[DEBUG] [TPPO] Enable AW-BC")
            # print(f"[DEBUG] [TPPO] Advantage weights: {advantage_weights.mean():.4f}")
            # print(f"[DEBUG] [TPPO] High weight ratio: {(advantage_weights > 0.7).float().mean():.4f}")

            # Plan A: export a small sample for offline inspection
            self._maybe_awbc_audit(minibatch, advantage_weights, dist_loss)
            
        else:

            # print("[DEBUG] [TPPO] AW-BC disabled, using unweighted distillation loss")

            final_dist_loss = dist_loss.mean()
            if hasattr(self, 'writer') and self.writer is not None:
                self.writer.add_scalar('AW-BC/enabled', float(getattr(self, 'awbc_weighting', False)), self.current_learning_iteration)

        if "tanh" in self.distill_target:
            stats["l1distance"] = torch.norm(
                self.actor_critic.action_mean - minibatch.action_labels,
                dim= -1,
                p= 1,
            ).mean().detach()
            stats["l1_before_tanh"] = torch.norm(
                torch.tan(self.actor_critic.action_mean) - torch.tan(minibatch.action_labels),
                dim= -1,
                p= 1
            ).mean().detach()

        # update distillation loss coef if applicable
        self.distillation_loss_coef = self.distillation_loss_coef_func(self.current_learning_iteration) if hasattr(self, "distillation_loss_coef_func") else self.__distillation_loss_coef
        losses["distillation_loss"] = final_dist_loss

        # distill latent embedding
        if self.distill_latent_obs_component_mapping is not None:
            for k, v in self.distill_latent_obs_component_mapping.items():
                # get the latent embedding
                latent = self.actor_critic.get_encoder_latent(
                    minibatch.obs,
                    k,
                )
                with torch.no_grad():
                    target_latent = self.teacher_actor_critic.get_encoder_latent(
                        minibatch.critic_obs,
                        v,
                    )
                if self.actor_critic.is_recurrent:
                    latent = unpad_trajectories(latent, minibatch.masks)
                    target_latent = unpad_trajectories(target_latent, minibatch.masks)
                if self.distill_latent_target == "real":
                    dist_loss = torch.norm(
                        latent - target_latent,
                        dim= -1,
                    )
                elif self.distill_latent_target == "l1":
                    dist_loss = torch.norm(
                        latent - target_latent,
                        dim= -1,
                        p= 1,
                    )
                elif self.distill_latent_target == "tanh":
                    dist_loss = F.binary_cross_entropy(
                        (latent + 1) * 0.5,
                        (target_latent + 1) * 0.5,
                    ) * 2
                elif self.distill_latent_target == "scaled_tanh":
                    l1 = torch.norm(
                        latent - target_latent,
                        dim= -1,
                        p= 1,
                    )
                    dist_loss = F.binary_cross_entropy(
                        (latent + 1) * 0.5,
                        (target_latent + 1) * 0.5, # (n, t, d)
                        reduction= "none",
                    ).mean(-1) * 2 * l1 / latent.shape[-1] # (n, t)
                else:
                    dist_loss = torch.norm(
                        latent - target_latent,
                        dim=-1,
                    )
                setattr(self, f"distill_latent_{k}_coef", self.distill_latent_coef)
                losses[f"distill_latent_{k}"] = dist_loss.mean()

        return losses, inter_vars, stats

    def compute_advantage_weights(self, positive_advantages, method=None):
        """计算优势权重，使用配置中的类型/分位数/截断。"""
        if positive_advantages.numel() == 0:
            return torch.ones_like(positive_advantages)
        
        positive_advantages = positive_advantages.squeeze(-1)
        non_zero_mask = positive_advantages > 0
        if not non_zero_mask.any():
            return torch.ones_like(positive_advantages)
        
        method = method or getattr(self, 'awbc_weight_type', 'percentile')
        if method == 'percentile':
            q = float(getattr(self, 'awbc_percentile', 95)) / 100.0
            p = torch.quantile(positive_advantages[non_zero_mask], q)
            clip = float(getattr(self, 'awbc_weight_clip', 1.0))
            weights = torch.clamp(positive_advantages / (p + 1e-8), max=clip)
        elif method == 'softmax':
            tau = torch.quantile(positive_advantages[non_zero_mask], 0.9)
            weights = torch.softmax(positive_advantages / (tau + 1e-8), dim=0) * len(positive_advantages)
            clip = float(getattr(self, 'awbc_weight_clip', 1.0))
            weights = torch.clamp(weights, max=clip)
        else:
            weights = torch.ones_like(positive_advantages)
        
        return weights.detach()

    def _maybe_awbc_audit(self, minibatch, weights, dist_loss):
        """Export a small batch sample for offline inspection of AW-BC (Plan A)."""
        if not (self._awbc_audit_enable and self._awbc_audit_save_dir):
            return
        if (self.current_learning_iteration % max(1, self._awbc_audit_every)) != 0:
            return
        try:
            os.makedirs(self._awbc_audit_save_dir, exist_ok=True)
            K = min(int(self._awbc_audit_per_iter), weights.shape[0])
            idx = torch.arange(K, device=weights.device)

            npz = {
                'iter': int(self.current_learning_iteration),
                'indices': idx.detach().cpu().numpy(),
                'weights': weights[idx].detach().cpu().numpy(),
                'pos_adv': getattr(minibatch, 'positive_advantages', None)[idx].detach().cpu().numpy().squeeze(-1)
                    if hasattr(minibatch, 'positive_advantages') and minibatch.positive_advantages is not None else None,
                'dist_loss': dist_loss[idx].detach().cpu().numpy(),
                'dist_loss_mean_unweighted': float(dist_loss.mean().detach().cpu().item()),
                'dist_loss_mean_weighted': float((dist_loss * weights).mean().detach().cpu().item()),
            }
            # If a numeric position slice is configured (start,end), include it
            if self._awbc_audit_position_key is not None:
                key = self._awbc_audit_position_key
                try:
                    if isinstance(key, (tuple, list)) and len(key) == 2:
                        start, end = int(key[0]), int(key[1])
                        src = minibatch.critic_obs if hasattr(minibatch, 'critic_obs') and minibatch.critic_obs is not None else minibatch.obs
                        npz['position_slice'] = src[idx, start:end].detach().cpu().numpy()
                except Exception:
                    pass

            path = os.path.join(self._awbc_audit_save_dir,
                                 f"awbc_audit_iter{int(self.current_learning_iteration):07d}_{self._awbc_audit_file_count:03d}.npz")
            np.savez_compressed(path, **{k: v for k, v in npz.items() if v is not None})
            self._awbc_audit_file_count += 1
            print(f"[AWBC-AUDIT] saved: {path}")
        except Exception as e:
            print(f"[AWBC-AUDIT] failed to save: {e}")

    def log_advantage_weights(self, weights, losses):
        if not hasattr(self, 'writer') or self.writer is None:
            return
        self.writer.add_scalar('AW-BC/weight_mean', weights.mean(), self.current_learning_iteration)
        self.writer.add_scalar('AW-BC/weight_std', weights.std(), self.current_learning_iteration)
        self.writer.add_scalar('AW-BC/weight_max', weights.max(), self.current_learning_iteration)
        self.writer.add_scalar('AW-BC/weight_min', weights.min(), self.current_learning_iteration)
        high_weight_ratio = (weights > 0.7).float().mean()
        self.writer.add_scalar('AW-BC/high_weight_ratio', high_weight_ratio, self.current_learning_iteration)
        # 归一化后计算有效样本量 (ESS) 与 Gini-like 集中度
        try:
            w = weights.detach().float()
            if w.numel() > 0:
                w_norm = w / (w.sum() + 1e-8)
                ess = (w_norm.sum() ** 2) / ( (w_norm ** 2).sum() + 1e-8)
                # Gini = 1 - 2 * sum_i ( (n - i + 0.5)/n * w_sorted_i )  (此处近似，不做严格偏差修正)
                w_sorted, _ = torch.sort(w_norm)
                n = w_sorted.numel()
                idx = torch.arange(1, n+1, device=w.device, dtype=w.dtype)
                gini = 1. - 2. * ( ( (n - idx + 0.5)/n * w_sorted ).sum() )
                self.writer.add_scalar('AW-BC/effective_sample_size', ess, self.current_learning_iteration)
                self.writer.add_scalar('AW-BC/effective_sample_ratio', ess / n, self.current_learning_iteration)
                self.writer.add_scalar('AW-BC/gini_like', gini, self.current_learning_iteration)
        except Exception:
            pass
        try:
            weight_quantiles = torch.quantile(weights, torch.tensor([0.5, 0.75, 0.9], device=weights.device))
            for i, q in enumerate([0.5, 0.75, 0.9]):
                mask = weights >= weight_quantiles[i]
                if mask.sum() > 0:
                    bucket_loss = losses[mask].mean()
                    self.writer.add_scalar(f'AW-BC/loss_Q{int(q*100)}', bucket_loss, self.current_learning_iteration)
        except Exception:
            pass