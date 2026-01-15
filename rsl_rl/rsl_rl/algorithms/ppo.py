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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 estimator_loss_coef=0.25,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.use_grad_penalty = False
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=learning_rate * 0.5)
        self.critic_optimizer = optim.Adam([
            {'params': self.actor_critic.critic.parameters(), 'lr': learning_rate * 3},
            {'params': self.actor_critic.estimator.parameters(), 'lr': learning_rate},
        ])
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.estimator_loss_coef = estimator_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape, action_shape, num_target_obs):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape, action_shape, num_target_obs, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, privileged_obs, obs_history):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, obs_history).detach()
        self.transition.values = self.actor_critic.evaluate_target(obs, privileged_obs, obs_history).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs # for Decoder
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_history = obs_history
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, next_obs):
        self.transition.next_observations = next_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
    
    def compute_returns(self, last_critic_obs, last_critic_privileged_obs, last_obs_history):
        last_values= self.actor_critic.evaluate_target(last_critic_obs, last_critic_privileged_obs, last_obs_history).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def check_grad_norm(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2) # L2 norm
                total_norm += param_norm.item()**2
        total_norm = total_norm**0.5
        return total_norm

    def _calc_grad_penalty(self, obs_batch, actions_log_prob_batch):
        grad_log_prob = torch.autograd.grad(actions_log_prob_batch.sum(), obs_batch, create_graph=True)[0]
        gradient_penalty_loss = torch.sum(torch.square(grad_log_prob), dim=-1).mean()
        return gradient_penalty_loss

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_decoder_loss = 0
        mean_info_loss = 0
        mean_consistency_loss = 0

        mean_actor_grad_norm = 0
        mean_critic_grad_norm = 0
        mean_estimator_grad_norm = 0
        mean_grad_penalty = 0
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, privileged_obs_batch, obs_history_batch, next_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hidden_state_batch, masks_batch in generator:
                
                # EMA update (Target Encoder)
                if self.actor_critic.use_target:
                    self.actor_critic.estimator_ema.update(self.actor_critic.estimator.state_dict())

                # PPO Step
                if self.use_grad_penalty:
                    obs_batch.requires_grad_()
                self.actor_critic.act(obs_batch, obs_history_batch, hidden_state_batch, masks_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch, latent_batch, enc_mu_batch, enc_logvar_batch = self.actor_critic.evaluate(obs_batch.clone(), 
                                                                                                       privileged_obs_batch.clone(), 
                                                                                                       obs_history_batch.clone(), 
                                                                                                       hidden_state_batch, 
                                                                                                       masks_batch, 
                                                                                                       return_stats=True)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.actor_optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                        for param_group in self.critic_optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                
                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # Estimator Update
                estimation_loss, decoder_loss, info_loss, consistency_loss = self.actor_critic.estimator.update(obs_batch.clone().detach(), 
                                                                                                                latent_batch, 
                                                                                                                actions_batch.clone().detach(), 
                                                                                                                privileged_obs_batch, 
                                                                                                                next_observations_batch, 
                                                                                                                obs_history_batch.clone(),
                                                                                                                enc_mu_batch,
                                                                                                                enc_logvar_batch,
                                                                                                                hidden_state_batch,
                                                                                                                masks_batch
                                                                                                            )

                # Gradient step
                estimator_loss = self.estimator_loss_coef * (estimation_loss + decoder_loss + info_loss + consistency_loss)
                critic_loss = self.value_loss_coef * value_loss

                abstraction_loss = critic_loss + estimator_loss
                self.critic_optimizer.zero_grad()
                abstraction_loss.backward()
                critic_grad_norm = self.check_grad_norm(self.actor_critic.critic)
                estimator_grad_norm = self.check_grad_norm(self.actor_critic.estimator)
                nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actor_critic.estimator.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()      

                if self.use_grad_penalty:
                    grad_penalty = self._calc_grad_penalty(obs_batch, actions_log_prob_batch)
                    mean_grad_penalty += grad_penalty.item()
                    actor_loss = surrogate_loss - self.entropy_coef * entropy_batch.mean() + 0.0000025 * grad_penalty
                else:
                    actor_loss = surrogate_loss - self.entropy_coef * entropy_batch.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad_norm = self.check_grad_norm(self.actor_critic.actor)
                nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.max_grad_norm * 0.7)
                self.actor_optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_estimation_loss += estimation_loss.item()
                mean_decoder_loss += decoder_loss.item()
                mean_info_loss += info_loss.item()
                mean_consistency_loss += consistency_loss.item()

                mean_actor_grad_norm += actor_grad_norm
                mean_critic_grad_norm += critic_grad_norm
                mean_estimator_grad_norm += estimator_grad_norm

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_decoder_loss /= num_updates
        mean_info_loss /= num_updates
        mean_consistency_loss /= num_updates
        mean_actor_grad_norm /= num_updates
        mean_critic_grad_norm /= num_updates
        mean_estimator_grad_norm /= num_updates
        mean_grad_penalty /= num_updates if self.use_grad_penalty else 1
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_decoder_loss, mean_info_loss, mean_consistency_loss, \
            mean_actor_grad_norm, mean_critic_grad_norm, mean_estimator_grad_norm, mean_grad_penalty
