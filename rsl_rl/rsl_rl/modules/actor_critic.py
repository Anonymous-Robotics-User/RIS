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

import numpy as np
import math

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.autograd import Variable
import torch.nn.functional as F

from rsl_rl.modules.encoder import StateHistoryEncoderDecoder
from rsl_rl.modules.modules import Weight_EMA_Update

class ActorCritic(nn.Module):
    def __init__(self,  num_props,
                        num_obs,
                        num_privileged_obs,
                        num_latent,
                        num_actions,
                        num_target_obs,
                        tsteps,
                        device,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)
        self.num_props = num_props
        self.num_obs = num_obs
        self.num_priv_obs = num_privileged_obs
        self.tsteps = tsteps
        
        # I/O Attributions
        mlp_input_dim_a = num_obs + num_latent + num_privileged_obs
        mlp_input_dim_c = num_obs + num_latent + num_privileged_obs
        mlp_input_dim_e = num_props
        mlp_input_dim_d = num_obs + num_latent + num_privileged_obs + num_actions

        mlp_output_dim_a = num_actions
        mlp_output_dim_e = num_latent + num_privileged_obs
        mlp_output_dim_d = num_target_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # Estimator
        self.use_target = True
        common_kwargs = dict(
            activation_fn=nn.ReLU(),
            tsteps=tsteps,
            num_privileged_obs=num_privileged_obs,
            mlp_input_dim_e=mlp_input_dim_e,
            mlp_input_dim_d=mlp_input_dim_d,
            mlp_output_dim_e=mlp_output_dim_e,
            mlp_output_dim_d=mlp_output_dim_d,
            device=device,
        )
        self.estimator = StateHistoryEncoderDecoder(**common_kwargs)
            
        if self.use_target:
            self.estimator_ema = Weight_EMA_Update(self.estimator, self.estimator.state_dict(), decay=0.999)
            self.estimator.set_target(self.estimator_ema)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, observation_history, hidden_states=None, masks=None):
        with torch.no_grad():
            latent = self.estimator(observation_history)
        
        mean = self.actor(torch.cat((observations, latent.clone().detach()), dim=-1))
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observation, observation_history, hidden_states=None, masks=None, **kwargs):
        self.update_distribution(observation, observation_history, hidden_states, masks)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, observation_history, hidden_states=None, masks=None):
        with torch.no_grad():
            latent = self.estimator.encode(observation_history)
        actions_mean = self.actor(torch.cat((observations, latent.clone().detach()), dim=-1))
        return actions_mean

    def evaluate(self, observations, privileged_observations, observation_history, hidden_states=None, masks=None, return_stats=False, **kwargs):
        if return_stats:
            latent, mu, logvar = self.estimator(observation_history, return_stats=True)
            value = self.critic(torch.cat((observations, latent[:, :-self.num_priv_obs], privileged_observations), dim=-1))
            return value, latent, mu, logvar
        else:
            latent = self.estimator(observation_history)
            value = self.critic(torch.cat((observations, latent[:, :-self.num_priv_obs], privileged_observations), dim=-1))
            return value, latent
    
    def evaluate_target(self, observations, privileged_observations, observation_history, hidden_states=None, masks=None, **kwargs):
        with torch.no_grad():
            if self.use_target:
                latent = self.estimator_ema.model.encode(observation_history)
            else:
                latent = self.estimator.encode(observation_history)
        value = self.critic(torch.cat((observations, latent[:, :-self.num_priv_obs], privileged_observations), dim=-1))
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
