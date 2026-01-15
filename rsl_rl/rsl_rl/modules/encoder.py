import numpy as np
import math

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.autograd import Variable
import torch.nn.functional as F

from rsl_rl.modules.modules import Weight_EMA_Update

class StateHistoryEncoderDecoder(nn.Module):
    def __init__(self, 
                 activation_fn, 
                 tsteps, 
                 num_privileged_obs, 
                 mlp_input_dim_e, 
                 mlp_input_dim_d, 
                 mlp_output_dim_e, 
                 mlp_output_dim_d, 
                 device, 
                 encoder_dims = (32, 32, 32),
                 decoder_dims = [256, 128, 64], 
                 beta = 1e-4):
        super(StateHistoryEncoderDecoder, self).__init__()

        self.activation_fn = activation_fn
        self.tsteps = tsteps

        self.num_props = mlp_input_dim_e
        self.num_privileged_obs = num_privileged_obs
        self.latent_size = mlp_output_dim_e

        self.beta = beta

        # Encoder
        channel_size = (128, 64)
        # hidden_size = 128
        self.encoder = nn.Module()
        self.encoder.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels = mlp_input_dim_e, out_channels = channel_size[0], kernel_size = 4, stride = 2, dilation=1), self.activation_fn,
            nn.Conv1d(in_channels = channel_size[0], out_channels = channel_size[1], kernel_size = 3, stride = 2, dilation=1), self.activation_fn,
            nn.Conv1d(in_channels = channel_size[1], out_channels = channel_size[1], kernel_size = 1, stride = 1, dilation=2), self.activation_fn,
            nn.Flatten())
        self.encoder.linear_output = nn.Linear(channel_size[1] * 4, self.latent_size * 2 - num_privileged_obs)

        # Decoder
        self.mitigate_collapse = True
        self.estimate_delta = True
        if self.mitigate_collapse:
            self.decoder = nn.Sequential(
                nn.Linear(mlp_input_dim_d - self.num_props, decoder_dims[0]),
                self.activation_fn,
                nn.Linear(decoder_dims[0], decoder_dims[1]),
                self.activation_fn,
                nn.Linear(decoder_dims[1], decoder_dims[2]),
                self.activation_fn
            )
            if self.estimate_delta:
                self.decoder_linear = nn.Linear(decoder_dims[2], mlp_output_dim_d)
            else:
                self.decoder_linear = nn.Linear(decoder_dims[2] + self.num_props, mlp_output_dim_d)
        else:
            self.decoder = nn.Sequential(
                nn.Linear(mlp_input_dim_d, decoder_dims[0]),
                self.activation_fn,
                nn.Linear(decoder_dims[0], decoder_dims[1]),
                self.activation_fn,
                nn.Linear(decoder_dims[1], decoder_dims[2]),
                self.activation_fn,
                nn.Linear(decoder_dims[2], mlp_output_dim_d),
            )
        self.use_true_priv_target = True
        self.use_online_decoder = True
        self.device = device

    def forward(self, obs_history, return_stats=False):
        batch_size = obs_history.size(0)
        output = self.encoder.conv_layers(obs_history.reshape([batch_size, self.tsteps, self.num_props]).permute((0, 2, 1)).contiguous())
        output = self.encoder.linear_output(output)
        mu, logvar = torch.split(output, self.latent_size, dim=-1)
        logvar = torch.clamp(logvar, min=-6.0, max=2.0)
        encodings = self.reparametrize(mu[:, :-self.num_privileged_obs], logvar)

        z = torch.cat((encodings, mu[:, -self.num_privileged_obs: ]), dim=-1)
        if return_stats:
            return z, mu, logvar
        return z
    
    def encode(self, obs_history):
        batch_size = obs_history.size(0)
        output = self.encoder.conv_layers(obs_history.reshape([batch_size, self.tsteps, self.num_props]).permute((0, 2, 1)).contiguous())
        output = self.encoder.linear_output(output)
        return output[:, :self.latent_size]

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps
    
    def kl_bits(self, mu, logvar):
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl.sum(dim=1).mean() / math.log(2)

    def symlog(self, x, eps=1e-6):
        # sign(x) * log(1+|x|)
        return torch.sign(x) * torch.log1p(torch.abs(x) + eps)

    def symexp(self, y):
        # sign(y) * (exp(|y|) - 1)
        return torch.sign(y) * (torch.exp(torch.abs(y)) - 1.)

    def beta_schedule(self,
                      step,
                      total_steps,
                      beta_max=0.05,
                      warmup_frac=0.15,
                      ramp_frac=0.30
                      ):
        self.step = step
        self.total_steps = total_steps
        warmup_steps = int(total_steps * warmup_frac)
        ramp_steps = int(total_steps * ramp_frac)
        ramp_start = warmup_steps
        ramp_end = warmup_steps + ramp_steps

        if step < ramp_start:
            return 0.0

        if step < ramp_end:
            progress = (step - ramp_start) / max(1, ramp_steps - 1)
            return beta_max * progress

        return beta_max

    def decode(self, identity, decoder_input):
        if self.mitigate_collapse:
            if self.estimate_delta:
                target_inference = self.decoder(decoder_input)
                target = self.decoder_linear(target_inference)
            else:
                target_inference = self.decoder(decoder_input)
                target = self.decoder_linear(torch.cat((identity, target_inference), dim=-1))
        else:
            target = self.decoder(torch.cat((identity, decoder_input), dim=-1))
        return target

    def update(self, decoder_obs, latents, detached_actions, privileged_obs, inference_target, obs_history, mu, logvar, hiddem_states=None, masks=None):
        # predict privileged_obs
        estimation_loss = 0.5 * F.l1_loss(latents[:, -privileged_obs.shape[1]: ], privileged_obs, reduction='mean')

        # predict next obs and reward
        if self.use_true_priv_target:
            decoder_inputs = torch.cat((decoder_obs[:, self.num_props:], latents[:, :-privileged_obs.shape[1]], privileged_obs, detached_actions), dim=-1)
        else:
            estimated_priv_obs = latents[:, -privileged_obs.shape[1]:].clone().detach()
            decoder_inputs = torch.cat((decoder_obs[:, self.num_props:], latents[:, :-privileged_obs.shape[1]], estimated_priv_obs, detached_actions), dim=-1)
            
        if self.estimate_delta:
            target_shape = inference_target.shape[1]
            inference = self.decode(None, decoder_inputs)
            decoder_loss = F.l1_loss(
                                    self.symlog(inference).clamp(-6, 6), 
                                    self.symlog(inference_target - decoder_obs[:, :target_shape]).clamp(-6, 6), reduction='mean')
        else:
            inference = self.decode(decoder_obs[:, :self.num_props], decoder_inputs)
            decoder_loss = F.l1_loss(self.symlog(inference).clamp(-6, 6), self.symlog(inference_target).clamp(-6, 6), reduction='mean')

        # latent consisteny
        inference_obs = inference + decoder_obs[:, :(self.num_props-12)] if self.estimate_delta else inference
        det_actions = detached_actions.detach()
        prev_history = obs_history[:, self.num_props:]
        if self.use_online_decoder:
            estimated_history = torch.cat((prev_history, inference_obs, det_actions), dim=-1)
        else:
            pred_obs = inference_obs.detach()
            estimated_history = torch.cat((prev_history, pred_obs, det_actions), dim=-1)      
        target_history = torch.cat((prev_history, inference_target, det_actions), dim=-1)

        estimated_latent = self.encode(estimated_history)
        with torch.inference_mode():
            target_latent = self.ema_target.model.encode(target_history)

        lambda_c = 0.02 * torch.exp(-3.0 * decoder_loss.detach()).clamp(1e-4, 1.0)
        consistency_loss = lambda_c * F.smooth_l1_loss(estimated_latent, target_latent.clone(), reduction='mean')

        # KL divergence loss
        info_loss = self.kl_bits(mu[:, :-self.num_privileged_obs], logvar)

        return estimation_loss, decoder_loss, self.beta * info_loss, consistency_loss

    def set_target(self, ema_target):
        self.ema_target = ema_target