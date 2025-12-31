import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .encoder import Encoder
import torch.nn.functional as F

class AttnActorCritic(nn.Module):
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_one_step_obs,
                        num_actions,
                        num_height_map_scans = 187,
                        height_map_shape = (1, 17, 11),
                        height_map_real_H = 17,
                        d_model = 64,
                        actor_hidden_dims=[256, 128],
                        critic_hidden_dims=[256, 128],
                        activation='elu',
                        init_noise_std=1.0,
                        self_attn=False,
                        vel_est = False,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(AttnActorCritic, self).__init__()

        self.num_history = int(num_actor_obs/num_one_step_obs)
        self.height_map_shape = height_map_shape
        self.height_map_real_H = height_map_real_H
        self.num_height_map_scans = num_height_map_scans
        self.num_one_step_obs = num_one_step_obs
        self.vel_est = vel_est
        
        activation = get_activation(activation)

        # Policy
        
        self.encoder_actor = Encoder(num_his=self.num_history, num_one_step_obs=num_one_step_obs, d_model=d_model, self_attn=self_attn)
        if vel_est:
            self.vel_head = nn.Linear((num_one_step_obs + d_model)*self.num_history, 3)  # predict x,y velocity from encoder output
                               
        actor_input_dim = (num_one_step_obs + d_model)*self.num_history
        actor_layers = []
        for l in range(len(actor_hidden_dims)):
            actor_layers += [nn.Linear(actor_input_dim, actor_hidden_dims[l]), activation]
            actor_input_dim = actor_hidden_dims[l]
        actor_layers += [nn.Linear(actor_input_dim, num_actions)]
        self.actor = nn.Sequential(*actor_layers)
        
        self.encoder_critic = Encoder(num_his=1, num_one_step_obs=num_critic_obs-num_height_map_scans, d_model=d_model, self_attn=self_attn)
        
        critic_input_dim = d_model + num_critic_obs - num_height_map_scans
        critic_layers = []
        for l in range(len(critic_hidden_dims)):
            critic_layers += [nn.Linear(critic_input_dim, critic_hidden_dims[l]), activation]
            critic_input_dim = critic_hidden_dims[l]
        critic_layers += [nn.Linear(critic_input_dim, 1)]
        self.critic = nn.Sequential(*critic_layers)
        
        print(f'Encoder Actor: {self.encoder_actor}')
        print(f'Encoder Critic: {self.encoder_critic}')
        print(f'Actor MLP: {self.actor}')
        print(f'Critic MLP: {self.critic}')

        # print(f"Actor MLP: {self.actor}")
        # print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

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

    def update_distribution(self, obs, critic_obs):
        height_map = critic_obs[:, -self.num_height_map_scans:].view(-1, self.height_map_shape[0], self.height_map_shape[1], self.height_map_shape[2])
        height_map_actor = height_map[:, :, -self.height_map_real_H:, :]
        out_encoder = self.encoder_actor(obs, height_map_actor)
        out_encoder = torch.cat((out_encoder, obs.view(-1, self.num_history, self.num_one_step_obs)), dim=-1)
        mean = self.actor(out_encoder.flatten(start_dim=1))
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, obs, critic_obs, **kwargs):
        self.update_distribution(obs, critic_obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs, critic_obs):
        height_map = critic_obs[:, -self.num_height_map_scans:].view(-1, self.height_map_shape[0], self.height_map_shape[1], self.height_map_shape[2])
        height_map_actor = height_map[:, :, -self.height_map_real_H:, :]
        out_encoder = self.encoder_actor(obs, height_map_actor)
        out_encoder = torch.cat((out_encoder, obs.view(-1, self.num_history, self.num_one_step_obs)), dim=-1)
        actions_mean = self.actor(out_encoder.flatten(start_dim=1))
        return actions_mean

    def evaluate(self, critic_obs, **kwargs):
        height_map = critic_obs[:, -self.num_height_map_scans:].view(-1, self.height_map_shape[0], self.height_map_shape[1], self.height_map_shape[2])
        critic_obs = critic_obs[:, :-self.num_height_map_scans]
        out_encoder = self.encoder_critic(critic_obs, height_map)
        out_encoder = torch.cat((out_encoder, critic_obs.view(out_encoder.shape[0], out_encoder.shape[1], -1)), dim=-1)
        value = self.critic(out_encoder.flatten(start_dim=1))
        return value
    
    def get_vel_loss(self, obs, critic_obs, next_critic_obs):
        if self.vel_est:
            height_map = critic_obs[:, -self.num_height_map_scans:].view(-1, self.height_map_shape[0], self.height_map_shape[1], self.height_map_shape[2])
            height_map_actor = height_map[:, :, -self.height_map_real_H:, :]
            out_encoder = self.encoder_actor(obs, height_map_actor)
            out_encoder = torch.cat((out_encoder, obs.view(-1, self.num_history, self.num_one_step_obs)), dim=-1)
            est_vel = self.vel_head(out_encoder.flatten(start_dim=1))
            true_vel = next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs+3].detach()
            estimation_loss = F.mse_loss(est_vel, true_vel)
            return estimation_loss
        else:
            return 0.0
        
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
