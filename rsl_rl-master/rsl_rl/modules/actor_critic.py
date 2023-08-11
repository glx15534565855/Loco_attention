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
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 use_mlp = False,
                 use_transformer = True,
                 use_mlpmixer=False,
                 use_gnn=False,
                 use_distribute_mlp=False,
                 hidden_size=160, torso_size=12, token_num=6, joint_size=24,
                 leg_size=27, patch_size=64, patch_num=12, n_block=2, n_head=2,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
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

        self.use_mlpmixer = use_mlpmixer
        self.use_transformer = use_transformer
        self.use_gnn = use_gnn
        self.use_mlp = use_mlp
        self.use_distribute_mlp = use_distribute_mlp

        if not self.use_mlp and not self.use_distribute_mlp:
            self.torso_embed = nn.Linear(torso_size, hidden_size)
            self.joint_embed = nn.Linear(joint_size, hidden_size)
            self.leg_embed = nn.Linear(leg_size, hidden_size)
            self.patch_embed = nn.Linear(patch_size, hidden_size)
            self.pos_embed = nn.Parameter(torch.normal(mean=0, std=0.1, size=((1, patch_num, hidden_size))))
            self.token_num = token_num
            self.patch_num = patch_num
            self.joint_size = joint_size
            self.torso_size = torso_size
            self.leg_size = leg_size
            if self.use_transformer:
                self.actor_transformer = TransformerEncoder(hidden_size, n_block, hidden_size, n_head, token_num + patch_num)
                #self.actor_transformer = TransformerEncoder(hidden_size,n_block,hidden_size,n_head,token_num)
                # self.actor_transformer = TransformerEncoder(hidden_size,n_block,hidden_size,n_head,1)
                # self.obs_embed = nn.Linear(48, hidden_size)
            if self.use_gnn:
                # Anymal/a1 edge_index:
                self.edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                                                [1, 2, 3, 4, 0, 4, 2, 0, 1, 3, 0, 2, 4, 0, 3, 1]], dtype=torch.long,
                                               device='cuda')
                # Littledog_index:
                # self.edge_index = torch.tensor([[0,0,0,0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6],
                #                                [1,2,3,4,5,6,0,6,2,0,1,3,0,2,4,0,3,5,0,4,6,0,5,1]], dtype=torch.long, device='cuda')
                self.actor_gnn = GNN(hidden_size, hidden_size, hidden_size, 3)

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

    def update_distribution(self, observations):
        if self.use_mlp:
            # occlusion height map:
             #height_map = observations[:, 66:]
             #height_map = height_map.reshape(height_map.shape[0], 16, 10)
             #p = torch.full((16,), 11 / 16).cuda()  # mask 1 patch
             #m = torch.bernoulli(p).cuda().unsqueeze(0).unsqueeze(-1)
             # mask height_map
             #height_map.masked_fill_(m == 1, 0)
             #observations = torch.cat((observations[:,:66],height_map.reshape(height_map.shape[0],-1)),dim=-1)
             mean = self.actor(observations)

        elif  self.use_transformer:
            obs_re = observations[:, :66].unsqueeze(1)
            # obs_re = self.obs_embed(obs_re)
            # self.value, mean = self.actor_transformer(obs_re)

            pos_embd = torch.zeros([obs_re.shape[0], 12, obs_re.shape[1], 6], device='cuda')
            # for anymal and a1:4 for littledog: 6
            for i in range(4):
                pos_embd[:, 3 * i, :] = torch.tensor([i, 0, i, 1, i, 2], device='cuda')
                pos_embd[:, 3 * i + 1, :] = torch.tensor([i, 1, i, 2, i, 3], device='cuda')
                pos_embd[:, 3 * i + 2, :] = torch.tensor([i, 2, i, 3, i, -1], device='cuda')

            torso = obs_re[:, :, :12].reshape(obs_re.shape[0], -1)
            height_map = observations[:, 66:]
            height_map = height_map.reshape(height_map.shape[0], 12, 64)
            # train: 45
            # perform occlusion 1/12:    2/12:    3/12:   4/12:    5/12:    6/12:
            p = torch.full((12,), 9/12).cuda()  # mask 1 patch
            m = torch.bernoulli(p).cuda().unsqueeze(0).unsqueeze(-1)
            # mask height_map
            # height_map.masked_fill_(m == 1, 0)
            '''
            joints = [torch.cat(
                (obs_re[:, :, 12 + (11 + i) % 12].unsqueeze(-1), obs_re[:, :, 12 + (11 + i) % 12 + 12].unsqueeze(-1),
                 obs_re[:, :, 12 + (11 + i) % 12 + 24].unsqueeze(-1), obs_re[:, :, 12 + i].unsqueeze(-1),
                 obs_re[:, :, 12 + i + 12].unsqueeze(-1),
                 obs_re[:, :, 12 + i + 24].unsqueeze(-1), obs_re[:, :, 12 + (13 + i) % 12].unsqueeze(-1),
                 obs_re[:, :, 12 + (13 + i) % 12 + 12].unsqueeze(-1),
                 obs_re[:, :, 12 + (13 + i) % 12 + 24].unsqueeze(-1), obs_re[:, :, 0:6], obs_re[:, :, 9:12] * 5,
                 pos_embd[:, i, :]), dim=-1) for i in range(12)]
            
            joints = [torch.cat(
                (obs_re[:, :, 12 + (17 + i) % 18].unsqueeze(-1), obs_re[:, :, 12 + (17 + i) % 18 + 18].unsqueeze(-1),
                 obs_re[:, :, 12 + (17 + i) % 18 + 36].unsqueeze(-1), obs_re[:, :, 12 + i].unsqueeze(-1),
                 obs_re[:, :, 12 + i + 18].unsqueeze(-1),
                 obs_re[:, :, 12 + i + 36].unsqueeze(-1), obs_re[:, :, 12 + (19 + i) % 18].unsqueeze(-1),
                 obs_re[:, :, 12 + (19 + i) % 18 + 18].unsqueeze(-1),
                 obs_re[:, :, 12 + (19 + i) % 18 + 36].unsqueeze(-1), obs_re[:, :, 0:6], obs_re[:, :, 9:12] * 5,
                 pos_embd[:, i, :]), dim=-1) for i in range(18)]
           
            joints = torch.stack(joints)
            joints = joints.permute(1, 0, 2, 3)
            joints = joints.reshape(joints.shape[0], joints.shape[1], -1)

            torso_token = self.torso_embed(torso)
            joints_token = self.joint_embed(joints)
            heightmap_patch = self.patch_embed(height_map) + self.pos_embed
            # tokens = torch.cat([torso_token.unsqueeze(1), joints_token], dim=1)
            tokens = torch.cat([torso_token.unsqueeze(1), joints_token, heightmap_patch], dim=1)
            '''
            # if use leg as a token:
            leg_pos_embd = torch.zeros([obs_re.shape[0], 6, obs_re.shape[1], 6], device='cuda')
            for i in range(6):
                leg_pos_embd[:, i, :] = torch.tensor([i, 0, i, 1, i, 2], device='cuda')

            legs = [torch.cat((obs_re[:, :, 12 + 3 * i:12 + 3 * i + 3], obs_re[:, :, 30 + 3 * i:30 + 3 * i + 3],
                               obs_re[:, :, 48 + 3 * i:48 + 3 * i + 3], obs_re[:, :, :12], leg_pos_embd[:, i, :]),
                              dim=-1) for i in range(6)]
            #legs = [torch.cat((obs_re[:, :, 12 + 3 * i:12 + 3 * i + 3], obs_re[:, :, 24 + 3 * i:24 + 3 * i + 3],
            #                  obs_re[:, :, 36 + 3 * i:36 + 3 * i + 3], obs_re[:, :, :12], leg_pos_embd[:, i, :]),
            #                 dim=-1) for i in range(4)]
            legs = torch.stack(legs)
            legs = legs.permute(1, 0, 2, 3)
            legs = legs.reshape(legs.shape[0], legs.shape[1], -1)
            legs_token = self.leg_embed(legs)
            # not use patch embedding:
            heightmap_patch = self.patch_embed(height_map)
            tokens = torch.cat([legs_token, heightmap_patch], dim=1)
            self.value, mean = self.actor_transformer(tokens)
            mean = mean.reshape(mean.shape[0], -1)

        elif self.use_gnn:
            obs_re = observations[:, :48].unsqueeze(1)
            torso = obs_re[:, :, :12].reshape(obs_re.shape[0], -1)
            leg_pos_embd = torch.zeros([obs_re.shape[0], 4, obs_re.shape[1], 6], device='cuda')
            for i in range(4):
                leg_pos_embd[:, i, :] = torch.tensor([i, 0, i, 1, i, 2], device='cuda')

            legs = [torch.cat((obs_re[:, :, 12 + 3 * i:12 + 3 * i + 3], obs_re[:, :, 24 + 3 * i:24 + 3 * i + 3],
                               obs_re[:, :, 36 + 3 * i:36 + 3 * i + 3], obs_re[:, :, :12], leg_pos_embd[:, i, :]),
                              dim=-1) for i in range(4)]
            legs = torch.stack(legs)
            legs = legs.permute(1, 0, 2, 3)
            legs = legs.reshape(legs.shape[0], legs.shape[1], -1)
            legs_token = self.leg_embed(legs)
            torso_token = self.torso_embed(torso)
            gnn_tokens = torch.cat([torso_token.unsqueeze(1), legs_token], dim=1)
            self.value, mean = self.actor_gnn(gnn_tokens, self.edge_index)
            #self.value, mean = self.actor_gnn(legs_token, self.edge_index)
            mean = mean.reshape(mean.shape[0], -1)

        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        if self.use_mlp:
            value = self.critic(critic_observations)
        else:
            value = torch.mean(self.value, dim=1)
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


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_tokens, dropout = 0.1, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_tokens + 1, n_tokens + 1))
                             .view(1, 1, n_tokens + 1, n_tokens + 1))
        self.dropout = nn.Dropout(dropout)

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y

class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_tokens, dropout):
        super(EncodeBlock, self).__init__()

        # self.attn = SelfAttention(n_embd, n_head, n_tokens, masked=True)
        self.attn = SelfAttention(n_embd, n_head, n_tokens, dropout,masked=False)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 1 * n_embd),
            nn.GELU()
        )
        self.mmm=nn.Sequential(nn.Linear(n_embd,n_embd),nn.GELU(),nn.Linear(n_embd,n_embd),nn.GELU())
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x):
        # dropout during rollout:
        if x.shape[0] == 4096:
            x = x + self.dropout1(self.attn(x, x, x))
            x = x + self.dropout2(self.mlp(x))
        else:
            x = x + self.attn(x,x,x)
            x = x + self.mlp(x)
        # x = self.mmm(x)
        # x = self.mlp(x)
        # x = x + self.attn(self.ln1(x),self.ln1(x))

        return x

class TransformerEncoder(nn.Module):

    def __init__(self, obs_dim, n_block, n_embd, n_head, n_tokens, dropout = [0.0001,0.0001,0.05]):  # [0.0001 0.0001, 0.05]
        super(TransformerEncoder, self).__init__()
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_tokens = n_tokens


        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),nn.GELU())

        self.blocks1 = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_tokens, dropout[i]) for i in range(1)])
        self.blocks2 = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_tokens, dropout[i]) for i in range(1)])
        self.head = nn.Sequential(nn.Linear(n_embd, n_embd), nn.GELU(),nn.Linear(n_embd, 1))
        self.output = nn.Linear(n_embd, 3)



    def forward(self, obs):

        # obs: (batch, n_tokens, obs_dim)

        obs_embeddings = self.obs_encoder(obs)
        x = obs_embeddings
        # x = self.blocks1(self.ln(x))
        x = self.blocks1(x)
        v_loc = self.head(x)
        x = self.blocks2(x)
        # out = self.blocks2(self.ln(x))

        means = self.output(x)


        return v_loc, means[:,:6] #means[:, 1:19]

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convn = GCNConv(hidden_channels, out_channels)
        self.head = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.GELU(),
                                  nn.Linear(hidden_channels, 1))
        self.output = nn.Linear(hidden_channels, 3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        v_loc = self.head(x)
        x = self.convn(x, edge_index)
        means = self.output(x)
        return v_loc, means[:,1:]


class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin0 = nn.Linear(in_channels, out_channels)
        self.lin1 = nn.Linear(2*out_channels, out_channels)
        self.lin2 = nn.Linear(out_channels, out_channels)
        self.output = nn.Linear(out_channels, out_channels)


    def forward(self, x, edge_index):
        adj = edge_index_to_adj(edge_index, x.size(1))
        for i in range(adj.shape[0]):
            adj[i, i] = 0
        output = self.lin0(x)
        output = torch.bmm(adj.unsqueeze(0).expand(x.size(0), *adj.size()), output)
        output = torch.concat([x,output], dim=-1)
        output =self.output(nn.functional.gelu(self.lin2(self.lin1(output))))
        #output = self.output(nn.functional.gelu(self.lin2(output)))
        return x + output

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                    self.out_channels)

def edge_index_to_adj(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes),device='cuda')
    adj[edge_index[0], edge_index[1]] = 1
    adj[edge_index[1], edge_index[0]] = 1
    return adj