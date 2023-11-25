import math
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from algorithm.attention import Attention
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = 1e-7


class Guide_policy(nn.Module):
    def __init__(self, state_dim, log_std_multiplier=1.0, log_std_offset=-1.0, hidden_num=256):
        super(Guide_policy, self).__init__()
        self.attention=Attention(head=8,embeding_dim=80,droput=0.2)
        self.fc1 = nn.Linear(state_dim, hidden_num)
        self.fc2 = nn.Linear(hidden_num, hidden_num)
        self.mu_head = nn.Linear(hidden_num, state_dim)
        self.sigma_head = nn.Linear(hidden_num, state_dim)

        self.log_sigma_multiplier = log_std_multiplier
        self.log_sigma_offset = log_std_offset

    def _get_outputs(self, state):
        a = self.attention(state)
        a = F.relu(self.fc1(a))
        a = F.relu(self.fc2(a))
        mu = self.mu_head(a)
        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = self.sigma_head(a)
        # log_sigma = self.log_sigma_multiplier * log_sigma + self.log_sigma_offset

        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = TransformedDistribution(
            Normal(mu, sigma), TanhTransform(cache_size=1)
        )
        a_tanh_mode = torch.tanh(mu)
        return a_distribution, a_tanh_mode

    def forward(self, state):
        a_dist, a_tanh_mode = self._get_outputs(state)
        action = a_dist.rsample()
        logp_pi = a_dist.log_prob(action).sum(axis=-1)
        return action, logp_pi, a_tanh_mode

    def get_log_density(self, state, action):
        a_dist, _ = self._get_outputs(state)
        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        logp_action = a_dist.log_prob(action_clip)
        return logp_action


class Execute_policy(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_multiplier=1.0, log_std_offset=-1.0, hidden_num=512):
        super(Execute_policy, self).__init__()

        self.fc1 = nn.Linear(state_dim*2, hidden_num)
        self.fc2 = nn.Linear(hidden_num, hidden_num)
        self.mu_head = nn.Linear(hidden_num, action_dim)
        self.sigma_head = nn.Linear(hidden_num, action_dim)

        self.log_sigma_multiplier = log_std_multiplier
        self.log_sigma_offset = log_std_offset

    def _get_outputs(self, state, goal):
        concat_state = torch.concat([state, goal], dim=1)
        a = F.relu(self.fc1(concat_state))
        a = F.relu(self.fc2(a))

        mu = self.mu_head(a)
        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = self.sigma_head(a)
        # log_sigma = self.log_sigma_multiplier * log_sigma + self.log_sigma_offset

        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = TransformedDistribution(
            Normal(mu, sigma), TanhTransform(cache_size=1)
        )
        a_tanh_mode = torch.tanh(mu)
        return a_distribution, a_tanh_mode

    def forward(self, state, goal):
        a_dist, a_tanh_mode = self._get_outputs(state, goal)
        action = a_dist.rsample()
        logp_pi = a_dist.log_prob(action).sum(axis=-1)
        return action, logp_pi, a_tanh_mode

    def get_log_density(self, state, goal, action):
        a_dist, _ = self._get_outputs(state, goal)
        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        logp_action = a_dist.log_prob(action_clip)
        return logp_action


class Double_Critic(nn.Module):
    def __init__(self, state_dim):
        super(Double_Critic, self).__init__()

        self.attention=Attention(head=8,embeding_dim=state_dim,droput=0.2)
        # V1 architecture
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # V2 architecture
        self.l4 = nn.Linear(state_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state):
        v=self.attention(state)
        v1 = F.relu(self.l1(v))
        v1 = F.relu(self.l2(v1))
        v1 = self.l3(v1)

        v2 = F.relu(self.l4(v))
        v2 = F.relu(self.l5(v2))
        v2 = self.l6(v2)
        return v1, v2

    def V1(self, state):
        v=self.attention(state)
        v1 = F.relu(self.l1(v))
        v1 = F.relu(self.l2(v1))
        v1 = self.l3(v1)
        return v1


def loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class POR(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            eta=0.005,
            tau=0.9,
            alpha=10.0,
            lmbda=10.0,
            g_v=False,
            e_weight=True,
            Nuber_channel=3
    ):
        self.policy_e=[]
        self.policy_e_optimizer=[]
        for i in range(Nuber_channel):
            self.policy_e.append(Execute_policy(state_dim, action_dim).to(device))
            self.policy_e_optimizer.append(torch.optim.Adam(self.policy_e[i].parameters(), lr=3e-4))

        self.policy_g = Guide_policy(state_dim).to(device)
        self.policy_g_optimizer = torch.optim.Adam(self.policy_g.parameters(), lr=3e-4)

        self.critic = Double_Critic(state_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.tau = tau
        self.alpha = alpha
        self.lmbda = lmbda
        self.g_v = g_v
        self.e_weight = e_weight

        self.discount = discount
        self.eta = eta
        self.total_it = 0
        self.number_channel=Nuber_channel

    def select_action(self, state,cur_c=0):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        _, _, goal = self.policy_g(state)
        _, _, action = self.policy_e[cur_c](state, goal)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer,numb_iter=3000, batch_size=256,channel_number=3, log_writer=None):

        channel_iter = 0
        for it in range(numb_iter):
            self.total_it += 1
            channel_iter+=1

            # state=torch.randn(128,3,80).to(device)
            # action=torch.randn(128,3,1).to(device)
            # next_state=torch.randn(128,3,80).to(device)
            # reward=torch.randn(128,3,1).to(device)
            # not_done=torch.randn(128,3,1).to(device)

            # Sample replay buffer
            state, action, next_state, reward, not_done, costs, gmvs,rois = replay_buffer.sample(batch_size)

            # Update V
            with torch.no_grad():
                next_v1, next_v2 = self.critic_target(next_state)
                next_v = torch.minimum(next_v1, next_v2).detach()
                # next_v = next_v1
                target_v = (reward + self.discount * (1-not_done) * next_v).detach()

            v1, v2 = self.critic(state)
            critic_loss = (loss(target_v - v1, self.tau) + loss(target_v - v2, self.tau)).mean()
            # critic_loss = loss(target_v - v1, self.tau).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10, norm_type=2)
            self.critic_optimizer.step()

            # Update guide-policy
            with torch.no_grad():
                next_v1, next_v2 = self.critic_target(next_state)
                next_v = torch.minimum(next_v1, next_v2).detach()
                target_v = (reward + self.discount * not_done * next_v).detach()
                v1, v2 = self.critic(state)
                residual = target_v - v1
                weight = torch.exp(residual * self.alpha)
                weight = torch.clamp(weight, max=100.0).squeeze(-1).detach()

            log_pi_g = self.policy_g.get_log_density(state, next_state)
            log_pi_g = torch.sum(log_pi_g, dim=-1)

            if not self.g_v:
                p_g_loss = -(weight * log_pi_g).mean()
            else:
                g, _, _ = self.policy_g(state)
                v1_g, v2_g = self.critic(g)
                min_v_g = torch.squeeze(torch.min(v1_g, v2_g))
                lmbda = self.lmbda / min_v_g.abs().mean().detach()
                p_g_loss = -(weight * log_pi_g + lmbda * min_v_g).mean()

            self.policy_g_optimizer.zero_grad()
            p_g_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_g.parameters(), max_norm=10,norm_type=2)
            self.policy_g_optimizer.step()

            # Update execute-policy
            peloss=[]
            for cur_c in range(channel_number):
                log_pi_a = self.policy_e[cur_c].get_log_density(state[:,cur_c,:], next_state[:,cur_c,:], action[:,cur_c,:])
                log_pi_a = torch.sum(log_pi_a, dim=-1)

                if self.e_weight:
                    p_e_loss = -(weight[:,cur_c] * log_pi_a).mean()
                else:
                    p_e_loss = -log_pi_a.mean()

                self.policy_e_optimizer[cur_c].zero_grad()
                p_e_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_e[cur_c].parameters(), max_norm=10, norm_type=2)
                self.policy_e_optimizer[cur_c].step()
                peloss.append(p_e_loss.item())

            if log_writer is not None:
                log_writer.add_scalar('Loss/Guide_Loss', p_g_loss.item(), self.total_it)
                for cur_c in range(channel_number):
                    log_writer.add_scalar('Loss/Exec_'+str(cur_c)+'_Loss', peloss[cur_c], channel_iter)
                log_writer.add_scalar('Loss/Critic_Loss', critic_loss.item(), self.total_it)

            if self.total_it % 50 == 0:
                print(f'mean target v value is {target_v.mean()}')
                print(f'mean v1 value is {v1.mean()}')
                print(f'mean residual is {residual.mean()}')

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.eta * param.data + (1 - self.eta) * target_param.data)

    def save(self, filename):
        torch.save(self.policy_g.state_dict(), filename + "_policy_g")
        torch.save(self.policy_g_optimizer.state_dict(), filename + "_policy_g_optimizer")
        for i in range(self.number_channel):
            torch.save(self.policy_e[i].state_dict(), filename + "_policy_e_"+str(i))
            torch.save(self.policy_e_optimizer[i].state_dict(), filename + "_policy_e_optimizer_"+str(i))

    def load(self, filename):
        self.policy_g.load_state_dict(torch.load(filename + "_policy_g"))
        self.policy_g_optimizer.load_state_dict(torch.load(filename + "_policy_g_optimizer"))
        for i in range(self.number_channel):
            self.policy_e[i].load_state_dict(torch.load(filename + "_policy_e_"+str(i)))
            self.policy_e_optimizer[i].load_state_dict(torch.load(filename + "_policy_e_optimizer_"+str(i)))


if __name__ == '__main__':
    agent=POR(state_dim=80,action_dim=1,max_action=10)
    agent.train(replay_buffer=None,numb_iter=1)