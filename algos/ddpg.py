import copy
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from components.models import Actor, get_activation
from utils.radam import RAdam


class ActionValueFunction(nn.Module):
    def __init__(self, d_state, d_action, n_layers, n_units, activation):
        super().__init__()
        assert n_layers >= 1, "# of hidden layers"

        layers = [nn.Linear(d_state + d_action, n_units), get_activation(activation)]
        for lyr_idx in range(1, n_layers):
            layers += [nn.Linear(n_units, n_units), get_activation(activation)]
        layers += [nn.Linear(n_units, 1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, state, action):
        x = th.cat([state, action], dim=1)
        return self.layers(x)


class DDPG(nn.Module):
    def __init__(
            self,
            d_state,
            d_action,
            device,
            gamma,
            tau,
            policy_lr,
            value_lr,
            value_loss,
            value_n_layers,
            value_n_units,
            value_activation,
            policy_n_layers,
            policy_n_units,
            policy_activation,
            grad_clip,
            policy_noise=0.2,
            noise_clip=0.5,
            expl_noise=0.1,
    ):
        super(DDPG, self).__init__()

        self.actor = Actor(d_state, d_action, policy_n_layers, policy_n_units, policy_activation).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=policy_lr)

        self.critic = ActionValueFunction(d_state, d_action, value_n_layers, value_n_units, value_activation).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=value_lr)

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.expl_noise = expl_noise
        self.normalizer = None
        self.value_loss = value_loss
        self.grad_clip = grad_clip
        self.device = device

    def setup_normalizer(self, normalizer):
        self.normalizer = copy.deepcopy(normalizer)

    def get_action(self, states, deterministic=False):
        states = states.to(self.device)
        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
        actions = self.actor(states)
        if not deterministic:
            actions = actions + th.randn_like(actions) * self.expl_noise
        return actions.clamp(-1, +1)

    def get_action_with_logp(self, states):
        states = states.to(self.device)
        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
        a = self.actor(states)
        return a, th.ones(a.shape[0], device=a.device) * np.inf  # inf: should not be used

    def update(self, states, actions, rewards, next_states, masks):
        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
            next_states = self.normalizer.normalize_states(next_states)

        # Select action according to policy and add clipped noise
        noise = (th.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        raw_next_actions = self.actor_target(next_states)
        next_actions = (raw_next_actions + noise).clamp(-1, 1)

        # Compute the target Q value
        next_q = self.critic_target(next_states, next_actions)
        q_target = rewards.unsqueeze(1) + self.gamma * masks.float().unsqueeze(1) * next_q
        zero_targets = th.zeros_like(q_target, device=self.device)

        q = self.critic(states, actions)  # Q(s,a)
        q_td_error = q_target - q
        critic_loss = th.tensor(0, device=self.device)

        if self.value_loss == 'huber':
            standard_loss = 0.5 * F.smooth_l1_loss(q_td_error, zero_targets)
        elif self.value_loss == 'mse':
            standard_loss = 0.5 * F.mse_loss(q_td_error, zero_targets)
        else:
            raise ValueError(f"Unknown loss function {self.value_loss}")
        critic_loss = critic_loss + standard_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        th.nn.utils.clip_grad_value_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # Compute actor loss
        q = self.critic(states, self.actor(states))  # Q(s,pi(s))
        actor_loss = -q.mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        th.nn.utils.clip_grad_value_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        param_dict = {
            "critic": self.critic.state_dict(),
            "actor": self.actor.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict()
        }
        th.save(param_dict, filename)

    def load(self, filename):
        param_dict = th.load(filename)
        # Get parameters from save_dict
        self.critic.load_state_dict(param_dict["critic"])
        self.actor.load_state_dict(param_dict["actor"])
        self.critic_optimizer.load_state_dict(param_dict["critic_optimizer"])
        self.actor_optimizer.load_state_dict(param_dict["actor_optimizer"])
        # Copy the eval networks to target networks
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
