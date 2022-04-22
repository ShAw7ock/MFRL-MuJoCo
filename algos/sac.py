import copy
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from components.models import GaussianActor, ActionValueFunction
from utils.radam import RAdam


class SAC(nn.Module):
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
            policy_delay=2,
            policy_noise=0.2,
            noise_clip=0.5,
            expl_noise=0.1,
    ):
        super(SAC, self).__init__()
        self.actor = GaussianActor(d_state, d_action, policy_n_layers, policy_n_units, policy_activation).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=policy_lr)

        self.critic = ActionValueFunction(d_state, d_action, value_n_layers, value_n_units, value_activation).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=value_lr)

        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.expl_noise = expl_noise
        self.normalizer = None
        self.value_loss = value_loss
        self.grad_clip = grad_clip
        self.device = device

        self.step_counter = 0

    def setup_normalizer(self, normalizer):
        self.normalizer = copy.deepcopy(normalizer)

    def get_action(self, states, deterministic=False):
        states = states.to(self.device)
        with th.no_grad():
            if self.normalizer is not None:
                states = self.normalizer.normalize_states(states)
            actions, _, = self.actor.sample(states)

        return actions

    def update(self, states, actions, rewards, next_states, masks):
        if self.normalizer is not None:
            states = self.normalizer.normalize_states(states)
            next_states = self.normalizer.normalize_states(next_states)
        self.step_counter += 1
        # TODO: Automatic Entropy Tuning
        alpha = 1

        # Critic Update
        next_actions, new_log_pi = self.actor.sample(next_states)
        new_log_pi = new_log_pi.unsqueeze(-1)
        next_Q1, next_Q2 = self.critic_target(next_states, next_actions)
        next_Q = th.min(next_Q1, next_Q2) - alpha * new_log_pi
        q_target = rewards + self.gamma * masks.float().unsqueeze(1) * next_Q
        zero_targets = th.zeros_like(q_target, device=self.device)

        # Get current Q estimates
        q1, q2 = self.critic(states, actions)
        q1_td_error, q2_td_error = q_target - q1, q_target - q2

        critic_loss = th.tensor(0, device=self.device)
        # Compute standard critic loss
        if self.value_loss == 'huber':
            standard_loss = 0.5 * (
                        F.smooth_l1_loss(q1_td_error, zero_targets) + F.smooth_l1_loss(q2_td_error, zero_targets))
        elif self.value_loss == 'mse':
            standard_loss = 0.5 * (F.mse_loss(q1_td_error, zero_targets) + F.mse_loss(q2_td_error, zero_targets))
        else:
            raise ValueError(f"Unknown loss function {self.value_loss}")
        critic_loss = critic_loss + standard_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        th.nn.utils.clip_grad_value_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # Actor Update
        if self.step_counter % self.policy_delay == 0:
            # Compute actor loss
            new_actions, log_pi = self.actor.sample(states)
            q1, q2 = self.critic(states, new_actions)  # originally in TD3 we had here q1 only
            q_min = th.min(q1, q2)
            actor_loss = (alpha * log_pi - q_min).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            th.nn.utils.clip_grad_value_(self.actor.parameters(), self.grad_clip)
            self.actor_optimizer.step()

        # Update the frozen target value function
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
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
