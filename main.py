import numpy as np
import torch
import os
import gym
from dotmap import DotMap
from pathlib import Path
from tensorboardX import SummaryWriter

import envs
import envs.gymmb
from algos import TD3, SAC, DDPG
from components.arguments import common_args, policy_function, value_function
from components.env_loop import EnvLoop
from components.buffer import Buffer
from components.normalizer import TransitionNormalizer
from utils.wrappers import BoundedActionsEnv, IsDoneEnv, MuJoCoCloseFixWrapper, RecordedEnv
from utils.misc import to_np, EpisodeStats


def get_random_agent(d_action, device):
    class RandomAgent:
        @staticmethod
        def get_action(states, deterministic=False):
            return torch.rand(size=(states.shape[0], d_action), device=device) * - 1
    return RandomAgent()


def get_deterministic_agent(agent):
    class DeterministicAgent:
        @staticmethod
        def get_action(states):
            return agent.get_action(states, deterministic=True)
    return DeterministicAgent()


def get_env(env_name, record=False):
    env = gym.make(env_name)
    env = BoundedActionsEnv(env)

    env = IsDoneEnv(env)
    env = MuJoCoCloseFixWrapper(env)
    if record:
        env = RecordedEnv(env)

    env.seed(np.random.randint(np.iinfo(np.uint32).max))
    if hasattr(env.action_space, 'seed'):  # Only for more recent gym
        env.action_space.seed(np.random.randint(np.iinfo(np.uint32).max))
    if hasattr(env.observation_space, 'seed'):  # Only for more recent gym
        env.observation_space.seed(np.random.randint(np.iinfo(np.uint32).max))

    return env


class MainLoopTraining:
    def __init__(self, logger, args):
        self.step_i = 0
        # env_config
        tmp_env = gym.make(args.env_name)
        self.is_done = tmp_env.unwrapped.is_done
        self.eval_tasks = {args.task_name: tmp_env.tasks()[args.task_name]}
        self.exploitation_task = tmp_env.tasks()[args.task_name]
        self.d_state = tmp_env.observation_space.shape[0]
        self.d_action = tmp_env.action_space.shape[0]
        self.max_episode_steps = tmp_env.spec.max_episode_steps
        del tmp_env
        if args.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.logger = logger
        self.env_loop = EnvLoop(get_env, env_name=args.env_name, render=args.render)
        # Buffer and Normalizer
        self.buffer = Buffer(self.d_state, self.d_action, args.n_total_steps)
        if args.normalize_data:
            self.buffer.setup_normalizer(TransitionNormalizer(self.d_state, self.d_action, self.device))

        if args.algo == 'td3':
            self.agent = TD3(
                d_state=self.d_state, d_action=self.d_action, device=self.device, gamma=args.gamma, tau=args.tau,
                policy_lr=args.policy_lr, value_lr=args.value_lr,
                value_loss=args.value_loss, value_n_layers=args.value_n_layers, value_n_units=args.value_n_units,
                value_activation=args.value_activation,
                policy_n_layers=args.policy_n_layers, policy_n_units=args.policy_n_units,
                policy_activation=args.policy_activation, grad_clip=args.grad_clip, policy_delay=args.policy_delay,
                expl_noise=args.td3_expl_noise
            )
        elif args.algo == 'ddpg':
            self.agent = DDPG(
                d_state=self.d_state, d_action=self.d_action, device=self.device, gamma=args.gamma, tau=args.tau,
                policy_lr=args.policy_lr, value_lr=args.value_lr,
                value_loss=args.value_loss, value_n_layers=args.value_n_layers, value_n_units=args.value_n_units,
                value_activation=args.value_activation,
                policy_n_layers=args.policy_n_layers, policy_n_units=args.policy_n_units,
                policy_activation=args.policy_activation, grad_clip=args.grad_clip, expl_noise=args.td3_expl_noise
            )
        elif args.algo == 'sac':
            self.agent = SAC(
                d_state=self.d_state, d_action=self.d_action, device=self.device, gamma=args.gamma, tau=args.tau,
                policy_lr=args.policy_lr, value_lr=args.value_lr,
                value_loss=args.value_loss, value_n_layers=args.value_n_layers, value_n_units=args.value_n_units,
                value_activation=args.value_activation,
                policy_n_layers=args.policy_n_layers, policy_n_units=args.policy_n_units,
                policy_activation=args.policy_activation, grad_clip=args.grad_clip, policy_delay=args.policy_delay,
                expl_noise=args.td3_expl_noise
            )
        else:
            raise ValueError(f"Unknown Algorithm {args.algo} ...")
        # TODO: stats store the ep_returns and ep_lengths
        self.stats = EpisodeStats(self.eval_tasks)
        self.last_avg_eval_score = None
        self.random_agent = get_random_agent(self.d_action, self.device)

        self.args = args
        self.plot_rews, self.plot_steps = [], []

    def evaluate_on_task(self):
        env = get_env(self.args.env_name, record=False)
        task = env.unwrapped.tasks()[self.args.task_name]
        env.close()

        episode_returns, episode_length = [], []
        env_loop = EnvLoop(get_env, env_name=self.args.env_name, render=False)
        agent = get_deterministic_agent(self.agent)

        # Test agent on real environment by running an episode
        for ep_i in range(self.args.n_eval_episodes):
            with torch.no_grad():
                states, actions, next_states = env_loop.episode(agent)
                rewards = task(states, actions, next_states)

            ep_return = rewards.sum().item()
            ep_len = len(rewards)
            # print(f"MainLoopStep {self.step_i}| evaluate | EvaluateEpisode {ep_i} | EpReturns {ep_return} | EpLength {ep_len}")
            episode_returns.append(ep_return)
            episode_length.append(ep_len)
        env_loop.close()

        avg_ep_return = np.mean(episode_returns)
        avg_ep_length = np.mean(episode_length)
        print(f"MainLoopStep {self.step_i} | evaluate | AverageReturns {avg_ep_return: 5.2f} | AverageLength {avg_ep_length: 5.2f}")

        return avg_ep_return

    def train(self):
        self.step_i += 1

        behavior_agent = self.random_agent if self.step_i <= self.args.n_warm_up_steps else self.agent
        with torch.no_grad():
            action = behavior_agent.get_action(self.env_loop.state, deterministic=False).to('cpu')

        state, next_state, done = self.env_loop.step(to_np(action))
        reward = self.exploitation_task(state, action, next_state).item()
        self.buffer.add(state, action, next_state, torch.from_numpy(np.array([[reward]], dtype=np.float32)))
        self.stats.add(state, action, next_state, done)

        if done:
            for task_name in self.eval_tasks:
                last_ep_return = self.stats.ep_returns[task_name][-1]
                last_ep_length = self.stats.ep_lengths[task_name][-1]
                print(f"MainLoopStep {self.step_i} | train | EpReturns {last_ep_return: 5.2f} | EpLength {last_ep_length: 5.2f}")

        # Print TaskName StepReward
        for task_name in self.eval_tasks:
            step_reward = self.stats.get_recent_reward(task_name)
            # print(f"Step {self.step_i}\tReward: {step_reward}")
            self.logger.add_scalar("StepReward", step_reward, self.step_i)

        # Training agent
        if self.step_i >= self.args.n_warm_up_steps and self.step_i % self.args.learning_freq == 0:
            self.agent.setup_normalizer(self.buffer.normalizer)
            for _ in range(self.args.n_policy_update_iters):
                states, actions, next_states, rewards = self.buffer.sample(self.args.batch_size, self.device)
                dones = self.is_done(next_states)

                self.agent.update(states, actions, rewards, next_states, ~dones)

        if self.args.eval_freq is not None and self.step_i % self.args.eval_freq == 0:
            self.last_avg_eval_score = self.evaluate_on_task()
            self.logger.add_scalar("EvaluateReturn", self.last_avg_eval_score, self.step_i)

        # Save agent parameters
        if self.step_i >= self.args.n_warm_up_steps and self.step_i % self.args.save_freq == 0:
            os.makedirs(str(args.run_dir / 'incremental'), exist_ok=True)
            self.agent.save(str(args.run_dir / 'incremental' / ('model_step%i.pt' % self.step_i)))
            self.agent.save(str(self.args.run_dir / 'model.pt'))

        experiment_finished = self.step_i >= self.args.n_total_steps
        return DotMap(
            done=experiment_finished,
            step_i=self.step_i
        )

    def stop(self):
        self.env_loop.close()


if __name__ == "__main__":
    args = common_args()
    args = policy_function(args)
    args = value_function(args)

    # Save Directory
    model_dir = Path('./models') / args.env_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir()
                         if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    args.run_dir = model_dir / curr_run
    args.log_dir = args.run_dir / 'logs'
    os.makedirs(str(args.log_dir))
    logger = SummaryWriter(str(args.log_dir))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not args.use_cuda and args.n_training_threads is not None:
        torch.set_num_threads(args.n_training_threads)

    training = MainLoopTraining(logger, args)
    # MainLoop
    res = DotMap(done=False)
    while not res.done:
        res = training.train()

    training.stop()
