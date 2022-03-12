import argparse


def common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="GYMMB_HalfCheetah-v2", type=str, help="environment name: GYMMB_* or Magellan*")
    parser.add_argument("--task_name", default="standard", type=str, help="assert standard")
    parser.add_argument("--seed", default=1, type=int, help="random seed")
    parser.add_argument("--normalize_data", default=False, type=bool, help="whether to normalize the data for training")
    parser.add_argument("--n_training_threads", default=None, type=int, help="while not GPU, set CPU threads")
    parser.add_argument("--n_total_steps", default=50000, type=int, help="total number of steps in real environment (including warm up)")
    parser.add_argument("--n_warm_up_steps", default=1000, type=int, help="number of steps to initialized the buffer")
    parser.add_argument("--render", default=False, type=bool, help="rendering the env")
    parser.add_argument("--use_cuda", default=True, type=bool, help="use GPU")
    parser.add_argument("--batch_size", default=1024, type=int, help="Training batch size for agent")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--tau', default=0.005, type=float, help="soft update factor")
    parser.add_argument('--learning_freq', default=1, type=int, help="agent update frequency")
    parser.add_argument('--eval_freq', default=1000, type=int, help="evaluating policy frequency")
    parser.add_argument('--grad_clip', default=5, type=int, help="gradient clip")
    parser.add_argument('--save_freq', default=10000, type=int, help="parameters save frequency")

    args = parser.parse_args()
    return args


def policy_function(args):
    args.policy_n_layers = 2            # number of hidden layers (>=1)
    args.policy_n_units = 384           # number of units in each hidden layer
    args.policy_activation = 'swish'
    args.policy_lr = 1e-4               # learning rate
    # Parameters for TD3
    args.policy_delay = 2
    args.td3_expl_noise = 0.1
    # Parameters for running
    args.n_policy_update_iters = 1      # training times for each agent updating
    args.n_eval_episodes = 10           # the number of evaluation times
    return args


def value_function(args):
    args.value_n_layers = 2             # number of hidden layers (>=1)
    args.value_n_units = 384            # number of units in each hidden layer
    args.value_activation = 'swish'
    args.value_lr = 1e-4
    args.value_loss = 'mse'             # 'huber' or 'mse'
    return args
