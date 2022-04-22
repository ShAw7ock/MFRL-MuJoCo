import torch as th
import torch.nn as nn
from torch.distributions import Normal


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
def linear(x):
    return x


def swish(x):
    return x * th.sigmoid(x)


class Swish(nn.Module):
    def forward(self, input):
        return swish(input)


def init_weights(layer):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.constant_(layer.bias, 0)


def get_activation(activation):
    if activation == 'swish':
        return Swish()
    if activation == 'relu':
        return nn.ReLU()
    if activation == 'tanh':
        return nn.Tanh()
    if activation == 'leaky_relu':
        return nn.LeakyReLU()
    # TODO: I should also initialize depending on the activation
    raise NotImplementedError(f"Unknown activation {activation}")


class ParallelLinear(nn.Module):
    def __init__(self, n_in, n_out, ensemble_size):
        super().__init__()

        weights = th.zeros(ensemble_size, n_in, n_out).float()
        biases = th.zeros(ensemble_size, 1, n_out).float()

        for weight in weights:
            weight.transpose_(1, 0)
            nn.init.xavier_uniform_(weight)
            weight.transpose_(1, 0)

        self.weights = nn.Parameter(weights)
        self.biases = nn.Parameter(biases)

    def forward(self, inp):
        return th.baddbmm(self.biases, inp, self.weights)


class ActionValueFunction(nn.Module):
    """
    With ensemble=2 to output twin q values for TD3.
    """
    def __init__(self, d_state, d_action, n_layers, n_units, activation):
        super().__init__()

        if n_layers == 0:
            # Use linear q function
            self.layers = ParallelLinear(d_state + d_action, 1, ensemble_size=2)
        else:
            layers = [ParallelLinear(d_state + d_action, n_units, ensemble_size=2), get_activation(activation)]
            for lyr_idx in range(1, n_layers):
                layers += [ParallelLinear(n_units, n_units, ensemble_size=2), get_activation(activation)]
            layers += [ParallelLinear(n_units, 1, ensemble_size=2)]
            self.layers = nn.Sequential(*layers)

    def forward(self, state, action):
        x = th.cat([state, action], dim=1)
        x = x.unsqueeze(0).repeat(2, 1, 1)
        return self.layers(x)


class Actor(nn.Module):
    def __init__(self, d_state, d_action, n_layers, n_units, activation):
        super(Actor, self).__init__()

        layers = [nn.Linear(d_state, n_units), get_activation(activation)]
        for _ in range(1, n_layers):
            layers += [nn.Linear(n_units, n_units), get_activation(activation)]
        layers += [nn.Linear(n_units, d_action)]

        [init_weights(layer) for layer in layers if isinstance(layer, nn.Linear)]

        self.layers = nn.Sequential(*layers)

    def forward(self, state):
        # Bounded action (-1, 1)
        return th.tanh(self.layers(state))


class GaussianActor(nn.Module):
    def __init__(self, d_state, d_action, n_layers, n_units, activation):
        super(GaussianActor, self).__init__()

        layers = [nn.Linear(d_state, n_units), get_activation(activation)]
        for _ in range(1, n_layers):
            layers += [nn.Linear(n_units, n_units), get_activation(activation)]
        # layers += [nn.Linear(n_units, d_action)]

        [init_weights(layer) for layer in layers if isinstance(layer, nn.Linear)]

        self.layers = nn.Sequential(*layers)
        self.mean_linear = nn.Linear(n_units, d_action)
        self.log_std_linear = nn.Linear(n_units, d_action)

    def forward(self, state):
        x = self.layers(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = th.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = th.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        return action, log_prob
