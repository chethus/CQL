import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from .utils import get_cifar_head
import pickle as pkl


def extend_and_repeat(tensor, dim, repeat):
    # Extend the tensor along dim axis and repeat it
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)


def soft_target_update(network, target_network, soft_target_update_rate):
    target_network_params = {k: v for k, v in target_network.named_parameters()}
    for k, v in network.named_parameters():
        target_network_params[k].data = (
            (1 - soft_target_update_rate) * target_network_params[k].data
            + soft_target_update_rate * v.data
        )


def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        if isinstance(observations, dict):
            batch_size = next(iter(observations.values())).shape[0]
            observations_ndim = min(v.ndim for v in observations.values())
        else:
            batch_size = observations.shape[0]
            observations_ndim = observations.ndim
        if actions.ndim == 3 and observations_ndim == 2:
            multiple_actions = True
            if isinstance(observations, dict):
                for k, v in observations.items():
                    observations[k] = extend_and_repeat(v, 1, actions.shape[1]).reshape(-1, v.shape[-1])
            else:
                observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values
    return wrapped


class FullyConnectedNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, arch='256-256', orthogonal_init=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init

        d = input_dim
        modules = []
        hidden_sizes = [int(h) for h in arch.split('-')]

        for hidden_size in hidden_sizes:
            fc = nn.Linear(d, hidden_size)
            if orthogonal_init:
                nn.init.orthogonal_(fc.weight, gain=np.sqrt(2))
                nn.init.constant_(fc.bias, 0.0)
            modules.append(fc)
            modules.append(nn.ReLU())
            d = hidden_size

        last_fc = nn.Linear(d, output_dim)
        if orthogonal_init:
            nn.init.orthogonal_(last_fc.weight, gain=1e-2)
        else:
            nn.init.xavier_uniform_(last_fc.weight, gain=1e-2)

        nn.init.constant_(last_fc.bias, 0.0)
        modules.append(last_fc)

        self.network = nn.Sequential(*modules)

    def forward(self, input_tensor):
        return self.network(input_tensor)


class ReparameterizedTanhGaussian(nn.Module):

    def __init__(self, log_std_min=-20.0, log_std_max=2.0, no_tanh=False):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(self, mean, log_std, sample):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(self, mean, log_std, deterministic=False):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)


        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(
            action_distribution.log_prob(action_sample), dim=-1
        )

        return action_sample, log_prob

class TanhGaussianPolicy(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256',
                 log_std_multiplier=1.0, log_std_offset=-1.0,
                 orthogonal_init=False, no_tanh=False):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = FullyConnectedNetwork(
            observation_dim, 2 * action_dim, arch, orthogonal_init
        )
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    @classmethod
    def build_from_obs(cls, example_obs, action_dim, arch='256-256',
                 log_std_multiplier=1.0, log_std_offset=-1.0,
                 orthogonal_init=False, no_tanh=False):
        obs_is_dict = isinstance(example_obs, dict)
        if obs_is_dict:
            cifar_heads = nn.ModuleDict()
            flat_observation_dim = 0
            for k, v in sorted(example_obs.items()):
                if v.ndim == 3:
                    assert v.shape == (3, 32, 32), 'Only CIFAR images allowed.'
                    cifar_heads[k] = get_cifar_head()
                    # We will use a convolutional head with 10 outputs
                    flat_observation_dim += 10
                elif v.ndim == 1:
                    flat_observation_dim  += v.shape[0]
                else:
                    raise Exception("Only image and vector observation entries supported.")
        else:
            flat_observation_dim = example_obs.shape[0]
        policy = cls(flat_observation_dim, action_dim, arch, log_std_multiplier,
            log_std_offset, orthogonal_init, no_tanh)
        if obs_is_dict:
            policy.cifar_heads = cifar_heads
        return policy

    def log_prob(self, observations, actions):
        if isinstance(observations, dict):
            observations = self.flatten_obs(observations)
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(self, observations, deterministic=False, repeat=None):
        if isinstance(observations, dict):
            observations = self.flatten_obs(observations)
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian(mean, log_std, deterministic)
    
    def flatten_obs(self, observations):
        observations_copy = {k: v for k, v in observations.items()}
        for k, v in observations_copy.items():
            if v.ndim != 4:
                continue
            for layer in self.cifar_heads[k]:
                observations_copy[k] = layer(observations_copy[k])
        return torch.hstack([v for _, v in sorted(observations_copy.items())])

class CIFAROneHotPolicy(nn.Module):

    def __init__(self, cifar_head, one_hot_fc_net):
        super().__init__()
        self.modules_dict = nn.ModuleDict()
        self.modules_dict['cifar_head'] = cifar_head
        self.modules_dict['one_hot_fc_net'] = one_hot_fc_net
    
    def forward(self, observations, deterministic=False):
        if isinstance(observations, dict):
            flattened_obs = self.conv_obs(observations)
        return self.modules_dict['one_hot_fc_net'](flattened_obs, deterministic)

    def conv_obs(self, observations):
        observations_copy = {k: v for k, v in observations.items()}
        for k, v in observations_copy.items():
            if v.ndim != 4:
                continue
            observations_copy[k] = F.one_hot(self.modules_dict['cifar_head'](observations_copy[k])[:,:4].argmax(dim=1).reshape(-1), 4)
        return observations_copy


class SamplerPolicy(object):

    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def __call__(self, observations, deterministic=False):
        obs_is_arr = type(observations).__module__ == np.__name__
        with torch.no_grad():
            if obs_is_arr:
                if observations.ndim == 1:
                    observations = np.expand_dims(observations, 0).astype('float32')
                observations = torch.tensor(
                    observations, dtype=torch.float32, device=self.device
                )
            elif isinstance(observations, dict):
                observations = {k: torch.tensor(v, dtype=torch.float32,
                    device=self.device) for k, v in observations.items()}
                    
            actions, _ = self.policy(observations, deterministic)
            actions = actions.cpu().numpy()
        return actions


class FullyConnectedQFunction(nn.Module):

    def __init__(self, observation_dim, action_dim, arch='256-256', orthogonal_init=False):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.network = FullyConnectedNetwork(
            observation_dim + action_dim, 1, arch, orthogonal_init
        )
    
    @classmethod
    def build_from_obs(cls, example_obs, action_dim, arch='256-256', orthogonal_init=False):
        obs_is_dict = isinstance(example_obs, dict)
        if obs_is_dict:
            cifar_heads = nn.ModuleDict()
            flat_observation_dim = 0
            for k, v in sorted(example_obs.items()):
                if v.ndim == 3:
                    assert v.shape == (3, 32, 32), 'Only CIFAR images allowed.'
                    cifar_heads[k] = get_cifar_head()
                    # We will use a convolutional head with 10 outputs
                    flat_observation_dim += 10
                elif v.ndim == 1:
                    flat_observation_dim  += v.shape[0]
                else:
                    raise Exception('Only image and vector observation entries supported.')
        else:
            flat_observation_dim = example_obs.shape[0]
        q_func = cls(flat_observation_dim, action_dim, arch, orthogonal_init)
        if obs_is_dict:
            q_func.cifar_heads = cifar_heads
        return q_func

    @multiple_action_q_function
    def forward(self, observations, actions):
        if isinstance(observations, dict):
            observations_flattened = self.flatten_obs(observations)
        input_tensor = torch.cat([observations_flattened, actions], dim=-1)
        return torch.squeeze(self.network(input_tensor), dim=-1)
    
    def flatten_obs(self, observations):
        observations_copy = {k: v for k, v in observations.items()}
        for k, v in observations_copy.items():
            if v.ndim != 4:
                continue
            for layer in self.cifar_heads[k]:
                observations_copy[k] = layer(observations_copy[k])
        return torch.hstack([v for _, v in sorted(observations_copy.items())])


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32)
        )

    def forward(self):
        return self.constant
