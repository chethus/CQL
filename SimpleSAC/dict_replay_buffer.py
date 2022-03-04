from copy import copy, deepcopy
from queue import Queue
from select import select
from .utils import flatten_dict
import threading

import numpy as np
import torch


class DictReplayBuffer(object):
    def __init__(self, max_size):
        self._max_size = max_size
        self._next_idx = 0
        self._size = 0
        self._initialized = False
        self._total_steps = 0

    def __len__(self):
        return self._size

    def _init_storage(self, example_obs, action_dim):
        example_obs = flatten_dict(example_obs)
        self._observations, self._next_observations = dict(), dict()
        for k, v in example_obs.items():
            self._observations[k] = np.zeros((self._max_size, *np.array(v).shape), dtype=np.float32)
            self._next_observations[k] = np.zeros((self._max_size, *v.shape), dtype=np.float32)
        self._action_dim = action_dim
        self._actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._max_size, dtype=np.float32)
        self._dones = np.zeros(self._max_size, dtype=np.float32)
        self._next_idx = 0
        self._size = 0
        self._initialized = True

    def add_sample(self, observation, action, reward, next_observation, done):
        if not self._initialized:
            self._init_storage(observation, action.size)
        observation = flatten_dict(observation)
        for k, v in observation.items():
            self._observations[k][self._next_idx, ...] = np.array(v, dtype=np.float32)
        next_observation = flatten_dict(next_observation)
        for k, v in next_observation.items():
            self._next_observations[k][self._next_idx, ...] = np.array(v, dtype=np.float32)
        self._actions[self._next_idx, :] = np.array(action, dtype=np.float32)
        self._rewards[self._next_idx] = reward
        self._dones[self._next_idx] = float(done)

        if self._size < self._max_size:
            self._size += 1
        self._next_idx = (self._next_idx + 1) % self._max_size
        self._total_steps += 1

    def add_traj(self, observations, actions, rewards, next_observations, dones):
        for o, a, r, no, d in zip(observations, actions, rewards, next_observations, dones):
            self.add_sample(o, a, r, no, d)

    def add_batch(self, batch):
        self.add_traj(
            batch['observations'], batch['actions'], batch['rewards'],
            batch['next_observations'], batch['dones']
        )

    def sample(self, batch_size):
        indices = np.random.randint(len(self), size=batch_size)
        return self.select(indices)

    def select(self, indices):
        selected_observations = {k: v[indices, ...] for k, v in self._observations.items()}
        selected_next_observations = {k: v[indices, ...] for k, v in self._next_observations.items()}
        return flatten_dict(dict(
            observations=selected_observations,
            actions=self._actions[indices, ...],
            rewards=self._rewards[indices, ...],
            next_observations=selected_next_observations,
            dones=self._dones[indices, ...],
        ))

    def generator(self, batch_size, n_batchs=None):
        i = 0
        while n_batchs is None or i < n_batchs:
            yield self.sample(batch_size)
            i += 1

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def data(self):
        return flatten_dict(dict(
            observations=self._observations,
            actions=self._actions[:self._size, ...],
            rewards=self._rewards[:self._size, ...],
            next_observations=self._next_observations,
            dones=self._dones[:self._size, ...]
        ))


def batch_to_torch(batch, device):
    return {
        k: torch.from_numpy(v).to(device=device, non_blocking=True)
        for k, v in batch.items()
    }

def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed


def parition_batch_train_test(batch, train_ratio):
    train_indices = np.random.rand(batch['observations'].shape[0]) < train_ratio
    train_batch = index_batch(batch, train_indices)
    test_batch = index_batch(batch, ~train_indices)
    return train_batch, test_batch


def subsample_batch(batch, size):
    indices = np.random.randint(batch['observations'].shape[0], size=size)
    return index_batch(batch, indices)


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        concatenated[key] = np.concatenate([batch[key] for batch in batches], axis=0).astype(np.float32)
    return concatenated


def split_batch(batch, batch_size):
    batches = []
    length = batch['observations'].shape[0]
    keys = batch.keys()
    for start in range(0, length, batch_size):
        end = min(start + batch_size, length)
        batches.append({key: batch[key][start:end, ...] for key in keys})
    return batches


def split_data_by_traj(data, max_traj_length):
    dones = data['dones'].astype(bool)
    start = 0
    splits = []
    for i, done in enumerate(dones):
        if i - start + 1 >= max_traj_length or done:
            splits.append(index_batch(data, slice(start, i + 1)))
            start = i + 1

    if start < len(dones):
        splits.append(index_batch(data, slice(start, None)))

    return splits