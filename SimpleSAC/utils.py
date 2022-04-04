import random
import pprint
import time
import datetime
import uuid
import tempfile
import os
from copy import copy
from socket import gethostname
import pickle
import torch.nn as nn
import wandb
import torch
import numpy as np
import pickle as pkl

import absl.flags
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import config_dict
from collections import defaultdict



class Timer(object):

    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


class WandBLogger(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.online = False
        config.prefix = 'SimpleSAC'
        config.project = 'sac'
        config.output_dir = '/tmp/SimpleSAC'
        config.random_delay = 0.0
        config.experiment_id = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant, exp_prefix='', exp_descriptor='', unique_identifier=None):
        self.config = self.get_default_config(config)

        if not unique_identifier:
            unique_identifier = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        if self.config.experiment_id is None:
            self.config.experiment_id = f'{exp_prefix}_{exp_descriptor}_{unique_identifier}'

        if self.config.prefix != '':
            self.config.project = '{}--{}'.format(self.config.prefix, self.config.project)

        if self.config.output_dir == '':
            self.config.output_dir = tempfile.mkdtemp()
        else:
            self.config.output_dir = os.path.join(self.config.output_dir, self.config.experiment_id)
            os.makedirs(self.config.output_dir, exist_ok=True)

        self._variant = copy(variant)

        if 'hostname' not in self._variant:
            self._variant['hostname'] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        self.run = wandb.init(
            reinit=True,
            config=self._variant,
            project=self.config.project,
            tags=[exp_prefix],
            group=exp_prefix,
            dir=self.config.output_dir,
            id=self.config.experiment_id,
            anonymous=self.config.anonymous,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            mode='online' if self.config.online else 'offline',
        )

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        with open(os.path.join(self.config.output_dir, filename), 'wb') as fout:
            pickle.dump(obj, fout)

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, 'automatically defined flag')
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, 'automatically defined flag')
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, 'automatically defined flag')
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, 'automatically defined flag')
        else:
            raise ValueError('Incorrect value type')
    return kwargs


def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def print_flags(flags, flags_def):
    logging.info(
        'Running training with hyperparameters: \n{}'.format(
            pprint.pformat(
                ['{}: {}'.format(key, val) for key, val in get_user_flags(flags, flags_def).items()]
            )
        )
    )


def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            if prefix is not None:
                output['{}.{}'.format(prefix, key)] = val
            else:
                output[key] = val
    return output

def flatten_dict(d, separator='.'):
    d_copy = d.copy()
    for k in d:
        if isinstance(d[k], dict):
            dk = flatten_dict(d[k], separator)
            del d_copy[k]
            d_copy.update({f'{k}{separator}{new_k}': v for new_k, v in dk.items()})
    return d_copy

def unflatten_dict(d, separator='.'):
    d_copy = defaultdict(defaultdict)
    sep_found = False
    for k, v in d.items():
        if separator in k:
            top_key = k[:k.index(separator)]
            rest_key = k[k.index(separator) + 1:]
            d_copy[top_key][rest_key] = v
            found = True
        else:
            d_copy[k] = v
    for k, v in d_copy.items():
        if type(v) == defaultdict:
            d_copy[k] = unflatten_dict(v)
    return dict(d_copy)

def sequential_head():
    return nn.ModuleList([
        nn.Conv2d(3, 6, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5),
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    ])

def prefix_metrics(metrics, prefix):
    return {
        '{}/{}'.format(prefix, key): value for key, value in metrics.items()
    }

def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        return pkl.load(model_file)

def load_sac(model_path):
    with open(model_path, 'rb') as model_file:
        model = pkl.load(model_file)
        return model['sac']