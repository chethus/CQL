import os
import time
from copy import deepcopy
import uuid

import numpy as np
import pprint

import gym
import torch

import absl.app
import absl.flags

from .sac import SAC
from .replay_buffer import ReplayBuffer, batch_to_torch
from .dict_replay_buffer import DictReplayBuffer
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy
from .sampler import StepSampler, TrajSampler
from .utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from .utils import WandBLogger
from viskit.logging import logger, setup_logger
import locked_doors.room_world.multiroom


FLAGS_DEF = define_flags_with_default(
    exp_prefix='test',
    exp_descriptor='sac',
    train_env='HalfCheetah-v2',
    eval_env='HalfCheetah-v2',
    max_traj_length=1000,
    replay_buffer_size=1000000,
    seed=42,
    device='cuda',
    save_model=True,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,
    resnet_head=False,

    n_epochs=2000,
    n_env_steps_per_epoch=1000,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,

    batch_size=256,

    sac=SAC.get_default_config(),
    logging=WandBLogger.get_default_config(),
    snapshot_period=0,
    env_logging=False,

    flatten_obs=False,
)


def main(argv):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    
    variant['exp_descriptor'] = variant['exp_descriptor'].format(**variant)
    FLAGS.exp_descriptor = variant['exp_descriptor']
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant,
        exp_prefix=variant['exp_prefix'],
        exp_descriptor=variant['exp_descriptor'])
    base_log_dir = os.path.join(FLAGS.logging.output_dir, variant['exp_prefix'])
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=base_log_dir,
        include_exp_prefix_sub_dir=False
    )

    set_random_seed(FLAGS.seed)

    train_env = gym.make(FLAGS.train_env).unwrapped
    eval_env = gym.make(FLAGS.eval_env).unwrapped

    if FLAGS.flatten_obs:
        train_env = gym.wrappers.FlattenObservation(train_env)
        eval_env = gym.wrappers.FlattenObservation(eval_env)

    train_sampler = StepSampler(train_env, FLAGS.max_traj_length)
    eval_sampler = TrajSampler(eval_env, FLAGS.max_traj_length)

    if FLAGS.flatten_obs:
        replay_buffer = ReplayBuffer(FLAGS.replay_buffer_size)
    else:
        replay_buffer = DictReplayBuffer(FLAGS.replay_buffer_size)
    
    example_obs = train_env.reset()
    policy = TanhGaussianPolicy.build_from_obs(
        example_obs,
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.policy_arch,
        log_std_multiplier=FLAGS.policy_log_std_multiplier,
        log_std_offset=FLAGS.policy_log_std_offset,
        orthogonal_init=FLAGS.orthogonal_init,
        resnet_head=FLAGS.resnet_head,
    )

    qf1 = FullyConnectedQFunction.build_from_obs(
        example_obs,
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
        resnet_head=FLAGS.resnet_head,
    )
    target_qf1 = deepcopy(qf1)

    qf2 = FullyConnectedQFunction.build_from_obs(
        example_obs,
        eval_sampler.env.action_space.shape[0],
        arch=FLAGS.qf_arch,
        orthogonal_init=FLAGS.orthogonal_init,
        resnet_head=FLAGS.resnet_head,
    )
    target_qf2 = deepcopy(qf2)

    if FLAGS.sac.target_entropy >= 0.0:
        FLAGS.sac.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = SAC(FLAGS.sac, policy, qf1, qf2, target_qf1, target_qf2)
    sac.torch_to_device(FLAGS.device)

    sampler_policy = SamplerPolicy(policy, FLAGS.device)

    viskit_metrics = {}
    recent_traj_infos = []
    for epoch in range(FLAGS.n_epochs):
        metrics = {}
        with Timer() as rollout_timer:
            _, traj_info = train_sampler.sample(
                sampler_policy, FLAGS.n_env_steps_per_epoch,
                deterministic=False, replay_buffer=replay_buffer
            )
            recent_traj_infos.append(traj_info)
            if len(recent_traj_infos) > FLAGS.eval_n_trajs:
                del recent_traj_infos[0:len(recent_traj_infos) - FLAGS.eval_n_trajs]
            metrics['env_steps'] = replay_buffer.total_steps
            metrics['epoch'] = epoch
            if len(recent_traj_infos) == FLAGS.eval_n_trajs:
                for k in recent_traj_infos[0].keys():
                    metrics[f'train_env/runavg_last_{k}'] = np.mean([traj_info[k][-1] for traj_info in recent_traj_infos])
                    metrics[f'train_env/runavg_min_{k}'] = np.mean([min(traj_info[k]) for traj_info in recent_traj_infos])
                    metrics[f'train_env/runavg_max_{k}'] = np.mean([max(traj_info[k]) for traj_info in recent_traj_infos])
                    metrics[f'train_env/runavg_{k}'] = np.mean([np.mean(traj_info[k]) for traj_info in recent_traj_infos])

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                batch = batch_to_torch(replay_buffer.sample(FLAGS.batch_size), FLAGS.device)
                if batch_idx + 1 == FLAGS.n_train_step_per_epoch:
                    metrics.update(
                        prefix_metrics(sac.train(batch), 'sac')
                    )
                else:
                    sac.train(batch)

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs, traj_infos = eval_sampler.sample(
                    sampler_policy, FLAGS.eval_n_trajs, deterministic=True
                )

                metrics['eval_env/average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['eval_env/average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                info_keys = traj_infos[0].keys()
                for k in info_keys:
                    metrics[f'eval_env/avg_last_{k}'] = np.mean([traj_info[k][-1] for traj_info in traj_infos])
                    metrics[f'eval_env/avg_min_{k}'] = np.mean([min(traj_info[k]) for traj_info in traj_infos])
                    metrics[f'eval_env/avg_max_{k}'] = np.mean([max(traj_info[k]) for traj_info in traj_infos])
                    metrics[f'eval_env/avg_{k}'] = np.mean([np.mean(traj_info[k]) for traj_info in traj_infos])

                if FLAGS.save_model:
                    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, 'model.pkl')
                    if FLAGS.snapshot_period > 0 and (epoch + 1) % FLAGS.snapshot_period == 0:
                        wandb_logger.save_pickle(save_data, f'model_epoch_{epoch}.pkl')

        metrics['rollout_time'] = rollout_timer()
        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = rollout_timer() + train_timer() + eval_timer()

        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_model:
        save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')


if __name__ == '__main__':
    absl.app.run(main)
