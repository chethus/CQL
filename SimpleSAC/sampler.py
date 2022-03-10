import numpy as np
from collections import defaultdict
from .utils import flatten_dict


class StepSampler(object):

    def __init__(self, env, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._current_observation = self.env.reset()

    def sample(self, policy, n_steps, deterministic=False, replay_buffer=None):
        traj = defaultdict(list)
        traj_info = defaultdict(list)
        for _ in range(n_steps):
            self._traj_steps += 1
            observation = self._current_observation
            obs_is_arr = type(observation).__module__ == np.__name__
            if obs_is_arr:
                observations_batch = np.expand_dims(observation, 0)
            elif isinstance(observation, dict):
                observations_batch = {k: np.expand_dims(v, 0) for k, v in observation.items()}
            else:
                raise Exception('Only array and dictionary observations supported.')
            action = policy(
                observations_batch, deterministic=deterministic
            )[0, :]
            next_observation, reward, done, info = self.env.step(action)
            transition = dict(
                **flatten_dict(dict(observations=observation)),
                actions=action,
                rewards=reward,
                **flatten_dict(dict(next_observations=next_observation)),
                dones=done,
            )
            for k, v in transition.items():
                traj[k].append(v)
            
            for k, v in info.items():
                traj_info[k].append(v)

            if replay_buffer is not None:
                replay_buffer.add_sample(
                    observation, action, reward, next_observation, done
                )

            self._current_observation = next_observation

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0
                self._current_observation = self.env.reset()

        traj = flatten_dict(traj)
        for k, v in traj.items():
            traj[k] = np.array(v, dtype=np.float32)
        return traj, traj_info

    @property
    def env(self):
        return self._env


class TrajSampler(object):

    def __init__(self, env, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env

    def sample(self, policy, n_trajs, deterministic=False, replay_buffer=None, reset_id=None):
        trajs, traj_infos = [], []
        for _ in range(n_trajs):
            traj, traj_info = defaultdict(list), defaultdict(list)
            
            if reset_id is not None:
                assert callable(getattr(self.env, 'reset_specific')), 'Specific resets not supported.'
                self.env.reset()
                observation, _ = self.env.env.env.reset_specific(reset_id)
            else:
                observation = self.env.reset()

            for _ in range(self.max_traj_length):
                obs_is_arr = type(observation).__module__ == np.__name__
                if obs_is_arr:
                    observations_batch = np.expand_dims(observation, 0)
                elif isinstance(observation, dict):
                    observations_batch = {k: np.expand_dims(v, 0) for k, v in observation.items()}
                else:
                    raise Exception('Only array and dictionary observations supported.')
                action = policy(
                    observations_batch, deterministic=deterministic
                )[0, :]
                next_observation, reward, done, info = self.env.step(action)
                transition = dict(
                    **flatten_dict(dict(observations=observation)),
                    actions=action,
                    rewards=reward,
                    **flatten_dict(dict(next_observations=next_observation)),
                    dones=done,
                )
                for k, v in transition.items():
                    traj[k].append(v)
                for k, v in info.items():
                    traj_info[k].append(v)

                if replay_buffer is not None:
                    replay_buffer.add_sample(
                        observation, action, reward, next_observation, done
                    )

                observation = next_observation

                if done:
                    break

            for k, v in traj.items():
                traj[k] = np.array(v, dtype=np.float32)
            for k, v in traj_info.items():
                traj_info[k] = np.array(v, dtype=np.float32)

            trajs.append(traj)
            traj_infos.append(traj_info)
        return trajs, traj_infos

    @property
    def env(self):
        return self._env