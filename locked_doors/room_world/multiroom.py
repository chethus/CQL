import gym
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Dict, Box
from .rooms import Room
import numpy as np


class RoomWithObstacles(Room):
    
    def __init__(self, base='pm', length=0.6, width=0.6, n_slots=2, free_one=0, start=None, target=None):
        if start is None:
            start = np.array((0, -width / 2 + width / 12))
        
        if target is None:
            target = np.array((0, width / 4))

        self.n_slots = n_slots
        self.free_one = free_one
        super().__init__(base, length, width, start, target)

    def get_boundary(self):
        bottom_left = (-self.length / 2, - self.width / 2)
        top_right = (self.length / 2 , self.width / 2)
        return bottom_left, top_right
    
    def get_starting_boundary(self):
        return self.get_boundary()

    def get_walls(self):
        posts = np.linspace(-self.length / 2, self.length / 2, self.n_slots + 1)
        columns = [[(x, - self.width / 8), (x, self.width / 8)] for x in posts[1:-1]]
        barriers = [[(posts[t], 0), (posts[t+1], 0)] for t in range(self.n_slots) if t != self.free_one]
        return columns + barriers
    
    def get_shaped_distance(self,a,b):
        return np.linalg.norm(a-b)
    
class SlottedRoomEnv(mujoco_env.MujocoEnv):

    FRAME_SKIP = 5

    def __init__(self, n_slots, free_one,):

        # Initialize
        utils.EzPickle.__init__(**locals())        
        self._room = RoomWithObstacles(n_slots=n_slots, free_one=free_one, length=0.6, width=0.6)
        self.posts = np.linspace(-0.6 / 2, 0.6 / 2, n_slots + 1)

        model = self.room.get_mjcmodel()

        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, frame_skip=self.FRAME_SKIP)
        
        # Observation Space
        example_obs = self._get_obs()
        obs_size = example_obs.shape[0]
        self.observation_space = Box(-1 * np.ones(obs_size), np.ones(obs_size))

        self.reset()

    @property
    def room(self):
        return self._room

    def _get_obs(self):
        xy = self.get_body_com("particle")[:2].copy()
        return xy

    def _compute_reward(self, obs):
        return obs[1]

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        reward = self._compute_reward(obs)
        info = dict(x=obs[0], y=obs[1], is_success=obs[1]>0)
        done = False
        return obs, reward, done, info

    def reset_to_position(self, position):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.01, high=0.01, size=self.model.nq
        )
        qpos[0] = position[0]#np.random.rand() * 1.2 - 0.6
        qpos[1] = position[1] + self.room.length * 5 / 12

        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.01, high=0.01, size=self.model.nq
        )
        qpos[0] = (np.random.rand() - 1/2) * self.room.length 

        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()




import os.path as osp

class LockedDoorsEnv(gym.Env):
    """
    Environment in which, every episode, the agent gets placed in one of `n_slots` rooms,
    where the i-th room has only the i-th slot open.

    Agent observation: dict(state=position, descriptor=one hot vector)
    Agent action:
    """
    def __init__(self, n_slots):
        self.n_slots = n_slots
        self.sub_envs = [SlottedRoomEnv(n_slots, i) for i in range(self.n_slots)]

        example_descriptor = self.sample_descriptor(0)
        self.example_obs = self.combine_obs(self.sub_envs[0].reset(), example_descriptor)

        self.env_id = np.random.choice(self.n_slots)
        self.descriptor = self.sample_descriptor(self.env_id)
        self.action_space = self.sub_envs[0].action_space

    @property
    def observation_space(self):
        return Dict(spaces={
            k: Box(-1.0, 1.0, shape=v.shape, dtype=np.float32)
            for k, v in self.example_obs.items()
        })

    def reset(self):
        self.env_id = np.random.choice(self.n_slots)
        self.descriptor = self.sample_descriptor(self.env_id)
        return self.combine_obs(self.env.reset(), self.descriptor)
    
    def combine_obs(self, inner_obs, descriptor):
        return {
            'state': inner_obs,
            'descriptor': descriptor,
        }
    
    def sample_descriptor(self, env_id):
        descriptor = np.zeros(self.n_slots)
        descriptor[env_id] = 1
        return descriptor

    @property
    def env(self):
        return self.sub_envs[self.env_id]
    
    @property
    def room(self):
        return self.env.room
    
    @property
    def sim(self):
        return self.env.sim

    def render(self, mode, width=128, height=128):
        return self.sim.render(width, height, camera_name='topview')

    def step(self, a):
        inner_obs, r, done, info = self.env.step(a)
        obs = self.combine_obs(inner_obs, self.descriptor)
        info['correct_slot'] = self.env_id
        return obs, r, done, info

from ..room_world import datasets
dataset_dir = osp.abspath(osp.join(osp.dirname(__file__), '../datasets'))


class LockedDoorsImageEnv(LockedDoorsEnv):
    def __init__(self, dataset='MNIST', train=False, dataset_size=None):
        self.dataset_name = dataset
        dataset_loader = getattr(datasets, dataset)
        self.dataset = dataset_loader.get_dataset(dataset_dir, train=train)
        self.valid_indices = [np.nonzero(np.array(self.dataset.targets) == label)[0].flat[:] for label in [0, 1, 2, 3]]
        if dataset_size is not None:
            per_class_size = dataset_size // 4
            self.valid_indices = [
                idxs[:per_class_size]
                for idxs in self.valid_indices
            ]
        self.dataset_size = sum([len(idxs) for idxs in self.valid_indices])
        super().__init__(n_slots=4)

    @property
    def observation_space(self):
        return Dict({
            k: Box(low=-1, high=1, shape=v.shape,)
            for k, v in self.example_obs.items()
        })    
            
    def sample_descriptor(self, env_id):
        idx = np.random.choice(self.valid_indices[env_id])
        descriptor = self.dataset[idx][0].numpy()
        return descriptor
    
    def reset_specific(self, id=None):
        if id is None:
            self.env_id = np.random.choice(self.n_slots)
            idx = np.random.choice(self.valid_indices[self.env_id])
        else:
            (self.env_id, idx) = id
        self.descriptor = self.dataset[idx][0].numpy()
        return self.combine_obs(self.env.reset(), self.descriptor), (self.env_id, idx)


if True:
    gym.register(f'LockedDoorsEnv-v2',
            entry_point=LockedDoorsEnv,
            max_episode_steps=100,
            kwargs=dict(n_slots=4)
    )
    gym.register('CIFARLockedDoorsEnvTest-v2',
        entry_point=LockedDoorsImageEnv,
        max_episode_steps=100,
        kwargs=dict(dataset='CIFAR10', train=False),
    )
    gym.register('CIFARLockedDoorsEnvTrain-v2',
        entry_point=LockedDoorsImageEnv,
        max_episode_steps=100,
        kwargs=dict(dataset='CIFAR10', train=True),
    )
    for dataset_size in [100, 1000, 10000]:
        gym.register(f'CIFARLockedDoorsEnvTrain{dataset_size}-v2',
            entry_point=LockedDoorsImageEnv,
            max_episode_steps=100,
            kwargs=dict(dataset='CIFAR10', train=True, dataset_size=dataset_size),
        )
# except:

#     print('Already registered LockedDoors envs')
    