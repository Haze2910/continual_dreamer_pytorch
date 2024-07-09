import threading

import gymnasium as gym
import numpy as np


class GymWrapper:
    """
    A wrapper for Minigrid environments to handle observations and actions.
    """
    def __init__(self, env, obs_key='image', act_key='action'):
        self._env = env
        self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
        self._act_is_dict = hasattr(self._env.action_space, 'spaces')
        self._obs_key = obs_key
        self._act_key = act_key

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)
    
    @property
    def obs_space(self):
        if self._obs_is_dict:
            spaces = self._env.observation_space.spaces.copy()
        else:
            spaces = {self._obs_key: self._env.observation_space}
        spaces = self._remove_mission_field(spaces)
        return {
            **spaces,
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=bool),
        }

    @property
    def act_space(self):
        if self._act_is_dict:
            return self._env.action_space.spaces.copy()
        else:
            return {self._act_key: self._env.action_space}

    def step(self, action):
        if not self._act_is_dict:
            action = action[self._act_key]
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs['reward'] = float(reward)
        obs['is_first'] = False
        obs['is_last'] = done
        obs['is_terminal'] = info.get('is_terminal', done)
        obs = self._remove_mission_field(obs)
        return obs

    def reset(self):
        obs, _ = self._env.reset()
        if not self._obs_is_dict:
            obs = {self._obs_key: obs}
        obs['reward'] = 0.0
        obs['is_first'] = True
        obs['is_last'] = False
        obs['is_terminal'] = False
        obs = self._remove_mission_field(obs)
        return obs
    
    def _remove_mission_field(self, fields):
        if "mission" in fields:
            del fields["mission"]
        return fields

class Atari:
    LOCK = threading.Lock()

    def __init__(
            self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
            life_done=False, sticky=True, all_actions=False):
        assert size[0] == size[1]
        if name == 'james_bond':
            name = 'jamesbond'
        with self.LOCK:
            env = gym.make(
                id=name, obs_type='image', frameskip=1,
                repeat_action_probability=0.25 if sticky else 0.0,
                full_action_space=all_actions)
        env._get_obs = lambda: None
        self._env = gym.wrappers.AtariPreprocessing(env, noops, action_repeat, size[0], life_done, grayscale)
        self._size = size
        self._grayscale = grayscale

    @property
    def obs_space(self):
        shape = self._size + (1 if self._grayscale else 3,)
        return {
            'image': gym.spaces.Box(0, 255, shape, np.uint8),
            'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=bool),
        }

    @property
    def act_space(self):
        return {'action': self._env.action_space}

    def step(self, action):
        image, reward, done, info = self._env.step(action['action'])
        if self._grayscale:
            image = image[..., None]
        return {
            'image': image,
            'ram': self._env.env._get_ram(),
            'reward': reward,
            'is_first': False,
            'is_last': done,
            'is_terminal': done,
        }

    def reset(self):
        with self.LOCK:
            image = self._env.reset()
        if self._grayscale:
            image = image[..., None]
        return {
            'image': image,
            'ram': self._env.env._get_ram(),
            'reward': 0.0,
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
        }

    def close(self):
        return self._env.close()

class TimeLimit:
    """
    A wrapper for adding a time limit to an env.
    """
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        if self._step is None:
            print('Must reset environment.')
            raise RuntimeError
        else:
            obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            obs['is_last'] = True
            self._step = None
        return obs

    def reset(self):
        self._step = 0
        return self._env.reset()

class NormalizeAction:
    def __init__(self, env, key='action'):
        self._env = env
        self._key = key
        space = env.act_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})

class OneHotAction:
    def __init__(self, env, key='action'):
        assert hasattr(env.act_space[key], 'n')
        self._env = env
        self._key = key
        self._random = np.random.RandomState()

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        shape = (self._env.act_space[self._key].n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        space.n = shape[0]
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        index = np.argmax(action[self._key]).astype(int)
        reference = np.zeros_like(action[self._key])
        reference[index] = 1
        if not np.allclose(reference, action[self._key]):
            raise ValueError(f'Invalid one-hot action:\n{action}')
        return self._env.step({**action, self._key: index})

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.act_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference

class ResizeImage:
    """
    A wrapper for resizing the obs image to a fixed size.
    """
    def __init__(self, env, size=(64, 64)):
        self._env = env
        self._size = size
        self._keys = [k for k, v in env.obs_space.items()
            if len(v.shape) > 1 and v.shape[:2] != size]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
            from PIL import Image
            self._Image = Image

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        for key in self._keys:
            shape = self._size + spaces[key].shape[2:]
            spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def reset(self):
        obs = self._env.reset()
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image
