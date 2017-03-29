"""basic wrappers, useful for reinforcement learning on gym envs"""
import numpy as np
from scipy.misc import imresize
from gym.core import ObservationWrapper,Wrapper
from gym.spaces.box import Box
from copy import copy

class PreprocessImage(ObservationWrapper):
    def __init__(self, env, height=64, width=64, grayscale=True,
                 crop=lambda img: img):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        super(PreprocessImage, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop

        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [height, width, n_colors])

    def _observation(self, img):
        """what happens to the observation"""
        img = self.crop(img)
        img = imresize(img, self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims=True)
        # img = np.transpose(img, (2, 0, 1))  # reshape from (h,w,colors) to (colors,h,w)
        img = img.astype('float32') / 255.
        return img


class FrameBuffer(Wrapper):
    def __init__(self, env, n_frames=4, reshape_fn=lambda x: x):
        """A gym wrapper that returns last n_frames observations as a single observation.
        Useful for games like Atari and Doom with screen as input."""
        super(FrameBuffer, self).__init__(env)
        self.reshape_fn = reshape_fn
        self.framebuffer = np.zeros([n_frames,]+list(env.observation_space.shape))
        self.observation_space = Box(0.0, 1.0, self.reshape_fn(self.framebuffer).shape)

    def reset(self):
        """resets breakout, returns initial frames"""
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset())
        return self.reshape_fn(self.framebuffer)

    def step(self, action):
        """plays breakout for 1 step, returns 4-frame buffer"""
        new_obs, r, done, info = self.env.step(action)
        self.update_buffer(new_obs)
        return self.reshape_fn(self.framebuffer), r, done, info

    def update_buffer(self, obs):
        """push new observation to the buffer, remove the earliest one"""
        self.framebuffer = np.vstack([obs[None], self.framebuffer[:-1]])


class EnvPool(Wrapper):
    def __init__(self, env, n_envs=16, min_n_envs=None):
        super(EnvPool, self).__init__(env)
        self.initial_env = env
        self.n_envs = n_envs
        self.min_n_envs = (min_n_envs or 1)
        self.env_shape = env.observation_space.shape
        self.envs = []
        self.dones_env = []
        self.recreate_envs()

    def recreate_envs(self, done_mask=None):
        self.close()
        if done_mask is None:
            self.envs = np.array([copy(self.initial_env) for _ in range(self.n_envs)])
        else:
            self.envs = np.array(
                [copy(self.initial_env) if done_mask[i] else self.envs[i]
                for i in range(self.n_envs)])
        self.dones_env = np.array([False] * self.n_envs)

    def reset(self):
        result = np.zeros(shape=[self.n_envs, ] + list(self.env_shape), dtype=np.float32)
        for i, env in enumerate(self.envs):
            result[i] = env.reset()
        return result

    def step(self, actions):
        curr_n_env = np.sum(np.invert(self.dones_env))
        if len(actions) == curr_n_env:
            new_states = np.zeros(shape=(curr_n_env, ) + tuple(self.env_shape), dtype=np.float32)
            rewards = np.zeros(shape=curr_n_env, dtype=np.float32)
            dones = np.ones(shape=curr_n_env, dtype=np.bool)
            for i, env in enumerate(self.envs[self.dones_env]):
                new_s, r, done, _ = env.step(actions[i])
                new_states[i] = new_s
                rewards[i] = r
                dones[i] = done
        else:
            raise NotImplemented()
        return new_states, rewards, dones, None

    def close(self):
        for env in self.envs:
            env.close()
