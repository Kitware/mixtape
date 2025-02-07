import importlib
import types

import gymnasium as gym
from pettingzoo import AECEnv
from pettingzoo.utils import ParallelEnv
from ray.tune.registry import register_env
import supersuit as ss

from mixtape.ray_utils.constants import ExampleEnvs
from mixtape.ray_utils.wrappers import ParallelPZWrapper, PZWrapper


def is_gymnasium_env(env_name: str) -> bool:
    return ExampleEnvs.type(env_name) == 'Gymnasium'


def reshape_if_necessary(env) -> AECEnv | ParallelEnv:
    # Re-shape the environment if necessary
    allowed = [[42, 42], [84, 84], [64, 64], [10, 10]]
    agent = env.possible_agents[0]
    shape = list(env.observation_space(agent).shape)
    if len(shape) >= 3 and shape[:2] not in allowed:
        env = ss.color_reduction_v0(env, mode='B')
        env = ss.dtype_v0(env, 'float32')
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
        env = ss.frame_stack_v1(env, 3)
    return env


def pz_env_creator(env_module: types.ModuleType, config: dict) -> AECEnv:
    # Create the selected PettingZoo environment
    env = env_module.env(render_mode='rgb_array', **config)
    return reshape_if_necessary(env)


def parallel_env_creator(env_module: types.ModuleType, config: dict) -> ParallelEnv:
    # Create the selected PettingZoo parallel environment
    env = env_module.parallel_env(render_mode='rgb_array', **config)
    return reshape_if_necessary(env)


def gym_env_creator(env_module: str, config: dict) -> gym.Env:
    # Create the selected Gymnasium environment
    return gym.make(f'ale_py:ALE/{env_module}', render_mode='rgb_array', **config)


def register_environment(
    env_name: str, config: dict, parallel: bool
) -> AECEnv | ParallelEnv | gym.Env:
    # Register the selected environment with RLlib
    if is_gymnasium_env(env_name):
        register_env(env_name, lambda config: gym_env_creator(env_name, config))
        return gym_env_creator(env_name, config)
    else:
        env_module = importlib.import_module(f'pettingzoo.butterfly.{env_name}')
        if parallel:
            env = parallel_env_creator(env_module, config)
            register_env(env_name, lambda config: ParallelPZWrapper(env))
        else:
            env = pz_env_creator(env_module, config)
            register_env(env_name, lambda config: PZWrapper(env))
        return env
