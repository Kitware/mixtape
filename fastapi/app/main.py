"""Endpoints that can be used to train or run inference on RL environments."""

import importlib
import numpy as np

from enum import Enum
from pathlib import Path
from typing import Dict
from fastapi import FastAPI
from datetime import datetime
import supersuit as ss

from app.callbacks import CustomLoggingCallbacks
from app.logger import Logger

from PIL import Image

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune import run
from ray.tune.registry import register_env

import pettingzoo.butterfly as butterfly

app = FastAPI()

# Initialize Ray
ray.init(ignore_reinit_error=True)


class PZWrapper(PettingZooEnv):
    """Temporary wrapper for PettingZooEnv.

    Correct an issue with the latest release of PettingZoo environments: The
    RLlib PettingZoo wrapper still passes in the `render_mode` argument to the
    render function, but PettingZoo environments no longer accept an argument.
    """

    def __init__(self, env: PettingZooEnv):
        """Create an interface to the PettingZoo MARL environment library.

        Args:
            env (PettingZooEnv): Inherits from MultiAgentEnv and exposes a
                                 given AEC game from the PettingZoo project.
        """
        super().__init__(env)

    def render(self) -> np._typing.NDArray[np.uint8]:
        """RGB image given the current observation.

        Returns:
            np._typing.NDArray[np.uint8]: A numpy uint8 3D array (image) to
                                          render.
        """
        return self.env.render()


class ParallelPZWrapper(ParallelPettingZooEnv):
    """Temporary wrapper for ParallelPettingZooEnv.

    Correct an issue with the latest release of PettingZoo environments: The
    RLlib PettingZoo wrapper still passes in the `render_mode` argument to the
    render function, but PettingZoo environments no longer accept an argument.
    """

    def __init__(self, env: ParallelPettingZooEnv):
        """Create an interface to the PettingZoo MARL environment library.

        Args:
            env (ParallelPettingZooEnv): Inherits from MultiAgentEnv and
                                         exposes a given AEC game from the
                                         PettingZoo project.
        """
        super().__init__(env)

    def render(self) -> np._typing.NDArray[np.uint8]:
        """RGB image given the current observation.

        Returns:
            np._typing.NDArray[np.uint8]: A numpy uint8 3D array (image) to
                                          render.
        """
        return self.par_env.render()


class ButterflyEnvs(str, Enum):
    """Available PettingZoo Environments."""

    KnightsArchersZombies = 'knights_archers_zombies_v10'
    Pistonball = 'pistonball_v6'
    Pong = 'cooperative_pong_v5'


# Re-shape the environment if necessary
def _reshape_if_necessary(env):
    allowed = [[42, 42], [84, 84], [64, 64], [10, 10]]
    agent = env.possible_agents[0]
    shape = list(env.observation_space(agent).shape)
    if (len(shape) >= 3 and shape[:2] not in allowed):
        env = ss.color_reduction_v0(env, mode='B')
        env = ss.dtype_v0(env, 'float32')
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
        env = ss.frame_stack_v1(env, 3)
    return env


# Create the selected environment
def _env_creator(env_module: butterfly, config: Dict):
    env = env_module.env(render_mode='rgb_array', **config)
    return _reshape_if_necessary(env)


# Create the selected parallel environment
def _parallel_env_creator(env_module: butterfly, config: Dict):
    env = env_module.parallel_env(render_mode='rgb_array', **config)
    return _reshape_if_necessary(env)


# Register the selected environment with RLlib
def _register_environment(env_name: str, config: Dict, parallel: bool):
    env_module = importlib.import_module(f'pettingzoo.butterfly.{env_name}')

    if parallel:
        register_env(env_name, lambda config: ParallelPZWrapper(
            _parallel_env_creator(env_module, config)))
    else:
        register_env(env_name, lambda config: PZWrapper(
            _env_creator(env_module, config)))

    return _env_creator(env_module, config)


###############################################################################
# API ENDPOINTS
###############################################################################

@app.post('/train')
def train(
    env_to_register: ButterflyEnvs,
    env_config: Dict,
    parallel: bool = True,
    num_gpus: float = 0,
    timesteps_total: int = 100,
    env_args: Dict = None,
    training_args: Dict = None,
    framework_args: Dict = None,
    run_args: Dict = None
) -> None:
    """Train an RL PettingZoo Butterfly environment.

    Args:
        env_to_register (ButterflyEnvs): The PettingZoo Butterfly environment
                                         to use.
        env_config (Dict): Arguments to configure the environment.
        parallel (bool, optional): Use the parallel API where all agents have
                                   simultaneous actions and observations.
                                   Defaults to True.
        num_gpus (float, optional): Number of GPUs to allocate to the algorithm
                                    process. Defaults to 0.
        timesteps_total (int, optional): Number of timesteps to stop after.
                                         Defaults to 100.
        env_args (Dict, optional): Set the config’s RL-environment settings.
                                   Defaults to None.
        training_args (Dict, optional): Set the training related configuration.
                                        Defaults to None.
        framework_args (Dict, optional): Sets the config’s DL framework
                                         settings. Defaults to None.
        run_args (Dict, optional): Additional arguments to be passed to the
                                   training function. Defaults to None.
    """
    env_name = env_to_register.value
    _register_environment(env_name, env_config, parallel)

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True, **env_args)
        .rollouts(num_rollout_workers=4, rollout_fragment_length=128)
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
            **training_args
        )
        .debugging(log_level='ERROR')
        .framework(framework='torch', **framework_args)
        .resources(num_gpus=num_gpus)
        .callbacks(callbacks_class=CustomLoggingCallbacks)
    )

    date_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    run(
        'PPO',
        name='PPO',
        stop={'timesteps_total': timesteps_total},
        checkpoint_freq=10,
        checkpoint_at_end=True,
        storage_path=Path(f'./logs/metrics/{date_time}').resolve(),
        config=config.to_dict(),
        **run_args
    )


@app.post('/inference')
def inference(
    env_to_register: ButterflyEnvs,
    env_config: Dict,
    checkpoint_path: str,
    parallel: bool = True
) -> None:
    """Perform prediction on a trained model.

    Args:
        env_to_register (ButterflyEnvs): The PettingZoo Butterfly environment
                                         to use.
        env_config (Dict): Arguments to configure the environment.
        checkpoint_path (str): The path to a checkpoint directory to restore
                               from.
        parallel (bool, optional): Use the parallel API where all agents have
                                   simultaneous actions and observations.
                                   Defaults to True.
    """
    logger = Logger()
    env_name = env_to_register.value
    env = _register_environment(env_name, env_config, parallel)
    if parallel:
        env = env.aec_env

    ppo_agent = PPO.from_checkpoint(Path(checkpoint_path).resolve())
    reward_sum = 0
    frame_list = []

    env.reset()
    i = 0
    data = {}
    for agent in env.agent_iter():
        data.setdefault(i, {'actions': {}, 'rewards': {}, 'observation': {}})
        observation, reward, termination, truncation, info = env.last()
        data[i]['rewards'][agent] = reward
        data[i]['observation'][agent] = observation
        reward_sum += reward
        if termination or truncation:
            action = None
        else:
            action = ppo_agent.compute_single_action(observation)
        data[i]['actions'][agent] = action

        env.step(action)
        i += 1
        if i % (len(env.possible_agents) + 1) == 0:
            img = Image.fromarray(env.render())
            frame_list.append(img)
    data['total_reward'] = reward_sum
    env.close()

    logger.write_to_log('inference.json', data)
    gif_file = f'{logger.log_path}/inference.gif'
    frame_list[0].save(gif_file, save_all=True,
                       append_images=frame_list[1:], duration=3, loop=0)
