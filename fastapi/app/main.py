"""Endpoints that can be used to train or run inference on RL environments."""

import importlib
import types
from enum import Enum
from pathlib import Path
from typing import Dict

import numpy as np
import pettingzoo.butterfly as butterfly
import ray
import supersuit as ss
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune import run
from ray.tune.registry import register_env

from app.callbacks import CustomLoggingCallbacks, InferenceLoggingCallbacks
from fastapi import FastAPI

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
    if len(shape) >= 3 and shape[:2] not in allowed:
        env = ss.color_reduction_v0(env, mode='B')
        env = ss.dtype_v0(env, 'float32')
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
        env = ss.frame_stack_v1(env, 3)
    return env


# Create the selected environment
def _env_creator(env_module: types.ModuleType, config: Dict):
    env = env_module.env(render_mode='rgb_array', **config)
    return _reshape_if_necessary(env)


# Create the selected parallel environment
def _parallel_env_creator(env_module: butterfly, config: Dict):
    env = env_module.parallel_env(render_mode='rgb_array', **config)
    return _reshape_if_necessary(env)


# Register the selected environment with RLlib
def _register_environment(env_name: str, config: Dict, parallel: bool):
    env_module = importlib.import_module(f'pettingzoo.butterfly.{env_name}')
    env = None

    if parallel:
        env = _parallel_env_creator(env_module, config)
        register_env(env_name, lambda config: ParallelPZWrapper(env))
    else:
        env = _env_creator(env_module, config)
        register_env(env_name, lambda config: PZWrapper(env))

    return env


###############################################################################
# API ENDPOINTS
###############################################################################


class SupportedAlgorithm(str, Enum):
    PPO = 'PPO'  # noqa: F811
    DQN = 'DQN'  # noqa: F811


@app.post('/train')
def train(
    env_to_register: ButterflyEnvs,
    env_config: Dict,
    algorithm: SupportedAlgorithm = SupportedAlgorithm.PPO,
    parallel: bool = True,
    num_gpus: float = 0,
    timesteps_total: int = 5000,
    env_args: Dict | None = None,
    training_args: Dict | None = None,
    framework_args: Dict | None = None,
    run_args: Dict | None = None,
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

    # Set general purpose defaults
    training_args.setdefault('train_batch_size', 512)
    training_args.setdefault('lr', 2e-5)
    training_args.setdefault('gamma', 0.99)

    # Set algorithm specific defaults
    if algorithm == SupportedAlgorithm.PPO:
        ppo_defaults = {
            'lambda_': 0.9,
            'use_gae': True,
            'clip_param': 0.4,
            'grad_clip': None,
            'entropy_coeff': 0.1,
            'vf_loss_coeff': 0.25,
            'sgd_minibatch_size': 64,
            'num_sgd_iter': 10,
        }
        training_args = ppo_defaults | training_args

    # Setup config with callbacks, debugging, training, etc.
    alg = PPOConfig() if algorithm == SupportedAlgorithm.PPO else DQNConfig()
    config = (
        alg.callbacks(callbacks_class=CustomLoggingCallbacks)
        .debugging(log_level='ERROR')
        .environment(
            env=env_name,
            clip_actions=env_args.get('clip_actions', True),
            **env_args,
        )
        .env_runners(num_env_runners=4, rollout_fragment_length='auto')
        .framework(
            framework=framework_args.get('framework', 'torch'),
            **framework_args,
        )
        .resources(num_gpus=num_gpus)
        .training(**training_args)
    )

    # Set run arg defaults
    run_args = run_args or {}
    run_args.setdefault('stop', {'timesteps_total': timesteps_total})
    run_args.setdefault('checkpoint_freq', 10)
    run_args.setdefault('checkpoint_at_end', True)
    run_args.setdefault('storage_path', Path('./logs').resolve())
    run_args.setdefault('config', config.to_dict())

    # Dispatch run
    run(
        algorithm.value,
        name=algorithm.value,
        **run_args,
    )


@app.post('/inference')
def inference(
    env_to_register: ButterflyEnvs,
    env_config: Dict,
    checkpoint_path: str,
    parallel: bool = True,
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
    env_name = env_to_register.value
    env = _register_environment(env_name, env_config, parallel)

    # FIXME: Temporary work-around
    # Ray currently has no way to natively hook up callbacks like they do for
    # training. In the future this should be a custom class that will hook into
    # the inference workflow so that logs are automatically produced like they
    # are for training.
    callback = InferenceLoggingCallbacks(env)

    config = PPOConfig().environment(env=env_name)
    algorithm = config.build()
    algorithm.restore(checkpoint_path)  # Restore a trained model checkpoint for inference

    callback.on_begin_inference()  # FIXME: Manually signal the start of the inference process via callback

    if parallel:
        observations, _ = (
            env.reset()
        )  # `reset` returns observations for all agents in parallel envs
        done = {agent: False for agent in env.agents}

        # Loop until all agents are done (terminated or truncated)
        while not all(done.values()):
            actions = {
                agent: algorithm.compute_single_action(obs) for agent, obs in observations.items()
            }
            # Step the environment forward with the computed actions
            observations, rewards, terminations, truncations, infos = env.step(actions)
            done = {agent: terminations[agent] or truncations[agent] for agent in env.agents}
            # FIXME: Manually log the actions, rewards, and observations for analysis
            callback.on_compute_action(actions, rewards, observations)
        # FIXME: Manually signal that inference is complete
        callback.on_complete_inference(env_name)
    else:
        # AEC environment (agent-iterating environment)
        env.reset()

        # Loop over agents in the environment using the agent iterator
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None  # No action needed if the agent is done
            else:
                action = algorithm.compute_single_action(observation)
            env.step(action)  # Step the environment forward with the action
            # FIXME: Manually log the action, reward, and observation for the current agent
            callback.on_compute_action({agent: action}, {agent: reward}, {agent: observation})
        # FIXME: Manually signal that inference for the AEC environment is complete
        callback.on_complete_inference(env_name, parallel=False)
    env.close()  # Close the environment to free resources
