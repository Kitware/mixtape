import contextlib
from pathlib import Path
from typing import TextIO

import djclick as click
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
import yaml

from mixtape.ray_utils.callbacks import InferenceLoggingCallbacks
from mixtape.ray_utils.constants import ExampleEnvs
from mixtape.ray_utils.utils import is_gymnasium_env, register_environment


@click.command()
@click.argument(
    'checkpoint_path',
    type=click.Path(exists=True, file_okay=False, resolve_path=True, path_type=Path),
)
@click.option(
    '-e',
    '--env_name',
    type=click.Choice(
        [
            'knights_archers_zombies_v10',
            'pistonball_v6',
            'cooperative_pong_v5',
            'BattleZone-v5',
            'Berzerk-v5',
            'ChopperCommand-v5',
        ]
    ),
    default=ExampleEnvs.PZ_KnightsArchersZombies.value,
    help='The PettingZoo or Gymnasium environment to use.',
)
@click.option(
    '-f',
    '--config_file',
    type=click.File('r'),
    default=None,
    help='Arguments to configure the environment.',
)
@click.option(
    '-p',
    '--parallel',
    is_flag=True,
    help='All agents have simultaneous actions and observations.',
)
def inference(
    checkpoint_path: Path, env_name: str, config_file: TextIO | None, parallel: bool
) -> None:
    """Run inference on the specified trained environment."""
    config_dict: dict = yaml.safe_load(config_file) if config_file else {}
    env_config = config_dict.get('env_config', {})

    with contextlib.closing(register_environment(env_name, env_config, parallel)) as env:
        callback = InferenceLoggingCallbacks(env)

        # Restore a trained model checkpoint for inference
        algorithm = Algorithm.from_checkpoint(str(checkpoint_path))

        callback.on_begin_inference()
        if is_gymnasium_env(env_name):
            env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, frame_skip=1)
            env = FrameStack(env, num_stack=4)
            observation, _ = env.reset()  # `reset` returns observation for single agent in Gym envs

            while True:
                observation = np.transpose(observation, (1, 2, 0))
                action = algorithm.compute_single_action(observation)
                # Step the environment forward with the computed actions
                observation, reward, terminated, truncated, info = env.step(action)
                callback.on_compute_action(
                    {'agent_0': action}, {'agent_0': reward}, {'agent_0': np.array(observation)}
                )
                if terminated or truncated:
                    break
            callback.on_complete_inference(env_name, parallel=False)
        elif parallel:
            # `reset` returns observations for all agents in parallel envs
            observations, _ = env.reset()
            done = {agent: False for agent in env.agents}

            # Loop until all agents are done (terminated or truncated)
            while not all(done.values()):
                actions = {
                    agent: algorithm.compute_single_action(obs)
                    for agent, obs in observations.items()
                }
                # Step the environment forward with the computed actions
                observations, rewards, terminations, truncations, infos = env.step(actions)
                done = {agent: terminations[agent] or truncations[agent] for agent in env.agents}
                callback.on_compute_action(actions, rewards, observations)
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
                callback.on_compute_action({agent: action}, {agent: reward}, {agent: observation})
            callback.on_complete_inference(env_name, parallel=False)
