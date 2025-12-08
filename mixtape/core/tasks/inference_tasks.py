from __future__ import annotations

import contextlib
import itertools
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from celery import shared_task
from django.db import transaction
import gymnasium as gym
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import ParallelEnv
from ray.rllib.algorithms.algorithm import Algorithm

from mixtape.core.models import AgentStep, Episode, Inference
from mixtape.core.models.step import Step
from mixtape.core.ray_utils.environments import register_environment

if TYPE_CHECKING:
    import numpy.typing as npt


@shared_task
def run_inference_task(inference_pk: int):
    inference = Inference.objects.select_related('checkpoint__training').get(pk=inference_pk)
    env_config = (inference.config or {}).get('env_config', {})

    with contextlib.closing(
        register_environment(
            inference.checkpoint.training.environment,
            env_config,
            inference.parallel,
        )
    ) as env:

        with inference.checkpoint.archive_path() as checkpoint_dir:
            algorithm = Algorithm.from_checkpoint(checkpoint_dir)

            # Create the Episode early, as AgentSteps need to reference it
            with transaction.atomic():
                episode = Episode.objects.create(inference=inference)

                if isinstance(env, gym.Env):
                    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, frame_skip=1)
                    env = FrameStack(env, num_stack=4)
                    observation, _ = (
                        env.reset()
                    )  # `reset` returns observation for single agent in Gym envs
                    reward = 0.0

                    for step in itertools.count(start=0):
                        observation = np.transpose(observation, (1, 2, 0))
                        action, state, extras = algorithm.compute_single_action(
                            observation, full_fetch=True
                        )

                        rgb_image_array = cast('npt.NDArray', env.render())
                        with Step.rgb_array_to_file(
                            rgb_image_array, f'step/{uuid4()}'
                        ) as image_file:
                            step_model = Step.objects.create(
                                episode=episode, number=step, image=image_file
                            )

                        agent_step = AgentStep(
                            agent='agent_0',
                            step=step_model,
                            action=action,
                            rewards=[reward],
                            observation_space=observation.tolist(),
                            action_distribution=(
                                extras['action_dist_inputs'].tolist()
                                if 'action_dist_inputs' in extras
                                else None
                            ),
                        )
                        agent_step.full_clean()
                        agent_step.save()

                        # Step the environment forward with the computed actions
                        observation, reward, terminated, truncated, info = env.step(action)

                        if terminated or truncated:
                            break
                elif isinstance(env, ParallelEnv):
                    # `reset` returns observations for all agents in parallel envs
                    observations, _ = env.reset()
                    rewards = {agent: 0.0 for agent in env.agents}

                    # Loop until all agents are done (terminated or truncated)
                    for step in itertools.count(start=0):
                        actions = {}
                        action_distributions = {}
                        for agent, obs in observations.items():
                            action, state, extras = algorithm.compute_single_action(
                                obs, full_fetch=True
                            )
                            actions[agent] = action
                            action_distributions[agent] = extras.get('action_dist_inputs')
                        rgb_image_array = env.render()
                        with Step.rgb_array_to_file(
                            rgb_image_array, f'step/{uuid4()}'
                        ) as image_file:
                            step_model = Step.objects.create(
                                episode=episode, number=step, image=image_file
                            )
                        for agent in env.agents:
                            agent_step = AgentStep(
                                agent=agent,
                                step=step_model,
                                action=actions[agent],
                                observation_space=observations[agent].tolist(),
                                rewards=[rewards[agent]],
                                action_distribution=(
                                    action_distributions[agent].tolist()
                                    if action_distributions[agent] is not None
                                    else None
                                ),
                            )
                            agent_step.full_clean()
                            agent_step.save()

                        # Step the environment forward with the computed actions
                        observations, rewards, terminations, truncations, infos = env.step(actions)

                        if all([terminations[agent] or truncations[agent] for agent in env.agents]):
                            break
                elif isinstance(env, AECEnv):
                    # AEC environment (agent-iterating environment)
                    env.reset()
                    reward = 0.0

                    # Loop over agents in the environment using the agent iterator
                    for step, agent in enumerate(env.agent_iter()):
                        observation, reward, termination, truncation, info = env.last()
                        action, state, extras = algorithm.compute_single_action(
                            observation, full_fetch=True
                        )
                        # No action needed if the agent is done
                        action = None if termination or truncation else action
                        rgb_image_array = env.render()
                        with Step.rgb_array_to_file(
                            rgb_image_array, f'step/{uuid4()}'
                        ) as image_file:
                            step_model = Step.objects.create(
                                episode=episode, number=step, image=image_file
                            )
                        if action is not None:
                            agent_step = AgentStep(
                                agent=agent,
                                step=step_model,
                                action=action,
                                rewards=[reward],
                                observation_space=observation.tolist(),
                                action_distribution=(
                                    extras['action_dist_inputs'].tolist()
                                    if 'action_dist_inputs' in extras
                                    else None
                                ),
                            )
                            agent_step.full_clean()
                            agent_step.save()

                        env.step(action)  # Step the environment forward with the action
