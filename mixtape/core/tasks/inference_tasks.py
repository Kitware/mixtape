import contextlib
from io import BytesIO
import itertools

from PIL import Image
from celery import shared_task
from django.core.files import File
from django.db import transaction
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import ParallelEnv
from ray.rllib.algorithms.algorithm import Algorithm

from mixtape.core.models import AgentStep, Episode, InferenceRequest
from mixtape.core.ray_utils.environments import register_environment


@shared_task
def run_inference_task(inference_request_pk: int):
    inference_request = InferenceRequest.objects.select_related('checkpoint__training_request').get(
        pk=inference_request_pk
    )
    env_config = inference_request.config.get('env_config', {})

    with contextlib.closing(
        register_environment(
            inference_request.checkpoint.training_request.environment,
            env_config,
            inference_request.parallel,
        )
    ) as env:
        # callback = InferenceLoggingCallbacks(env)

        with inference_request.checkpoint.archive_path() as checkpoint_dir:
            algorithm = Algorithm.from_checkpoint(checkpoint_dir)

            # Create the Episode early, as AgentSteps need to reference it
            with transaction.atomic():
                episode = Episode.objects.create(inference_request=inference_request)

                if isinstance(env, gym.Env):
                    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, frame_skip=1)
                    env = FrameStack(env, num_stack=4)
                    observation, _ = (
                        env.reset()
                    )  # `reset` returns observation for single agent in Gym envs

                    for step in itertools.count(start=0):
                        observation = np.transpose(observation, (1, 2, 0))
                        action = algorithm.compute_single_action(observation)
                        # Step the environment forward with the computed actions
                        observation, reward, terminated, truncated, info = env.step(action)

                        agent_step = AgentStep(
                            episode=episode,
                            agent='agent_0',
                            step=step,
                            action=action,
                            reward=reward,
                            observation_space=observation,
                        )
                        rendering_image = Image.fromarray(env.render())
                        with BytesIO() as rendering_stream:
                            rendering_image.save(rendering_stream, format='PNG')
                            rendering_stream.seek(0)
                            agent_step.rendering = File(rendering_stream, name='rendering.png')
                            agent_step.rendering.save()
                        agent_step.save()

                        if terminated or truncated:
                            break
                elif inference_request.parallel and isinstance(env, ParallelEnv):
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
                        for agent in env.agents:
                            agent_step = AgentStep(
                                episode=episode,
                                agent=agent,
                                step=step,
                                action=actions[agent],
                                reward=rewards[agent],
                            )
                            rendering_image = Image.fromarray(env.render())
                            with BytesIO() as rendering_stream:
                                rendering_image.save(rendering_stream, format='PNG')
                                rendering_stream.seek(0)
                                agent_step.rendering = File(rendering_stream, name='rendering.png')
                                agent_step.rendering.save()
                            agent_step.save()
                        done = {
                            agent: terminations[agent] or truncations[agent] for agent in env.agents
                        }
                elif isinstance(env, AECEnv):
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
                        AgentStep.objects.create(
                            episode=episode, agent=agent, step=step, action=action, reward=reward
                        )
                        rendering_image = Image.fromarray(env.render())
                        with BytesIO() as rendering_stream:
                            rendering_image.save(rendering_stream, format='PNG')
                            rendering_stream.seek(0)
                            agent_step.rendering = File(rendering_stream, name='rendering.png')
                            agent_step.rendering.save()
                        agent_step.save()
