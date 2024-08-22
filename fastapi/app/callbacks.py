"""Customize callbacks to parse and track relevant data."""

from typing import Any, Dict, Optional, Tuple
from PIL import Image

from app.logger import Logger

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.evaluation import Episode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID


class CustomLoggingCallbacks(DefaultCallbacks):
    """Customized callbacks class.

    Used to track, parse, and log data throughout the training process.
    """

    def on_episode_start(
        self,
        *,
        worker: EnvRunner,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        env_index: int,
        episode: Episode,
        **kwargs
    ) -> None:
        """Run right after an Episode has been started.

        This method gets called after an episode instance has been reset with a
        call to env.reset() by the EnvRunner.

        Create a frame_list list to append to throughout the training. This
        will be used at the end to produce the playback gif.


        Args:
            worker (EnvRunner): Reference to the EnvRunner running the env and
                                episode.
            base_env (BaseEnv): The lowest-level env interface used by RLlib
                                for sampling.
            policies (Dict[PolicyID, Policy]): A dict mapping policy IDs to
                                               policy objects.
            env_index (int): The index of the sub-environment that is about to
                             be reset (within the vector of sub-environments of
                             the BaseEnv).
            episode (Episode): The newly created Episode or EpisodeV2 object.
                               This is the episode that is about to be started
                               with an upcoming env.reset(). Only after this
                               reset call, the on_episode_start callback will
                               be called.
        """
        worker.global_vars.setdefault('logger', Logger())
        episode.user_data['frame_list'] = []
        episode.user_data['completed_agents'] = []

    def on_episode_step(
        self,
        *,
        worker: EnvRunner,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ) -> None:
        """Run on each episode step (after the action(s) has/have been logged).

        The exact time of the call of this callback is after env.step([action])
        and also after the results of this step (observation, reward,
        terminated, truncated, infos) have been logged to the given episode
        object.

        Render and save the image for this step. These images will be used at
        the end to produce the playback gif.


        Args:
            worker (EnvRunner): Reference to the EnvRunner running the env and
                                episode.
            base_env (BaseEnv): The lowest-level env interface used by RLlib
                                for sampling.
            policies (Dict[PolicyID, Policy]): A dict mapping policy IDs to
                                               policy objects.
            episode (Episode): The just stepped episode object (after
                               env.step() and after returned obs, rewards,
                               etc.. have been logged to the episode object).
            env_index (int): The index of the sub-environment that is about to
                             be reset (within the vector of sub-environments of
                             the BaseEnv).
        """
        img = Image.fromarray(worker.env.render())
        episode.user_data['frame_list'].append(img)

    def on_postprocess_trajectory(
        self,
        *,
        worker: EnvRunner,
        episode: Episode,
        agent_id: Any,
        policy_id: str,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        postprocessed_batch: SampleBatch,
        original_batches: Dict[Any, Tuple[Policy, SampleBatch]],
        **kwargs,
    ) -> None:
        """Run immediately after a policy’s postprocess_fn is called.

        This callback ise useful for additional postprocessing for a policy,
        including looking at the trajectory data of other agents in multi-agent
        settings.

        Log the action, reward, and observation for each agent per step, as
        well as the total reward for the episode. Additionally, use saved
        frames from each step to produce a gif of the game playback.


        Args:
            worker (EnvRunner): Reference to the current rollout worker.
            episode (Episode): Episode object.
            agent_id (Any): Id of the current agent.
            policy_id (str): Id of the current policy for the agent.
            postprocessed_batch (SampleBatch): The postprocessed sample batch
                                               for this agent. This object can
                                               be mutated to apply custom
                                               trajectory postprocessing.
            original_batches (Dict[Any, Tuple[Policy, SampleBatch]]):
                    Dict mapping agent IDs to their unpostprocessed trajectory
                    data.
            policies (Optional[Dict[PolicyID, Policy]], optional):
                    Dict mapping policy IDs to policy objects.
        """
        logger = worker.global_vars['logger']
        episode.user_data['completed_agents'].append(agent_id)

        if episode.user_data['completed_agents'] == episode.get_agents():
            # Write out the actions, rewards and observation space for each
            # agent per step
            data = {}
            for step in range(3):
                for agent_id, values in original_batches.items():
                    _, _, batch = values
                    data.setdefault(step, {
                        'actions': {},
                        'rewards': {},
                        'observation': {}
                    })
                    if len(actions := batch.get('actions')) > step:
                        data[step]['actions'][agent_id] = actions[step]
                    if len(rewards := batch.get('rewards')) > step:
                        data[step]['rewards'][agent_id] = rewards[step]
                    if len(obs := batch.get('obs')) > step:
                        data[step]['observation'][agent_id] = obs[step]
            data['total_reward'] = episode.total_reward
            file_name = f'training_episode_{episode.episode_id}.json'
            logger.write_to_log(file_name, data)

            frame_list = episode.user_data['frame_list']
            fn = f'training_episode_{episode.episode_id}'
            gif_file = f'{logger.log_path}/{fn}.gif'
            frame_list[0].save(gif_file, save_all=True,
                               append_images=frame_list[1:],
                               duration=3, loop=0)
