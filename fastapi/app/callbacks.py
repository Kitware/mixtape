"""Customize callbacks to parse and track relevant data."""

from typing import Any, Dict, Optional, Tuple
from PIL import Image

from app.logger import Logger

from ray.rllib.algorithms import Algorithm
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

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

    def on_algorithm_init(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: MetricsLogger,
        **kwargs
    ) -> None:
        """Run when a new Algorithm instance has finished setup.

        This method gets called at the end of Algorithm.setup() after all the
        initialization is done, and before actually training starts.

        Create a Logger object for each worker. By instantiating on init we can
        create directories by date and time and keep a reference to those
        directories throughout all steps of the training.


        Args:
            algorithm (Algorithm): Reference to the Algorithm instance.
            metrics_logger (MetricsLogger): The MetricsLogger object inside the
                                            Algorithm. Can be used to log custom
                                            metrics after algo initialization.
        """
        def create_logger(w):
            w.global_vars['logger'] = Logger()

        algorithm.workers.foreach_worker(create_logger)

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
        episode.user_data['frame_list'] = []

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

        # Write out the actions, rewards and observation space for each agent
        # per step
        data = {}
        for step in range(postprocessed_batch.agent_steps()):
            for agent_id, values in original_batches.items():
                _, _, batch = values
                data.setdefault(step, {
                    'actions': {},
                    'rewards': {},
                    'observation': {}
                })
                if len(batch.get('actions')) > step:
                    data[step]['actions'][agent_id] = batch.get('actions')[step]
                if len(batch.get('rewards')) > step:
                    data[step]['rewards'][agent_id] = batch.get('rewards')[step]
                if len(batch.get('obs')) > step:
                    data[step]['observation'][agent_id] = batch.get('obs')[step]
        data['total_reward'] = episode.total_reward
        logger.write_to_log(f'training_episode_{episode.episode_id}.json', data)

        frame_list = episode.user_data['frame_list']
        fn = f'training_episode_{episode.episode_id}'
        gif_file = f'{logger.log_path}/{fn}.gif'
        frame_list[0].save(gif_file, save_all=True,
                           append_images=frame_list[1:], duration=3, loop=0)
