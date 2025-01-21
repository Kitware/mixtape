"""Customize callbacks to parse and track relevant data."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium
from PIL import Image
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.evaluation import Episode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import PolicyID

from app.logger import Logger


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
        **kwargs,
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
        episode.user_data['step_data'] = {}

    def on_episode_step(
        self,
        *,
        worker: EnvRunner,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        env_index: int,
        **kwargs,
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

    def on_episode_end(
        self,
        *,
        episode: Episode,
        env_runner: EnvRunner | None = None,
        metrics_logger: MetricsLogger | None = None,
        env: gymnasium.Env | None = None,
        env_index: int,
        rl_module: RLModule | None = None,
        worker: EnvRunner | None = None,
        base_env: BaseEnv | None = None,
        policies: Dict[str, Policy] | None = None,
        **kwargs,
    ) -> None:
        """Call when an episode is done.

        Args:
            episode (Episode): The terminated/truncated episode object after
                               the returned obs, rewards, etc.. have been
                               logged to the episode object.
            env_index (int): The index of the sub-environment that is about to
                             be reset (within the vector of sub-environments of
                             the BaseEnv).
            env_runner (EnvRunner | None, optional): Reference to the EnvRunner
                                                     running the env and
                                                     episode. Defaults to None.
            metrics_logger (MetricsLogger | None, optional): The MetricsLogger
                                                             object inside the
                                                             env_runner.
                                                             Defaults to None.
            env (gymnasium.Env | None, optional): The environment object
                                                  running the started episode.
                                                  Defaults to None.
            rl_module (RLModule | None, optional): The RLModule used to compute
                                                   actions for stepping the
                                                   env. Defaults to None.
            worker (EnvRunner | None, optional): Reference to the EnvRunner
                                                 running the env and episode.
                                                 Defaults to None.
            base_env (BaseEnv | None, optional): The lowest-level env interface
                                                 used by RLlib for sampling.
                                                 Defaults to None.
            policies (Dict[str, Policy] | None, optional): A dict mapping
                                                           policy IDs to policy
                                                           objects.
                                                           Defaults to None.
        """
        log_dir = worker.io_context.log_dir.rsplit('/', 1)[-1]
        log_dir = list(Path('./logs').glob(f'*/{log_dir}/'))[0]
        logger = Logger(log_dir)

        file_name = f'training_episode_{episode.episode_id}.json'
        episode.user_data['step_data']['total_reward'] = episode.total_reward
        logger.write_to_log(file_name, episode.user_data['step_data'])

        frame_list = episode.user_data['frame_list']
        fn = f'training_episode_{episode.episode_id}'
        gif_file = f'{logger.log_path}/{fn}.gif'
        frame_list[0].save(
            gif_file,
            save_all=True,
            append_images=frame_list[1:],
            duration=3,
            loop=0,
        )

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
        # Write out the actions, rewards and observation space for each
        # agent per step
        data = episode.user_data['step_data']
        offset = episode.total_env_steps - episode.active_env_steps

        for idx in range(episode.active_env_steps):
            step = idx + offset
            values = original_batches[agent_id]
            _, _, batch = values
            data.setdefault(step, {'actions': {}, 'rewards': {}, 'obss': {}})
            if idx < len(actions := batch.get('actions')):
                data[step]['actions'][agent_id] = actions[idx]
            if idx < len(rewards := batch.get('rewards')):
                data[step]['rewards'][agent_id] = rewards[idx]
            if idx < len(obs := batch.get('obs')):
                data[step]['obss'][agent_id] = obs[idx]


class InferenceLoggingCallbacks:
    def __init__(self, env):
        self.user_data = {}
        self.step = 0
        self.logger = Logger()
        self.env = env

    def on_begin_inference(self):
        self.user_data['frame_list'] = []
        self.user_data['step_data'] = {}
        self.user_data['step_data']['total_reward'] = 0

    def on_compute_action(self, actions, rewards, obss):
        data = self.user_data['step_data']
        data.setdefault(self.step, {'actions': {}, 'rewards': {}, 'obss': {}})

        for agent in actions.keys():
            data[self.step]['actions'][agent] = actions[agent]
        for agent in rewards.keys():
            data[self.step]['rewards'][agent] = rewards[agent]
            data['total_reward'] += rewards[agent]
        for agent in obss.keys():
            data[self.step]['obss'][agent] = obss[agent]

        img = Image.fromarray(self.env.render())
        self.user_data['frame_list'].append(img)

        self.step += 1

    def on_complete_inference(self, env_name, parallel=True):
        self.logger.write_to_log(
            f'{"parallel" if parallel else "aec"}_{env_name}_inference.json',
            self.user_data['step_data'],
        )
        gif_file = (
            f'{self.logger.log_path}/{"parallel" if parallel else "aec"}_{env_name}_inference.gif'
        )
        frame_list = self.user_data['frame_list']
        frame_list[0].save(
            gif_file, save_all=True, append_images=frame_list[1:], duration=3, loop=0
        )
