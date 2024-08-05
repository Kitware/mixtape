from typing import Any, Dict, Optional, Tuple
from PIL import Image

from app.logger import Logger


from ray.rllib.algorithms import Algorithm
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import Episode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID


class MyCallback(DefaultCallbacks):
    def on_algorithm_init(self, *, algorithm: Algorithm, metrics_logger: MetricsLogger, **kwargs):
        def create_logger(w):
            w.global_vars['logger'] = Logger()

        algorithm.workers.foreach_worker(create_logger)

    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker", # type: ignore
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        env_index: int,
        episode: Episode,
        **kwargs
    ) -> None:
        episode.user_data['frame_list'] = []

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker", # type: ignore
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ) -> None:
        img = Image.fromarray(worker.env.render())
        episode.user_data['frame_list'].append(img)

    def on_postprocess_trajectory(
        self,
        *,
        worker: "RolloutWorker", # type: ignore
        episode: Episode,
        agent_id: Any,
        policy_id: str,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        postprocessed_batch: SampleBatch,
        original_batches: Dict[Any, Tuple[Policy, SampleBatch]],
        **kwargs,
    ) -> None:
        logger = worker.global_vars['logger']

        # Write out the actions, rewards and observation space for each agent per step
        data = {}
        for step in range(postprocessed_batch.agent_steps()):
            for agent_id, values in original_batches.items():
                _, _, batch = values
                data.setdefault(step, {'actions': {}, 'rewards': {}, 'observation': {}})
                if len(batch.get('actions')) > step:
                    data[step]['actions'][agent_id] = batch.get('actions')[step]
                if len(batch.get('rewards')) > step:
                    data[step]['rewards'][agent_id] = batch.get('rewards')[step]
                if len(batch.get('obs')) > step:
                    data[step]['observation'][agent_id] = batch.get('obs')[step]
        data['total_reward'] = episode.total_reward
        logger.write_to_log(f'training_episode_{episode.episode_id}.json', data)

        frame_list = episode.user_data['frame_list']
        gif_file = f'{logger.log_path}/training_episode_{episode.episode_id}.gif'
        frame_list[0].save(gif_file, save_all=True, append_images=frame_list[1:], duration=3, loop=0)
