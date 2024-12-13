import ray
import json

import numpy as np
import supersuit as ss

from pathlib import Path
from PIL import Image

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune import run
from ray.tune.registry import register_env

from pettingzoo.butterfly import knights_archers_zombies_v10


class Logger:
    def __init__(self, parent: str | Path = './logs'):
        self.parent = Path(parent).resolve()

    @property
    def log_path(self) -> Path:
        return f'{self.parent}/custom_logs'

    def serialize_numpy(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

    def write_to_log(self, file_name, data):
        p = Path(f'{self.log_path}/{file_name}')
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(str(p), 'w') as log_file:
            json.dump(data, log_file, indent=2, default=self.serialize_numpy)


class CustomLoggingCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, env_index, episode, **kwargs):
        episode.user_data['frame_list'] = []
        episode.user_data['step_data'] = {}

    def on_episode_step(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        try:
            img = Image.fromarray(worker.env.render())
            episode.user_data['frame_list'].append(img)
        except:
            pass

    def on_episode_end(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index,
        rl_module=None,
        worker=None,
        base_env=None,
        policies=None,
        **kwargs,
    ):
        log_dir = worker.io_context.log_dir.rsplit('/', 1)[-1]
        log_dir = list(Path('./logs').glob(f'*/{log_dir}/'))[0]
        logger = Logger(log_dir)

        file_name = f'training_episode_{episode.episode_id}.json'
        episode.user_data['step_data']['total_reward'] = episode.total_reward
        logger.write_to_log(file_name, episode.user_data['step_data'])

        try:
            frame_list = episode.user_data['frame_list']
            fn = f'training_episode_{episode.episode_id}'
            gif_file = f'{logger.log_path}/{fn}.gif'
            frame_list[0].save(
                gif_file, save_all=True, append_images=frame_list[1:], duration=3, loop=0
            )
        except:
            pass

    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies=None,
        postprocessed_batch,
        original_batches,
        **kwargs,
    ):
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


class ParallelPZWrapper(ParallelPettingZooEnv):
    def __init__(self, env):
        super().__init__(env)

    def render(self):
        return self.par_env.render()


def _reshape_if_necessary(env):
    """
    Check the observation space of the environment and ensures it matches
    allowed dimensions and applies transformations like color reduction,
    resizing, normalization, and frame stacking.
    """
    # List of allowed observation space dimensions (height, width).
    allowed = [[42, 42], [84, 84], [64, 64], [10, 10]]
    agent = env.possible_agents[0]
    shape = list(env.observation_space(agent).shape)
    if len(shape) >= 3 and shape[:2] not in allowed:
        # Checks if the observation space has 3 or more dimensions and if the
        # (height, width) are not in the list of allowed dimensions.
        env = ss.color_reduction_v0(env, mode='B')  # Convert observation to grayscale
        env = ss.dtype_v0(env, 'float32')  # Change observation data type to float32
        env = ss.resize_v1(env, x_size=84, y_size=84)  # Resize to 84x84 pixels.
        env = ss.normalize_obs_v0(env, env_min=0, env_max=1)  # Normalize values to [0, 1]
        env = ss.frame_stack_v1(env, 3)  # Stack the last 3 frames into a single observation
    return env


def _parallel_env_creator(env_module, config):
    env = env_module.parallel_env(render_mode='rgb_array', **config)
    return _reshape_if_necessary(env)


def train(num_gpus=0, timesteps_total=50000):
    """
    Configure and train a PPO agent on the specified environment.

    Args:
        num_gpus (int): Number of GPUs to allocate for training.
        timesteps_total (int): Total number of timesteps to train the agent.
    """
    # Register the environment with RLlib
    env_name = 'knights_archers_zombies_v10'
    register_env(
        env_name,
        lambda config: ParallelPZWrapper(
            _parallel_env_creator(knights_archers_zombies_v10, config)
        ),
    )

    # Define the configuration for the algorithm
    config = (
        PPOConfig()
        .callbacks(
            callbacks_class=CustomLoggingCallbacks
        )  # Set up custom callbacks for logging during training.
        .debugging(log_level='ERROR')
        .environment(env=env_name, clip_actions=True)  # clip actions to the valid range
        .env_runners(num_env_runners=4, rollout_fragment_length='auto')
        .framework(framework='torch')  # use PyTorch.
        .resources(num_gpus=num_gpus)
        .training(
            train_batch_size=512,
            lr=2e-5,  # learning rate for the algorithm
            gamma=0.99,  # Discount factor for rewards
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,  # coefficient for entropy bonus to encourage exploration
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
    )

    run(
        'PPO',
        name='PPO',
        stop={
            'timesteps_total': timesteps_total
        },  # Training stops once the total timesteps reach `timesteps_total`
        checkpoint_freq=10,  # Save a checkpoint every 10 iterations.
        checkpoint_at_end=True,  # Saves a checkpoint at the end of training. Guarantees at least one checkpoint always.
        storage_path=Path(f'./logs').resolve(),  # Location to store logs and checkpoints
        config=config.to_dict(),
    )


if __name__ == "__main__":
    # Initialize the Ray runtime. `ignore_reinit_error` avoids warnings when restarting Ray.
    ray.init(ignore_reinit_error=True)
    # Calls the `train` function to begin training.
    train()
