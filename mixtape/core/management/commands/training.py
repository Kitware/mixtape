from pathlib import Path

import djclick as click
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import run
import yaml

from mixtape.ray_utils.callbacks import CustomLoggingCallbacks
from mixtape.ray_utils.constants import ExampleEnvs, SupportedAlgorithm
from mixtape.ray_utils.utils import is_gymnasium_env, register_environment


@click.command()
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
    '-a',
    '--algorithm',
    type=click.Choice(['PPO', 'DQN']),
    default=SupportedAlgorithm.PPO.value,
    help='The RLlib algorithm to use.',
)
@click.option(
    '-p',
    '--parallel',
    is_flag=True,
    help='All agents have simultaneous actions and observations.',
)
@click.option('-g', '--num_gpus', type=float, default=0, help='Number of GPUs to use.')
@click.option(
    '-t',
    '--training_iteration',
    type=int,
    default=100,
    help='Number of training iterations to run.',
)
@click.option(
    '-f',
    '--config_file',
    type=click.File('r'),
    default=None,
    help='Arguments to configure the environment.',
)
def training(env_name, algorithm, parallel, num_gpus, training_iteration, config_file):
    """Run training on the specified environment."""
    config_dict = {}
    if config_file is not None:
        config_dict = yaml.safe_load(config_file)
    env_config = config_dict.get('env_config', {})
    parallel = False if is_gymnasium_env(env_name) else parallel
    register_environment(env_name, env_config, parallel)

    # Ensure all arguments default to empty dict, not None
    training_args = config_dict.get('training_args', {})
    env_args = config_dict.get('env_args', {})
    framework_args = config_dict.get('framework_args', {})

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
        .environment(env=env_name, clip_actions=env_args.pop('clip_actions', True), **env_args)
        .env_runners(num_env_runners=4, rollout_fragment_length='auto')
        .framework(framework=framework_args.pop('framework', 'torch'), **framework_args)
        .resources(num_gpus=num_gpus)
        .training(**training_args)
    )

    # Set run arg defaults
    run_args = config_dict.get('run_args', {})
    run_args.setdefault('stop', {'training_iteration': training_iteration})
    run_args.setdefault('checkpoint_freq', 10)
    run_args.setdefault('checkpoint_at_end', True)
    run_args.setdefault('storage_path', Path('./logs').resolve())
    run_args.setdefault('config', config.to_dict())

    # Dispatch run
    result = run(
        algorithm,
        name=algorithm,
        **run_args,
    )

    checkpoint = result.get_last_checkpoint()
    if checkpoint is None:
        raise Exception('Last checkpoint not found!')

    # Normalize path to start at the `fastapi` directory
    checkpoint_path = checkpoint.path.removeprefix('/app/')
    return checkpoint_path
