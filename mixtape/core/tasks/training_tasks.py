from tempfile import mkdtemp
from uuid import uuid4

from celery import shared_task
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import run

from mixtape.core.models.checkpoint import Checkpoint
from mixtape.core.models.training import Training
from mixtape.core.ray_utils.constants import SupportedAlgorithm
from mixtape.core.ray_utils.environments import is_gymnasium_env, register_environment


@shared_task
def run_training_task(training_pk: int):
    training = Training.objects.get(pk=training_pk)

    training_config = training.config or {}
    env_config = training_config.get('env_config', {})

    parallel = False if is_gymnasium_env(training.environment) else training.parallel
    register_environment(training.environment, env_config, parallel)

    # Ensure all arguments default to empty dict, not None
    training_args = {
        # Set general purpose defaults
        'train_batch_size': 512,
        'lr': 2e-5,
        'gamma': 0.99,
        **training_config.get('training_args', {}),
    }
    env_args = training_config.get('env_args', {})
    framework_args = training_config.get('framework_args', {})

    # Set algorithm specific defaults
    if training.algorithm == SupportedAlgorithm.PPO:
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
    alg = PPOConfig() if training.algorithm == SupportedAlgorithm.PPO else DQNConfig()
    config = (
        alg.debugging(log_level='ERROR')
        .environment(
            env=training.environment,
            clip_actions=env_args.pop('clip_actions', True),
            **env_args,
        )
        .env_runners(num_env_runners=4, rollout_fragment_length='auto')
        .framework(framework=framework_args.pop('framework', 'torch'), **framework_args)
        .resources(num_gpus=training.num_gpus)
        .training(**training_args)
    )

    # Set run arg defaults
    run_args = {
        'stop': {'training_iteration': training.iterations},
        'checkpoint_freq': 10,
        'checkpoint_at_end': True,
        'config': config.to_dict(),
        **training_config.get('run_args', {}),
    }

    if 'storage_path' not in run_args:
        # TemporaryDirectory is a nicer API, but Python 3.11 lacks `delete`
        run_args['storage_path'] = mkdtemp(prefix='ray_training_')

    # Dispatch run
    result = run(
        training.algorithm,
        name=training.algorithm,
        **run_args,
    )

    checkpoint = result.get_last_checkpoint()
    if checkpoint is None:
        raise Exception('Last checkpoint not found!')
    with checkpoint.as_directory() as checkpoint_dir:
        with Checkpoint.directory_to_file(checkpoint_dir, f'checkpoint/{uuid4()}') as archive:
            Checkpoint.objects.create(
                training=training,
                last=True,
                archive=archive,
            )
