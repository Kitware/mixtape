from typing import TextIO

import djclick as click
import yaml

from mixtape.core.models import TrainingRequest
from mixtape.core.ray_utils.constants import ExampleEnvs, SupportedAlgorithm
from mixtape.core.ray_utils.environments import is_gymnasium_env
from mixtape.core.tasks.training_tasks import run_training_task


@click.command()
@click.option(
    '-e',
    '--env_name',
    type=click.Choice([choice.value for choice in ExampleEnvs]),
    default=ExampleEnvs.PZ_KnightsArchersZombies,
    help='The PettingZoo or Gymnasium environment to use.',
)
@click.option(
    '-a',
    '--algorithm',
    type=click.Choice([choice.value for choice in SupportedAlgorithm]),
    default=SupportedAlgorithm.PPO,
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
@click.option('--immediate', is_flag=True, help='Run the task immediately.')
def training(
    env_name: ExampleEnvs,
    algorithm: SupportedAlgorithm,
    parallel: bool,
    num_gpus: float,
    training_iteration: int,
    config_file: TextIO | None,
    immediate: bool,
) -> None:
    """Run training on the specified environment."""
    if is_gymnasium_env(env_name) and parallel:
        click.echo(
            click.style(
                'Warning: The parallel option is only available for PettingZoo environments. '
                + 'Ignoring --parallel.',
                fg='red',
                bold=True,
            )
        )
        parallel = False

    config_dict = yaml.safe_load(config_file) if config_file else {}

    training_request = TrainingRequest.objects.create(
        environment=env_name,
        algorithm=algorithm,
        parallel=parallel,
        num_gpus=num_gpus,
        iterations=training_iteration,
        config=config_dict,
    )

    task = run_training_task.s(training_request_pk=training_request.pk)
    if immediate:
        task.apply()
    else:
        task.delay()
