from typing import TextIO

import djclick as click
import yaml

from mixtape.core.management.commands._utils import check_parallel
from mixtape.core.models import Training
from mixtape.core.ray_utils.constants import ExampleEnvs, SupportedAlgorithm
from mixtape.core.tasks.training_tasks import run_training_task


def check_algo(
    ctx: click.Context, param: click.Option, value: SupportedAlgorithm | ExampleEnvs
) -> SupportedAlgorithm | ExampleEnvs:
    env_name = value if param.name == 'env_name' else ctx.params.get('env_name')
    algorithm = value if param.name == 'algorithm' else ctx.params.get('algorithm')
    if algorithm == SupportedAlgorithm.DQN and env_name == ExampleEnvs.PZ_Pistonball:
        raise click.ClickException(
            click.style(
                'DQN is only available for discrete action spaces, but Pisontball has a '
                + 'continuous action space. Please select another algorithm.',
                fg='red',
                bold=True,
            )
        )
    return value


def check_parallel_and_algo(
    ctx: click.Context, param: click.Option, value: ExampleEnvs
) -> ExampleEnvs:
    check_parallel(ctx, param, value)
    check_algo(ctx, param, value)
    return value


@click.command()
@click.option(
    '-e',
    '--env_name',
    type=click.Choice([choice.value for choice in ExampleEnvs]),
    default=ExampleEnvs.PZ_KnightsArchersZombies,
    callback=check_parallel_and_algo,
    help='The PettingZoo or Gymnasium environment to use.',
)
@click.option(
    '-a',
    '--algorithm',
    type=click.Choice([choice.value for choice in SupportedAlgorithm]),
    default=SupportedAlgorithm.PPO,
    callback=check_algo,
    help='The RLlib algorithm to use.',
)
@click.option(
    '-p',
    '--parallel',
    is_flag=True,
    callback=check_parallel,
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
    config_dict = yaml.safe_load(config_file) if config_file else {}

    training = Training(
        environment=env_name,
        algorithm=algorithm,
        parallel=parallel,
        num_gpus=num_gpus,
        iterations=training_iteration,
        config=config_dict,
        is_external=False,
    )
    training.full_clean()
    training.save()

    task = run_training_task.s(training_pk=training.pk)
    if immediate:
        task.apply()
    else:
        task.delay()
