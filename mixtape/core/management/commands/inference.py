from typing import TextIO

import djclick as click
import yaml

from mixtape.core.models.checkpoint import Checkpoint
from mixtape.core.models.inference_request import InferenceRequest
from mixtape.core.ray_utils.constants import ExampleEnvs
from mixtape.core.ray_utils.environments import is_gymnasium_env
from mixtape.core.tasks.inference_tasks import run_inference_task


@click.command()
@click.argument('checkpoint_pk', type=int)
@click.option(
    '-e',
    '--env_name',
    type=click.Choice([choice.value for choice in ExampleEnvs]),
    default=ExampleEnvs.PZ_KnightsArchersZombies,
    help='The PettingZoo or Gymnasium environment to use.',
)
@click.option(
    '-f',
    '--config_file',
    type=click.File('r'),
    default=None,
    help='Arguments to configure the environment.',
)
@click.option(
    '-p',
    '--parallel',
    is_flag=True,
    help='All agents have simultaneous actions and observations.',
)
@click.option('--immediate', is_flag=True, help='Run the task immediately.')
def inference(
    checkpoint_pk: int,
    env_name: ExampleEnvs,
    config_file: TextIO | None,
    parallel: bool,
    immediate: bool,
) -> None:
    """Run inference on the specified trained environment."""
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

    checkpoint = Checkpoint.objects.get(pk=checkpoint_pk)
    inference_request = InferenceRequest.objects.create(
        checkpoint=checkpoint, parallel=parallel, config=config_dict
    )

    task = run_inference_task.s(inference_request_pk=inference_request.pk)
    if immediate:
        task.apply()
    else:
        task.delay()
