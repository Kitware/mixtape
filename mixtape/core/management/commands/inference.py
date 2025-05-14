from typing import TextIO

import djclick as click
import yaml

from mixtape.core.management.commands._utils import check_parallel
from mixtape.core.models.checkpoint import Checkpoint
from mixtape.core.models.inference import Inference
from mixtape.core.ray_utils.constants import ExampleEnvs
from mixtape.core.tasks.inference_tasks import run_inference_task


@click.command()
@click.argument('checkpoint_pk', type=int)
@click.option(
    '-e',
    '--env_name',
    type=click.Choice([choice.value for choice in ExampleEnvs]),  # type: ignore
    default=ExampleEnvs.PZ_KnightsArchersZombies,
    callback=check_parallel,
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
    callback=check_parallel,
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
    config_dict = yaml.safe_load(config_file) if config_file else {}

    checkpoint = Checkpoint.objects.get(pk=checkpoint_pk)
    inference = Inference.objects.create(
        checkpoint=checkpoint, parallel=parallel, config=config_dict
    )

    task = run_inference_task.s(inference_pk=inference.pk)
    if immediate:
        task.apply()
    else:
        task.delay()
