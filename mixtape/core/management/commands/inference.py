from typing import TextIO

import djclick as click
import yaml

from mixtape.core.models.checkpoint import Checkpoint
from mixtape.core.models.inference_request import InferenceRequest
from mixtape.core.ray_utils.constants import ExampleEnvs
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
def inference(
    checkpoint_pk: int, env_name: ExampleEnvs, config_file: TextIO | None, parallel: bool
) -> None:
    """Run inference on the specified trained environment."""
    config_dict = yaml.safe_load(config_file) if config_file else {}

    checkpoint = Checkpoint.objects.get(pk=checkpoint_pk)
    inference_request = InferenceRequest.objects.create(
        environment=env_name, checkpoint=checkpoint, parallel=parallel, config=config_dict
    )

    run_inference_task.delay(inference_request_pk=inference_request.pk)
