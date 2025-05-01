import base64
import json
from typing import TextIO

from django.core.files.base import ContentFile
from django.db import transaction
import djclick as click

from mixtape.core.models import (
    AgentStep,
    Episode,
    Step,
    TrainingRequest,
)
from mixtape.core.models.checkpoint import Checkpoint
from mixtape.core.ingest.models.external_episode import ExternalImport
from mixtape.core.models.inference_request import InferenceRequest


@click.command()
@click.argument('json_file', type=click.File('r'))
def ingest_episode(json_file: TextIO) -> None:
    """Ingest an external episode from a JSON file."""
    data = json.load(json_file)

    # Validate the data using our Pydantic model
    external_import = ExternalImport(**data)
    external_training = external_import.training
    external_inference = external_import.inference

    with transaction.atomic():
        # Create the training request
        training_request = TrainingRequest(
            environment=external_training.environment,
            algorithm=external_training.algorithm,
            parallel=external_training.parallel,
            num_gpus=external_training.num_gpus,
            iterations=external_training.iterations,
            config=external_training.config,
            is_external=True,
        )
        training_request.full_clean()
        training_request.save()

        # Create the checkpoint
        checkpoint = Checkpoint.objects.create(
            training_request=training_request,
            best=False,
            last=False,
        )

        # Create the inference request
        inference_request = InferenceRequest.objects.create(
            checkpoint=checkpoint,
            parallel=external_inference.parallel,
            config=external_inference.config,
        )

        # Create the episode
        episode = Episode.objects.create(
            inference_request=inference_request,
        )

        # Create the steps and agent steps
        for step_data in external_inference.steps:
            # Create the step
            step = Step.objects.create(
                episode=episode,
                number=step_data.number,
            )

            # Save the image if provided
            if step_data.image:
                # Decode base64 image data back to binary
                image_binary = base64.b64decode(step_data.image)
                step.image.save(
                    f'step_{step_data.number}.png',
                    ContentFile(image_binary),
                    save=True,
                )

            # Create the agent steps
            if step_data.agent_steps:
                for agent_step_data in step_data.agent_steps:
                    AgentStep.objects.create(
                        step=step,
                        agent=agent_step_data.agent,
                        action=agent_step_data.action,
                        reward=agent_step_data.reward,
                        observation_space=agent_step_data.observation_space,
                    )
