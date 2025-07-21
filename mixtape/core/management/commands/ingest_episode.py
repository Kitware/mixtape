import json
from typing import TextIO

from django.core.files.base import ContentFile
from django.db import transaction
import djclick as click

from mixtape.core.management.ingest.external_episode import ExternalImport
from mixtape.core.models import (
    AgentStep,
    Episode,
    Step,
    Training,
)
from mixtape.core.models.action_mapping import ActionMapping
from mixtape.core.models.checkpoint import Checkpoint
from mixtape.core.models.inference import Inference
from mixtape.core.ray_utils.utility_functions import get_environment_mapping


@click.command()
@click.argument('json_file', type=click.File('r'))
@click.option(
    '--allow_existing',
    is_flag=True,
    help=(
        'If true, use existing action mapping if it exists. '
        'If false and existing action mapping is found, the ingestion will fail.'
    ),
)
def ingest_episode(json_file: TextIO, allow_existing: bool) -> None:
    """Ingest an external episode from a JSON file."""
    data = json.load(json_file)

    # Validate the data
    external_import = ExternalImport(**data)
    external_training = external_import.training
    external_inference = external_import.inference

    if external_import.action_mapping:
        environment = external_training.environment
        existing_mapping = get_environment_mapping(environment)

        if existing_mapping and existing_mapping != external_import.action_mapping:
            if not allow_existing:
                click.echo(
                    click.style(
                        'Error: Existing action mapping found and --allow_existing is false. '
                        'Please rename your environment or use --allow_existing to use the '
                        'existing mapping.',
                        fg='red',
                        bold=True,
                    )
                )
                return
        elif not existing_mapping:
            # Create new custom mapping
            ActionMapping.objects.create(
                environment=environment,
                mapping=external_import.action_mapping,
            )

    with transaction.atomic():
        # Create the training request
        training = Training(
            environment=external_training.environment,
            algorithm=external_training.algorithm,
            parallel=external_training.parallel,
            num_gpus=external_training.num_gpus,
            iterations=external_training.iterations,
            config=external_training.config,
            reward_mapping=external_training.reward_mapping,
            is_external=True,
        )
        training.full_clean()
        training.save()

        # Create the checkpoint
        checkpoint = Checkpoint.objects.create(
            training=training,
            best=False,
            last=False,
        )

        # Create the inference request
        inference = Inference.objects.create(
            checkpoint=checkpoint,
            parallel=external_inference.parallel,
            config=external_inference.config,
        )

        # Create the episode
        episode = Episode.objects.create(
            inference=inference,
        )

        # Create the steps and agent steps
        for step_data in external_inference.steps:
            # Create the step
            step = Step.objects.create(
                episode=episode,
                number=step_data.number,
            )

            # Save the image if provided - key is optional
            if step_data.image:
                # Decode base64 image data back to binary
                image_binary = step_data.image
                step.image.save(
                    f'step_{step_data.number}.png',
                    ContentFile(image_binary),
                    save=True,
                )

            # Create the agent steps if they exist - key is optional
            if step_data.agent_steps:
                for agent_step_data in step_data.agent_steps:
                    # Support either single reward or multiple rewards
                    rewards = agent_step_data.rewards
                    if rewards is None:
                        if agent_step_data.reward is None:
                            raise ValueError('Either reward or rewards must be defined')
                        rewards = [agent_step_data.reward]

                    AgentStep.objects.create(
                        step=step,
                        agent=agent_step_data.agent,
                        action=agent_step_data.action,
                        rewards=rewards,
                        observation_space=agent_step_data.observation_space,
                    )
