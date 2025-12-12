from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import PIL.Image
import PIL.ImageDraw
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.db.models import signals
import factory.django
import factory.fuzzy

from mixtape.core.models import AgentStep, Checkpoint, Episode, Inference, Step, Training
from mixtape.core.ray_utils.constants import SupportedAlgorithm


class UserFactory(factory.django.DjangoModelFactory[User]):
    class Meta:
        model = User

    username = factory.SelfAttribute('email')
    email = factory.Faker('safe_email')
    first_name = factory.Faker('first_name')
    last_name = factory.Faker('last_name')


class TrainingFactory(factory.django.DjangoModelFactory[Training]):
    class Meta:
        model = Training

    environment = 'knights_archers_zombies_v10'
    algorithm = factory.fuzzy.FuzzyChoice(choices=SupportedAlgorithm.values)


class CheckpointFactory(factory.django.DjangoModelFactory[Checkpoint]):
    class Meta:
        model = Checkpoint

    training = factory.SubFactory(TrainingFactory)
    last = True
    archive = factory.django.FileField(
        from_path=Path(__file__).parent / 'data' / 'checkpoint_archive.tar.bz2',
        # TODO: Change to f'checkpoint/{uuid4()}.tar.bz2'
        filename=f'checkpoint/archive.tar.bz2'
    )


class InferenceFactory(factory.django.DjangoModelFactory[Inference]):
    class Meta:
        model = Inference

    checkpoint = factory.SubFactory(CheckpointFactory)


# Disable post_save signal to avoid clustering
@factory.django.mute_signals(signals.post_save)
class EpisodeFactory(factory.django.DjangoModelFactory[Episode]):
    class Meta:
        model = Episode
        skip_postgeneration_save = True

    inference = factory.SubFactory(InferenceFactory)

    steps = factory.RelatedFactoryList(
        'mixtape.core.tests.factories.StepFactory', factory_related_name='episode', size=150
    )


class StepFactory(factory.django.DjangoModelFactory[Step]):
    class Meta:
        model = Step
        skip_postgeneration_save = True

    # Don't set `episode`, we will almost always want multiple invocations of
    # `StepFactory` to share a common `Episode`

    # TODO: this sequence is global, so creating Steps for a different Episode won't reset the value
    number = factory.Sequence(lambda n: n)

    @factory.lazy_attribute
    def image(self):
        image = PIL.Image.new('RGB', (100, 100), color=(127, 127, 127))
        draw = PIL.ImageDraw.Draw(image)
        draw.text((10, 40), f'Step {self.number}', fill=(255, 255, 255))
        with BytesIO() as image_stream:
            image.save(image_stream, format='PNG')
            image_stream.seek(0)
            return ContentFile(image_stream.getvalue(), name=f'step_{self.number}.png')

    agent_steps = factory.RelatedFactoryList(
        'mixtape.core.tests.factories.AgentStepFactory', factory_related_name='step', size=4
    )


class AgentStepFactory(factory.django.DjangoModelFactory[AgentStep]):
    class Meta:
        model = AgentStep

    # Don't set `step`, we will almost always want multiple invocations of
    # `AgentStepFactory` to share a common `Step`

    agent = factory.Iterator(['archer_0', 'archer_1', 'knight_0', 'knight_1'])
    action = factory.fuzzy.FuzzyChoice(choices={0, 1, 2, 3, 4, 5})
    # Return `[0]` 99% of the time, `[1]` 1% of the time
    rewards = factory.Faker(
        'random_elements', elements=OrderedDict([(0, 0.99), (1, 0.01)]), length=1
    )
    observation_space = factory.LazyFunction(
        lambda: [[factory.fuzzy.FuzzyFloat(-1.0, 1.0).fuzz() for _i in range(5)] for _j in range(5)]
    )
    # This should match the size of the options in `action`
    action_distribution = factory.LazyFunction(
        lambda: [factory.fuzzy.FuzzyFloat(-2.0, 2.0).fuzz() for _i in range(6)]
    )
