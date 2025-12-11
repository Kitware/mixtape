from django.test import Client
from django.urls import reverse
import pytest
from pytest_django.asserts import assertContains

from .factories import EpisodeFactory


@pytest.mark.django_db
def test_view_home(client: Client) -> None:
    episode = EpisodeFactory.create()
    training = episode.inference.checkpoint.training

    response = client.get(reverse('home'))
    assertContains(response, training.environment)
