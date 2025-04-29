from collections.abc import Generator
from contextlib import contextmanager
from io import BytesIO

from PIL import Image
from django.core.files import File
from django.db import models
import numpy as np

from .episode import Episode


class Step(models.Model):
    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['episode', 'number'], name='unique_episode_number'),
        ]

    episode = models.ForeignKey(Episode, on_delete=models.CASCADE, related_name='steps')

    number = models.PositiveIntegerField()
    image = models.ImageField(null=True, blank=True)

    @contextmanager
    @staticmethod
    def rgb_array_to_file(rgb_array: np.ndarray, file_base_name: str) -> Generator[File]:
        """Yield a Django File containing an encoded Image."""
        with BytesIO() as rendering_stream:
            Image.fromarray(rgb_array).save(rendering_stream, format='PNG')
            yield File(rendering_stream, f'{file_base_name}.png')
