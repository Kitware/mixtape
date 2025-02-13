from django.db import models

from .episode import Episode


class RenderedStep(models.Model):
    episode = models.ForeignKey(Episode, on_delete=models.CASCADE)

    step = models.PositiveIntegerField()
    image = models.ImageField()
