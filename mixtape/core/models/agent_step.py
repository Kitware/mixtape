from django.db import models

from mixtape.core.models.episode import Episode
from mixtape.core.ray_utils.logger import NumpyJSONEncoder

from .episode import Episode


class AgentStep(models.Model):
    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['episode', 'agent', 'step'], name='unique_agent_step')
        ]

    episode = models.ForeignKey(Episode, on_delete=models.CASCADE)
    agent = models.CharField(max_length=200)
    step = models.PositiveIntegerField()

    action = models.FloatField()
    reward = models.FloatField()
    observation_space = models.JSONField(encoder=NumpyJSONEncoder)

    rendering = models.ImageField()
