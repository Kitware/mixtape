from django.db import models

from mixtape.core.ray_utils.logger import NumpyJSONEncoder

from .step import Step


class AgentStep(models.Model):
    class Meta:
        constraints = [models.UniqueConstraint(fields=['agent', 'step'], name='unique_agent_step')]

    step = models.ForeignKey(Step, on_delete=models.CASCADE, related_name='agent_steps')
    agent = models.CharField(max_length=200)

    action = models.FloatField()
    reward = models.FloatField()
    observation_space = models.JSONField(encoder=NumpyJSONEncoder)
