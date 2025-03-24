from django.db import models

from mixtape.core.ray_utils.logger import NumpyJSONEncoder
from mixtape.environments.mappings import action_maps

from .step import Step


class AgentStep(models.Model):
    class Meta:
        constraints = [models.UniqueConstraint(fields=['agent', 'step'], name='unique_agent_step')]

    step = models.ForeignKey(Step, on_delete=models.CASCADE, related_name='agent_steps')
    agent = models.CharField(max_length=200)

    action = models.FloatField()
    reward = models.FloatField()
    observation_space = models.JSONField(encoder=NumpyJSONEncoder)

    @property
    def action_string(self) -> str:
        # Note: "prefetch_related" should be called on any AgentStep where this is used, otherwise
        # this property can create very inefficient queries
        environment = self.step.episode.inference_request.checkpoint.training_request.environment
        return action_maps[environment].get(int(self.action), f'{self.action}')
