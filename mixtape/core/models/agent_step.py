from django.core.exceptions import ValidationError
from django.db import models

from mixtape.core.ray_utils.utility_functions import get_environment_mapping

from .step import Step


def _rewards_default():
    return [0]


class AgentStep(models.Model):
    class Meta:
        constraints = [models.UniqueConstraint(fields=['agent', 'step'], name='unique_agent_step')]

    step = models.ForeignKey(Step, on_delete=models.CASCADE, related_name='agent_steps')
    agent = models.CharField(max_length=200)

    action = models.FloatField()
    rewards = models.JSONField(default=_rewards_default)
    observation_space = models.JSONField()

    action_distribution = models.JSONField(null=True, blank=True)

    def clean(self):
        # Validate that it matches the training's reward mapping length
        training = self.step.episode.inference.checkpoint.training
        if not training.reward_mapping and len(self.rewards) != 1:
            raise ValidationError(
                'Reward mapping must be set when there are multiple reward values.'
            )

        if training.reward_mapping:
            expected_length = len(training.reward_mapping)
            actual_length = len(self.rewards)
            if actual_length != expected_length:
                raise ValidationError(
                    f'Rewards array length ({actual_length}) must match reward '
                    f'mapping length ({expected_length}).'
                )

    @property
    def action_string(self) -> str:
        # Note: "select_related" should be called on any AgentStep where this is used, otherwise
        # this property can create very inefficient queries
        environment = self.step.episode.inference.checkpoint.training.environment
        mapping = get_environment_mapping(environment)
        return mapping.get(f'{int(self.action)}', f'{self.action}')

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    @property
    def reward_strings(self) -> list[str]:
        training = self.step.episode.inference.checkpoint.training
        reward_mapping = training.reward_mapping

        if not reward_mapping:
            return [f'{value}' for value in self.rewards]

        return reward_mapping
