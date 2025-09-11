from django.core.exceptions import ValidationError
from django.db import models

from mixtape.core.ray_utils.utility_functions import get_environment_mapping

from .step import Step


def _rewards_default():
    return [0]


class StepBase(models.Model):
    class Meta:
        abstract = True

    action = models.FloatField()
    rewards = models.JSONField(default=_rewards_default)

    action_distribution = models.JSONField(null=True, blank=True)
    health = models.FloatField(null=True, blank=True)
    value_estimate = models.FloatField(null=True, blank=True)
    predicted_reward = models.FloatField(null=True, blank=True)
    custom_metrics = models.JSONField(null=True, blank=True)

    def _clean(self, training):
        # Validate that it matches the training's reward mapping length
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

    def _action_string(self, environment) -> str:
        # Note: "select_related" should be called on any AgentStep where this is used, otherwise
        # this property can create very inefficient queries
        mapping = get_environment_mapping(environment)
        return mapping.get(f'{int(self.action)}', f'{self.action}')

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    def _reward_strings(self, training) -> list[str]:
        reward_mapping = training.reward_mapping

        if not reward_mapping:
            return [f'{value}' for value in self.rewards]

        return reward_mapping


class AgentStep(StepBase):
    class Meta:
        constraints = [models.UniqueConstraint(fields=['agent', 'step'], name='unique_agent_step')]

    step = models.ForeignKey(Step, on_delete=models.CASCADE, related_name='agent_steps')
    agent = models.CharField(max_length=200)
    observation_space = models.JSONField()
    enemy_agent_health = models.JSONField(null=True, blank=True)
    enemy_unit_health = models.JSONField(null=True, blank=True)

    def clean(self):
        training = self.step.episode.inference.checkpoint.training
        super()._clean(training)

    @property
    def action_string(self) -> str:
        environment = self.step.episode.inference.checkpoint.training.environment
        return super()._action_string(environment)

    @property
    def reward_strings(self) -> list[str]:
        training = self.step.episode.inference.checkpoint.training
        return super()._reward_strings(training)


class UnitStep(StepBase):
    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['agent_step', 'unit'], name='unique_unit_step')
        ]

    agent_step = models.ForeignKey(AgentStep, on_delete=models.CASCADE, related_name='unit_steps')
    unit = models.CharField(max_length=200)

    def clean(self):
        training = self.agent_step.step.episode.inference.checkpoint.training
        super()._clean(training)

    @property
    def action_string(self) -> str:
        environment = self.agent_step.step.episode.inference.checkpoint.training.environment
        return super()._action_string(environment)

    @property
    def reward_strings(self) -> list[str]:
        training = self.agent_step.step.episode.inference.checkpoint.training
        return super()._reward_strings(training)
