from typing import Literal

from django.core.exceptions import ValidationError
from django.db import models

from mixtape.core.json_encoder import CustomJSONEncoder


class Training(models.Model):
    created = models.DateTimeField(auto_now_add=True)

    environment = models.CharField(max_length=200)
    algorithm = models.CharField(max_length=200)
    parallel = models.BooleanField(default=False)
    num_gpus = models.FloatField(default=0.0)
    iterations = models.PositiveIntegerField(default=100)

    config = models.JSONField(default=dict, blank=True, null=True, encoder=CustomJSONEncoder)

    # Reward mapping for environments with multiple reward components
    reward_mapping = models.JSONField(null=True, blank=True)

    is_external = models.BooleanField(default=False)

    def clean(self):
        if not self.is_external:
            if self.environment not in ExampleEnvs.values:
                raise ValidationError({'environment': 'Invalid environment'})
            if self.algorithm not in SupportedAlgorithm.values:
                raise ValidationError({'algorithm': 'Invalid algorithm'})


class ExampleEnvs(models.TextChoices):
    # Example PettingZoo Environments
    PZ_KnightsArchersZombies = 'knights_archers_zombies_v10'
    PZ_Pistonball = 'pistonball_v6'
    PZ_Pong = 'cooperative_pong_v5'
    # Example Gymnasium Environments
    GYM_BattleZone = 'BattleZone-v5'
    GYM_Berzerk = 'Berzerk-v5'
    GYM_ChopperCommand = 'ChopperCommand-v5'

    @classmethod
    def type(cls, value) -> Literal['PettingZoo', 'Gymnasium', 'Unknown']:
        if cls(value).name.startswith('PZ'):
            return 'PettingZoo'
        elif cls(value).name.startswith('GYM'):
            return 'Gymnasium'
        return 'Unknown'

    @classmethod
    def is_gymnasium_env(cls, env_name: str) -> bool:
        return cls.type(env_name) == 'Gymnasium'


class SupportedAlgorithm(models.TextChoices):
    PPO = 'PPO'
    DQN = 'DQN'
