from django.db import models


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
    def type(cls, value):
        if cls(value).name.startswith('PZ'):
            return 'PettingZoo'
        elif cls(value).name.startswith('GYM'):
            return 'Gymnasium'


class SupportedAlgorithm(models.TextChoices):
    PPO = 'PPO'
    DQN = 'DQN'
