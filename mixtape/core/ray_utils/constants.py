from django.db import models


class ExampleEnvs(models.TextChoices):
    # Example PettingZoo Environments
    PZ_KnightsArchersZombies = 'knights_archers_zombies_v10'
    PZ_Pistonball = 'pistonball_v6'
    # Example Gymnasium Environments
    GYM_LunarLander = 'LunarLander-v2'

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
