from enum import Enum


class ExampleEnvs(str, Enum):
    # Example PettingZoo Environments
    PZ_KnightsArchersZombies = 'knights_archers_zombies_v10'
    PZ_Pistonball = 'pistonball_v6'
    PZ_Pong = 'cooperative_pong_v5'

    # Example Gymnasium Environments
    GYM_BattleZone = 'BattleZone-v5'
    GYM_Berzerk = 'Berzerk-v5'
    GYM_ChopperCommand = 'ChopperCommand-v5'

    @classmethod
    def type(cls, str):
        if cls(str).name.startswith('PZ'):
            return 'PettingZoo'
        elif cls(str).name.startswith('GYM'):
            return 'Gymnasium'


class SupportedAlgorithm(str, Enum):
    PPO = 'PPO'
    DQN = 'DQN'
