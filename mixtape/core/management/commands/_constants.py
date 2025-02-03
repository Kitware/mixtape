from enum import Enum


class ButterflyEnvs(str, Enum):
    """Available PettingZoo Environments."""

    KnightsArchersZombies = 'knights_archers_zombies_v10'
    Pistonball = 'pistonball_v6'
    Pong = 'cooperative_pong_v5'
