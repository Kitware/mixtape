import numpy as np
from ray.rllib.env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


class PZWrapper(PettingZooEnv):
    """Temporary wrapper for PettingZooEnv.

    Correct an issue with the latest release of PettingZoo environments: The
    RLlib PettingZoo wrapper still passes in the `render_mode` argument to the
    render function, but PettingZoo environments no longer accept an argument.
    """

    def render(self) -> np._typing.NDArray[np.uint8]:
        """RGB image given the current observation.

        Returns:
            np._typing.NDArray[np.uint8]: A numpy uint8 3D array (image) to
                                          render.
        """
        return self.env.render()


class ParallelPZWrapper(ParallelPettingZooEnv):
    """Temporary wrapper for ParallelPettingZooEnv.

    Correct an issue with the latest release of PettingZoo environments: The
    RLlib PettingZoo wrapper still passes in the `render_mode` argument to the
    render function, but PettingZoo environments no longer accept an argument.
    """

    def render(self) -> np._typing.NDArray[np.uint8]:
        """RGB image given the current observation.

        Returns:
            np._typing.NDArray[np.uint8]: A numpy uint8 3D array (image) to
                                          render.
        """
        return self.par_env.render()
