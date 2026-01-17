from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ray.rllib.env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

if TYPE_CHECKING:
    import numpy.typing as npt


class PZWrapper(PettingZooEnv):
    """Temporary wrapper for PettingZooEnv.

    Correct an issue with the latest release of PettingZoo environments: The
    RLlib PettingZoo wrapper still passes in the `render_mode` argument to the
    render function, but PettingZoo environments no longer accept an argument.
    """

    # Superclass return type is incorrect:
    # https://github.com/Farama-Foundation/Gymnasium/issues/845#issuecomment-2535827312
    def render(self) -> npt.NDArray[np.uint8]:  # type: ignore[override]
        """RGB image given the current observation.

        Returns:
            A numpy uint8 3D array (image) to render.
        """
        return self.env.render()


class ParallelPZWrapper(ParallelPettingZooEnv):
    """Temporary wrapper for ParallelPettingZooEnv.

    Correct an issue with the latest release of PettingZoo environments: The
    RLlib PettingZoo wrapper still passes in the `render_mode` argument to the
    render function, but PettingZoo environments no longer accept an argument.
    """

    # Superclass return type is incorrect:
    # https://github.com/Farama-Foundation/Gymnasium/issues/845#issuecomment-2535827312
    def render(self) -> npt.NDArray[np.uint8]:  # type: ignore[override]
        """RGB image given the current observation.

        Returns:
            A numpy uint8 3D array (image) to render.
        """
        return self.par_env.render()
