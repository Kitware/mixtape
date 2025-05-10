import json
from typing import Any

from gymnasium.wrappers import LazyFrames
import numpy as np

from mixtape.core.models.action_mapping import ActionMapping


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        """Serialize Numpy values.

        Args:
            obj: Object to serialize.

        Returns:
            Any: The converted, serializable object.
        """
        if isinstance(obj, LazyFrames):
            obj = np.array(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        return super().default(obj)


def get_environment_mapping(environment: str) -> dict:
    try:
        custom_mapping = ActionMapping.objects.get(environment=environment)
    except ActionMapping.DoesNotExist:
        return {}
    return custom_mapping.mapping
