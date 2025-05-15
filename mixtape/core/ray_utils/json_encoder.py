import json
from typing import Any

from gymnasium.wrappers import LazyFrames
import numpy as np


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        """Serialize values that are not serializable by default.

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
        if isinstance(obj, type):
            # Handle class objects
            return f'{obj.__module__}.{obj.__name__}'
        if callable(obj):
            # Handle function objects
            return f'{obj.__module__}.{obj.__name__}'
        return super().default(obj)
