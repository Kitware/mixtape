import json
from pathlib import Path
from typing import Any

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        """Serialize Numpy values.

        Args:
            obj: Object to serialize.

        Returns:
            Any: The converted, serializable object.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        return super().default(obj)


class Logger:
    """Custom class for writing data to a log."""

    def __init__(self, parent: str | Path = './logs'):
        """Initialize the Logger class.

        Args:
            parent: Parent directory for logs. Defaults to None.
        """
        self.parent = Path(parent).resolve()

    @property
    def log_path(self) -> Path:
        """Log directory.

        Returns:
            Path: Path to the directory files are logged to.
        """
        return Path(f'{self.parent}/custom_logs')

    def write_to_log(self, file_name: str, data: dict) -> None:
        """Write data to log file.

        Args:
            file_name: Log file name to use.
            data: The dict of data to write to the log file.
        """
        p = Path(f'{self.log_path}/{file_name}')
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(str(p), 'w') as log_file:
            json.dump(data, log_file, indent=2, cls=NumpyJSONEncoder)
