"""Log custom data to files."""

import json
from typing import Any, Dict
import numpy as np
from pathlib import Path
from datetime import datetime


class Logger:
    """Custom class for writing data to a log."""

    def __init__(self, parent: str | Path = './logs'):
        """Initialize the Logger class.

        Args:
            parent (str | Path): Parent directory for logs. Defaults to None.
        """
        self.parent = Path(parent).resolve()

    @property
    def log_path(self) -> Path:
        """Log directory.

        Returns:
            Path: Path to the directory files are logged to.
        """
        return f'{self.parent}/custom_logs'

    def serialize_numpy(self, obj: Any) -> Any:
        """Serialize Numpy values.

        Args:
            obj (Any): Object to serialize.

        Returns:
            Any: The converted, serializable object.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

    def write_to_log(self, file_name: str, data: Dict) -> None:
        """Write data to log file.

        Args:
            file_name (str): Log file name to use.
            data (Dict): The dict of data to write to the log file.
        """
        p = Path(f'{self.log_path}/{file_name}')
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(str(p), 'w') as log_file:
            json.dump(data, log_file, indent=2, default=self.serialize_numpy)
