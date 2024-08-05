import json
import numpy as np
from pathlib import Path
from datetime import datetime


class Logger:
    def __init__(self):
        self.date_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    @property
    def log_path(self):
        return Path(f'./logs/results/{self.date_time}').resolve()

    def serialize_numpy(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        try:
            return json.JSONEncoder.default(self, obj)
        except Exception as e:
            return f'{e} - {type(obj)}'

    def write_to_log(self, file_name, data):
        p = Path(f'{self.log_path}/{file_name}')
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(str(p), 'w') as log_file:
            json.dump(data, log_file, default=self.serialize_numpy)
