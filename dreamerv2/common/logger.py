import json
import pathlib

import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, step, logdir, multiplier=1):
        self._step = step
        self._logdir = pathlib.Path(logdir).expanduser()
        self._multiplier = multiplier
        self._last_step = None
        self._last_time = None
        self._metrics = []
        self._writer = None

    def add(self, mapping, prefix=None):
        step = int(self._step) * self._multiplier
        for name, value in dict(mapping).items():
            name = f'{prefix}_{name}' if prefix else name
            value = np.array(value)
            self._metrics.append((step, name, value))

    def scalar(self, name, value):
        self.add({name: value})

    def write(self):
        if not self._metrics:
            return
        
        step = max(s for s, _, _, in self._metrics)
        scalars = {k: float(v) for _, k, v in self._metrics if len(v.shape) == 0}
        
        # Print metrics on the console
        formatted = {k: self._format_value(v) for k, v in scalars.items()}
        print(f'[{step}]', ' | '.join(f'{k} {v}' for k, v in formatted.items()))
        
        # Save the metrics in a jsonl
        with (self._logdir/'metrics.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, **scalars}) + '\n')
        
        # Log into Tensorboard
        if not self._writer:
            self._writer = SummaryWriter(log_dir=self._logdir)
        for step, name, value in self._metrics:
            if len(value.shape) == 0:
                self._writer.add_scalar('scalars/' + name, value, step)
            elif len(value.shape) == 1:
                for i in range(value.shape[0]):
                    self._writer.add_scalar(f'scalars/{name}_{i}', value[i], step)
                    
        self._writer.flush()
        self._metrics.clear()
 
    def _format_value(self, value):
        if value == 0:
            return '0'
        elif 0.01 < abs(value) < 10000:
            value = f'{value:.2f}'
            value = value.rstrip('0')
            value = value.rstrip('0')
            value = value.rstrip('.')
            return value
        else:
            value = f'{value:.1e}'
            value = value.replace('.0e', 'e')
            value = value.replace('+0', '')
            value = value.replace('+', '')
            value = value.replace('-0', '-')
        return value
    