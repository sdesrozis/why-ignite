import numbers

import ignite.distributed as idist
import torch
from ignite.handlers import Checkpoint


class Context:
    def __init__(self, **kwargs):
        self.device = idist.device()
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        s = "Context:\n"
        for attr, value in self.__dict__.items():
            if not isinstance(value, (numbers.Number, str)):
                value = type(value)
            s += f"\t{attr}: {value}\n"
        return s

    def to_dict(self):
        return self.kwargs

    def load(self, resume):
        if resume is None:
            return
        Checkpoint.load_objects(to_load=self.kwargs, checkpoint=torch.load(resume))
