from abc import ABCMeta, abstractmethod
from enum import Enum

from .context import Context


class Usage(Enum):
    TRAINING = "Training"
    EVALUATION = "Evaluation"
    INFERENCE = "Inference"


class Function(metaclass=ABCMeta):
    def __init__(self, context, usage, required):
        self.__context = context
        self.__usage = usage
        self.__required = required
        # proxy of context restricted to required
        kwargs = {}
        for r in required:
            if not hasattr(self.__context, r):
                raise ValueError(
                    f"Missing required attribute '{r}' in context (actual: {self.__context})"
                )
            kwargs[r] = getattr(self.__context, r)
        self.ctx = Context(**kwargs)

    @abstractmethod
    def __call__(self, engine, batch):
        pass

    def usage(self):
        return self.__usage

    def context(self):
        return self.__context
