import torch
from ignite.handlers import Checkpoint

from .driver import Driver
from .function import Usage


class Trainer(Driver):
    def __init__(self, name, function, dataloader):

        if function.usage() is not Usage.TRAINING:
            raise ValueError(
                f"Mismatch function usage (got: {function.usage()}, expected: Training)"
            )

        super(Trainer, self).__init__(
            name=name, function=function, dataloader=dataloader
        )

    def load(self, resume=None):
        if resume is None:
            return
        Checkpoint.load_objects(
            to_load={
                "trainer": self.engine,
            },
            checkpoint=torch.load(resume),
        )
