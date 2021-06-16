from .driver import Driver
from .function import Usage


class Evaluator(Driver):
    def __init__(self, name, function, dataloader):

        if function.usage() is not Usage.EVALUATION:
            raise ValueError(
                f"Mismatch function usage (got: {function.usage()}, expected: Training)"
            )

        super(Evaluator, self).__init__(
            name=name, function=function, dataloader=dataloader
        )
