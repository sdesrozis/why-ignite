import ignite.distributed as idist
import torch
from ignite.utils import convert_tensor
from torch.cuda.amp import autocast

from .function import Function, Usage


class EvaluationFunction(Function):
    def __init__(self, context):
        super(EvaluationFunction, self).__init__(
            context=context,
            usage=Usage.EVALUATION,
            required=["network", "optimizer", "loss_fn", "device"],
        )

    def __call__(self, engine, batch):

        self.ctx.network.eval()

        with torch.no_grad():
            x, y = batch
            x = convert_tensor(x, device=self.ctx.device, non_blocking=True)
            y = convert_tensor(y, device=self.ctx.device, non_blocking=True)

            # autocast if idist is nccl
            with autocast(enabled=idist.backend() == "nccl"):
                y_pred = self.ctx.network(x)

            return y_pred, y
