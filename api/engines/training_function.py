import ignite.distributed as idist
from ignite.utils import convert_tensor
from torch.cuda.amp import autocast

from .function import Function, Usage


class TrainingFunction(Function):
    def __init__(self, context):
        super(TrainingFunction, self).__init__(
            context=context,
            usage=Usage.TRAINING,
            required=["network", "optimizer", "loss_fn", "device", "scaler"],
        )

    def __call__(self, engine, batch):

        self.ctx.network.train()
        self.ctx.optimizer.zero_grad()

        x, y = batch
        x = convert_tensor(x, device=self.ctx.device, non_blocking=True)
        y = convert_tensor(y, device=self.ctx.device, non_blocking=True)

        # autocast if scaler is enabled
        with autocast(enabled=idist.backend() == "nccl"):
            y_pred = self.ctx.network(x)

            loss = self.ctx.loss_fn(y_pred, y)

        self.ctx.scaler.scale(loss).backward()
        self.ctx.scaler.step(self.ctx.optimizer)
        self.ctx.scaler.update()

        return loss.item()
