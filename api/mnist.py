import argparse
import os

import ignite.distributed as idist
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from ignite.contrib.handlers import PiecewiseLinear
from ignite.metrics import (
    Accuracy,
    TopKCategoricalAccuracy,
    Precision,
    Recall,
    Fbeta,
    ConfusionMatrix,
    Loss,
    RunningAverage
)
from ignite.engine import Events
from ignite.handlers import TerminateOnNan
from torch.cuda.amp import GradScaler
from torch.optim import Adam

from api.engines import (
    Context,
    EvaluationFunction,
    Evaluator,
    Trainer,
    TrainingFunction,
)
from api.handlers import (
    ConfusionMatrixHelper,
    Checkpoint,
    LearningRateLogging,
    Loggers,
    MetricLogging,
    ScreenLogging,
    Validation,
)
from api.utils import create_dirname


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def training(_, config):

    # First set the seed for random
    torch.manual_seed(config["seed"])

    dirname = create_dirname()

    # create loggers once
    loggers = Loggers(dirname=dirname, clearml=config["clearml"])

    # connect argparse to loggers (clearml)
    loggers.connect(config, extratags=[config["backend"]])

    # ----------------------------------------------
    # Define PyTorch objects
    # ----------------------------------------------

    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    datasets = {
        "Train": MNIST(download=True, root=config["root"], transform=data_transform, train=True),
        "Valid": MNIST(download=True, root=config["root"], transform=data_transform, train=False),
    }

    train_batch_size = idist.get_world_size() * config["train_batch_size"]

    dataloader = idist.auto_dataloader(
        datasets["Train"],
        batch_size=train_batch_size,
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=True,
    )

    valid_batch_size = idist.get_world_size() * config["valid_batch_size"]

    dataloaders = {
        mode: idist.auto_dataloader(
            dataset,
            batch_size=valid_batch_size,
            num_workers=config["num_workers"],
            shuffle=False,
            drop_last=True,
        )
        for mode, dataset in datasets.items()
    }

    model = Net()
    model = idist.auto_model(model)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)

    lr_scheduler = PiecewiseLinear(
        optimizer,
        param_name="lr",
        milestones_values=[
            (0, 0.0),
            (len(dataloader) * 3, config["lr"]),
            (len(dataloader) * 10, config["lr"] / 10),
            (len(dataloader) * 15, config["lr"] / 100),
        ],
    )

    # enables amp if nccl
    scaler = GradScaler(enabled=idist.backend() == "nccl")

    # ----------------------------------------------
    # Define Context objects
    # ----------------------------------------------

    ctx = Context(
        network=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scaler=scaler,
        lr_scheduler=lr_scheduler,
    )

    # load context when resume
    ctx.load(resume=config["resume"])

    # ----------------------------------------------
    # Define Evaluator objects
    # ----------------------------------------------

    fn = EvaluationFunction(ctx)

    evaluators = {
        mode: Evaluator(name=f"{mode}Evaluator", function=fn, dataloader=dataloader)
        for mode, dataloader in dataloaders.items()
    }

    precision = Precision()
    recall = Recall()
    f1 = Fbeta(beta=1, average=False, precision=precision, recall=recall)

    # confusion matrix is expensive and should be computed only at the end
    cm = ConfusionMatrix(num_classes=10)

    for mode, evaluator in evaluators.items():
        evaluator.attach(
            handler=Loss(loss_fn, output_transform=lambda x: x),
            name="Loss"
        ).attach(
            handler=Accuracy(),
            name="Accuracy"
        ).attach(
            handler=TopKCategoricalAccuracy(k=3),
            name="Top3Accuracy"
        ).attach(
            handler=precision.mean(),
            name="MeanPrecision"
        ).attach(
            handler=recall.mean(),
            name="MeanRecall"
        ).attach(
            handler=f1.mean(),
            name="MeanF1"
        ).attach(
            handler=precision,
            name="Precision"
        ).attach(
            handler=recall,
            name="Recall"
        ).attach(
            handler=f1,
            name="F1"
        ).attach(
            handler=ConfusionMatrixHelper(cm),
            name="ConfusionMatrix",
        ).attach(
            handler=ScreenLogging(progress=not config["clearml"]),
        ).attach(
            handler=Checkpoint(
                prefix=mode.lower(),
                metric_name="MeanF1",
                dirname=os.path.join(dirname, "checkpoint"),
                clearml=config["clearml"],  # hack to upload artifact to clearml
            )
        ).attach(
            handler=MetricLogging(
                loggers=loggers,
                metric_names=["Loss", "Accuracy", "Top3Accuracy", "Precision", "MeanPrecision",
                              "Recall", "MeanRecall", "F1", "MeanF1"],
                tag=mode,
                event_name=Events.COMPLETED,
            )
        )

    # ----------------------------------------------
    # Define Trainer object
    # ----------------------------------------------

    fn = TrainingFunction(ctx)

    trainer = Trainer(
        name="Trainer",
        function=fn,
        dataloader=dataloader,
    )

    # load internal state of trainer when resume
    trainer.load(resume=config["resume"])

    trainer.attach_on_event(
        handler=lr_scheduler,
        handler_event=Events.ITERATION_STARTED,
    ).attach_on_event(
        handler=TerminateOnNan(),
        handler_event=Events.ITERATION_COMPLETED,
    ).attach(
        handler=RunningAverage(output_transform=lambda x: x),
        name="Loss",
    ).attach(
        handler=ScreenLogging(
            config=config, progress_metrics=["Loss"], progress=not config["clearml"]
        ),
    ).attach(
        handler=MetricLogging(
            loggers=loggers,
            metric_names=["Loss"],
            tag=f"Training",
            event_name=Events.ITERATION_COMPLETED(every=100),
        )
    ).attach(
        handler=Validation(evaluators=evaluators.values())
    ).attach(
        handler=LearningRateLogging(
            loggers=loggers,
            optimizer=optimizer,
            event_name=Events.ITERATION_COMPLETED(every=100),
        )
    )

    # ----------------------------------------------
    # Training
    # ----------------------------------------------

    trainer.fit(max_epochs=config["max_epochs"])

    loggers.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("MNIST DDP Training")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--valid_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--clearml", type=int, default=0)
    args = parser.parse_args()

    if args.backend == "nccl":
        assert torch.cuda.is_available()
        assert torch.backends.cudnn.enabled
        torch.backends.cudnn.benchmark = True

    with idist.Parallel(backend=args.backend) as parallel:
        try:
            parallel.run(training, vars(args))
        except KeyboardInterrupt:
            if idist.get_rank() == 0:
                print("Catched KeyboardInterrupt -> exit")
        except Exception as e:
            if idist.get_rank() == 0:
                print("Catched exception", e)
            raise e
