import socket

import ignite
import ignite.distributed as idist
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import GpuInfo
from ignite.engine import Events

from v0.engines import DriverEvents


class ScreenLogging:
    def __init__(
        self, config=None, progress_metrics=None, persist=False, progress=True
    ):
        self.config = config
        self.gpu = None
        self.progressbar = None
        self.metric_names = None
        self.gpu_metric_names = ["gpu:0 mem(%)", "gpu:0 util(%)"]
        self.progress = progress
        if idist.get_rank() == 0 and self.progress:
            self.gpu = GpuInfo()
            nb_gpu = len(self.gpu.compute())
            self.gpu_mem = [f"gpu:{i} mem(%)" for i in range(nb_gpu)]
            self.gpu_util = [f"gpu:{i} util(%)" for i in range(nb_gpu)]
            self.gpu_metric_names = self.gpu_mem + self.gpu_util
            progress_metrics = [] if progress_metrics is None else progress_metrics
            first_gpu_metrics = (
                [self.gpu_mem[0], self.gpu_util[0]] if nb_gpu > 0 else []
            )
            self.metric_names = progress_metrics + first_gpu_metrics
            self.progressbar = ProgressBar(persist=persist)

    def attach(self, engine):
        if idist.get_rank() == 0:
            if self.progress:
                self.gpu.attach(engine)
                self.progressbar.attach(engine, metric_names=self.metric_names)
            self.__attach_handlers_on_screen(engine)

    def __attach_handlers_on_screen(self, engine):

        if self.config and not engine.has_event_handler(
            self.dump_config, DriverEvents.BEFORE_STARTED
        ):
            engine.add_event_handler(DriverEvents.BEFORE_STARTED, self.dump_config)

        if not engine.has_event_handler(self.epoch_started, Events.EPOCH_STARTED):
            engine.add_event_handler(Events.EPOCH_STARTED, self.epoch_started)

        if not engine.has_event_handler(self.dump_metrics, Events.EPOCH_COMPLETED):
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.dump_metrics)

    @staticmethod
    def epoch_started(engine):
        if not hasattr(engine.state, "trainer"):
            epoch = engine.state.epoch
            engine.logger.info(f"Training Epoch[{epoch}] Started.")
        else:
            engine.logger.info(f"Validation Epoch Started.")

    def dump_metrics(self, engine):
        metrics = engine.state.metrics
        engine.logger.info("Metrics results")
        max_length = max(
            [len(v) for v in metrics.keys() if v not in self.gpu_metric_names]
        )
        for k, v in metrics.items():
            # we do not dump gpu metrics
            if k in self.gpu_metric_names:
                continue
            key = f"{k}".rjust(max_length)
            if isinstance(v, float):
                val = f"{v:.4f}"
                engine.logger.info(f"- {key} : {val: >7}")
            elif isinstance(v, torch.Tensor):
                v = [f"{r:.4f}" for r in v.numpy()]
                v = " ".join(v)
                engine.logger.info(f"- {key} : [{v}]")
            else:
                raise ValueError("Type is not handled by ScreenLogging")

    def dump_config(self, engine):
        engine.logger.info("Experiments :")
        engine.logger.info(f"-        Hostname : {socket.gethostname()}")
        engine.logger.info(f"- PyTorch version : {torch.__version__}")
        engine.logger.info(f"-  Ignite version : {ignite.__version__}")
        if torch.cuda.is_available() and idist.backend() == "nccl":
            engine.logger.info(
                f"- Cuda device name: {torch.cuda.get_device_name(idist.get_local_rank())}"
            )
        engine.logger.info("Parallel configuration :")
        engine.logger.info(f" -        backend : {idist.backend()}")
        engine.logger.info(f" -         device : {idist.device()}")
        engine.logger.info(f" -     world size : {idist.get_world_size()}")
        engine.logger.info(f" -     local rank : {idist.get_local_rank()}")
        engine.logger.info(f" -           rank : {idist.get_rank()}")
        engine.logger.info(f" -         nnodes : {idist.get_nnodes()}")
        engine.logger.info(f" -      node rank : {idist.get_node_rank()}")
        engine.logger.info(f" - nproc per rank : {idist.get_nproc_per_node()}")

        engine.logger.info("Argparse configuration :")
        max_length = max([len(v) for v in self.config.keys()])
        for k, v in self.config.items():
            key = f"{k}".rjust(max_length)
            engine.logger.info(f"- {key} : {v}")
