import os
import socket

import ignite.distributed as idist
from ignite.contrib.handlers.clearml_logger import ClearMLLogger
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger

from api.utils import gst


class Loggers:
    def __init__(self, dirname=None, clearml=False, tensorboard=True, project_name="ignite", task_name="default"):

        self.loggers = []

        self.clearml_logger = None
        if idist.get_rank() == 0 and clearml:
            self.clearml_logger = ClearMLLogger(
                project_name=project_name, task_name=task_name
            )
            self.loggers.append(self.clearml_logger)

        self.tensorboard_logger = None
        if idist.get_rank() == 0 and tensorboard:
            tensorboard_dir = os.path.join(dirname, "tensorboard")
            self.tensorboard_logger = TensorboardLogger(log_dir=tensorboard_dir)
            self.loggers.append(self.tensorboard_logger)

    def connect(self, config, extratags=None):
        if self.clearml_logger:
            self.clearml_logger._task.connect_configuration(config)
            self.clearml_logger._task.connect({k: config[k] for k in config})
            extratags = [] if extratags is None else extratags
            tags = extratags + [socket.gethostname()]
            if idist.get_world_size() > 1:
                tags += ["ddp"]
            if "SLURM_JOBID" in os.environ:
                tags += ["slurm"]
            self.clearml_logger._task.add_tags(tags)

    def close(self):
        for logger in self.loggers:
            logger.close()

    def to_list(self):
        return self.loggers


class MetricLogging:
    def __init__(self, loggers, tag, metric_names, event_name):
        self.loggers = loggers
        self.tag = tag
        self.metric_names = metric_names
        self.event_name = event_name

    def attach(self, engine):

        if idist.get_rank() == 0:
            for logger in self.loggers.to_list():
                logger.attach_output_handler(
                    engine,
                    event_name=self.event_name,
                    tag=self.tag,
                    metric_names=self.metric_names,
                    global_step_transform=gst,
                )


class LearningRateLogging:
    def __init__(self, loggers, optimizer, event_name):
        self.loggers = loggers
        self.optimizer = optimizer
        self.event_name = event_name

    def attach(self, engine):

        if idist.get_rank() == 0:
            for logger in self.loggers.to_list():

                logger.attach_opt_params_handler(
                    engine,
                    event_name=self.event_name,
                    optimizer=self.optimizer,
                    param_name="lr",
                )
