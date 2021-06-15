import os

import ignite.distributed as idist
import torch
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint

from api.engines import DriverEvents
from api.utils import gst


class Checkpoint:
    def __init__(self, prefix, metric_name, dirname, n_saved_models=1, clearml=False):

        self.dirname = dirname
        self.clearml = clearml

        def score_fn(engine):
            return engine.state.metrics[metric_name]

        self.checkpoint = ModelCheckpoint(
            dirname=self.dirname,
            filename_prefix=prefix,
            score_function=score_fn,
            score_name=metric_name,
            global_step_transform=gst,
            n_saved=n_saved_models,
            create_dir=True,
        )

    def attach(self, engine):
        if idist.get_rank() == 0:
            if not engine.has_event_handler(self.save, Events.COMPLETED):
                engine.add_event_handler(Events.COMPLETED, self.save)

    def save(self, engine):
        trainer = engine.state.trainer
        ctx = trainer.state.function.context()
        self.checkpoint(engine, {"trainer": trainer, **ctx.to_dict()})
        if not self.clearml:
            return
        # attach on trainer completed to hack clearml
        if not trainer.has_event_handler(
            self.hack_clearml, DriverEvents.AFTER_COMPLETED
        ):
            trainer.add_event_handler(DriverEvents.AFTER_COMPLETED, self.hack_clearml)

    def hack_clearml(self, engine):
        # checkpoint files are sent once at the end from disk saves
        clearml_cache_dir = os.path.join(self.dirname, "clearml-cache")
        os.makedirs(clearml_cache_dir, exist_ok=True)

        engine.logger.info("upload artifacts to clearml file server...")
        for save in self.checkpoint._saved:
            engine.logger.info(
                "checkpoint filename : {}, priority : {}".format(
                    save.filename, save.priority
                )
            )
            state = torch.load(os.path.join(self.dirname, save.filename))
            filename = "{}_r2_{:.4f}.pt".format(
                self.checkpoint.filename_prefix, save.priority
            )
            filename = os.path.join(clearml_cache_dir, filename)
            engine.logger.info("uploaded filename : {}".format(filename))
            # only model is saved and automatically updated to clearml fileserver
            torch.save(state["network"], filename)
