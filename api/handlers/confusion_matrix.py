from ignite.engine import Engine, Events
from ignite.metrics import ConfusionMatrix


class ConfusionMatrixHelper:

    def __init__(self, cm: ConfusionMatrix):
        self.cm = cm

    def attach(self, engine: Engine, name: str):

        def last_epoch_fn(engine, _):
            tr = engine.state.trainer
            return tr.state.epoch == tr.state.max_epochs

        # cm is attached during the STARTED event of the last epoch of trainer
        cm_attach_event = Events.STARTED(event_filter=last_epoch_fn)
        engine.add_event_handler(cm_attach_event, self.__attach_confusion_matrix, name)

        # cm is detached during the COMPLETED event of the last epoch of trainer
        cm_detach_event = Events.COMPLETED(event_filter=last_epoch_fn)
        engine.add_event_handler(cm_detach_event, self.__detach_confusion_matrix)

        engine.add_event_handler(cm_detach_event, self.__print_confusion_matrix, name)

        # hint
        # During the last epoch, ConfusionMatrix is attached in the engine at the end of
        # handlers list. It means that ConfusionMatrix is triggered AFTER screen logging,
        # checkpoint, etc.

    def __attach_confusion_matrix(self, engine, name):
        self.cm.attach(engine, name)

    def __detach_confusion_matrix(self, engine):
        self.cm.detach(engine)

    @staticmethod
    def __print_confusion_matrix(engine, name):
        engine.logger.info(f"- {name} :")
        print(engine.state.metrics[name].cpu().numpy())
