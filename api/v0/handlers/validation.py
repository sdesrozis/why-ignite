from ignite.engine import Events


class Validation:
    def __init__(self, evaluators):
        self.evaluators = evaluators

    def attach(self, engine):
        if not engine.has_event_handler(self.epoch_completed, Events.EPOCH_COMPLETED):
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.epoch_completed)

    def epoch_completed(self, trainer):
        for evaluator in self.evaluators:
            # state is now embedding the trainer
            # this could be usefull for checkpoint
            evaluator.engine.state.trainer = trainer
            evaluator.fit(max_epochs=1)
