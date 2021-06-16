import functools
import sys

import ignite.distributed as idist
from ignite.engine import Engine, EventEnum
from ignite.utils import setup_logger


class DriverEvents(EventEnum):
    BEFORE_STARTED = "dump_config"
    AFTER_COMPLETED = "after_completed"


class Driver:
    def __init__(self, name, function, dataloader):
        self.name = name
        self.engine = Engine(function)
        self.engine.state.driver = self
        self.engine.state.function = function
        self.engine.register_events(*DriverEvents)
        self.engine.logger = setup_logger(self.name, stream=sys.stdout)
        self.dataloader = dataloader
        self.handlers = {}

    def load_state_dict(self, state_dict):
        self.engine.load_state_dict(state_dict)

    def register_handler(func):
        @functools.wraps(func)
        def wrap(self, handler, predicat=True, handler_name=None, *args, **kwargs):
            if not predicat:
                return None
            if handler_name is None:
                handler_name = id(handler)
            if handler_name in self.handlers:
                raise ValueError(f"Already registered event '{handler_name}'")
            self.handlers[handler_name] = handler
            return func(self, handler, *args, **kwargs)

        return wrap

    @register_handler
    def attach_on_event(self, handler, handler_event, *args, **kwargs):
        self.engine.add_event_handler(
            event_name=handler_event, handler=handler, *args, **kwargs
        )
        return self

    @register_handler
    def attach(self, handler, *args, **kwargs):
        handler.attach(self.engine, *args, **kwargs)
        return self

    @register_handler
    def __attach_on_master(self, handler, *args, **kwargs):
        handler.attach(self.engine, *args, **kwargs)

    # register only on master
    def attach_on_master(self, *args, **kwargs):
        if idist.get_rank() == 0:
            self.__attach_on_master(*args, **kwargs)
        return self

    def on(self, event_name, *args, **kwargs):
        return self.engine.on(event_name, *args, **kwargs)

    def fit(self, max_epochs):
        self.engine.fire_event(DriverEvents.BEFORE_STARTED)
        self.engine.run(data=self.dataloader, max_epochs=max_epochs)
        self.engine.fire_event(DriverEvents.AFTER_COMPLETED)

    register_handler = staticmethod(register_handler)
