import os
import tempfile
from datetime import datetime

import ignite.distributed as idist


def create_dirname():
    dirname = ""
    if idist.get_rank() == 0:
        os.makedirs("experiments", exist_ok=True)
        if "SLURM_JOBID" in os.environ:
            dirname = f"experiments/{os.environ['SLURM_JOBID']}"
            os.makedirs(dirname, exist_ok=True)
        else:
            dirname = tempfile.mkdtemp(
                prefix=f"experiments/{datetime.now().strftime('%Y_%m_%d_%H:%M:%S_')}",
                dir=os.getcwd(),
            )
    if idist.get_world_size() > 1:
        dirname = idist.all_gather(dirname)[0]

    return dirname


def gst(engine, event_name):
    if hasattr(engine.state, "trainer"):
        state = engine.state.trainer.state
    else:
        state = engine.state
    return state.get_event_attrib_value(event_name)
