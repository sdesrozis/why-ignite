import os
import socket
import argparse
import torch
import torch.distributed as dist
import ignite.distributed as idist


def main_fn(_):
    hostname = socket.gethostname()
    tensor = torch.ones(1).cuda()
    dist.all_reduce(tensor)  # or use idist.all_reduce(tensor)
    addr = f"http://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    print(f"[{addr}] [{hostname}:{dist.get_backend()}] "
          f"process {dist.get_rank()}/{dist.get_world_size()} has data {tensor}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("single-node")
    parser.add_argument("--backend", type=str, default="nccl")
    args = parser.parse_args()

    # idist from ignite handles automatically backend (gloo, nccl, horovod, xla)
    # and launcher (slurm, torch.distributed.launch)
    with idist.Parallel(backend=args.backend) as parallel:
        parallel.run(main_fn)
