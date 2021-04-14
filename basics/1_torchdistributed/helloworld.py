import os
import socket
import argparse
import torch
import torch.distributed as dist


if __name__ == "__main__":

    assert dist.is_available()

    parser = argparse.ArgumentParser("single-node")
    parser.add_argument("--backend", type=str, default="nccl")
    # local rank is mandatory if torch.distributed.launch is used without --use_env
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    if args.backend == "nccl":
        assert dist.is_nccl_available()
        assert torch.cuda.is_available()
    elif args.backend == "gloo":
        assert dist.is_gloo_available()
    else:
        raise ValueError(f"unvalid backend `{args.backend}` (valid: `gloo` or `nccl`)")

    # these variables are mandatory to initialize a process group
    # automatically updated by torch.distributed.launch
    assert "MASTER_ADDR" in os.environ
    assert "MASTER_PORT" in os.environ
    assert "WORLD_SIZE" in os.environ
    assert "RANK" in os.environ
    assert "LOCAL_RANK" in os.environ

    # initialize world communicator
    dist.init_process_group(backend=args.backend)

    assert dist.is_initialized()

    # now torch.distributed module is initialized

    hostname = socket.gethostname()

    for current in range(dist.get_world_size()):
        if dist.get_rank() == current:
            addr = f"http://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
            print(f"[{addr}] hello from [{hostname}:{dist.get_backend()}] process {dist.get_rank()}/{dist.get_world_size()}")
        dist.barrier()

    dist.destroy_process_group()
