import os
import subprocess
import socket
import argparse
import torch
import torch.distributed as dist


if __name__ == "__main__":

    assert dist.is_available()

    parser = argparse.ArgumentParser("single-node")
    parser.add_argument("--backend", type=str, default="nccl")
    args = parser.parse_args()

    if args.backend == "nccl":
        assert dist.is_nccl_available()
        assert torch.cuda.is_available()
    elif args.backend == "gloo":
        assert dist.is_gloo_available()
    else:
        raise ValueError(f"unvalid backend `{args.backend}` (valid: `gloo` or `nccl`)")

    # configuration from slurm variables
    assert "SLURM_JOBID" in os.environ
    assert "SLURM_PROCID" in os.environ
    assert "SLURM_LOCALID" in os.environ
    assert "SLURM_NTASKS" in os.environ
    assert "SLURM_JOB_NODELIST" in os.environ

    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    # port should be the same over all process
    slurm_port = os.environ["SLURM_JOB_ID"]
    slurm_port = slurm_port[-4:]
    os.environ["MASTER_PORT"] = str(int(slurm_port) + 15000)
    # master address is the first hostname of nodes list
    hostnames = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]])
    os.environ["MASTER_ADDR"] = hostnames.split()[0].decode("utf-8")

    # these variables are mandatory to initialize a process group
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
