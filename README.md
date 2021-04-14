# Why use PyTorch-Ignite ?

## Installation

To install dependencies, use the following `pip` command

```commandline
pip install -r requirements.txt 
```

## Documentation and tutorials

Please read with attention the following links to official documentation and tutorials 

* https://pytorch.org/docs/master/notes/ddp.html
* https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html 
* https://pytorch.org/tutorials/beginner/dist_overview.html
* https://pytorch.org/tutorials/intermediate/dist_tuto.html
* https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

## Configuration

Use the script `check_config.py` to get some information about the configuration and environment

```commandline
python check_config.py
```

On a cluster with `slurm` manager, use `srun` command 

```commandline
srun --nodes=1
     --ntasks=1
     --job-name=check_config_Divers
     --time=00:01:00 
     --partition=gpgpu
     --gres=gpu:2
     python check_config.py
```

> The variable `SLURM_WCKEY` should be defined to a relevant project id value.

> `srun` can be used in a scripting mode and submit to scheduler by `sbatch` command.
> Scripting can help to configure the environment more precisely.

Please see [here](basics/0_environment/README.md) for relocated environments if needed.

Example of configuration (using `check_config.py`) on a GPU node :

    torch version : 1.7.1
    torch git version : 57bffc3a8e4fee0cce31e1ff1f662ccf7b16db57
    torch version cuda : 10.1
    number of cuda device(s) : 2
    - device 0 : _CudaDeviceProperties(name='Tesla P100-PCIE-16GB', major=6, minor=0, total_memory=16280MB, multi_processor_count=56)
    - device 1 : _CudaDeviceProperties(name='Tesla P100-PCIE-16GB', major=6, minor=0, total_memory=16280MB, multi_processor_count=56)

## `PyTorch` backends

It exists several backends available in PyTorch :
* `gloo` for GPUs and CPUs
* `nccl` for GPUs
* `mpi` for CPUs
* `xla` for TPUs

> The `nccl` backend should be prefered to handle GPUs but only one process per GPU is allowed.

