# A native tool : `torch.distributed`

Here is a minimal example using `torch.distributed.launch`. This solution should be prefered for simple use cases.

Example of configuration (using `check_config.py`)

    torch version : 1.7.1
    torch git version : 57bffc3a8e4fee0cce31e1ff1f662ccf7b16db57
    torch version cuda : 10.1
    number of cuda device(s) : 2
    - device 0 : _CudaDeviceProperties(name='Tesla P100-PCIE-16GB', major=6, minor=0, total_memory=16280MB, multi_processor_count=56)
    - device 1 : _CudaDeviceProperties(name='Tesla P100-PCIE-16GB', major=6, minor=0, total_memory=16280MB, multi_processor_count=56)

Sequential experiment

```commandline
python -m torch.distributed.launch --nproc_per_node 1 helloworld.py
```

Results 

    [http://127.0.0.1:29500] hello from nccl process 0/1

Single node experiment 

```commandline
python -m torch.distributed.launch --nproc_per_node 2 helloworld.py
```

Results

    [http://127.0.0.1:29500] hello from nccl process 0/2
    [http://127.0.0.1:29500] hello from nccl process 1/2
   
Multi node experiment 

on node 0
```commandline
python -m torch.distributed.launch --nnodes 2 --nproc_per_node 4 --node_rank 0 helloworld.py --backend=gloo
```

on node 1
```commandline
python -m torch.distributed.launch --nnodes 2 --nproc_per_node 4 --node_rank 1 helloworld.py --backend=gloo
```

Results

on node 0

    [http://127.0.0.1:29500] hello from gloo process 0/4
    [http://127.0.0.1:29500] hello from gloo process 1/4
    [http://127.0.0.1:29500] hello from gloo process 2/4
    [http://127.0.0.1:29500] hello from gloo process 3/4

on node 1

    [http://127.0.0.1:29500] hello from gloo process 4/8
    [http://127.0.0.1:29500] hello from gloo process 5/8
    [http://127.0.0.1:29500] hello from gloo process 6/8
    [http://127.0.0.1:29500] hello from gloo process 7/8

That's it for `torch.distributed.launch`. 

## More advanced : limited compatibility with `slurm`

The following section aims to give some information
about interoperability of `torch.distributed.launch` and `slurm`. This is an optional reading.

See [here](../2_slurm/README.md) to correctly use `slurm` for distributed computing with `PyTorch`. 

The `srun` command spawns as `torch.distributed.launch` does. Therefore, the interaction is not perfect.

Sequential experiment

```commandline
srun --nodes=1 
     --ntasks-per-node=1 
     --job-name=helloworld_Divers 
     --time=00:01:00  
     --partition=gpgpu 
     --gres=gpu:2 
     python -m torch.distributed.launch --nproc_per_node 1 helloworld.py 
```

Results

    [http://127.0.0.1:29500] hello from [mycluster024:nccl] process 0/1
   
Single node experiment 

```commandline
srun --nodes=1 
     --ntasks-per-node=1 
     --job-name=helloworld_Divers 
     --time=00:01:00  
     --partition=gpgpu 
     --gres=gpu:2 
     python -m torch.distributed.launch --nproc_per_node 2 helloworld.py 
```

Results

    [http://127.0.0.1:29500] hello from [mycluster024:nccl] process 0/2
    [http://127.0.0.1:29500] hello from [mycluster024:nccl] process 1/2

> Note that `--ntasks-per-node=1` but `--nproc_per_node 2`. That's a bit weird.

Multi node experiment 

```commandline
srun --nodes=4 
     --ntasks-per-node=1 
     --job-name=helloworld_Divers 
     --time=00:01:00  
     --partition=gpgpu 
     --gres=gpu:2
     --mem=10G 
     bash slurm_multinode_helper.sh 
```

where `slurm_multinode_helper.sh` is defined as follows 

```
#!/bin/bash -l

# list of hostnames
hosts=(`scontrol show hostnames`)
# pick the fisrt hostname as master_addr
master_addr=${hosts[0]}

# build a port using last 4 digits of jobid
key=${SLURM_JOB_ID: -4}
master_port=$((15000 + key))

python -m torch.distributed.launch --master_addr $master_addr 
                                   --master_port $master_port 
                                   --nnodes $SLURM_NNODES 
                                   --node_rank $SLURM_NODEID 
                                   --nproc_per_node 2 
                                   helloworld.py
```

Results 

    [http://mycluster025:18420] hello from [mycluster025:nccl] process 0/8
    [http://mycluster025:18420] hello from [mycluster025:nccl] process 1/8
    [http://mycluster025:18420] hello from [mycluster026:nccl] process 2/8
    [http://mycluster025:18420] hello from [mycluster026:nccl] process 3/8
    [http://mycluster025:18420] hello from [mycluster027:nccl] process 4/8
    [http://mycluster025:18420] hello from [mycluster027:nccl] process 5/8
    [http://mycluster025:18420] hello from [mycluster028:nccl] process 6/8
    [http://mycluster025:18420] hello from [mycluster028:nccl] process 7/8

> To use slurm in command line fashion, we need a script to define the hostname/port required by 
> `torch.distributed.launch`.

> Note the option `--mem=10G` to avoid memory issues.
