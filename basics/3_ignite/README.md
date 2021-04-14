# A high level deep learning library : `PyTorch-Ignite`

Sequential experiment locally, here no need `srun` neither `torch.launch.distributed`

```commandline
python helloworld.py 
```

Results

    [http://127.0.0.1:15000] hello from [mydesktop:nccl] process 0/1

Sequential experiment with `torch.launch.distributed`

```commandline
python -m torch.distributed.launch --nproc_per_node 1 --use_env helloworld.py 
```

> Use `--use_env` to avoid `--local_rank` option in `python` code.

Results

    [http://127.0.0.1:29500] hello from [mycluster023:nccl] process 0/1

> Run was launched on an `mycluster440` node.

Sequential experiment with `slurm`

```commandline
srun --nodes=1 
     --ntasks-per-node=1 
     --job-name=helloworld_Divers 
     --time=00:01:00  
     --partition=gpgpu 
     --gres=gpu:2 
     python helloworld.py 
```

Results

    [http://mycluster023:18464] hello from [mycluster023:nccl] process 0/1

Single node experiment with `torch.launch.distributed`

```commandline
python -m torch.distributed.launch --nproc_per_node 2 --use_env helloworld.py 
```

Results

    [http://127.0.0.1:29500] hello from [mycluster023:nccl] process 0/2
    [http://127.0.0.1:29500] hello from [mycluster023:nccl] process 1/2

Single node experiment with `slurm`

```commandline
srun --nodes=1 
     --ntasks-per-node=2 
     --job-name=helloworld_Divers 
     --time=00:01:00  
     --partition=gpgpu 
     --gres=gpu:2 
     python helloworld.py 
```

Results

    [http://mycluster023:18465] hello from [mycluster023:nccl] process 1/2
    [http://mycluster023:18465] hello from [mycluster023:nccl] process 0/2

Multi node experiment with `slurm`

```commandline
srun --nodes=4
     --ntasks-per-node=2 
     --job-name=helloworld_Divers 
     --time=00:01:00  
     --partition=gpgpu 
     --gres=gpu:2
     --mem=10G 
     python helloworld.py
```

Results

    [http://mycluster025:18466] hello from [mycluster025:nccl] process 0/8
    [http://mycluster025:18466] hello from [mycluster026:nccl] process 3/8
    [http://mycluster025:18466] hello from [mycluster025:nccl] process 1/8
    [http://mycluster025:18466] hello from [mycluster026:nccl] process 2/8
    [http://mycluster025:18466] hello from [mycluster028:nccl] process 6/8
    [http://mycluster025:18466] hello from [mycluster028:nccl] process 7/8
    [http://mycluster025:18466] hello from [mycluster027:nccl] process 5/8
    [http://mycluster025:18466] hello from [mycluster027:nccl] process 4/8

To sum up, the module `idist` of `PyTorch-Ignite` allows handling distribution of process
using `slurm` and `torch.distributed.launch` in an agnotic way. The `python` code is 
minimal.

Use script `reduction.py` rather than `helloworld.py` to try a reduction using `all_reduce`.

```commandline
srun --nodes=2
     --ntasks-per-node=2 
     --job-name=reduction_Divers 
     --time=00:01:00  
     --partition=gpgpu 
     --gres=gpu:2
     --mem=10G 
     python reduction.py
```

Results

     [http://mycluster029:15493] [mycluster030:nccl] process 3/4 has data tensor([4.], device='cuda:1')
     [http://mycluster029:15493] [mycluster030:nccl] process 2/4 has data tensor([4.], device='cuda:0')
     [http://mycluster029:15493] [mycluster029:nccl] process 0/4 has data tensor([4.], device='cuda:0')
     [http://mycluster029:15493] [mycluster029:nccl] process 1/4 has data tensor([4.], device='cuda:1')
