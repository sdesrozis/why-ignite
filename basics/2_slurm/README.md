# An HPC tool : `slurm`

Sequential experiment

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

    [http://mycluster024:18423] hello from [mycluster024:nccl] process 0/1

Single node experiment 

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

    [http://mycluster024:18424] hello from [mycluster024:nccl] process 1/2
    [http://mycluster024:18424] hello from [mycluster024:nccl] process 0/2

> The `nccl` backend is used : so maximum 2 tasks can be defined because `nccl` handles exactly one 
> process per GPU and no more.

Single node experiment with `gloo`

```commandline
srun --nodes=1 
     --ntasks-per-node=8 
     --job-name=helloworld_Divers 
     --time=00:01:00  
     --partition=gpgpu 
     --gres=gpu:2 
     python helloworld.py --backend=gloo
```

Results
    
    [http://mycluster024:18425] hello from [mycluster024:gloo] process 0/8
    [http://mycluster024:18425] hello from [mycluster024:gloo] process 1/8
    [http://mycluster024:18425] hello from [mycluster024:gloo] process 2/8
    [http://mycluster024:18425] hello from [mycluster024:gloo] process 3/8
    [http://mycluster024:18425] hello from [mycluster024:gloo] process 4/8
    [http://mycluster024:18425] hello from [mycluster024:gloo] process 5/8
    [http://mycluster024:18425] hello from [mycluster024:gloo] process 6/8
    [http://mycluster024:18425] hello from [mycluster024:gloo] process 7/8

> Using `gloo` backend, it is not mandatory to have GPUs and any partition could be reached

Single node experiment with `gloo` without GPU

```commandline
srun --nodes=1 
     --ntasks-per-node=8 
     --job-name=helloworld_Divers 
     --time=00:01:00  
     python helloworld.py --backend=gloo
```

Results

    [http://mycluster135:18429] hello from [mycluster135:gloo] process 0/8
    [http://mycluster135:18429] hello from [mycluster135:gloo] process 1/8
    [http://mycluster135:18429] hello from [mycluster135:gloo] process 2/8
    [http://mycluster135:18429] hello from [mycluster135:gloo] process 3/8
    [http://mycluster135:18429] hello from [mycluster135:gloo] process 4/8
    [http://mycluster135:18429] hello from [mycluster135:gloo] process 5/8
    [http://mycluster135:18429] hello from [mycluster135:gloo] process 6/8
    [http://mycluster135:18429] hello from [mycluster135:gloo] process 7/8

Multi node experiment 

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

    [http://mycluster025:18430] hello from [mycluster025:gloo] process 0/8
    [http://mycluster025:18430] hello from [mycluster026:gloo] process 2/8
    [http://mycluster025:18430] hello from [mycluster025:gloo] process 1/8
    [http://mycluster025:18430] hello from [mycluster026:gloo] process 3/8
    [http://mycluster025:18430] hello from [mycluster027:gloo] process 4/8
    [http://mycluster025:18430] hello from [mycluster028:gloo] process 6/8
    [http://mycluster025:18430] hello from [mycluster028:gloo] process 7/8
    [http://mycluster025:18430] hello from [mycluster027:gloo] process 5/8

To sum up, it is easy to distribute process using `slurm` for a `PyTorch` code. However, it
needs to programmatically handle the `slurm` configuration. 
