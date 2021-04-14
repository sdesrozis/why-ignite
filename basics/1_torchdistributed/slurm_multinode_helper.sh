#!/bin/bash -l

# list of hostnames
hosts=(`scontrol show hostnames`)
# pick the fisrt hostname as master_addr
master_addr=${hosts[0]}

# build a port using last 4 digits of jobid
key=${SLURM_JOB_ID: -4}
master_port=$((15000 + key))

python -m torch.distributed.launch --master_addr $master_addr --master_port $master_port --nnodes $SLURM_NNODES --node_rank $SLURM_NODEID --nproc_per_node 2 helloworld.py
