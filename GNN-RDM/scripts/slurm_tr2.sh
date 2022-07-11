#!/bin/bash

rank="$SLURM_PROCID"
echo rank=$rank
graph=$1
#root=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
ifconfig | grep inet\ 10| tail -n1 | awk '{print $2}'> ip.txt
root=$(head -n1 ip.txt)
echo root=$root
output=$2
hidden=128

cmd="python -m torch.distributed.run --nproc_per_node=2 --nnodes=$SLURM_NTASKS --node_rank=$rank --master_addr=$root --master_port=1234 ../src/gcn_distr_transpose.py --accperrank=2 --epochs=10 --graphname=$graph --timing=True --midlayer=$hidden --runcount=1 --normalization=True --activations=True --accuracy=True --csv=$output"
echo $cmd
$cmd

