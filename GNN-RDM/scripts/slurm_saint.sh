#!/bin/bash
rank=0
#"$SLURM_PROCID"
export OMP_NUM_THREADS=1
echo rank=$rank
graph=$1
#root="10.242.2.106"
root="127.0.0.1"
echo root=$root
hidden=128
np=$2
e=200
act="--activations=True"
norm="--normalization=True"
#act=""
norm=""
#cmd="python -m torch.distributed.run --nproc_per_node=$np --nnodes=$SLURM_NTASKS --node_rank=$rank --master_addr=$root --master_port=12394 src/gcn_distr_graphsaint_meta.py  --accperrank=$np --epochs=$e --graphname=$graph --timing=True --midlayer=$hidden --runcount=1 $act $norm --accuracy=True "
cmd="python -m torch.distributed.run --nproc_per_node=$np --nnodes=1 --node_rank=$rank --master_addr=$root --master_port=12394 src/gcn_distr_graphsaint_load.py  --accperrank=$np --epochs=$e --graphname=$graph --timing=True --midlayer=$hidden --runcount=1 $act $norm --accuracy=True "
#cmd="python -m torch.distributed.run --nproc_per_node=$np --nnodes=$SLURM_NTASKS --node_rank=$rank --master_addr=$root --master_port=12394 src/gcn_distr_transpose.py  --accperrank=$np --epochs=$e --graphname=$graph --timing=True --midlayer=$hidden --runcount=1 $act $norm --accuracy=True"
echo $cmd
$cmd

