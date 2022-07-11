#!/bin/bash

rank="$SLURM_PROCID"
echo rank=$rank
#bash /uufs/chpc.utah.edu/common/home/u1288779/.bashrc
#export PATH="/uufs/chpc.utah.edu/common/home/u1288779/anaconda3/bin:$PATH"
#cd  /uufs/chpc.utah.edu/common/home/u1288779/GNN-RDM
#cd ..
#bash
#cd GNN-RDM
#source env.sh
graph=$1
root="10.242.66.105"
echo root=$root
hidden=$2
rep=$3
mm=$4
e=100
act="--activations=True"
norm="--normalization=True"
#act=""
norm=""
acc="--accuracy=True"
acc=""
cmd="python -m torch.distributed.run --nproc_per_node=2 --nnodes=$SLURM_NTASKS --node_rank=$rank --master_addr=$root --master_port=12394 src/gcn_distr_transpose_15d.py  --accperrank=2 --epochs=$e --graphname=$graph --timing=True --midlayer=$hidden --runcount=1 $act $norm $acc --mmorder=$mm --replication=$rep"
echo $cmd
$cmd

