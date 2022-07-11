#!/bin/bash

#SBATCH --time=3:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

for graph in  "ogbn-arxiv"  Reddit  # "ogbn-products" "ogbn-mag"
do
    for hidden in 128 256 512 #1024
    do
        srun slurm_tr.sh $graph $hidden
        srun slurm_tr2.sh $graph $hidden
        for replication in 1 2 4
        do
            srun slurm_15d.sh $graph $hidden $replication
        done
    done
done
