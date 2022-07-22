#!/bin/bash -x
#SBATCH -M notchpeak
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH -c 16
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH -t 48:00:00
__conda_setup="$('/uufs/chpc.utah.edu/common/home/u1320844/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/uufs/chpc.utah.edu/common/home/u1320844/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/uufs/chpc.utah.edu/common/home/u1320844/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/uufs/chpc.utah.edu/common/home/u1320844/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate ogb
module load gcc/9.2.0
module load cuda/11.0

sh run.sh python pyg/inf_sampling_nonogb.py reddit
sh run.sh python pyg/inf_sampling.py ogbn-papers100M
#sh run.sh sh dgl/run_dgl.sh full_new.csv
#sh run.sh sh GNN-RDM/genacc_rdm.sh 