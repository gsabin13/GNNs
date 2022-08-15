#!/bin/bash -x
#SBATCH -M kingspeak
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=32000
#SBATCH -t 72:00:00
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
sh run_saint.sh