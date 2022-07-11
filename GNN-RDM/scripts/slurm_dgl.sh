graph=$1
conf=$2
bs=$3
output=$4
rk=$SLURM_PROCID

ifconfig | grep inet\ 10| head -n1 | awk '{print $2}' >> ip.txt
root=$(head -n1 ip.txt)

echo "Launching server process"

DGL_ROLE=server DGL_NUM_SAMPLER=0 OMP_NUM_THREADS=1 DGL_NUM_CLIENT=$SLURM_NTASKS DGL_CONF_PATH=$conf DGL_IP_CONFIG=ip.txt DGL_NUM_SERVER=1 DGL_GRAPH_FORMAT=csc,coo  DGL_SERVER_ID=$rk python ../src/dgl_batched/train_dist.py --graph_name $graph --ip_config ip.txt --num_epochs 1000 --batch_size $bs --num_gpus 1

echo "Launching client process"

DGL_DIST_MODE=distributed DGL_ROLE=client DGL_NUM_SAMPLER=0 DGL_NUM_CLIENT=1 DGL_CONF_PATH=$conf DGL_IP_CONFIG=ip.txt DGL_NUM_SERVER=1 DGL_GRAPH_FORMAT=csc,coo OMP_NUM_THREADS=8  python -m torch.distributed.launch --nproc_per_node=2 --nnodes=4 --node_rank=$rk --master_addr=$root --master_port=61234 ../src/dgl_batched/train_dist.py --graph_name $graph --ip_config ip.txt --num_epochs 1000 --batch_size $bs --num_gpus 2 --csv $output
