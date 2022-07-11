graph=$1
conf=$2
bs=$3
rk=$SLURM_PROCID

DGL_ROLE=server DGL_NUM_SAMPLER=0 OMP_NUM_THREADS=1 DGL_NUM_CLIENT=4 DGL_CONF_PATH=$conf DGL_IP_CONFIG=ip_4.txt DGL_NUM_SERVER=1 DGL_GRAPH_FORMAT=csc,coo  DGL_SERVER_ID=$rk python train_dist.py --graph_name $graph --ip_config ip_4.txt --num_epochs 1000 --batch_size $bs --num_gpus 1&

DGL_DIST_MODE=distributed DGL_ROLE=client DGL_NUM_SAMPLER=0 DGL_NUM_CLIENT=1 DGL_CONF_PATH=$conf DGL_IP_CONFIG=ip_4.txt DGL_NUM_SERVER=1 DGL_GRAPH_FORMAT=csc,coo OMP_NUM_THREADS=8  python -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=$rk --master_addr=10.242.66.105 --master_port=61234 train_dist.py --graph_name $graph --ip_config ip_4.txt --num_epochs 1000 --batch_size $bs --num_gpus 1 &
