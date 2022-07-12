export OMP_NUM_THREADS=1
rk=0
ws=1
echo "rank:"$rk
root=127.0.0.1
csv=$1
for graph in ogbn-arxiv reddit meta arctic25 oral ogbn-products
do
    for ngpu in 1
    do
        python -m torch.distributed.run --nnodes=$ws --nproc_per_node=$ngpu --node_rank=$rk --master_addr=$root --master_port=12355  dgl/train_full_load.py  --dataset $graph --csv $csv
    done
done
