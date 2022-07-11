export OMP_NUM_THREADS=1
rk=0
ws=1
echo "rank:"$rk
root=127.0.0.1
csv=$1
for graph in ogbn-products
do
    for ngpu in 8
    do
        python -m torch.distributed.run --nnodes=$ws --nproc_per_node=$ngpu --node_rank=$rk --master_addr=$root --master_port=12355  train_sampling_load.py  --dataset $graph --csv $csv
    done
done
