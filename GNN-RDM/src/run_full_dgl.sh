for data in ogbn-arxiv reddit ogbn-products meta arctic25 oral
do
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12355  train_full_load.py  --dataset $data --csv test
done