# dgl
python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12355  testdgl.py  --dataset arctic25 --csv l
# rdm
python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=1234 testrdm.py --accperrank=1 --epochs=10  --graphname=arctic25 --timing=True --midlayer=128 --runcount=1 --normalization=True --activations=True --mmorder=sdsd --replication=1
