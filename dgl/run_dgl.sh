export OMP_NUM_THREADS=1
rk=0
ws=1
echo "rank:"$rk
root=127.0.0.1
csv=$1
#for graph in arxiv reddit products
#do
#    for topo in lower upper sym
#    do
#        python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=1234 testrdm.py --accperrank=1 --epochs=200 --timing=True --midlayer=128 --runcount=1  --activations=True --mmorder=ssss --replication=1 --acc_csv aug8_kp361.csv --accuracy=True --graphname=$graph  --lr=0.01 --topo $topo
#    done
#done
#for graph in ogbn-arxiv reddit ogbn-products
#do
#    for topo in lower upper sym
#    do
#        python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12355  testdgl.py  --dataset $graph  --csv aug8_kp361.csv   --topo $topo   --lr 0.01  --n_hidden 128 --self_loop --load
#    done
#done
#for graph in meta
#do
#    for topo in lower upper sym
#    do
#        python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12355  testdgl.py  --dataset $graph  --csv aug8_kp361.csv   --topo $topo   --lr 0.01  --n_hidden 128 --self_loop
#    done
#done
for graph in meta
do
    for topo in upper lower sym
    do
        python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=1234 testrdm.py --accperrank=2 --epochs=1000 --timing=True --midlayer=128 --runcount=1  --activations=True --mmorder=ssss --replication=1 --acc_csv aug11_kp361.csv --accuracy=True --graphname=$graph  --lr=0.01 --topo $topo --load
    done
done

for graph in arctic25 oral
do
    for topo in upper lower sym
    do
        python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=1234 testrdm.py --accperrank=2 --epochs=1000 --timing=True --midlayer=128 --runcount=1  --activations=True --mmorder=ssss --replication=1 --acc_csv aug11_kp361.csv --accuracy=True --graphname=$graph  --lr=0.001 --topo $topo --load
    done
done
#python -m torch.distributed.run --nnodes=$ws --nproc_per_node=$ngpu --node_rank=$rk --master_addr=$root --master_port=12355  train_full_load.py  --dataset $graph --csv $csv

#for graph in meta
#do
#    for topo in upper lower sym
#    do
#        python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=1234 gcn_distr_graphsaint_load.py --accperrank=1 --epochs=10   --timing=True --midlayer=128 --runcount=1  --activations=True  --accuracy=True --graphname=arctic25
#    done
#done

#for graph in ogbn-arxiv reddit ogbn-products
#do
#    for topo in lower upper sym
#    do
#        python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=1234 gcn_distr_graphsaint_load.py --accperrank=1 --epochs=10   --timing=True --midlayer=128 --runcount=1  --activations=True  --accuracy=True --graphname=arctic25
#    done
#done