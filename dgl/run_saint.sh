export OMP_NUM_THREADS=1
rk=0
ws=1
echo "rank:"$rk
root=127.0.0.1
#for graph in meta arxiv reddit products arctic25 oral
#do
#    for topo in lower upper sym
#    do
#        for ngpu in 2
#        do
#            python -m torch.distributed.run --nproc_per_node=$ngpu --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=1234 gcn_distr_graphsaint_load.py --accperrank=$ngpu --epochs=500   --timing=True --midlayer=128 --runcount=1  --activations=True  --accuracy=True --graphname=$graph --topo=$topo --load --acc_csv rdmsaint_with_dgl_weights.csv
#        done
#    done
#done

for graph in oral
do
    for topo in upper sym 
    # upper sym
    do
        for ngpu in 2
        do
            python -m torch.distributed.run --nproc_per_node=$ngpu --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=1234  train_sampling_load.py --csv dglsaint3.csv  --dataset $graph --topo $topo 
            #python -m torch.distributed.run --nproc_per_node=$ngpu --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=1234 gcn_distr_graphsaint_load.py --accperrank=$ngpu --epochs=500   --timing=True --midlayer=128 --runcount=1  --activations=True  --accuracy=True --graphname=$graph --topo=$topo --load --acc_csv dglsaint3.csv
        done
    done
done
