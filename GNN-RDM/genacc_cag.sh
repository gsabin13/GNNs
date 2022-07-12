#for data in ogbn-arxiv Reddit
#do
#    for ngpu in 1
#    do
#        for hidden in 128
#        do
#            #python -m torch.distributed.run --nproc_per_node=$ngpu --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12394 GNN-RDM/src/gcn_distr.py --accperrank=$ngpu --epochs=2000 --graphname=$data --timing=True --midlayer=$hidden --runcount=1 --activations=True  --accuracy=True --normalization=True --acc_csv full_348.csv
#            python -m torch.distributed.run --nproc_per_node=$ngpu --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12394 GNN-RDM/src/gcn_distr_15d.py --accperrank=$ngpu --epochs=2000 --graphname=$data --timing=True --midlayer=$hidden --runcount=1 --activations=True  --replication=2 --accuracy=True --normalization=True --acc_csv full_348.csv
#        done
#   done
#done

for data in meta arctic25 oral
do
    for ngpu in 1
    do
        for hidden in 128
        do
            python -m torch.distributed.run --nproc_per_node=$ngpu --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12394 GNN-RDM/src/gcn_distr.py --accperrank=$ngpu --epochs=2000 --graphname=$data --timing=True --midlayer=$hidden --runcount=1 --activations=True  --accuracy=True --normalization=True --acc_csv full_348.csv
            #python -m torch.distributed.run --nproc_per_node=$ngpu --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12394 GNN-RDM/src/gcn_distr_15d.py --accperrank=$ngpu --epochs=2000 --graphname=$data --timing=True --midlayer=$hidden --runcount=1 --activations=True  --replication=2 --accuracy=True --normalization=True --acc_csv full_348.csv
        done
   done
done
