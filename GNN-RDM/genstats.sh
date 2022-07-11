#for data in arctic25 
#do
#    for ngpu in 2 4 8
#    do
#        python -m torch.distributed.run --nproc_per_node=$ngpu --nnodes=1 --node_rank=0 --master_addr=10.242.76.74 --master_port=12394 src/gcn_distr_15d.py --accperrank=$ngpu --epochs=10 --graphname=$data --timing=True --midlayer=128 --runcount=1 --activations=True  --replication=2
#    done
#done
#
#for data in ogbn-arxiv Reddit ogbn-products meta arctic25 oral
#for data in ogbn-papers100M

#export OMP_NUM_THREADS=1
#rank=$SLURM_PROCID
#echo rank=$rank
#graph=$1
#root="10.242.2.106"
root="127.0.0.1"
#echo root=$root

for data in web-google ogbn-mag com-orkut 
do
    for ngpu in 2 4 8
    do
        python -m torch.distributed.run --nproc_per_node=$ngpu --nnodes=1 --node_rank=0 --master_addr=$root --master_port=12394 src/gcn_distr_15d.py --accperrank=$ngpu --epochs=10 --graphname=$data --timing=True --midlayer=128 --runcount=1 --activations=True  --replication=2
    done
done
