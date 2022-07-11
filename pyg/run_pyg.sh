export OMP_NUM_THREADS=1
rk=0
echo "rank:"$rk
root=127.0.0.1
csv=$1

for graph in reddit ogbn-products arctic25 oral meta
do
    for ws in 2 4 8
    do
        cmd="python -m torch.distributed.run --nnodes=1 --nproc_per_node=$ws --node_rank=$rk --master_addr=$root --master_port=12355  graphsaint_ddp.py  --dataset=$graph --log=$csv --batch_size=4000 --num_subgs=32 --load"
        echo $cmd
        $cmd
    done
done

# TODO write to csv
#for data in meta arctic25 oral 
#do
#    #python -m torch.distributed.run --nnodes=2 --nproc_per_node=2 --node_rank=$rk --master_addr='10.242.2.105' --master_port='12355'  graphsaint_ddp.py  --dataset $data --log ddp_kp_$data.csv --batch_size 2000 --num_subgs 32 --load
#    python  -m torch.distributed.run --nnodes=$ws --nproc_per_node=2 --node_rank=$rk --master_addr=$root --master_port='12355'   meta_gnn_overlap_multigpu.py --input /scratch/general/nfs1/u1320844/dataset/metagnn --name $data --output . --loader s --batch_size 4000 --subgs 32
#done
