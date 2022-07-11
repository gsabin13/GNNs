g=$1
output=$2
ws=`expr 2 \* $SLURM_NTASKS`
rk=$SLURM_PROCID
echo "World size is $ws"
ifconfig | grep inet\ 10| tail -n1 | awk '{print $2}'> ip.txt
root=$(head -n1 ip.txt)
echo root=$root

python ../src/dgcl/train_gcn.py  --master_addr=$root  --nproc_per_node=2 --dataset $g  --world_size $ws --comm=dgcl --feat_size=128 --n-layers=2 --n-hidden=128 --n-epochs=10  --node_rank=$rk --csv=$output
