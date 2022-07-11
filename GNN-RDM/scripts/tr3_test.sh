#input=$1
#hidden=$2
rep=1
ngpu=8
for mmorder in ssss sssd ssds ssdd sdss sdsd sdds sddd dsss dssd dsds dsdd ddss ddsd ddds dddd
do
    for hidden in 512 256 128
    do
        #cmd="srun slurm_tr3.sh $input $hidden $rep $mmorder"
        python -m torch.distributed.run --nproc_per_node=$ngpu --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=12349 src/gcn_distr_transpose_15d.py --accperrank=$ngpu --epochs=10 --graphname=$input --timing=True --midlayer=$hidden --runcount=1 --activations=True --mmorder=$mm --replication=1
        #echo $cmd
        #$cmd
    done
done
