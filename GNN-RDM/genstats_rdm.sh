for data in ogbn-mag 
do
    for ngpu in 8
    do
        for hidden in 128 
        do
            #for mmorder in sdsd
            for mmorder in ssss sssd ssds ssdd sdss sdsd sdds sddd dsss dssd dsds dsdd ddss ddsd ddds dddd
            do
                python -m torch.distributed.run --nproc_per_node=$ngpu --nnodes=1 --node_rank=0 --master_addr=10.242.76.74 --master_port=12394 src/gcn_distr_transpose_15d.py --accperrank=$ngpu --epochs=10 --graphname=$data --mmorder=$mmorder --timing=True --midlayer=$hidden --runcount=1 --activations=True  --replication=1
            done
        done
    done
done
