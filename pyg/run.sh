for data in ogbn-mag 
do
    python graphsaint_sequential.py --dataset $data --batch_size 4000 --num_subgs 32 --log ddp_kp_$data.csv --save
done

#for data in meta arctic25
#do
#    python meta_gnn_save.py --input /scratch/general/nfs1/u1320844/dataset/metagnn --name $data --output . --loader s --batch_size 4000 --subgs 32
#done
