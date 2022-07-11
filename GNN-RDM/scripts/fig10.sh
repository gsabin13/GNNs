# run all 6 datasets
for graph in arxiv reddit web-google ogbn-mag ogbn-products com-orkut
# run CAGNET-1D CAGNET-1.5D DGCL GNN-RDM
do
    output=$graph.csv
    # run on 2,4,8 GPUs
    for nmachines in 1 2 4
    do
        # run CAGNET-1D CAGNET-1.5D GNN-RDM DGCL
        srun -N $nmachines bash slurm_base.sh $graph $output
        srun -N $nmachines bash slurm_tr3.sh $graph  $output
        srun -N $nmachines bash slurm_dgcl.sh $graph $output
    done
    # generate bar chart
    barplot=$graph.fig10.pdf
    python gen_bar.py $output $barplot
done
