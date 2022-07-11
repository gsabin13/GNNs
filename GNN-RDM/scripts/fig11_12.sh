# partition graphs
bash run_partition.sh
# partitioned graphs are at ./data
for graph in ogbn-arxiv
#reddit ogbn-products
do
    output="dgl_$graph.csv"
    # run DGL distributed training on 8 GPUs
    for bs in 64 128 256 512 1024 2048
    # As we are running on 8 GPUs, actual batch size = 512, 1K, 2K, 4K, 8K, 16K
    do
        cmd="srun bash slurm_dgl.sh $graph ./data/$graph/$graph.json $bs $output &"
        echo $cmd
        $cmd
        rm ip.txt
    done
    wait
    # generate bar chart
    barplot=$graph.fig11.pdf
    python gen_line $output $barplot e
    barplot=$graph.fig12.pdf
    python gen_line $output $barplot t
done
