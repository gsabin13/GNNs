mkdir -p ./data/ogbn-arxiv
mkdir -p ./data/reddit
mkdir -p ./data/ogbn-products
python ../src/dgl_batched/partition_graph.py --dataset ogbn-arxiv --output ./data/ogbn-arxiv
#python ../src/dgl_batched/partition_graph.py --dataset reddit --output ./data/reddit
#python ../src/dgl_batched/partition_graph.py --dataset ogbn-products --output ./data/ogbn-products
#echo "DONE"
