# Forked CAGNET: Communication-Avoiding Graph Neural nETworks

## Dependencies
- Python 3.6.10
- PyTorch 1.3.1
- PyTorch Geometric (PyG) 1.3.2
- CUDA 10.1
- GCC 6.4.0

Newer packages also working, tested on KP360: 
- Python 3.7.11
- torch                   1.9.1+cu111
- torch-cluster           1.5.9
- torch-geometric         2.0.1
- torch-scatter           2.0.8
- torch-sparse            0.6.12
- CUDA 11.0
- GCC 9.2.0

On OLCF Summit, all of these dependencies can be accessed with the following
```bash
module load cuda # CUDA 10.1
module load gcc # GCC 6.4.0
module load ibm-wml-ce/1.7.0-3 # PyTorch 1.3.1, Python 3.6.10

# PyG and its dependencies
conda create --name gnn --clone ibm-wml-ce-1.7.0-3
conda activate gnn
pip install --no-cache-dir torch-scatter==1.4.0
pip install --no-cache-dir torch-sparse==0.4.3
pip install --no-cache-dir torch-cluster==1.4.5
pip install --no-cache-dir torch-geometric==1.3.2

# OGB dataset integration
pip install ogb

```
It's sometimes tricky to install pytorch-geometric... Even after installing them, there could be some error importing GCNConv. As this GCNConv isn't used, I think this line can be deleted.

## Compiling

This code uses C++ extensions. To compile these, run

```bash
cd sparse-extension
python setup.py install
```

If Ninja doesn't compile because cuda_runtime_api.h is not found, please check CUDA-related environment variables. For example
'''
export CUDA_HOME=$HOME/tools/cuda-9.0 # change to your path
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
'''

## Documentation

Each algorithm in CAGNET is implemented in a separate file.
- `gcn_distr.py` : 1D algorithm
- `gcn_distr_15d.py` : 1.5D algorithm
- `gcn_distr_2d.py` : 2D algorithm
- `gcn_distr_3d.py` : 3D algorithm

Each file also as the following flags:

- `--accperrank <int>` : Number of GPUs on each node
- `--epochs <int>`  : Number of epochs to run training
- `--graphname <Reddit/Amazon/subgraph3>` : Graph dataset to run training on
- `--timing <True/False>` : Enable timing barriers to time phases in training
- `--midlayer <int>` : Number of activations in the hidden layer
- `--runcount <int>` : Number of times to run training
- `--normalization <True/False>` : Normalize adjacency matrix in preprocessing
- `--activations <True/False>` : Enable activation functions between layers
- `--accuracy <True/False>` : Compute and print accuracy metrics (Reddit only)
- `--replication <int>` : Replication factor (1.5D algorithm only)
- `--download <True/False>` : Download the Reddit dataset

Some of these flags do not currently exist for the 3D algorithm.

Amazon/Protein datasets must exist as COO files in `../data/<graphname>/processed/`, compressed with pickle. 
For Reddit, PyG handles downloading and accessing the dataset (see below).

## Running with slurm on RI2 

Run the following command to download the ogbn-products dataset:
`python gcn_distr_15d.py --graphname='ogbn-products' --download=True`

This will download ogbn-products into `../data`. After downloading the ogbn-products dataset, run the following command to run 1.5D and transoposing benchmarks

`bash run_slurm.sh`

This script outamatically runs benchmarks for 1.5D and transpose. However it is set for 1 GPU per node, which might not be the case in other systems. Also this script tests for Reddit, ogbn-products, ogbn-mag and ogbn-arxiv. If some of these are not downloaded before it will cause runtime errors. Accelerator per gpu parameters can be changed in slurm_tr.sh and slurm_15d.sh scripts for other systems.

## Running with torch.distributed.launch on CHPC (example)

Run the following command to download the Reddit dataset:

`python gcn_distr_15d.py --graphname=Reddit --download=True`

This will download Reddit into `../data`. After downloading the Reddit dataset, run the following command to run training

To run with torch.distributed.launch, MASTER_PORT, MASTER_ADDR, WORLD_SIZE, RANK are required. The training script is setting them and this may cause some issues. I disabled the lines setting these environment variables and only passed them through the command below in an interactive job:

`python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=10.242.66.106 --master_port=61234 gcn_distr_15d.py --accperrank=1 --epochs=100 --graphname=Reddit --timing=True --midlayer=128 --runcount=1 --replication=2`

In a non-interactive job, the required environment variables can be obtained by 
`master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
master_port=12345
rank=$SLURM_PROCID
world_size=$SLURM_NTASKS`

Then they can be passed through a python command:

`python -m torch.distributed.launch --nproc_per_node=1 --nnodes=$world_size --node_rank=$rank --master_addr=$master_addr --master_port=$master_port gcn_distr_15d.py --accperrank=1 --epochs=100 --graphname=Reddit --timing=True --midlayer=128 --runcount=1 --replication=2`

## Running on OLCF Summit (example)

To run the CAGNET 1.5D algorithm on Reddit with
- 16 processes
- 100 epochs
- 16 hidden layer activations
- 2-factor replication

run the following command to download the Reddit dataset:

`python gcn_distr_15d.py --graphname=Reddit --download=True`

This will download Reddit into `../data`. After downloading the Reddit dataset, run the following command to run training

`ddlrun -x WORLD_SIZE=16 -x MASTER_ADDR=$(echo $LSB_MCPU_HOSTS | cut -d " " -f 3) -x MASTER_PORT=1234 -accelerators 6 python gcn_distr_15d.py --accperrank=6 --epochs=100 --graphname=Reddit --timing=False --midlayer=16 --runcount=1 --replication=2`

## Citation

To cite CAGNET, please refer to:

> Alok Tripathy, Katherine Yelick, Aydın Buluç. Reducing Communication in Graph Neural Network Training. Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis (SC’20), 2020.
