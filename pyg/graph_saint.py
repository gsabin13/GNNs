import argparse
import os

import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch_geometric.data import GraphSAINTRandomWalkSampler, NeighborSampler
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.utils import subgraph
from torch_geometric.datasets import Reddit2, Reddit, AmazonProducts, Flickr

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger
from time import time
import numpy as np
def setup(rank, world_size):
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return torch.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv(x, edge_index)
                #x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all



class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        return torch.log_softmax(x, dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all


def train(model, loader, optimizer, device):
    model.train()

    total_loss = 0
    for i, data in enumerate(loader):
        #print('Saving subg {}'.format(i))
        #pref = 'prod_subgs/'
        #torch.save(data.x, pref+'feat{}.pt'.format(i))
        #torch.save(data.edge_index, pref+'adj{}.pt'.format(i))
        #torch.save(data.y, pref+'label{}.pt'.format(i))
        #torch.save(data.train_mask, pref+'mask{}.pt'.format(i))
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        #print('label shape:', data.y.shape)
        y = data.y#.squeeze(1)
        #y = data.y
        loss = F.nll_loss(out[data.train_mask], y[data.train_mask].long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    #print('All subgs saved')
    return total_loss / len(loader)

@torch.no_grad()
def testr(model, data):
    model.eval()
    x = data.x.to(torch.device('cuda:0'))
    y = data.y.to(torch.device('cuda:0'))
    #print('x dtype:', x.dtype)
    #out = model(x.float(), data.edge_index.to(torch.device('cuda:0')))
    data = T.ToSparseTensor()(data)
    out = model(x.float(), data.edge_index.to(torch.device('cuda:0')))
    #out = model(x.float(), data.adj_t.to(torch.device('cuda:0')))

    y_true = y.cpu()#.unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True).detach().cpu()
    y_pred = torch.squeeze(y_pred, dim=1)
    print('pred shape:', y_pred.shape)
    print('gt shape:', y_true.shape)
    results = []
    for mask in [data.train_mask.cpu().bool(), data.val_mask.cpu().bool(), data.test_mask.cpu().bool()]:
        print('correct:',int(y_pred[mask].eq(y_true[mask]).sum()))
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]
    train_acc, valid_acc, test_acc = results

    return train_acc, valid_acc, test_acc

@torch.no_grad()
def test(model, data, subgraph_loader, device, args):
    model.eval()
    out = model.module.inference(data.x, subgraph_loader, device)
    #out = model(data.x, data.edge_index)
    y_true = data.y
    y_pred = out.argmax(dim=-1, keepdim=True)
    y_pred = torch.squeeze(y_pred, dim=1)
    results = []
    k = data.val_mask if not 'ogb' in args.dataset else data.valid_mask
    for mask in [data.train_mask.cpu().bool(), k.cpu().bool(), data.test_mask.cpu().bool()]:
        #print('correct:',int(y_pred[mask].eq(y_true[mask]).sum()))
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]
    train_acc, valid_acc, test_acc = results

    #train_acc = evaluator.eval({
    #    'y_true': y_true[data.train_mask],
    #    'y_pred': y_pred[data.train_mask]
    #})['acc']
    #valid_acc = evaluator.eval({
    #    'y_true': y_true[data.valid_mask],
    #    'y_pred': y_pred[data.valid_mask]
    #})['acc']
    #test_acc = evaluator.eval({
    #    'y_true': y_true[data.test_mask],
    #    'y_pred': y_pred[data.test_mask]
    #})['acc']

    return train_acc, valid_acc, test_acc


def to_inductive(data):
    mask = data.train_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = data.train_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (GraphSAINT)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--inductive', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--walk_length', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_steps', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--eval_steps', type=int, default=2)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='ogbn-products')
    parser.add_argument("--local_rank", type=int, default=30)
    args = parser.parse_args()
    print(args)

    rk = int(os.environ['RANK'])
    ws = int(os.environ['WORLD_SIZE']) 
    #ds = args['dataset']
    setup(rank=rk, world_size=ws)
    print('Inited proc group')
    #device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    #device = torch.device(device)

    #dataset = AmazonProducts('/scratch/general/nfs1/u1320844/dataset/amazon')#,transform=T.ToSparseTensor())
    #data = dataset[0]
    if args.dataset == 'ogbn-products':
        dataset = PygNodePropPredDataset('ogbn-products',root='/scratch/general/nfs1/u1320844/dataset')#, transform=T.ToSparseTensor())
    elif args.dataset == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset('ogbn-arxiv',root='/scratch/general/nfs1/u1320844/dataset')#, transform=T.ToSparseTensor())
    elif args.dataset == 'ogbn-papers100M':
        dataset = PygNodePropPredDataset('ogbn-papers100M',root='/scratch/general/nfs1/u1320844/dataset' )
    elif args.dataset == 'reddit':
        dataset = Reddit2('/scratch/general/nfs1/u1320844/dataset')#,transform=T.ToSparseTensor())
    elif args.dataset == 'flickr':
        dataset = Flickr('/scratch/general/nfs1/u1320844/dataset/flickr')#,transform=T.ToSparseTensor())
    data = dataset[0]
    if args.dataset == 'ogbn-arxiv':
        from torch_geometric.utils import to_undirected
        data.edge_index = to_undirected(data.edge_index)
    #edge_idx = data.edge_index
    #data = T.ToSparseTensor()(data)
    #data.edge_index = edge_idx
    if 'ogb' in args.dataset:
        split_idx = dataset.get_idx_split()
        data.y = data.y.squeeze(dim=1)
        ## Convert split indices to boolean masks and add them to `data`.
        for key, idx in split_idx.items():
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            data[f'{key}_mask'] = mask

    # We omit normalization factors here since those are only defined for the
    # inductive learning setup.
    sampler_data = data
    if args.inductive:
        sampler_data = to_inductive(data)

    loader = GraphSAINTRandomWalkSampler(sampler_data,
                                         batch_size=args.batch_size,
                                         walk_length=args.walk_length,
                                         num_steps=args.num_steps,
                                         sample_coverage=0,
                                         save_dir=dataset.processed_dir,
                                         num_workers=0)
    loader = []
    for data in l:
        loader.append(data)
    chunk_size = len(loader)//ws
    if rk == ws-1:
        loader = loader[rk*chunk_size:]
    else:
        loader = loader[rk*chunk_size:(rk+1)*chunk_size]

    #model = GCN(data.x.size(-1), args.hidden_channels, dataset.num_classes,
    #             args.num_layers, args.dropout).to(device)
    model = SAGE(data.x.size(-1), args.hidden_channels, dataset.num_classes,
                 args.num_layers, args.dropout)#.to(device)
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    print("Running GNN on: "+str(rank))
    model = model.to(rank)
    model = DDP(model, device_ids=[device_id])
    device = torch.device(device_id)
    print(model)
    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                      batch_size=4096, shuffle=False,
                                      num_workers=12)
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        #model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        ep_time = []
        for epoch in range(args.epochs):
            #print('{} subgraphs'.format(loader))
            ep_s = time()
            loss = train(model, loader, optimizer, device)
            ep_time.append(time()-ep_s)
            if rank == 0 and epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}')
            if rank == 0 and epoch > 0:
                #result = testr(model, data)
                result = test(model, data, subgraph_loader, device, args)
                logger.add_result(run, result)
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
                with open('jun26/{}_{}GPU_online_logs.csv'.format(args.dataset, ws), 'a') as f:
                    l = f'{args.dataset},{ws},{epoch},{np.sum(ep_time):.4f},{train_acc:.4f},{valid_acc:.4f},{test_acc:.4f}\n'
                    f.write(l)
        logger.add_result(run, result)
        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
