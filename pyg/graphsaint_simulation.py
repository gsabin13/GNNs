import argparse
import enum
import os.path as osp

from tqdm import tqdm
from time import time
from random import shuffle

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.data import Data
from torch_geometric.datasets import Flickr, Reddit2, Reddit,PPI
from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborSampler
from torch_geometric.nn import GraphConv, SAGEConv
from torch_geometric.utils import degree
from ogb.nodeproppred import PygNodePropPredDataset


parser = argparse.ArgumentParser()
parser.add_argument('--use_normalization', action='store_true')
parser.add_argument('--load', action='store_true')
parser.add_argument('--dataset', type=str, default='flickr')
parser.add_argument('--batch_size', type=int, default=2000)
parser.add_argument('--walk_length', type=int, default=3)
parser.add_argument('--num_subgs', type=int, default=32)
parser.add_argument('--freq', type=int, default=2)
parser.add_argument('--sample_coverage', type=int, default=0)
parser.add_argument('--log', type=str, default='jun28/log.csv')
args = parser.parse_args()

if args.dataset == 'flickr':
    path = '/scratch/general/nfs1/u1320844/dataset/flickr'
    dataset = Flickr(path)
    data = dataset[0]
elif args.dataset == 'reddit':
    path = '/scratch/general/nfs1/u1320844/dataset'
    dataset = Reddit2(path)
    data = dataset[0]
elif args.dataset == 'ppi':
    path = '/scratch/general/nfs1/u1320844/dataset'
    dataset = PPI(path)
    data = dataset[0]
elif args.dataset == 'ogbn-arxiv':
    path = '/scratch/general/nfs1/u1320844/dataset'
    dataset = PygNodePropPredDataset(name=args.dataset,root=path)
    data = dataset[0]
    data.y = data.y.squeeze(dim=1)
    from torch_geometric.utils import to_undirected
    data.edge_index = to_undirected(data.edge_index)
    split_idx = dataset.get_idx_split()
    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        if key=='valid':
            key = 'val'
        data[f'{key}_mask'] = mask
elif args.dataset == 'ogbn-products':
    path = '/scratch/general/nfs1/u1320844/dataset'
    dataset = PygNodePropPredDataset(name=args.dataset, root=path)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    data.y = data.y.squeeze(dim=1)
    ## Convert split indices to boolean masks and add them to `data`.
    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        if key=='valid':
            key = 'val'
        data[f'{key}_mask'] = mask

row, col = data.edge_index
data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.
if args.load:
    loader = [] 
    pref = '{}_subgs/'.format(args.dataset)
    for i in range(args.num_subgs):
        adj = torch.load(pref+'adj_{}.pt'.format(i)) 
        tm = torch.load(pref+'train_mask_{}.pt'.format(i)) 
        x = torch.load(pref+'x_{}.pt'.format(i)) 
        y = torch.load(pref+'y_{}.pt'.format(i)) 
        ew = torch.load(pref+'edge_weight_{}.pt'.format(i)) 
        d = Data()
        d.edge_index = adj
        d.x = x 
        d.y = y 
        d.edge_weight = ew
        d.train_mask = tm
        #d = T.ToSparseTensor()(d)
        loader.append(d)
else:
    loader = GraphSAINTRandomWalkSampler(data, batch_size=args.batch_size, walk_length=args.walk_length,
                                     num_steps=args.num_subgs, sample_coverage=args.sample_coverage,
                                     save_dir=dataset.processed_dir,
                                     num_workers=0)

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

class Net(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        in_channels = dataset.num_node_features
        out_channels = dataset.num_classes
        conv1 = GraphConv(in_channels, hidden_channels)
        conv2 = GraphConv(hidden_channels, hidden_channels)
        conv3 = GraphConv(hidden_channels, hidden_channels)
        self.convs = nn.Sequential(conv1, 
                                   conv2,
                                   conv3)
        self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        for conv in self.convs:
            conv.aggr = aggr

    def forward(self, x0, edge_index, edge_weight=None):
        xs = []
        x = x0
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
            xs.append(x)
        #x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        #x1 = F.dropout(x1, p=0.2, training=self.training)
        #x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        #x2 = F.dropout(x2, p=0.2, training=self.training)
        #x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        #x3 = F.dropout(x3, p=0.2, training=self.training)
        x = torch.cat(xs, dim=-1)
        x = self.lin(x)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        #for i, conv in enumerate(self.convs):
        #    xs = []
        #    for batch_size, n_id, adj in subgraph_loader:
        #        edge_index, _, size = adj.to(device)
        #        x = x_all[n_id].to(device)
        #        x_target = x[:size[1]]
        #        x = conv((x, x_target), edge_index)
        #        if i != len(self.convs) - 1:
        #            x = F.relu(x)
        #        xs.append(x.cpu())
        #        pbar.update(batch_size)
        #    x_all = torch.cat(xs, dim=0)
        xs = []
        for batch_size, n_id, adj in subgraph_loader:
            edge_index, _, size = adj.to(device)
            x = x_all[n_id].to(device)
            x_target = x[:size[1]]
            x = self(x, edge_index)
            #x = self((x, x_target), edge_index)
            #if i != len(self.convs) - 1:
            #    x = F.relu(x)
            xs.append(x.cpu())
            pbar.update(batch_size)
        x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(data.x.size(-1),128,dataset.num_classes,2,0.2).to(device)
#model = Net(hidden_channels=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    #model.set_aggr('add' if args.use_normalization else 'mean')

    total_loss = total_examples = 0
    loss_ = 0
    for i, data in enumerate(loader):
        #data = T.ToSparseTensor()(data)
        data = data.to(device)
        if args.use_normalization:
            edge_weight = data.edge_norm * data.edge_weight
            #out = model(data.x, data.adj_t, edge_weight)
            out = model(data.x, data.edge_index, edge_weight)
            loss = F.nll_loss(out, data.y, reduction='none')
            loss = (loss * data.node_norm)[data.train_mask].sum()
        else:
            #out = model(data.x, data.adj_t)
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        #loss /= args.freq
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
        loss.backward()
        if i%args.freq == 0:
            optimizer.step()
            optimizer.zero_grad()
    return total_loss / total_examples


@torch.no_grad()
def test(args, data):
    model.eval()
    if args.dataset in ['s']:# ['reddit','ogbn-products']:
        # use batched inference
        subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                      batch_size=1024, shuffle=False,
                                      num_workers=12)
        out = model.inference(data.x, subgraph_loader, device)
        y_true = data.y
        y_pred = out.argmax(dim=-1, keepdim=True)
        y_pred = torch.squeeze(y_pred, dim=1)
        accs = []
        for mask in [data.train_mask.cpu().bool(), data.val_mask.cpu().bool(), data.test_mask.cpu().bool()]:
            accs.append(int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum()))
        return accs
    else:
        # use full graph inference
        #model.set_aggr('mean')
        data = T.ToSparseTensor()(data)
        out = model(data.x.to(device), data.adj_t.to(device))
        #out = model(data.x.to(device), data.edge_index.to(device))
        pred = out.argmax(dim=-1)
        correct = pred.eq(data.y.to(device))
        accs = []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            accs.append(correct[mask].sum().item() / mask.sum().item())
        return accs


total_time = 0
for epoch in range(50):
    #if args.load:
    #    shuffle(loader)
    ep_start = time()
    loss = train()
    total_time += time()-ep_start
    torch.cuda.empty_cache()
    accs = test(args, data)
    logline = f'{args.dataset},Simulation-{args.freq},{epoch:02d},{total_time:.4f},{loss:.4f},{accs[0]:.4f},{accs[1]:.4f},{accs[2]:.4f}\n'
    print(logline)
    with open(args.log, 'a') as f:
        f.write(logline)
