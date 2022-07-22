# Reaches around 0.7870 ± 0.0036 test accuracy.

import enum
from operator import sub
import sys
import os.path as osp

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from tqdm import tqdm

from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import Reddit

def get_degree_distribution(data):
    deg = degree(data.edge_index[0], data.num_nodes)
    return deg

def calc_acc(y_true, y_pred):
    correct = y_pred.eq(y_true).sum().item()
    res = correct / len(y_true) 
    return res 

root = '/scratch/general/nfs1/u1320844/dataset/reddit'
graphname = sys.argv[1]
dataset = Reddit(root=root)
data = dataset[0]
train_idx = data.train_mask
print(train_idx.shape)
train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                               sizes=[15, 10, 5], batch_size=1024,
                               shuffle=True, num_workers=12)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=4096, shuffle=False,
                                  num_workers=12)

subgraph_loader_5 = NeighborSampler(data.edge_index, node_idx=None, sizes=[5],
                                  batch_size=4096, shuffle=False,
                                  num_workers=12)

subgraph_loader_10 = NeighborSampler(data.edge_index, node_idx=None, sizes=[10],
                                  batch_size=4096, shuffle=False,
                                  num_workers=12)
subgraph_loader_20 = NeighborSampler(data.edge_index, node_idx=None, sizes=[20],
                                  batch_size=4096, shuffle=False,
                                  num_workers=12)

deg = get_degree_distribution(data)[data.test_mask]
print(deg.shape)
#exit()
deg_b0 = (deg<2).nonzero()
deg_b1 = ((deg>=2) & (deg<4)).nonzero()
deg_b2 = ((deg>=4) & (deg<8)).nonzero()
deg_b3 = ((deg>=8) & (deg<16)).nonzero()
deg_b4 = ((deg>=16) & (deg<32)).nonzero()
deg_b5 = ((deg>=32) & (deg<64)).nonzero()
deg_b6 = ((deg>=64) & (deg<128)).nonzero()
deg_b7 = ((deg>=128) & (deg<256)).nonzero()
deg_b8 = ((deg>=256) & (deg<512)).nonzero()
deg_b9 = ((deg>=512) & (deg<1024)).nonzero()
deg_b10 = ((deg>=1024) & (deg<2048)).nonzero()
deg_b11 = ((deg>=2048) & (deg<4096)).nonzero()
deg_b12 = ((deg>=4096) & (deg<8192)).nonzero()
deg_b13 = (deg>=8192).nonzero()
print(deg_b0.shape)
print(deg_b1.shape)
print(deg_b2.shape)
print(deg_b3.shape)
print(deg_b4.shape)
print(deg_b5.shape)
print(deg_b6.shape)
print(deg_b7.shape)
print(deg_b8.shape)
print(deg_b9.shape)
print(deg_b10.shape)
print(deg_b11.shape)
print(deg_b12.shape)
print(deg_b13.shape)
buckets = [deg_b0, deg_b1, deg_b2, deg_b3, deg_b4, deg_b5, deg_b6, deg_b7, deg_b8, deg_b9, deg_b10,deg_b11, deg_b12, deg_b13]
print('{} buckets'.format(len(buckets)))
line0 = f'{sys.argv[1]},size:,0,1,2,3,4,5,6,7,8,9,10,11,12,13\n'
line1 = f'{sys.argv[1]},-,'
for b in buckets:
    line1 += f'{b.shape[0]},'
line1 += '\n'
with open('inf_s.csv', 'a') as f:
    f.write(line0)
    f.write(line1)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        pass
        #for conv in self.convs:
        #    conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, loader=subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)
model = model.to(device)

x = data.x.to(device)
y = data.y.squeeze().to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc



@torch.no_grad()
def test():
    model.eval()
    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc

@torch.no_grad()
def test_bucket(loader=subgraph_loader):
    model.eval()
    out = model.inference(x, loader=loader)
    y_true = y.cpu().unsqueeze(-1)
    print('y_true:', y_true.shape)
    y_pred = out.argmax(dim=-1, keepdim=True)
    print('y_pred:', y_pred.shape)
    accs = []
    for idx in buckets:
        #print('idx:',idx.shape)
        if idx.shape[0] == 0:
            acc = 0
        else:
            idx = idx.squeeze(1)
            #print('idx:',idx.shape)
            #exit()
            acc =  calc_acc(y_true[data.test_mask][idx], y_pred[data.test_mask][idx])
        accs.append(acc)
    acc_f = calc_acc(y_true[data.test_mask], y_pred[data.test_mask])
    accs.append(acc_f)
    return accs

test_accs = []
for run in range(1):
    print('')
    print(f'Run {run:02d}:')
    print('')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    best_val_acc = final_test_acc = 0
    for epoch in range(1, 21):
        loss, acc = train(epoch)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        if True :
            xx = ['full', '5', '10', '20']
            for i, loader in enumerate([subgraph_loader, subgraph_loader_5, subgraph_loader_10, subgraph_loader_20]):
                acc_b0, acc_b10_25, acc_b25_50, acc_b50_75, acc_b75_100, acc_b100_250, acc_b250_500, acc_b500_750, acc_b750_1000, acc_b1000_2500, acc_b2500_5000, acc_b5000_7500, acc_b7500_10000, acc_b10000, acc_full = test_bucket(loader=loader)
                log = f'{sys.argv[1]}, {xx[i]},ep{epoch},{acc_b0:.4f},{acc_b10_25:.4f},{acc_b25_50:.4f},{acc_b50_75:.4f},{acc_b75_100:.4f},{acc_b100_250:.4f},{acc_b250_500:.4f},{acc_b500_750:.4f},{acc_b750_1000:.4f},{acc_b1000_2500:.4f},{acc_b2500_5000:.4f},{acc_b5000_7500:.4f},{acc_b7500_10000:.4f},{acc_b10000:.4f},{acc_full:.4f}\n'
                if not osp.exists('inf_s.csv'):
                    with open('inf_s.csv', 'a') as f:
                        header = 'Dataset,Eval_fanout,10^0,10^1,10^2,10^3,10^4,All\n'
                        f.write(header)
                with open('inf_s.csv', 'a') as f:
                    f.write(log)

        #if epoch > 5:
        #    train_acc, val_acc, test_acc = test()
        #    print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
        #          f'Test: {test_acc:.4f}')

        #    if val_acc > best_val_acc:
        #        best_val_acc = val_acc
        #        final_test_acc = test_acc
    #test_accs.append(final_test_acc)

#test_acc = torch.tensor(test_accs)
#print('============================')
#print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
#