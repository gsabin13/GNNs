import argparse
#from importlib.metadata import requires
import os
from sys import prefix
import time
import dgl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sampler import SAINTNodeSampler, SAINTEdgeSampler, SAINTRandomWalkSampler
from config import CONFIG
from modules import GCNNet
from utils import Logger, evaluate, save_log_dir, load_data, calc_f1, eval
import warnings
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from torch import nn
from dgl.nn.pytorch import GraphConv
import random
torch.manual_seed(0)
#random.seed(0)
#np.random.seed(0)

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        torch.manual_seed(0)
        #random.seed(0)
        #np.random.seed(0)
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation,norm='none',bias=False))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation,norm='none',bias=False))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes,norm='none',bias=False))
        self.dropout = nn.Dropout(p=dropout)
        #w1 = torch.nn.Parameter(torch.rand(self.layers[0].weight.shape), requires_grad=True)
        #w2 = torch.nn.Parameter(torch.rand(self.layers[1].weight.shape), requires_grad=True)
        w1 = torch.nn.Parameter(torch.load('w1.pt'), requires_grad=True)
        w2 = torch.nn.Parameter(torch.load('w2.pt'), requires_grad=True)
        print(w1.shape)
        print(w1)
        print(w2.shape)
        print(w2)
        self.layers[0].weight = w1
        self.layers[1].weight = w2
        for l in self.layers:
            print(l.weight)
        print('*'*50)

    def forward(self, g):
        h = g.ndata['feat']
        print(g)
        print('adj:',g.adj())
        #torch.save(g.adj(), 'DGL_ADJ.pt')
        print('x:',h)
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            print(f'layer{i}W:', layer.weight)
            print(f'layer{i}:', h.shape, h)
            print('#'*30)
        exit()
        return h



def main(args):
    rk = int(os.environ['RANK'])
    ws = int(os.environ['WORLD_SIZE']) 
    setup(rank=rk, world_size=ws)
    print('Inited proc group')
    warnings.filterwarnings('ignore')
    if args.dataset in ['amazon', 'reddit', 'ogbn-products']:
        cpu_flag = False 
    else:
        cpu_flag = False
    pref_reddit = '../../reddit_subgs/'.format(args.dataset)
    pref = '/scratch/general/nfs1/u1320844/dataset/asplos/{}_subgs/'.format(args.dataset)
    subgraphs = []
    adj_full = torch.load(pref+'adj_full.pt')
    x_ = torch.load(pref+'x_full.pt')
    y_ = torch.load(pref+'y_full.pt')
    trm_ = torch.load(pref+'train_mask_full.pt')
    tem_ = torch.load(pref+'test_mask_full.pt')
    g = dgl.graph((adj_full[0], adj_full[1]))
#    g = dgl.add_self_loop(g)
    if args.dataset == 'ogbn-arxiv':
        g = dgl.to_bidirected(g)
    #g.ndata['feat'] = x_
    g.ndata['feat'] = torch.ones(x_.shape)
    g.ndata['label'] = y_
    print('Max label:', torch.max(y_))
    g.ndata['train_mask'] = trm_
    g.ndata['test_mask'] = tem_
    in_feats = g.ndata['feat'].shape[1]
    nclassdict = {}
    nclassdict['reddit'] = 41
    nclassdict['ogbn-arxiv'] = 40
    nclassdict['ogbn-products'] = 47
    nclassdict['meta'] = 25 
    nclassdict['arctic25'] = 33
    nclassdict['oral'] = 32 
    n_classes = nclassdict[args.dataset]#int(torch.max(g.ndata['label']))
    n_nodes = g.num_nodes()
    n_edges = g.num_edges()
    n_train_samples = trm_.int().sum().item()
    n_val_samples = 0#vam_.int().sum().item()
    n_test_samples = tem_.int().sum().item()

    print("""----Data statistics------'
    #Nodes %d
    #Edges %d
    #Classes/Labels (multi binary labels) %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
          (n_nodes, n_edges, n_classes,
           n_train_samples,
           n_val_samples,
           n_test_samples))
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()
        print("Running on: "+str(device_id))
        device = torch.device(device_id)
        test_mask = tem_.to(device)
        if not cpu_flag:
            g = g.to(device)

    model = GCN(in_feats=in_feats, n_hidden=args.n_hidden, n_classes=n_classes, n_layers=1,activation=nn.functional.relu, dropout=0.1)
    print(model)
    # TODO mv model to rank
    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id])

    # logger and so on
    log_dir = save_log_dir(args)
    logger = Logger(os.path.join(log_dir, 'loggings'))
    logger.write(args)

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr)

    start_time = time.time()
    best_f1 = -1

    dur = []
    g = g.to(device)
    trm = trm_.to(device)
    for epoch in range(2000):
        ep_start = time.time()
        model.train()
        # forward
        pred = model(g)
        print(pred)
        exit()
        batch_labels = g.ndata['label']
        loss = F.cross_entropy(pred[trm], batch_labels[trm].long(), reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()
        dur.append(time.time()-ep_start)
        torch.cuda.empty_cache()
        if rk == 0:# and j == len(loader) - 1 and epoch%10==0:
            model.eval()
            with torch.no_grad():
                #train_f1_mic, train_f1_mac = calc_f1(batch_labels.cpu().numpy(),
                #                                     pred.cpu().numpy(), False)
                train_f1_mic = eval(batch_labels[trm].cpu().numpy(),
                                                     pred[trm].cpu().numpy(), False)
                print(f"epoch:{epoch + 1}/{args.n_epochs}, training loss:{loss.item()}")
                print("Train Acc {:.4f}".format(train_f1_mic))
        # evaluate
        model.eval()
        if epoch % 1 == 0 and rk == 0:
            if cpu_flag and cuda:  # Only when we have shifted model to gpu and we need to shift it back on cpu
                model = model.to('cpu')
            #val_f1_mic, val_f1_mac = evaluate(
            #    model, g, labels, val_mask, multilabel)
            #g = g.to(device)
            val_f1_mic= evaluate(
                model, g, g.ndata['label'], test_mask, False)
            print(
                " Acc {:.4f}".format(val_f1_mic))
            if val_f1_mic > best_f1:
                best_f1 = val_f1_mic
                print('new best val f1:', best_f1)
                #torch.save(model.state_dict(), os.path.join(
                #    log_dir, 'best_model_{}.pkl'.format(task)))
            logline = f'{args.dataset},dgl,{ws},{epoch},{np.sum(dur):.4f},{val_f1_mic:.4f}\n'
            #logline = f'{args.dataset},dgl,{ws},{epoch},{np.sum(dur):.4f},{loss.item():.4f},{val_f1_mic:.4f}\n'
            print(logline)
            if rk == 0:
                with open(args.csv,'a') as f :
                    f.write(logline)
            if cpu_flag and cuda:
                model.cuda()

    end_time = time.time()
    print(f'training using time {end_time - start_time}')

    # test
    #if args.use_val:
    #    model.load_state_dict(torch.load(os.path.join(
    #        log_dir, 'best_model_{}.pkl'.format(task))))
    if cpu_flag and cuda:
        model = model.to('cpu')
    test_f1_mic, test_f1_mac = evaluate(
        model, g, g.ndata['label'], test_mask, False)
    print("Test F1-mic {:.4f}, Test F1-mac {:.4f}".format(test_f1_mic, test_f1_mac))

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='GraphSAINT')
    parser.add_argument("--dataset", type=str, default="ppi_n", help="type of tasks")
    parser.add_argument("--online", dest='online', action='store_true', help="sampling method in training phase")
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=50, help="the gpu index")
    parser.add_argument("--gpu", type=int, default=0, help="the gpu index")
    parser.add_argument("--lr", type=float, default=0.001, help="the gpu index")
    parser.add_argument("--csv", type=str, default='test.csv')
    parser.add_argument("--log_dir", type=str, default='test')
    args = parser.parse_args()
    #task = parser.parse_args().task
    #args = argparse.Namespace(**CONFIG[task])
    #args.online = parser.parse_args().online
    #args.gpu = parser.parse_args().gpu
    #args.csv = parser.parse_args().csv
    print(args)

    main(args)
