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
        if args.nonorm:
            self.layers.append(GraphConv(in_feats, n_hidden, activation=activation,norm='none',bias=False))
        else:
            self.layers.append(GraphConv(in_feats, n_hidden, activation=activation,norm='both',bias=False))
        # hidden layers
        #for i in range(n_layers - 1):
        #    self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation,norm='both',bias=False))
        # output layer
        if args.nonorm:
            self.layers.append(GraphConv(n_hidden, n_classes,norm='none',bias=False))
        else:
            self.layers.append(GraphConv(n_hidden, n_classes,norm='both',bias=False))

        self.dropout = nn.Dropout(p=dropout)

        if args.load:
            g = args.dataset.replace('ogbn-','')
            w1 = torch.nn.Parameter(torch.load('rdm_{}_w1.pt'.format(g)), requires_grad=True)
            w2 = torch.nn.Parameter(torch.load('rdm_{}_w2.pt'.format(g)), requires_grad=True)
            print(w1.shape)
            print(w1)
            print(w2.shape)
            print(w2)
            # w1 = torch.nn.Parameter(torch.ones(self.layers[0].weight.shape), requires_grad=True)
            # w2 = torch.nn.Parameter(torch.ones(self.layers[1].weight.shape), requires_grad=True)
            self.layers[0].weight = w1
            self.layers[1].weight = w2
        else:
            g = args.dataset.replace('ogbn-','')
            torch.save(self.layers[0].weight, 'dgl_{}_w1.pt'.format(g))
            torch.save(self.layers[1].weight, 'dgl_{}_w2.pt'.format(g))
        for l in self.layers:
            print(l.weight)
        print('*'*50)

    def forward(self, g):
        h = g.ndata['feat']
        #print(g)
        #print('adj:',g.adj())
        #torch.save(g.adj(), 'DGL_ADJ.pt')
        print('x:',h)
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            print(f'layer{i}W:', layer.weight)
            print(f'layer{i}:', h.shape, h)
            print('#'*30)
        #exit()
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
    if args.dataset == 'toy':
        adj_full = torch.tensor([[1,2,3,4,5],[0,1,2,3,4]]).long()
        x_ = torch.ones(6,1)
        y_ = torch.ones(6).long()
        trm_ = torch.ones(6).bool()
        tem_ = torch.ones(6).bool()   
    else:
        pref = '/scratch/general/nfs1/u1320844/dataset/asplos/{}_subgs/'.format(args.dataset)
        if args.topo=='sym':
            print('loading rdm saved adj')
            adj_full = torch.load(pref+'adj_full_sym.pt')
        else:
            adj_full = torch.load(pref+'adj_full.pt')
        x_ = torch.load(pref+'x_full.pt')
        y_ = torch.load(pref+'y_full.pt')
        trm_ = torch.load(pref+'train_mask_full.pt')
        tem_ = torch.load(pref+'test_mask_full.pt')
    in_ = adj_full[0]
    out_ = adj_full[1]
    if args.topo == 'upper':
        g = dgl.graph((in_, out_))
    elif args.topo == 'lower':
        g = dgl.graph((out_, in_))
    else: # symmetric
        g = dgl.graph((in_, out_))
    #g = dgl.to_bidirected(g)
    if args.self_loop:
        g = dgl.add_self_loop(g)
    print(g.adj())
    #exit()
    in_degrees = g.in_degrees()
    out_degrees = g.out_degrees()
    #torch.save(g.adj(), 'dgl_added_self_loop.pt')
    print(in_degrees)
    print(out_degrees)
    print(max(in_degrees))
    print(max(out_degrees))
    print(in_degrees==out_degrees)
    #if args.dataset == 'ogbn-arxiv':
    g.ndata['feat'] = x_
    # g.ndata['feat'] = torch.ones(x_.shape)
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
    nclassdict['toy'] = 2
    n_classes = nclassdict[args.dataset]#int(torch.max(g.ndata['label']))
    n_nodes = g.num_nodes()
    n_edges = g.num_edges()
    n_train_samples = trm_.int().sum().item()
    n_val_samples = 0
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
    #exit()
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
    for epoch in range(200):
        ep_start = time.time()
        model.train()
        if args.dump:
            torch.save(model.module.layers[0].weight, 'dgl_dumped_w1.pt')
            torch.save(model.module.layers[1].weight, 'dgl_dumped_w2.pt')
            #exit()
        # forward
        pred = model(g)
        print(pred)
        torch.save(pred, 'dgl_output.pt')
        #exit()
        batch_labels = g.ndata['label']
        loss = F.cross_entropy(pred[trm], batch_labels[trm].long(), reduction='mean')
        #print('loss:', loss.item())
        #print('grad:', loss.grad)
        #exit()
        optimizer.zero_grad()
        loss.backward()
#        torch.nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()
        dur.append(time.time()-ep_start)
        torch.cuda.empty_cache()
        #if epoch >=1:
        #    exit()
        
        #if rk == 0:# and j == len(loader) - 1 and epoch%10==0:
        #    model.eval()
        #    with torch.no_grad():
        #        #train_f1_mic, train_f1_mac = calc_f1(batch_labels.cpu().numpy(),
        #        #                                     pred.cpu().numpy(), False)
        #        train_f1_mic = eval(batch_labels[trm].cpu().numpy(),
        #                                             pred[trm].cpu().numpy(), False)
        #        print(f"epoch:{epoch + 1}/{args.n_epochs}, training loss:{loss.item()}")
        #        print("Train Acc {:.4f}".format(train_f1_mic))
        # evaluate
        if epoch % 1 == 0 and rk == 0:
            model.eval()
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
            
            logline = f'{args.dataset},dgl,{ws},{args.topo},{epoch},{np.sum(dur):.4f},{val_f1_mic:.4f}\n'
            #logline = f'{args.dataset},dgl,{ws},{epoch},{np.sum(dur):.4f},{loss.item():.4f},{val_f1_mic:.4f}\n'
            print(logline)
            if rk == 0:
                with open(args.csv,'a') as f :
                    f.write(logline)
            if cpu_flag and cuda:
                model.cuda()

    end_time = time.time()
    print(f'training using time {end_time - start_time}')

    if cpu_flag and cuda:
        model = model.to('cpu')

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='GraphSAINT')
    parser.add_argument("--dataset", type=str, default="ppi_n", help="type of tasks")
    parser.add_argument("--online", dest='online', action='store_true', help="sampling method in training phase")
    parser.add_argument("--self_loop", action='store_true', help="")
    parser.add_argument("--load", action='store_true', help="")
    parser.add_argument("--dump", action='store_true', help="")
    parser.add_argument("--nonorm", action='store_true', help="")
    parser.add_argument("--symmetric", action='store_true', help="")
    parser.add_argument("--n_hidden", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=50, help="the gpu index")
    parser.add_argument("--gpu", type=int, default=0, help="the gpu index")
    parser.add_argument("--lr", type=float, default=0.001, help="the gpu index")
    parser.add_argument("--csv", type=str, default='test.csv')
    parser.add_argument("--topo", type=str, default='upper')
    parser.add_argument("--log_dir", type=str, default='test')
    args = parser.parse_args()
    #task = parser.parse_args().task
    #args = argparse.Namespace(**CONFIG[task])
    #args.online = parser.parse_args().online
    #args.gpu = parser.parse_args().gpu
    #args.csv = parser.parse_args().csv
    print(args)

    main(args)
