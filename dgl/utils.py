import json
import os
from functools import namedtuple
import scipy.sparse
from sklearn.preprocessing import StandardScaler
import dgl
import numpy as np
import torch
from sklearn.metrics import f1_score
def pcp_mag(g):
    src_cites, dst_cites = g.all_edges(etype="cites")
    new_g = dgl.heterograph({
        ("paper", "cites", "paper"): (src_cites, dst_cites)
    })
    new_g.nodes["paper"].data["feat"] = g.nodes["paper"].data["feat"]
    target_type_id = g.get_ntype_id("paper")
    #g = dgl.to_homogeneous(g, ndata=["feat"])
    #g = dgl.add_reverse_edges(g, copy_ndata=True)
    ## Mask for paper nodes
    #g.ndata["target_mask"] = g.ndata[dgl.NTYPE] == target_type_id
    print(g.ndata['labels'])
    new_g.ndata['label'] = g.ndata['labels']['paper']
    return new_g



class Logger(object):
    '''A custom logger to log stdout to a logging file.'''
    def __init__(self, path):
        """Initialize the logger.

        Parameters
        ---------
        path : str
            The file path to be stored in.
        """
        self.path = path

    def write(self, s):
        with open(self.path, 'a') as f:
            f.write(str(s))
        print(s)
        return


def save_log_dir(args):
    log_dir = './log/{}/{}'.format(args.dataset, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def eval(y_true, y_pred, multilabel):
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)

    acc = y_pred.eq(y_true).sum().item() / y_true.bool().sum().item()
    return acc#f1_score(y_true, y_pred, average="micro")

def calc_f1(y_true, y_pred, multilabel):
    if multilabel:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average="micro"), \
        f1_score(y_true, y_pred, average="macro")


def evaluate(model, g, labels, mask, multilabel=False):
    model.eval()
    with torch.no_grad():
        logits = model(g)
        logits = logits[mask]
        labels = labels[mask]
        logits = torch.argmax(logits, dim=1)
        #print('logits:', logits)
        #print('labels:', labels)
        #print('mask:', mask.bool().sum().item())
        #print('correct:',logits.eq(labels).sum().item() )
        #exit()
        acc = logits.eq(labels).sum().item() / mask.bool().sum().item()
        #acc = eval(labels.cpu().numpy(),
        #                         logits.cpu().numpy(), multilabel)
        #y_pred = np.argmax(logits.cpu().numpy(), axis=1)
        #y_true = labels
        #acc = f1_score(y_true.cpu().numpy(), y_pred, average='micro')
        return acc 


# load data of GraphSAINT and convert them to the format of dgl
def load_mag(args, multilabel):
    DataType = namedtuple('Dataset', ['num_classes', 'train_nid', 'g'])
    from ogb.nodeproppred import DglNodePropPredDataset
    print('load', args.dataset)
    data = DglNodePropPredDataset(name=args.dataset,root='/scratch/general/nfs1/u1320844/dataset')
    print('finish loading', args.dataset)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    graph.ndata['features'] = graph.ndata['feat']
    graph.ndata['labels'] = labels
    graph.ndata['label'] = labels
    in_feats = 128
    num_labels = 349
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    labels = labels["paper"]
    train_nid = train_nid["paper"]
    val_nid = val_nid["paper"]
    test_nid = test_nid["paper"]
    graph = pcp_mag(graph)
    train_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    feats = graph.ndata['feat']#np.load('{}/feats.npy'.format(prefix))
    scaler = StandardScaler()
    scaler.fit(feats[train_nid])
    feats = scaler.transform(feats)

    #class_map = json.load(open('{}/class_map.json'.format(prefix)))
    #class_map = {int(k): v for k, v in class_map.items()}

    num_nodes = graph.num_nodes()
    labels = graph.ndata['label']
    class_map = {}
    for i in range(num_nodes):
        class_map[i] = labels[i]

    #num_classes = max(class_map.values()) - min(class_map.values()) + 1
    num_classes=349
    class_arr = np.zeros((num_nodes,))
    for k, v in class_map.items():
        class_arr[k] = v

    graph.ndata['feat'] = torch.tensor(feats, dtype=torch.float)
    graph.ndata['label'] = torch.tensor(class_arr, dtype=torch.float if multilabel else torch.long)
    graph.ndata['train_mask'] = torch.tensor(train_mask, dtype=torch.bool)
    graph.ndata['val_mask'] = torch.tensor(val_mask, dtype=torch.bool)
    graph.ndata['test_mask'] = torch.tensor(test_mask, dtype=torch.bool)

    data = DataType(g=graph, num_classes=num_classes, train_nid=train_nid)
    return data

def load_data(args, multilabel):
    if 'mag' in args.dataset:
        return load_mag(args,multilabel)
    if not os.path.exists('graphsaintdata') and not os.path.exists('data'):
        raise ValueError("The directory graphsaintdata does not exist!")
    elif os.path.exists('graphsaintdata') and not os.path.exists('data'):
        os.rename('graphsaintdata', 'data')
    prefix = "/scratch/general/nfs1/u1320844/dataset/graphsaint/{}".format(args.dataset)
    DataType = namedtuple('Dataset', ['num_classes', 'train_nid', 'g'])

    adj_full = scipy.sparse.load_npz('{}/adj_full.npz'.format(prefix)).astype(np.bool)
    g = dgl.from_scipy(adj_full)
    num_nodes = g.num_nodes()

    adj_train = scipy.sparse.load_npz('{}/adj_train.npz'.format(prefix)).astype(np.bool)
    train_nid = np.array(list(set(adj_train.nonzero()[0])))

    role = json.load(open('{}/role.json'.format(prefix)))
    mask = np.zeros((num_nodes,), dtype=bool)
    train_mask = mask.copy()
    train_mask[role['tr']] = True
    val_mask = mask.copy()
    val_mask[role['va']] = True
    test_mask = mask.copy()
    test_mask[role['te']] = True

    feats = np.load('{}/feats.npy'.format(prefix))
    scaler = StandardScaler()
    scaler.fit(feats[train_nid])
    feats = scaler.transform(feats)

    class_map = json.load(open('{}/class_map.json'.format(prefix)))
    class_map = {int(k): v for k, v in class_map.items()}
    if multilabel:
        # Multi-label binary classification
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_nodes, num_classes))
        for k, v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_nodes,))
        for k, v in class_map.items():
            class_arr[k] = v

    g.ndata['feat'] = torch.tensor(feats, dtype=torch.float)
    g.ndata['label'] = torch.tensor(class_arr, dtype=torch.float if multilabel else torch.long)
    g.ndata['train_mask'] = torch.tensor(train_mask, dtype=torch.bool)
    g.ndata['val_mask'] = torch.tensor(val_mask, dtype=torch.bool)
    g.ndata['test_mask'] = torch.tensor(test_mask, dtype=torch.bool)

    data = DataType(g=g, num_classes=num_classes, train_nid=train_nid)
    return data
