import dgl
import torch
adj = torch.load('../pyg/arctic25_subgs/adj_full.pt')
print(adj)
x= torch.load('../pyg/arctic25_subgs/x_full.pt')
y = torch.load('../pyg/arctic25_subgs/y_full.pt')
g = dgl.graph((adj[0],adj[1]))
#ndata = {}
#ndata['feat'] = x
#ndata['label'] = y
#g.ndata = ndata
g.ndata['feat'] = x#torch.tensor(feats, dtype=torch.float)
g.ndata['label'] = y#torch.tensor(class_arr, dtype=torch.float if multilabel else torch.long)
#g.ndata['train_mask'] = torch.tensor(train_mask, dtype=torch.bool)
#g.ndata['val_mask'] = torch.tensor(val_mask, dtype=torch.bool)
#g.ndata['test_mask'] = torch.tensor(test_mask, dtype=torch.bool)
#
#data = DataType(g=g, num_classes=num_classes, train_nid=train_nid)
#return data
print(g)