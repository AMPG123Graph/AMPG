import os
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Actor, WebKB, WikipediaNetwork

def get_trainer(params):
    dataset_name = params['task']
    split = params['index_split']

    if dataset_name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root='datasets/datasets_pyg/', geom_gcn_preprocess=True, name=dataset_name, transform=T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()]))
        data = dataset[0]
        data.train_mask = dataset[0].train_mask[:,int(split)]
        data.val_mask = dataset[0].val_mask[:,int(split)]
        data.test_mask = dataset[0].test_mask[:,int(split)]
        data.x = torch.eye(data.x.shape[0])
        data.adj_t = data.adj_t.t()
        params['in_channel']=data.x.shape[0]
        params['out_channel']=dataset.num_classes

    if dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root='datasets/datasets_pyg/', name=dataset_name, transform=T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()]))
        data = dataset[0]
        data.train_mask = dataset[0].train_mask[:,int(split)]
        data.val_mask = dataset[0].val_mask[:,int(split)]
        data.test_mask = dataset[0].test_mask[:,int(split)]
        params['in_channel']=data.num_features
        params['out_channel']=dataset.num_classes
    
    if dataset_name in ['Actor']:
        dataset = Actor(root='datasets/datasets_pyg/Actor', transform=T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()]))
        data = dataset[0]
        data.train_mask = dataset[0].train_mask[:,int(split)]
        data.val_mask = dataset[0].val_mask[:,int(split)]
        data.test_mask = dataset[0].test_mask[:,int(split)]
        params['in_channel']=data.num_features
        params['out_channel']=dataset.num_classes
    
    if dataset_name in ['Cora_full','CiteSeer_full','PubMed_full']:
        dataset = Planetoid(root='datasets/datasets_pyg/', name='%s'%(dataset_name.split('_')[0]), split=dataset_name.split('_')[-1], transform=T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()]))
        data = dataset[0]
        params['in_channel']=data.num_features
        params['out_channel']=dataset.num_classes

    if dataset_name in ['Cora_geom','CiteSeer_geom','PubMed_geom']:
        dataset = Planetoid(root='datasets/datasets_pyg/', name='%s'%(dataset_name.split('_')[0]), split='public', transform=T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()]))
        split_str = "%s_split_0.6_0.2_%s.npz"%(dataset_name.split('_')[0].lower(), str(split))
        split_file = np.load(os.path.join('datasets/datasets_geomgcn/', split_str))
        data = dataset[0]
        data.train_mask = torch.Tensor(split_file['train_mask'])==1
        data.val_mask = torch.Tensor(split_file['val_mask'])==1
        data.test_mask = torch.Tensor(split_file['test_mask'])==1
        params['in_channel']=data.num_features
        params['out_channel']=dataset.num_classes

    device = torch.device('cuda:%s'%(params['gpu_index']) if torch.cuda.is_available() else 'cpu')
    print("GPU device: [%s]"%(device))
    
    if params['model'] in ['ONGNN']:
        from model import GONN as Encoder
        model = Encoder(params).to(device)

    criterion = torch.nn.NLLLoss()
    
    if params['weight_decay2']=="None":
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    else:
        optimizer = torch.optim.Adam([dict(params=model.params_conv, weight_decay=params['weight_decay']),
                                      dict(params=model.params_others, weight_decay=params['weight_decay2'])],
                                     lr=params['learning_rate'])

    trainer = dict(zip(['data', 'device', 'model', 'criterion', 'optimizer', 'params'], [data, device, model, criterion, optimizer, params]))

    return trainer

def get_metric(trainer, stage):
    data, device, model, criterion, optimizer, params = trainer.values()

    if stage=='train':
        torch.set_grad_enabled(True)
        model.train()
    else:
        torch.set_grad_enabled(False)
        model.eval()
    
    data = data.to(device)

    for _, mask_tensor in data(stage+'_mask'):
        mask = mask_tensor
    encode_values = model(data.x, data.adj_t)
    vec = encode_values['x']
    pred = F.log_softmax(vec, dim=-1)
    loss = criterion(pred[mask], data.y[mask])
    
    if stage=='train':
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    acc = float((pred[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
    metrics = dict(zip(['metric', 'loss', 'encode_values'], [acc, loss.item(), encode_values]))

    return metrics