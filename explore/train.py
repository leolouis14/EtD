import os
import argparse
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel


parser = argparse.ArgumentParser(description="Parser for DualR")
parser.add_argument('--dataset', type=str, default='MetaQA/1-hop')
parser.add_argument('--load', action='store_true')
parser.add_argument('--seed', type=str, default=1234)
parser.add_argument('--K', type=int, default=50)
parser.add_argument('--gpu', type=int, default=0)


args = parser.parse_args()

class Options(object):
    pass

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.dataset
   
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    opts = Options
    opts.perf_file = os.path.join(results_dir,  dataset.replace('/','-') + '_perf.txt')

    gpu = args.gpu
    torch.cuda.set_device(gpu)
    # print('gpu:', gpu)

    if dataset == 'MetaQA/1-hop':
        opts.lr = 0.00005
        opts.decay_rate = 0.996
        opts.lamb = 0.00001
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 1
        opts.dropout = 0.1
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 40
        loaders = [DataLoader(args.dataset)]
    elif dataset == 'MetaQA/2-hop':
        opts.lr = 0.00004
        opts.decay_rate = 0.998
        opts.lamb = 0.00014
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 2
        opts.dropout = 0.1
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 60
        loaders = [DataLoader(args.dataset)]
    elif dataset == 'MetaQA/3-hop':
        opts.lr = 0.00002
        opts.decay_rate = 0.994
        opts.lamb = 0.00014
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.1
        opts.act = 'idd' 
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 100
        loaders = [DataLoader(args.dataset)]
    elif dataset == 'webqsp':  
        opts.lr = 0.00001
        opts.decay_rate = 0.9991
        opts.lamb = 0.00001
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.1
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 200
        loaders = [DataLoader(args.dataset)]
    elif dataset == 'CWQ':
        opts.lr = 0.00001
        opts.decay_rate = 0.993
        opts.lamb = 0.0001
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.1
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 200
        loaders = [DataLoader(args.dataset)]
    elif dataset == 'WebCWQ':    # pretrain
        opts.lr = 0.0001
        opts.decay_rate = 0.9968
        opts.lamb = 0.00001
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.2
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 200
        loaders = [DataLoader('webqsp_nsm'), DataLoader('CWQ')]
    else:    # all
        opts.lr = 0.0001
        opts.decay_rate = 0.9968
        opts.lamb = 0.00008
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.1
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 200
        loaders = [DataLoader('MetaQA/1-hop'),DataLoader('MetaQA/2-hop'),DataLoader('MetaQA/3-hop'),DataLoader('webqsp_nsm'), DataLoader('CWQ')]

    opts.sample = 1
    opts.n_ent = loaders[0].n_ent
    opts.n_rel = loaders[0].n_rel
    
    config_str = '%d, %d, %.6f, %.4f, %.6f,  %d, %d, %d, %d, %.4f,%s\n' % (opts.sample, opts.K, opts.lr, opts.decay_rate, opts.lamb, opts.hidden_dim, opts.attn_dim, opts.n_layer, opts.n_batch, opts.dropout, opts.act)
    print(config_str.strip())
    model = BaseModel(opts, loaders[0])
    
    with open(opts.perf_file, 'a+') as f:
        f.write(config_str)
    if args.load:
        checkpoint = 'WebCWQ_saved_model.pt'
        model.load_model(checkpoint)
        with open(opts.perf_file, 'a+') as f:
            f.write('[load-WebCWQ]'+checkpoint+ '\n')
    
    best_h1 = 0
    for epoch in range(40):
        evaluate = epoch % 1 == 0
        for i in range(len(loaders)):
            loader = loaders[i]
            if len(loaders) != 1:
                model.change_loader(loader=loader)
            h1, out_str, out_str2 = model.train_batch(evaluate=evaluate)
            with open(opts.perf_file, 'a+') as f:
                se = f'{epoch:<3d}+[{i+1}]'
                f.write(se+out_str)
                print(se+out_str)
        if dataset != 'WebCWQ':
            if h1 > best_h1:
                dataset = dataset.replace('/','-')
                model.get_path(mode='test', filepath=dataset + '-path.txt')                
        else:
            if epoch == 19:
                model.save_model(dataset, out_str=f'hit@1:{h1}')
                break
        if h1 > best_h1:
            best_h1 = h1   
            best_str = out_str
            print(str(epoch) + '\t' + best_str)
        
    with open(opts.perf_file, 'a+') as f:
        f.write('best:\n'+ best_str)
    print(best_str)

