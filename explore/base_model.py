import torch
import numpy as np
import time
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from models import Explore
from utils import cal_ranks, cal_performance, cal_top1


class BaseModel(object):
    def __init__(self, args, loader):
        self.model = Explore(args, loader)
        self.model.cuda()

        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_rel = loader.n_rel
        self.n_batch = args.n_batch
        self.n_tbatch = args.n_tbatch

        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test  = loader.n_test
        self.n_layer = args.n_layer
        self.args = args

        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.smooth = 1e-5
        self.t_time = 0

    def change_loader(self, loader):
        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_rel = loader.n_rel
        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test  = loader.n_test
        self.model.change_loader(loader)

    def train_batch(self, evaluate=False):
        self.loader.shuffle_train()
        epoch_loss = 0
        i = 0

        batch_size = self.n_batch
        n_batch = self.loader.n_train // batch_size + (self.loader.n_train % batch_size > 0)
        print('n_batch:', n_batch)
        num_nodes = np.zeros((self.n_layer, 2))
        num_edges = np.zeros((self.n_layer, 2))
        t_time = time.time()
        self.model.train()

        if 'MetaQA/2-hop' in self.loader.task_dir or 'MetaQA/3-hop' in self.loader.task_dir:
            n_batch = n_batch // 10

        for i in tqdm(range(n_batch)):
            start = i*batch_size
            end = min(self.loader.n_train, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx) #subs, rels, objs

            self.model.zero_grad()
            n_nodes, n_edges, scores = self.model(subs, rels)

            num_nodes += n_nodes / n_batch
            num_edges += n_edges / n_batch

 
            pos_scores = scores[[torch.arange(len(scores)).cuda(),torch.LongTensor(objs).cuda()]]
            max_n = torch.max(scores, 1, keepdim=True)[0]
            loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1))) 

            loss.backward()
            self.optimizer.step()
            '''if i % 1000 == 0 :
                print('batch:',i, 'loss:', loss.item())'''
            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()
        self.scheduler.step()
        self.t_time += time.time() - t_time
        print('epoch_loss:',epoch_loss, 'time:', self.t_time)
        out_str2 = str(num_nodes.reshape(1,-1).astype(int)) + '\n'+ str(num_edges.reshape(1,-1).astype(int))+'\n' #'i1:%d  u2:%d  i2:%d  e2:%d  u3:%d  i3:%d  e3:%d\n'%(i1,u2,i2,e2,u3,i3,e3)
        # print(out_str2)
        if evaluate:
            t_h1, out_str = self.evaluate()
            return t_h1, out_str, out_str2
        else:
            return 0, ''

    def evaluate(self, ):
        batch_size = self.n_tbatch
        print('valid:')
        n_data = self.n_valid
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()
        i_time = time.time()
        num_nodes = np.zeros((self.n_layer, 2))
        num_edges = np.zeros((self.n_layer, 2))
        for i in tqdm(range(n_batch)):
            start = i*batch_size
            end = min(n_data, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='valid')
            scores = self.model(subs, rels, mode='valid').data.cpu().numpy()
            
            filters = 0

            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks
        ranking = np.array(ranking)
        v_mrr, v_h1,v_h3, v_h10 = cal_performance(ranking)

        print('test:')
        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()
        for i in tqdm(range(n_batch)):
            start = i*batch_size
            end = min(n_data, (i+1)*batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='test')
            scores = self.model(subs, rels, mode='test').data.cpu().numpy()
             
            filters = 0
            # ranks = cal_ranks(scores, objs, filters)
            ranks = cal_top1(scores, objs, filters)
            ranking += ranks
        ranking = np.array(ranking)
        t_mrr, t_h1, _, _ = cal_performance(ranking)
        assert len(ranking) == self.n_test
        i_time = time.time() - i_time

        out_str = '[VAL] H@1:%.4f H@10:%.4f\t[TEST] H@1:%.4f \t[TIME] train:%.4f inference:%.4f\n'%( v_h1, v_h10, t_h1, self.t_time, i_time)

        return t_h1, out_str


    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def get_path(self, mode='test', filepath='path.txt'):

        print('generate path:')
        batch_size = 1 
        if mode == 'test':
            n_data = self.n_test
        else:
            n_data = 500
        n = 10
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        f = open(filepath,'w')
        self.model.eval()
        for i in tqdm(range(n_batch)):
            start = i*batch_size 
            end = min(n_data, (i+1)*batch_size) 
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data=mode)
            self.model.visual_path(subs, rels, objs, filepath=filepath, mode=mode)
        
        print('path generate done in ' + filepath + '\n')
        

    def save_model(self, dataset='', out_str=''):
        torch.save(self.model.state_dict(), dataset+'_saved_model.pt')
        print(out_str, 'model saved')

    def load_model(self, model_name='_saved_model.pt'):
        self.model.load_state_dict(torch.load(model_name))

