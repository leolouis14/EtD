import os, re, csv
import json, pickle
import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict

class DataLoader:
    def __init__(self, task_dir):  # task_dir: 'MetaQA/1-hop'
        self.task_dir = '../data/' + task_dir

        self.entity2id = dict()
        self.id2entity = dict()
        self.kbid2id = dict()
        self.relation2id = dict()
        self.question2id = dict()
        self.n_ent, self.n_rel = 0,0
        self.n_qs = 0

        if task_dir.startswith('MetaQA'):
            self.fact_triple = self.read_metakb('../data/MetaQA/kb.txt')    # [[h,r,t],]

            self.train_data = self.read_qa(self.task_dir+'/ntm/qa_train.txt') # [[h,q,a1],]
            self.n_train_qs = self.n_qs
            #print(self.n_qs)
            self.valid_data = self.read_qa(self.task_dir+'/ntm/qa_dev.txt')
            self.n_valid_qs = self.n_qs - self.n_train_qs
            #print(nqs_val)
            self.test_data = self.read_qa(self.task_dir+'/ntm/qa_test.txt')
            self.n_test_qs = self.n_qs - self.n_valid_qs - self.n_train_qs
            #print(nqs_tes)
            self.id2entity = {v:k for k,v in self.entity2id.items()}
            self.id2relation = {v:k for k,v in self.relation2id.items()}
            self.id2question = {v:k for k,v in self.question2id.items()} 
            
        elif task_dir.startswith('webqsp') :
            
            web = 1
            self.fact_triple = defaultdict(lambda:list())
            self.n_ent, self.n_rel = 1441420, 6102
            self.map_ent_and_rel(web)
            print('map ent and rel done')
            # print(len(self.entity2id),len(self.relation2id))  
            self.train_data = self.read_web_qa('train', web)
            print('n_train:', len(self.train_data))
            self.n_train_qs = self.n_qs
            print('train qs: ',self.n_qs)
            self.valid_data = self.read_web_qa('dev', web)
            self.n_valid_qs = self.n_qs - self.n_train_qs
            print('valid qs: ', self.n_valid_qs)
            self.test_data = self.read_web_qa('test', web)
            self.n_test_qs = self.n_qs - self.n_valid_qs - self.n_train_qs
            print('test qs:', self.n_test_qs)
            # self.id2entity = {v:k for k,v in self.entity2id.items()}
            self.id2relation = {v:k for k,v in self.relation2id.items()}
            self.id2question = {v:k for k,v in self.question2id.items()} 
        
        elif task_dir.startswith('CWQ') :
            
            web = 2
            self.fact_triple = defaultdict(lambda:list())
            self.n_ent, self.n_rel = 2429346, 6649
            self.map_ent_and_rel(web)
            print('map ent and rel done')
            # print(len(self.entity2id),len(self.relation2id))  # 2428148 13299
            self.train_data = self.read_web_qa('train', web)
            print('n_train:', len(self.train_data))
            self.n_train_qs = self.n_qs
            print('train qs: ',self.n_qs)
            self.valid_data = self.read_web_qa('dev',web)
            self.n_valid_qs = self.n_qs - self.n_train_qs
            print('valid qs: ', self.n_valid_qs)
            self.test_data = self.read_web_qa('test', web)
            self.n_test_qs = self.n_qs - self.n_valid_qs - self.n_train_qs
            print('test qs:', self.n_test_qs)
            # self.id2entity = {v:k for k,v in self.entity2id.items()}
            self.id2relation = {v:k for k,v in self.relation2id.items()}
            self.id2question = {v:k for k,v in self.question2id.items()} 


        if task_dir.startswith('webqsp') or task_dir.startswith('CWQ'):
            self.kgs = self.load_subgraph(self.fact_triple)
        else:
            self.fact_data  = self.double_triple(self.fact_triple)
            self.load_graph(self.fact_data)


        self.train_q, self.train_e, self.train_a = self.load_query(self.train_data)
        self.valid_q, self.valid_e, self.valid_a = self.load_query(self.valid_data)
        self.test_q,  self.test_e,  self.test_a  = self.load_query(self.test_data)
        
        self.n_train = len(self.train_data) 
        self.n_valid = len(self.valid_q)
        self.n_test  = len(self.test_q)

        print('n_ent:',self.n_ent,'n_rel:',self.n_rel)
        print('n_train:', self.n_train, 'n_valid:', self.n_valid, 'n_test:', self.n_test)


    def read_metakb(self,filename):
        triples = []
        n_ent, n_rel = 0, 0
        with open(filename) as f:
            for line in f:
                h, r, t = line.strip().split('|')   # str
                if h not in self.entity2id.keys():
                    self.entity2id[h] = n_ent
                    n_ent += 1
                if t not in self.entity2id.keys():
                    self.entity2id[t] = n_ent
                    n_ent += 1
                if r not in self.relation2id.keys():
                    self.relation2id[r] = n_rel
                    n_rel += 1
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]  # id
                triples.append([h,r,t])

        self.n_ent, self.n_rel = n_ent, n_rel
        # add reverse and self-loop
        r2id = {}
        for k,v in self.relation2id.items():
            r2id['reverse_of_'+k] = v + self.n_rel
        self.relation2id.update(r2id)
        self.relation2id['self_loop'] = 2* self.n_rel
        return triples
    
    def read_qa(self, filename):
        qa_triple = []
        
        with open(filename) as f:
            for line in f:
                line = line.strip()
                match = re.search(r'\[([^]]+)\]', line)
    
                assert match, f"Failed to match: {line}"
                
                question = line.split('\t')[0]
                entity = match.group(1)
                answer_str = line.split('\t')[1]
                answers = answer_str.split('|')
                
                try:
                    h = self.entity2id[entity]
                except:
                    print(entity, 'not exist!')

                self.question2id[question] = self.n_qs
                self.n_qs += 1
                
                for ans in answers:
                    try:
                        t = self.entity2id[ans]
                        qa_triple.append([[h],self.question2id[question],t])
                    except:
                        pass
                        
        return qa_triple
                

    def read_web_qa(self, mode, nsm):

        qa_triple = []

        if nsm == 1:
            filepath = f'../data/webqsp/'+mode+'_simple.json'
        elif nsm == 2:
            filepath = f'../data/CWQ/'+mode+'_simple.json'
        json_file = open(filepath, 'r') 
        data = json_file.readlines()

        for line in data:

            entry = json.loads(line.strip())
            
            question = entry.get('question', '')
            self.question2id[question] = self.n_qs
            subgraph = entry.get("subgraph", {}).get("tuples", [])
            self.fact_triple[self.n_qs] = subgraph

            self.n_qs += 1
            if len(subgraph) == 0:
                continue

            try:
                h = entry.get('entities', [])
            except:
                continue
            new_h = []
            for e in h:
                if e >= self.n_ent:   # unseen entity
                    print(e, question)
                else:
                    new_h.append(e)
            if len(new_h) != 0: 
                answers = entry.get('answers',[{}])
                ans = []
                for an in answers:
                    tkbid = an['kb_id']
                    if tkbid in self.kbid2id.keys():
                        t = self.kbid2id[tkbid]
                        qa_triple.append([h,self.question2id[question],t])
                    else:
                        print(tkbid,'not in kbid')
                        print(question)

        return qa_triple
            
    def map_ent_and_rel(self, nsm):
        
        if nsm == 1:  # webqsp

            ent_file = '../data/webqsp/entity_name.txt'
            rel_file = '../data/webqsp/relations.txt'

        elif nsm == 2:  # CWQ
            
            ent_file = '../data/CWQ/entity_name.txt'
            rel_file = '../data/CWQ/relations.txt'

        n_ent, n_rel = 0, 0
        with open(rel_file, 'r', encoding='utf-8') as f:
            for line in f:
                rel = line.strip()
                self.relation2id[rel] = n_rel
                n_rel += 1
        r2id = {}
        for k,v in self.relation2id.items():
            r2id['reverse_of_'+k] = v + self.n_rel
        self.relation2id.update(r2id)
        self.relation2id['self_loop'] = 2* self.n_rel

        kbid2ent = {}
        with open(ent_file, 'r', encoding='utf-8') as f:
            for line in f:
                kbid, name = line.strip().split('\t')
                self.kbid2id[kbid] = n_ent
                if name == 'None':
                    name = kbid
                self.entity2id[name] = n_ent
                self.id2entity[n_ent] = name
                n_ent += 1        

        return 1


    def double_triple(self, triples):

        triple = np.array(triples,dtype=np.int64)
        inverse_triple = np.column_stack((triple[:, 2], triple[:, 1] + self.n_rel, triple[:, 0]))
        combined_triples = np.concatenate((triple, inverse_triple), axis=0)

        return combined_triples

    def load_graph(self, triples):
        idd = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)

        self.KG = np.concatenate([triples, idd], 0)
        self.n_fact = len(self.KG)
        self.M_sub = csr_matrix((np.ones((self.n_fact,)), (np.arange(self.n_fact), self.KG[:,0])), shape=(self.n_fact, self.n_ent))

    def load_subgraph(self, fact_triple):
        kgs = {}
        M_subs = {}
        for qid in fact_triple.keys():

            triples = fact_triple[qid]
            if len(triples) == 0:
                skg = []
            else:
                triple = np.array(triples, dtype=np.int64)
                inverse_triple = np.column_stack((triple[:, 2], triple[:, 1] + self.n_rel, triple[:, 0]))
                combined_triples = np.concatenate((triple, inverse_triple), axis=0)
                skg = combined_triples

            n = len(skg)
            # sM_sub = csr_matrix((np.ones((n,)), (np.arange(n), self.KG[:,0])), shape=(n, self.n_ent))
            kgs[qid] = skg

        return kgs
    
    def load_query(self, triples):
        # triples.sort(key=lambda x:(x[0], x[1]))
        trip_qh = defaultdict(lambda:list())
        trip_qt = defaultdict(lambda:list())

        for trip in triples:
            h, q, t = trip    
            # trip_hr[(h,r)].append(t)
            trip_qh[q] = h
            trip_qt[q].append(t) 
        
        queries = []
        answers = []
        topics = []
        for key in trip_qh:
            queries.append(key)
            topics.append(np.array(trip_qh[key]))
            answers.append(np.array(trip_qt[key]))

        return queries, topics, answers

    def get_neighbors(self, nodes, qids = None):
        
        if 'webqsp' in self.task_dir or 'CWQ' in self.task_dir:
            n_ent = len(nodes)
            KG = np.concatenate([nodes[:,1].reshape(-1,1), 2*self.n_rel*np.ones((n_ent, 1)),nodes[:,1].reshape(-1,1)],1)
            for qid in qids:
                skg = self.kgs[qid]
                KG = np.concatenate([KG,skg],axis=0)
            # print(KG.shape)
            KG = np.unique(KG, axis = 0)
            n = len(KG)
            M_sub = csr_matrix((np.ones((n,)), (np.arange(n), KG[:,0])), shape=(n, self.n_ent))
        else: 
            KG = self.KG
            M_sub = self.M_sub

        # nodes: n_node x 2 with (batch_idx, node_idx)
        node_1hot = csr_matrix((np.ones(len(nodes)), (nodes[:,1], nodes[:,0])), shape=(self.n_ent, nodes.shape[0]))
        edge_1hot = M_sub.dot(node_1hot)
        edges = np.nonzero(edge_1hot)
        sampled_edges = np.concatenate([np.expand_dims(edges[1],1), KG[edges[0]]], axis=1)     # (batch_idx, head, rela, tail)
        sampled_edges = torch.LongTensor(sampled_edges).cuda()

        # index to nodes
        head_nodes, head_index = torch.unique(sampled_edges[:,[0,1]], dim=0, sorted=True, return_inverse=True)
        tail_nodes, tail_index = torch.unique(sampled_edges[:,[0,3]], dim=0, sorted=True, return_inverse=True)

        sampled_edges = torch.cat([sampled_edges, head_index.unsqueeze(1), tail_index.unsqueeze(1)], 1)
       
        mask = sampled_edges[:,2] == (self.n_rel*2)
        _, old_idx = head_index[mask].sort()
        old_nodes_new_idx = tail_index[mask][old_idx]

        return tail_nodes, sampled_edges, old_nodes_new_idx

    def get_batch(self, batch_idx, data='train'):
 
        if data=='train':
            subs = [self.train_data[id][0] for id in batch_idx]
            subs = np.array(subs, dtype = object)
            qids = [self.train_data[id][1] for id in batch_idx]
            qids = np.array(qids)
            objs = [self.train_data[id][2] for id in batch_idx]
            objs = np.array(objs)
            return subs, qids, objs  # return np.array(self.train_data)[batch_idx] 
            # query, topic, answer = np.array(self.train_q), np.array(self.train_a,dtype=object) # 
        if data=='valid':
            query, topic, answer = np.array(self.valid_q), np.array(self.valid_e,dtype=object), np.array(self.valid_a,dtype=object)
        if data=='test':
            query, topic, answer = np.array(self.test_q), np.array(self.test_e, dtype=object), np.array(self.test_a,dtype=object)

        subs = []
        rels = []
        objs = []
        
        subs = topic[batch_idx]
        rels = query[batch_idx]
        objs = np.zeros((len(batch_idx), self.n_ent))
        for i in range(len(batch_idx)):
            objs[i][answer[batch_idx[i]]] = 1
        return subs, rels, objs

    def shuffle_train(self):

        np.random.shuffle(self.train_data)


