from utils import *
import numpy as np
import json

def check(dataset):
    # dataset = 'webqsp'
    if dataset.startswith('MetaQA'):
        ta_file = '../data/'+ dataset + '/ntm/qa_test.txt'
        dataset = dataset.replace('/','-')
        path_file = '../explore/' + dataset + '-path.txt'
    elif dataset == 'webqsp':
        ta_file = '../data/webqsp/Webqsp.txt'
        path_file = '../explore/' + dataset + '-path.txt'
    elif dataset == 'CWQ':
        ta_file = '../data/CWQ/CWQ.txt'
        path_file = '../explore/' + dataset + '-path.txt'

    fta = open(ta_file)  # ta: true answer
    all_candi, all_score, all_p, all_ids = candidate_path(path_file)

    all_ta = []
    n_null = 0
    for line in fta: 
        line = line.strip().split('\t')
        try:
            if dataset.startswith('WC'):
                ta = line[2]
                ta = ta.replace('/','|')[:-1]
            else:
                _, ta = line[0], line[1]
            ta = ta.strip()
        except:
            ta = 'null'
        all_ta.append(ta)

    read_file = dataset + '-ans.jsonl'
    fa = open(read_file)   
    all_a = []
    all_a_id = []
    maxid = -1
    for line in fa:
        data = json.loads(line.strip())
        id = data['id']
        id = int(id)
        if id <= maxid:
            continue
        else:
            maxid = id
            all_a.append(data['answer'])
            all_a_id.append(id)
    print(len(all_ta),len(all_a))  

    check = []
    check_abc = []
    check_A = []
    n_true = 0
    flag = 0
    for i in range(len(all_a)):
        try:
            a = all_a[i]
        except:
            a = 'null'
        i = all_a_id[i]
        ta = all_ta[i]
        if ta == 'null':
            n_null += 1
            check.append(0)
            check_abc.append(0)
            continue
        ta = ta.split('|')
        flag = 0

        for oneta in ta:
            if oneta.lower() in a.lower():
                check.append(1)
                n_true += 1
                flag = 1
                break
        if flag == 0:
            check.append(0)
        flag = 0

        # check abc 
        s = a
        index_a = s.find('A. ')
        index_b = s.find('B. ')
        index_c = s.find('C. ')
        index_d = s.find('D. ')
        index_e = s.find('E. ')
        # print(index_a, index_b, index_c)
        if i not in all_ids:
            check_abc.append(0)
            check_A.append(0)
            continue
        i = all_ids[i]
        if 0 <= index_a and (index_b ==-1 or index_b > index_a) and (index_c == -1 or index_a < index_c) and (index_d == -1 or index_a < index_d):
            a = 'A. '  + all_candi[i][0].lower()
        elif 0 <= index_b and (index_a ==-1 or index_a > index_b) and (index_c == -1 or index_b < index_c) and (index_d == -1 or index_b < index_d):
            a = 'B. '+ all_candi[i][1].lower()
        elif 0 <= index_c and (index_a ==-1 or index_a > index_c) and (index_b == -1 or index_b > index_c) and (index_d == -1 or index_c < index_d):
            a = 'C. ' + all_candi[i][2].lower()
        elif 0 <= index_d and (index_a ==-1 or index_a > index_d) and (index_b == -1 or index_b > index_d) and (index_c == -1 or index_d < index_c):
            a = 'D. ' + all_candi[i][3].lower()
        else:
            a = a.lower()

        for oneta in ta:
            if oneta.lower() in a:
                check_abc.append(1)
                # n_true += 1
                flag = 1
                break
        if flag == 0:
            check_abc.append(0)


    hit1abc = np.array(check_abc).sum() / (len(check_abc) - n_null)
    print('HIT@1: ', hit1abc)
    hit = n_true / (len(check) - n_null)
    print('whether the correct answer is in the reply:' , hit)  

    fout = open('check-'+read_file,'w')
    for i in range(len(check)):
        data = {'id': all_a_id[i] , 'hit@1': check_abc[i], 'gold_answer': all_ta[all_a_id[i]], 'answer': all_a[i]}   
        fout.write(json.dumps(data) + '\n')
    fout.write(json.dumps({'HIT@1': hit1abc, 'HIT': hit})+ '\n')
    fout.close()