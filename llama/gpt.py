import json
import os, time, random
from tqdm import tqdm
from utils import *
from check import check
from openai import AzureOpenAI
import argparse

def gpt(dataset):
    client = AzureOpenAI(
    azure_endpoint = "https://***.openai.azure.com/", 
    api_key='***',  
    api_version="2024-02-15-preview"
    )

    # prompt-1
    # myprompt = {"role": "system", "content": "Given a question, and the reference answers with their correct probabilities and associated retrieved knowledge graph triples (entity, relation, entity) as related facts, you are asked to answer the question with this information and your own knowledge. Please output the answer label and content directly. Do not output the correct probability, related facts and other words."}
    # prompt-2
    # myprompt = {"role": "system", "content": "Given a question, and the reference answers with their correct probabilities and associated retrieved knowledge graph triples (entity, relation, entity) as related facts, you are asked to answer the question with this information and your own knowledge. If the reference answers contain the correct answer, please output the label and content of the answer; If not, please answer the question based on your own knowledge."}
    # prompt-3
    myprompt = {"role": "system", "content": "Given a question, and the reference answers with their correct probabilities and associated retrieved knowledge graph triples (entity, relation, entity) as related facts, you are asked to answer the question with this information and your own knowledge. If the reference answers contain the correct answer, please output the label and content of the answer; If not, please answer the question based on your own knowledge. Please end your reply with 'The answer is * * *'."}
    

    myqs = {"role": "user", "content":""}

    all_q = []
    x = 0
    # dataset = 'webqsp'  # 'MetaQA/1-hop
    if dataset.startswith('MetaQA'):  # MetaQA/1-hop
        f = open('../data/'+dataset+'/ntm/qa_test.txt')  
        for line in f: 
            line = line.strip().split('\t')
            q = line[0]
            q = q.replace('[','')
            q = q.replace(']','')
            q = q + "?"
            all_q.append(q)
        dataset = dataset.replace('/','-')
        path_file = '../explore/' + dataset + '-path.txt'

    elif dataset == 'webqsp':
        mode = 'test'
        filepath = f'../data/webqsp/'+mode+'_simple.json'
        json_file = open(filepath, 'r') 
        data = json_file.readlines()

        for line in data:
            entry = json.loads(line.strip())
            question = entry.get('question', '')
            question += '?'
            all_q.append(question)
        path_file = '../explore/webqsp-path.txt'

    elif dataset == 'CWQ':
        filepath = '../data/CWQ/test_simple.json'
        json_file = open(filepath, 'r') 
        data = json_file.readlines()
        for line in data:
            entry = json.loads(line.strip())
            question = entry.get('question', '')
            all_q.append(question)
        path_file = '../explore/CWQ-path.txt'


    all_candi, all_score, all_p, all_ids = candidate_path(path_file)


    slice = random.sample(range(0,len(all_q)), 1000)  

    for qid in tqdm(sorted(slice)):

        q = all_q[qid]
        myqs = {"role": "user", "content":""}

        if qid in all_ids:
            i = all_ids[qid]
            hintABCpp = ' A. ' + all_candi[i][0] + ' (correct probability: ' + str(all_score[i][0]) + ')  {relevant facts: '+ all_p[i][0] + '}  B. ' + all_candi[i][1] + ' (correct probability: ' + str(all_score[i][1]) + ')  {relevant facts: '+ all_p[i][1] + '} C. ' + all_candi[i][2] + ' (correct probability: ' + str(all_score[i][2]) + ')  {relevant facts: '+ all_p[i][2] + '}  Answer: '
            hintABCDpp = ' A. ' + all_candi[i][0] + ' (correct probability: ' + str(all_score[i][0]) + ')  {relevant facts: '+ all_p[i][0] + '}  B. ' + all_candi[i][1] + ' (correct probability: ' + str(all_score[i][1]) + ')  {relevant facts: '+ all_p[i][1] + '} C. ' + all_candi[i][2] + ' (correct probability: ' + str(all_score[i][2]) + ')  {relevant facts: '+ all_p[i][2] + '} D. ' + all_candi[i][3] + ' (correct probability: ' + str(all_score[i][3]) + ')  {relevant facts: '+ all_p[i][3] + '}  Answer: '
            hintABCDEpp = ' A. ' + all_candi[i][0] + ' (correct probability: ' + str(all_score[i][0]) + ')  {relevant facts: '+ all_p[i][0] + '}  B. ' + all_candi[i][1] + ' (correct probability: ' + str(all_score[i][1]) + ')  {relevant facts: '+ all_p[i][1] + '} C. ' + all_candi[i][2] + ' (correct probability: ' + str(all_score[i][2]) + ')  {relevant facts: '+ all_p[i][2] + '} D. ' + all_candi[i][3] + ' (correct probability: ' + str(all_score[i][3]) + ')  {relevant facts: '+ all_p[i][3] + '} E. ' + all_candi[i][4] + ' (correct probability: ' + str(all_score[i][4]) + ')  {relevant facts: '+ all_p[i][4]   + '}  Answer: '
                
            myqs['content'] = 'Question: ' + q  +  '\nReference answer: '+  hintABCDpp

        else:
            myqs['content'] = 'Question: ' + q + ' Answer:'

        i = qid
        try:
            response = client.chat.completions.create(
                model="g35", # model="gpt-3.5-turbo",
                messages=[
                    myprompt, 
                    myqs
                ],
                # temperature=0
            )
            output = response.choices[0].message.content
        except:
            output = 'NULL'   
        try:
            output = output.replace("\n", "  ")
        except:
            output = 'NULL'
        time.sleep(0.1)
        write_file = dataset+'-ans.jsonl'
        with open(write_file, 'a+',encoding='utf-8') as fout:
            data = {'id': i , 'answer': str(output), 'question': myqs['content']}   
            fout.write(json.dumps(data) + '\n')

    check(dataset=dataset)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Parser for DualR")
    parser.add_argument('--dataset', type=str, default='webqsp')

    args = parser.parse_args()
    
    gpt(dataset=args.dataset)
