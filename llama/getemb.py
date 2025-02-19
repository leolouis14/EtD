from llama import Llama, Dialog
from tqdm import tqdm
from typing import List, Optional
import fire
import os, json, pickle, csv
import torch
import numpy as np

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    dataset: str = 'webqsp'
):
    generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

    data = dataset #'webqsp'  # 'CWQ'  'MetaQA/1-hop'
    
    modes = ['test','train' ,'dev', 'rel'] 
    for mode in modes:
        all_q = []

        if data == 'CWQ':
            # CWQ
            if mode != 'rel':
                filepath = f'../data/CWQ/'+mode+'_simple.json'
                json_file = open(filepath, 'r') 
                data = json_file.readlines()
                for line in data:
                # 解析JSON行数据
                    entry = json.loads(line.strip())  
                    question = entry.get('question', '')
                    all_q.append(question)
            else:
                with open('../data/CWQ/relations.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        rel = line.strip()
                        all_q.append(rel)
        
        elif data == 'webqsp':
            if mode != 'rel':
                filepath = f'../data/webqsp/'+mode+'_simple.json'
                json_file = open(filepath, 'r') 
                data = json_file.readlines()
                for line in data:
                # 解析JSON行数据
                    entry = json.loads(line.strip())  
                    question = entry.get('question', '')
                    all_q.append(question)
            else:
                with open('../data/webqsp/relations.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        rel = line.strip()
                        all_q.append(rel)

        elif data.startswith('MetaQA'):
            if mode != 'rel':
                f = open('../data/'+data+'/ntm/qa_'+mode+'.txt')  
                for line in f: 
                    line = line.strip().split('\t')
                    q = line[0]
                    q = q.replace('[','')
                    q = q.replace(']','')
                    q = q + "?"
                    all_q.append(q)
            else:
                with open('../data/MetaQA/kb.txt') as f:
                    for line in f:
                        _, r, _ = line.strip().split('|')   
                        all_q.append(r)
        
        n = len(all_q)
        # print(n,n_rel,all_q)
        all_emb = np.zeros((n,5120))
        # print(n)
        with torch.no_grad():
            for i in tqdm(range(len(all_q))):
                q = all_q[i]
                prompts = [q]
                emb = get_emb(generator,
                            prompts,
                            max_gen_len=max_gen_len,
                            temperature=temperature,
                            top_p=top_p,
                        )
                all_emb[i, :] = emb.cpu().data.numpy()

        if mode == 'dev':
            mode = 'valid'
        if data.startswith('MetaQA'):
            data = 'Meta-' + data[7] + 'm'
            if mode == 'rel':
                data = 'Meta'

        np.save('../embedding/' + data + '-' + mode +'.npy',all_emb)
        print('save done')

def get_emb(generator,                  
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ):
    if max_gen_len is None:
        max_gen_len = generator.model.params.max_seq_len - 1
    prompt_tokens = [generator.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    params = generator.model.params
    bsz = len(prompt_tokens)
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
    assert bsz == 1
    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= params.max_seq_len
    total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

    pad_id = generator.tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    #print(tokens)
    h = get_h(generator.model, tokens[:,0:min_prompt_len], 0)
    emb = h[0].mean(0)

    return emb

def get_h(model, tokens, start_pos):   # forward() in model.py 

    _bsz, seqlen = tokens.shape
    try:
        h = model.tok_embeddings(tokens)  
    except:
        print('get token emb error!')
        exit(0)
    model.freqs_cis = model.freqs_cis.to(h.device)
    freqs_cis = model.freqs_cis[start_pos : start_pos + seqlen]

    mask = None
    if seqlen > 1:
        mask = torch.full(
            (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
        )
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
    x = 0
    for layer in model.layers:
        h = layer(h, start_pos, freqs_cis, mask)
        if x == 0:
            h_first = h
        x += 1
    h = model.norm(h)
    h_first = model.norm(h_first)
    h = 0.5*(h+h_first)
    return h
    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'  
    fire.Fire(main)