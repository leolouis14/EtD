# DualR

This is our Pytorch implementation for the paper (accepted by CPAL 2025):

> Liu, G., Zhang, Y., Li, Y., & Yao, Q. Dual Reasoning: A GNN-LLM Collaborative Framework for Knowledge Graph Question Answering. In *The Second Conference on Parsimony and Learning*

Link to paper: https://openreview.net/pdf?id=odnOkx8Qfj

## Instructions

A quick instruction is given for readers to reproduce the whole process.

## Environment Requirements

- torch == 2.5.1
- torch-cluster == 1.6.3
- torch-scatter == 2.1.2
- torchdrug == 0.2.1
- tokenizers == 0.20.3
- fairscale == 0.4.13
- fire == 0.5.0
- sentencepiece == 0.2.0

## Run the Codes

### Datasets

We follow the [NSM](https://github.com/RichardHGL/WSDM2021_NSM?tab=readme-ov-file) to preprocess the datasets.
You can download the datasets from NSM, and unzip them in the "data" folder.

### Experiments

#### Question and Relation Encoding

We use the Llama2-13B-chat as the encoder. Please download [it](https://github.com/meta-llama/llama) and put in the "llama" folder.

To get text embedding:

```
cd llama
bash getemb.sh
```

#### First-tier reasoning (knowledge exploration):

load pretrained model:

```
cd explore
python train.py  --dataset webqsp --load
```

or you can pretrain from scratch:

```
cd explore
python train.py  --dataset WebCWQ 
```

#### Second-tier reasoning (answer determination):

use Llama2-13B-chat:

```
cd llama
bash  chat13.sh
```

use ChatGPT:

```
cd llama
python gpt.py --dataset webqsp
```
