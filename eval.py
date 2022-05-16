import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy,softmax,relu
from torch.utils.data import DataLoader

import numpy as np
import jieba

import datasets
from models import Transformer
from utils import normalizeString,cht_to_chs

MAX_LEN = datasets.MAX_LENGTH

path = "/home/cslee/MyPythonCode/12_demo1/data/test_en-cn.txt"

lines = open(path, encoding='utf-8').readlines()

def eval(emb_dim=64,n_layer=3,n_head=4):
    dataset = datasets.SentenceData()

    dataset.x = [[i[j] if j<len(i) else datasets.PAD_ID for j in range(MAX_LEN)]   for i in dataset.x]
    dataset.y = [[i[j] if j<len(i) else datasets.PAD_ID for j in range(MAX_LEN+1)] for i in dataset.y]

    n_vocab_en, n_vocab_cn = dataset.num_word
    
    model = Transformer(
        n_vocab_en=n_vocab_en,
        n_vocab_cn=n_vocab_cn, 
        max_len=MAX_LEN, 
        n_layer=n_layer, 
        emb_dim=emb_dim, 
        n_head=n_head, 
        drop_rate=0.1, 
        padding_idx=0
    )

    model.load_state_dict(torch.load("/home/cslee/MyPythonCode/12_demo1/models/64-3-4/5000.pth"))
    
    if torch.cuda.is_available():
        print("GPU train avaliable")
        device =torch.device("cuda")
        model = model.cuda()
    else:
        device = torch.device("cpu")
        model = model.cpu()


    cnt,score_sum=0,0

    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
    #from nltk.translate.bleu_score import SmoothingFunction

    for line in lines:
        line = line.split("\t")
        sentence1 = normalizeString(line[0])
            
        sentence2 = cht_to_chs(line[1].strip())
        seg_list=jieba.cut(sentence2,cut_all=False)
        sentence2=" ".join(seg_list)   
        
        if len(sentence1.split(" "))>MAX_LEN:
           continue
        if len(sentence2.split(" "))>MAX_LEN:
           continue

        cnt += 1

        x = torch.tensor(
            [[dataset.v2i_en[i] if i in dataset.vocab_en else dataset.v2i_en["<UNK>"] for i in sentence1.split(" ")]]
        )
        y = torch.tensor(
             [[dataset.v2i_cn["<GO>"], ] + \
             [dataset.v2i_cn[i] if i in dataset.vocab_cn else dataset.v2i_cn["<UNK>"] for i in sentence2.split(" ")] + \
             [dataset.v2i_cn["<EOS>"], ]]
        )

        target = dataset.idx2str_cn(y[0, 1:-1].cpu().data.numpy())

        x = torch.tensor([[i[j] if j<len(i) else datasets.PAD_ID for j in range(MAX_LEN)]   for i in x])
        y = torch.tensor([[i[j] if j<len(i) else datasets.PAD_ID for j in range(MAX_LEN+1)] for i in y])

        pred = model.translate(x[0:1].cuda(),dataset.v2i_cn) 
        res = dataset.idx2str_cn(pred[0].cpu().data.numpy())
        src = dataset.idx2str_en(x[0].cpu().data.numpy())
        print(
            src, "------------------------", sentence2.split(" "),"\n",
            target,"\n",
            res[1:-1],"\n"
        )

        score_1 = sentence_bleu([target],res[1:-1],weights=(1,0,0,0))
        score_2 = sentence_bleu([target],res[1:-1],weights=(0,1,0,0))
        score_3 = sentence_bleu([target],res[1:-1],weights=(0,0,1,0))
        print(score_1,score_2,score_3)

        score_sum += sentence_bleu([target],res[1:-1],weights=(0.4,0.3,0.3,0))

    print("The avrage BLEU score of test batch is ",score_sum*1.0/cnt)


if __name__ == "__main__":
    eval()