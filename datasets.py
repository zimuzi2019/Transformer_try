import itertools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
import os
import re
import requests

import jieba
from utils import normalizeString
from utils import cht_to_chs


PAD_ID = 0
MAX_LENGTH = 10

path = "/home/cslee/MyPythonCode/12_demo1/data/train_en-cn.txt"

lines = open(path, encoding='utf-8').readlines()

class SentenceData(Dataset):
    def __init__(self):
        self.sentences_en = []
        self.sentences_cn = []

        for line in lines:
            line = line.split("\t")
            sentence1 = normalizeString(line[0])
            
            sentence2 = cht_to_chs(line[1].strip())
            seg_list=jieba.cut(sentence2,cut_all=False)
            sentence2=" ".join(seg_list)   
        
            if len(sentence1.split(" "))>MAX_LENGTH:
                continue
            if len(sentence2.split(" "))>MAX_LENGTH:
                continue

            self.sentences_en.append(sentence1)
            self.sentences_cn.append(sentence2)

        self.vocab_en = set(
            [ j for i in self.sentences_en for j in i.split(" ")] + ["<UNK>"]
        )

        self.vocab_cn = set(
            [j for i in self.sentences_cn for j in i.split(" ")] + ["<GO>","<EOS>","<UNK>"]
        )



        self.v2i_en ={
            v:i for i,v in enumerate(sorted(list(self.vocab_en)),start=1)
        }
        self.v2i_en["<PAD>"] = PAD_ID
        self.vocab_en.add("<PAD>")
        self.i2v_en = {
            i:v for v,i in self.v2i_en.items()
        }

        self.v2i_cn ={
            v:i for i,v in enumerate(sorted(list(self.vocab_cn)),start=1)
        }
        self.v2i_cn["<PAD>"] = PAD_ID
        self.vocab_cn.add("<PAD>")
        self.i2v_cn = {
            i:v for v,i in self.v2i_cn.items()
        }



        self.x, self.y = [], []
        for en, cn in zip(self.sentences_en, self.sentences_cn):
            self.x.append([self.v2i_en[v] for v in en.split(" ")])
            self.y.append(
                [self.v2i_cn["<GO>"], ] + 
                [self.v2i_cn[v] for v in cn.split(" ")] +
                [self.v2i_cn["<EOS>"], ]
            )
        #self.x, self.y = np.array(self.x), np.array(self.y)
        
        self.start_token = self.v2i_cn["<GO>"]
        self.end_token = self.v2i_cn["<EOS>"]



    def __len__(self):
        return len(self.x)

    @property
    def num_word(self):
        return len(self.vocab_en),len(self.vocab_cn)



    def __getitem__(self, index):
        return self.x[index],self.y[index]



    def idx2str_cn(self,idx):
        x=[]
        for i in idx:
            x.append(self.i2v_cn[i])
            if i == self.end_token:
                break
        return x
    def idx2str_en(self,idx):
        x=[]
        for i in idx:
            x.append(self.i2v_en[i])
        return x       

def pad_zero(seqs, max_len):
    padded = np.full((len(seqs), max_len), fill_value=PAD_ID, dtype=np.int32)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    return padded

if __name__ == "__main__":
    sentencedata=SentenceData()

    # print(sentencedata.num_word)
    # print(sentencedata.sentence_en)
    # print(sentencedata.sentence_cn)
    # print(sentencedata.vocab_en)
    # print(sentencedata.vocab_cn)
    print(
        sentencedata.idx2str_en(sentencedata.x[1]),sentencedata.x[1],"\n",
        sentencedata.idx2str_cn(sentencedata.y[1]),sentencedata.y[1]       
    )
    