import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy,softmax,relu
from torch.utils.data import DataLoader

import numpy as np
import argparse

import datasets
from models import Transformer

MAX_LEN = datasets.MAX_LENGTH

def train(emb_dim=64,n_layer=3,n_head=4):
    dataset = datasets.SentenceData()
    # print(dataset.sentences_en[:3])
    # print(dataset.sentences_cn[:3])
    # print(dataset.vocab_en)
    # print(dataset.vocab_cn)

    dataset.x = [[i[j] if j<len(i) else datasets.PAD_ID for j in range(MAX_LEN)]   for i in dataset.x]
    dataset.y = [[i[j] if j<len(i) else datasets.PAD_ID for j in range(MAX_LEN+1)] for i in dataset.y]

    # print(dataset.idx2str_en(dataset.x[0]),"\t",dataset.x[0],
    #       "\n",
    #       dataset.idx2str_cn(dataset.y[0]),"\t",dataset.y[0])   

    loader = DataLoader(dataset,batch_size=1000,shuffle=True)

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
    
    if torch.cuda.is_available():
        print("GPU train avaliable")
        device =torch.device("cuda")
        model = model.cuda()
    else:
        device = torch.device("cpu")
        model = model.cpu()
    
    epoch = 8000
    for i in range(1,epoch+1):
        for batch_idx , batch in enumerate(loader):
            bx, by = batch
            # bx [n, seq_len]
            # by [n, seq_len]

            bx = torch.stack(bx,dim=-1).type(torch.LongTensor).to(device)
            by = torch.stack(by,dim=-1).type(torch.LongTensor).to(device)
            # bx [n, MAX_LEN]
            # by [n, MAX_LEN+1]
            
            loss, logits = model.step(bx,by)
            
            if batch_idx%50 == 0:
                target = dataset.idx2str_cn(by[0, 1:-1].cpu().data.numpy())
                pred = model.translate(bx[0:1],dataset.v2i_cn) 
                res = dataset.idx2str_cn(pred[0].cpu().data.numpy())
                src = dataset.idx2str_en(bx[0].cpu().data.numpy())
                print(
                    "Epoch: ",i,
                    "| t: ", batch_idx,
                    "| loss: %.3f" % loss,
                    "| input: ", src,
                    "| target: ", target,
                    "| inference: ", res,
                )

        if i%1000 == 0:
            torch.save(model.state_dict(),
                   "/home/cslee/MyPythonCode/12_demo1/models/{}.pth".format(i))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--emb_dim", type=int, \
                        help="change the model dimension")

    parser.add_argument("--n_layer",type=int, \
                        help="change the number of layers in Encoder \
                                and Decoder")
    parser.add_argument("--n_head",type=int, \
                        help="change the number of heads in MultiHeadAttention")

    args = parser.parse_args()
    args = dict(filter(lambda x: x[1],vars(args).items()))  
    
    # 假设输入 python -u "transformer.py" --emb_dim 128 --n_layer 4
    # vars(args).items() 
    # dict_items([('emb_dim',128), ('n_layer',4), ('n_head',None)])
    
    # args {'emb_dim':128, 'n_layer':4}
    train(**args)  