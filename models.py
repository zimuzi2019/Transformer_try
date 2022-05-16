import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy,softmax,relu
from torch.utils.data import DataLoader
import numpy as np

import datasets

class MultiHead(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.head_dim = model_dim // n_head
        self.n_head = n_head
        self.model_dim = model_dim
  
        self.wq = nn.Linear(model_dim, n_head * self.head_dim)
        self.wk = nn.Linear(model_dim, n_head * self.head_dim)
        self.wv = nn.Linear(model_dim, n_head * self.head_dim)

        self.o_dense = nn.Linear(model_dim, model_dim)
        self.o_drop = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(normalized_shape=model_dim)
        self.attention = None

    def forward(self, q, k, v, mask, training):
        # residual connect
        residual = q
        dim_per_head = self.head_dim
        num_heads = self.n_head
        batch_size = q.size(0)

        # linear projection
        key = self.wk(k)    #  key [n, step, n_heads * h_dim]
        value = self.wv(v)  #  value [n, step, n_heads * h_dim]
        query = self.wq(q)  #  query [n, step, n_heads * h_dim]

        # split by head
        query = self.split_heads(query) # query [n, n_head, step, h_dim]
        key = self.split_heads(key)     # key [n, n_head, step, h_dim]
        value = self.split_heads(value) # value [n, n_head, step, h_dim]
        context = self.scaled_dot_product_attention(
                                query, key, value, mask)
                                # context [n, step, m_dim] 

        o = self.o_dense(context) #o [n, step, m_dim]
        o = self.o_drop(o)

        o = self.layer_norm(residual+o)
        return o



    def split_heads(self, x):
        x = torch.reshape(x,
                    (x.shape[0], x.shape[1], self.n_head, self.head_dim))

        return x.permute(0,2,1,3)
        # x [n, n_head, step, h_dim]



    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dk = torch.tensor(k.shape[-1]).type(torch.float) # dk h_dim
        
        # q [n, n_head, step, h_dim] 
        # k.permute(0,1,3,2) [n, n_head, h_dim, step]
        score = torch.matmul(q, k.permute(0,1,3,2)) /   \
                (torch.sqrt(dk) + 1e-8)
        # score [n, n_head, step, step]

        if mask is not None:
        # change the value at masked position to negative infinity, so the 
        # attention score at these positions after softmax will close to 0.
            score = score.masked_fill_(mask, -np.inf)
        self.attention = softmax(score, dim=-1)   
        # attention [n, n_head, step, step]
        context = torch.matmul(self.attention, v) 
        # context [n, n_head, step, h_dim]
        context = context.permute(0,2,1,3)
        # context [n, step, n_head, h_dim]

        context = context.reshape((context.shape[0], context.shape[1],-1))  
        return context  # [n, step, m_dim]






class PositionWiseFFN(nn.Module):
    def __init__(self, model_dim, dropout = 0.0):
        super().__init__()
        dff = model_dim*4
        self.l = nn.Linear(model_dim,dff)
        self.o = nn.Linear(dff,model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self,x):
        o = relu(self.l(x))
        o = self.o(o)
        o = self.dropout(o)

        o = self.layer_norm(x + o)
        return o    # [n, step, m_dim]




class EncoderLayer(nn.Module):
    def __init__(self, n_head, emb_dim, drop_rate):
        super().__init__()
        self.mh = MultiHead(n_head, emb_dim, drop_rate)
        self.ffn = PositionWiseFFN(emb_dim, drop_rate)

    def forward(self, xz, training, mask):
        # xz [n, step, emb_dim]
        context = self.mh(xz, xz, xz, mask, training) 
        # context [n, step, emb_dim]
        o = self.ffn(context)
        return o

class Encoder(nn.Module):
    def __init__(self, n_head, emb_dim, drop_rate, n_layer):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(n_head, emb_dim, drop_rate) for _ in range(n_layer)]
        )

    def forward(self, xz, training, mask):
        for encoder in self.encoder_layers:
           xz = encoder(xz, training, mask)
        return xz
        # xz [n, step, m_dim]



class DecoderLayer(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.mh = nn.ModuleList([MultiHead(n_head, model_dim, drop_rate) 
                                    for _ in range(2)])
        self.ffn = PositionWiseFFN(model_dim,drop_rate)

    def forward(self, yz, xz, training, yz_look_ahead_mask, xz_pad_mask):
        dec_output = self.mh[0](yz, yz, yz, yz_look_ahead_mask,training)
        # dec_output [n, step, m_dim]

        dec_output = self.mh[1](dec_output, xz, xz, xz_pad_mask,training)
        # dec_output [n, step, m_dim]

        dec_output = self.ffn(dec_output)
        # dec_output [n, step, m_dim]

        return dec_output

class Decoder(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()
        self.num_layers = n_layer

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(n_head, model_dim, drop_rate) 
                for _ in range(n_layer)]
        )

    def forward(self, yz, xz, training, yz_look_ahead_mask, xz_pad_mask):
        for decoder in self.decoder_layers:
            yz = decoder(yz, xz, 
                         training, yz_look_ahead_mask, xz_pad_mask)
        return yz
        # yz [n, step, m_dim]





class PositionEmbedding(nn.Module):
    def __init__(self, max_len, emb_dim, n_vocab):
        super().__init__()

        pos = np.expand_dims(np.arange(max_len), axis=1)
        # pos [step, 1] max_len=step

        pe = pos / np.power(1000, 2*np.expand_dims(np.arange \
                           (emb_dim)//2, axis=0)/emb_dim)
        # [step, 1] / [1, emb_dim]
        # pe [step, emb_dim]  
 
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = np.expand_dims(pe, axis=0) 
        # pe [1, step, emb_dim]

        self.pe = torch.from_numpy(pe).type(torch.float32)
        self.embeddings = nn.Embedding(n_vocab, emb_dim)
        self.embeddings.weight.data.normal_(0,0.1)

    def forward(self, x):
        device = self.embeddings.weight.device
        self.pe = self.pe.to(device)
        # x [n, step, emb_dim]  
        x_embed = self.embeddings(x) + self.pe     
        # x_embed [n, step, emb_dim]
        return x_embed 




class Transformer(nn.Module):
    def __init__(self,
                 n_vocab_en,
                 n_vocab_cn,
                 max_len,
                 n_layer=6,
                 emb_dim=512,
                 n_head=8,
                 drop_rate=0.1,
                 padding_idx=0):
        
        super().__init__()
        self.max_len = max_len
        self.padding_idx = torch.tensor(padding_idx)
        self.dec_v_emb = n_vocab_cn 

        self.embed_en = PositionEmbedding(max_len, emb_dim, n_vocab_en)
        self.embed_cn = PositionEmbedding(max_len, emb_dim, n_vocab_cn)

        self.encoder = Encoder(n_head, emb_dim, drop_rate, n_layer)
        self.decoder = Decoder(n_head, emb_dim, drop_rate, n_layer)
        self.o = nn.Linear(emb_dim,n_vocab_cn)
        self.opt = torch.optim.Adam(self.parameters(),lr=0.002)


    def forward(self, x, y, training= None):
        x_embed, y_embed = self.embed_en(x), self.embed_cn(y) 
        #x_embed, y_embed  [n, step, emb_dim]
        
        pad_mask = self._pad_mask(x)    
        #pad_mask [n, 1, step, step]    step=max_len
        
        encoded_z = self.encoder(x_embed, training, pad_mask) 
        #encoded_z [n, max_len, emb_dim]
        
        yz_look_ahead_mask = self._look_ahead_mask(y)   
        #yz_look_ahead_mask [n, 1, max_len, max_len]
       
        decoded_z = self.decoder(y_embed, encoded_z, \
                                 training, yz_look_ahead_mask, pad_mask) 
        #decoded_z [n, step, emb_dim]
        
        o = self.o(decoded_z)   
        #o [n, step, n_vocab]
        return o

    
    def step(self, x, y):
        self.opt.zero_grad()
        logits = self(x,y[:, :-1],training=True)
        # logits [n, step, n_vocab] n_vocab=dec_v_emb
        # x,y [n, step]
        
        # pad_mask = ~torch.eq(y[:,1:], self.padding_idx)  #
        # pad_mask [n, seq_len]

        loss = cross_entropy(logits.reshape(-1,self.dec_v_emb), 
                             y[:,1:].reshape(-1))
        loss.backward()
        
        self.opt.step()
        return loss.cpu().data.numpy(), logits

    

    def _pad_bool(self, seqs):
        o = torch.eq(seqs, self.padding_idx)
        # o [n, step]       
        return o

    def _pad_mask(self, seqs):
        # seqs [n, step]  step=max_len
        len_q = seqs.size(1) 
        mask = self._pad_bool(seqs).unsqueeze(1).expand(-1,len_q,-1)    
        #mask [n, step, step]
        return mask.unsqueeze(1)    
        #return [n, 1, step, step]
        
    def _look_ahead_mask(self, seqs):
        device = next(self.parameters()).device
        batch_size, seq_len = seqs.shape
        mask = torch.triu(torch.ones((seq_len,seq_len), dtype=torch.long), 
                                      diagonal=1).to(device)  
        #mask [step, step]  step=max_len
        mask = torch.where(self._pad_bool(seqs)[:,None,None,:],
                           # [n, 1, 1, step]
                           1,
                           mask[None,None,:,:]).to(device)  
                           # [1, 1, step, step]
        #mask [n, 1, step, step]
        return mask>0   
        #return [n, 1, step, step]       


    def translate(self, src, v2i):
        self.eval()
        device = next(self.parameters()).device
        src_pad = src

        # Initialize Decoder input by constructing a matrix M
        # M [n, self.max_len+1]), M[n,0] = start token id, M[n,:] = 0 
        target = torch.from_numpy(
        datasets.pad_zero(np.array([[v2i["<GO>"], ] for _ in range(len(src))]), 
        self.max_len+1)).to(device)
        
        x_embed = self.embed_en(src_pad)
        encoded_z = self.encoder(x_embed,False,mask=self._pad_mask(src_pad))
        for i in range(0,self.max_len):
            y = target[:,:-1]
            y_embed = self.embed_cn(y)
            
            decoded_z = self.decoder(y_embed,
                                     encoded_z,
                                     False,
                                     self._look_ahead_mask(y),
                                     self._pad_mask(src_pad))
            
            o = self.o(decoded_z)[:,i,:]
            idx = o.argmax(dim = 1).detach()
            # Update the Decoder input, to predict for the next position.
            target[:,i+1] = idx
        
        self.train()
        return target

if __name__ == "__main__":
        dataset = datasets.SentenceData()

        n_vocab_en, n_vocab_cn = dataset.num_word

        model = Transformer(
            n_vocab_en=n_vocab_en,
            n_vocab_cn=n_vocab_cn, 
            max_len=datasets.MAX_LENGTH, 
            n_layer=6, 
            emb_dim=512, 
            n_head=8, 
            drop_rate=0.1, 
            padding_idx=0
        )