import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import DataLoader,Dataset

class GPTDatasetV1(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        self.input_ids=[]
        self.target_ids=[]
        token_ids=tokenizer.encode(txt)

        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk=token_ids[i:i+max_length]
            target_chunk=token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self,index):
        return self.input_ids[index],self.target_ids[index]

def create_dataloader(txt,batch_size=4,max_length=4,stride=4,shuffle=True,num_workers=0,drop_last=True):
    tokenizer=tiktoken.get_encoding("gpt2")

    dataset=GPTDatasetV1(txt,tokenizer,max_length,stride)

    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,drop_last=drop_last)
    return dataloader
class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps=1e-5
        self.scale=nn.Parameter(torch.ones(emb_dim))
        self.shift=nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        var=x.var(dim=-1,keepdim=True,unbiased=False)
        norm_x = (x-mean)/torch.sqrt(var+self.eps)
        return self.scale*norm_x+self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044515*torch.pow(x,3))))

class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"])
        )
    def forward(self,x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,num_heads,dropout=0.0,qkv_bias=False):
        super().__init__()
        assert d_out%num_heads==0,"d_out must be divisible by num_heads"
        self.dout=d_out
        self.num_heads=num_heads
        self.head_dim=d_out//num_heads

        self.qkv=nn.Linear(d_in,3*d_out,bias=qkv_bias)
        self.proj=nn.Linear(d_out,d_out)
        self.dropout=nn.Dropout(dropout)

        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,x):
        batch_size,seq_len,embed_dim=x.shape
        qkv=self.qkv(x)
        qkv=qkv.view(batch_size,seq_len,3,self.num_heads,self.head_dim)
        qkv=qkv.permute(2,0,3,1,4)
        queries,keys,values=qkv.unbind(0)
        atten_scores=queries@keys.transpose(-1,-2)
        atten_scores=atten_scores.masked_fill_(self.mask.bool()[:seq_len,:seq_len],-torch.inf)
        atten_weights=torch.softmax(atten_scores/keys.shape[-1]**0.5,dim=-1)
        atten_weights=self.dropout(atten_weights)
        context_vec=(atten_weights@values).transpose(1,2)
        context_vec=context_vec.contiguous().view(batch_size,seq_len,self.dout)
        context_vec=self.proj(context_vec)
        return context_vec

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.att=MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["dropout"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ffn = FeedForward(cfg)
        self.norm1=LayerNorm(cfg["emb_dim"])
        self.norm2=LayerNorm(cfg["emb_dim"])
        self.drop=nn.Dropout(cfg["dropout"])

    def forward(self,x):
        shortcut=x
        x=self.norm1(x)
        x=self.att(x)
        x=self.drop(x)
        x=x+shortcut

        shortcut=x
        x=self.norm2(x)
        x=self.ffn(x)
        x=self.drop(x)
        x=x+shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb=nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb=nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_emb=nn.Dropout(cfg["dropout"])

        self.trf_blocks=nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm= LayerNorm(cfg["emb_dim"])
        self.out_head=nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)

    def forward(self,in_idx):
        batch_size,seq_len=in_idx.shape
        tok_embeds=self.tok_emb(in_idx)
        pos_ids = torch.arange(seq_len, device=in_idx.device)
        pos_embeds=self.pos_emb(pos_ids)
        x=tok_embeds+pos_embeds
        x=self.drop_emb(x)
        x=self.trf_blocks(x)
        x=self.final_norm(x)
        logits=self.out_head(x)
        return logits

def generate_text_simple(model,idx,max_new_tokens,context_size):
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond=idx[:,-context_size:]
            logits=model(idx_cond)
            logits=logits[:,-1,:]
            probs=torch.softmax(logits,dim=-1)
            idx_next=torch.argmax(probs,dim=-1,keepdim=True)
            idx=torch.cat((idx,idx_next),dim=1)

    return idx
