import math
import copy
import torch.nn.functional as F
import torch.nn as nn
import torch

def attention(query,key,value,mask=None,droput=None):
    d_k=query.size(-1)
    scorces=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not None:
        scorces=scorces.masked_fill(mask==0,-1e9)
    p_attn=F.softmax(scorces,dim=-1)
    if droput is not None:
        p_attn=droput(p_attn)

    return torch.matmul(p_attn,value),p_attn


def clone_func(model,N):
    return nn.ModuleList([copy.deepcopy(model) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self,head, embedding_dim,dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert embedding_dim%head==0

        self.d_k=embedding_dim//head
        self.head=head
        self.embedding_dim=embedding_dim
        self.liners=clone_func(nn.Linear(embedding_dim,embedding_dim),4)
        self.attn=None
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,query,key,value,mask=None):
        if mask is not None:
            mask=mask.unsqueeze(1)

        batch_size=query.shape[0]
        query,key,value=[model(x).view(batch_size,-1,self.head,self.d_k).transpose(1,2) for model,x in zip(self.liners,(query,key,value))]

        x,self.attn=attention(query,key,value,mask=mask,droput=self.dropout)
        x=x.transpose(1,2).contiguous().view(batch_size,-1,self.head*self.d_k)
        return self.liners[-1](x)

class PositionFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,droput=0.1):
        super(PositionFeedForward, self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.Mish(),
            nn.Dropout(p=droput),
            nn.Linear(d_ff,d_model)
        )

    def forward(self,x):
        return self.layers(x)

class Attention(nn.Module):
    def __init__(self,head,embeding_dim,droput):
        super(Attention, self).__init__()
        self.head=head
        self.embeding=embeding_dim
        self.droput=droput
        self.postion=PositionFeedForward(d_model=embeding_dim,d_ff=256,droput=droput)
        self.mult_hed=MultiHeadedAttention(head,embeding_dim,droput)

    def forward(self,x):
        x=self.postion(x)
        x=self.mult_hed(x,x,x)
        return x

if __name__ == '__main__':
    head=8
    embeding_dim=128
    droput=0.2
    query=torch.randn(128,4,128)
    att=Attention(head,embeding_dim,droput)
    x=att(query)
    print(x.shape)
