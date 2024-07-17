import torch 
import torch
import torch.nn as nn
import torch.nn.functional as F  


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_dim, prenorm=False, act=lambda x: x.relu(), dropout=0.1 ):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        self.prenorm, self.act  = prenorm, act 
        self.dropout = dropout 

        self.query = (torch.Tensor.uniform_(embed_dim, embed_dim), torch.zeros(embed_dim))
        self.key = (torch.Tensor.uniform_(embed_dim, embed_dim), torch.zeros(embed_dim))
        self.value = (torch.Tensor.uniform_(embed_dim, embed_dim), torch.zeros(embed_dim))

        self.out = (torch.Tensor.uniform_(embed_dim, embed_dim), torch.zeros(embed_dim))

        self.ff1 = (torch.Tensor.uniform_(embed_dim, ff_dim), torch.zeros(ff_dim))
        self.ff2 = (torch.Tensor.uniform_(ff_dim, embed_dim), torch.zeros(embed_dim))

        self.ln1 = (torch.ones(embed_dim), torch.zeros(embed_dim))
        self.ln2 = (torch.ones(embed_dim), torch.zeros(embed_dim))

        self.attention = ScaledDotProductAttention(temperature=self.head_size ** 0.5)
    def attn(self, x):

        query, key, value = [x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)).transpose(1,2) for y in [self.query, self.key, self.value]]
        
        attn_output, _ = self.attention(query, key, value)

        return attn_output.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.out)
    
    def __call__(self, x):
        if self.prenorm:
            x = x + self.attn(x.layernorm().linear(*self.ln1)).dropout(self.dropout)
            x = x + self.act(x.layernorm().linear(*self.ln2).linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
        else:
            x = x + self.attn(x).dropout(self.dropout)
            x = x.layernorm().linear(*self.ln1)
            x = x + self.act(x.linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
            x = x.layernorm().linear(*self.ln2)
        return x
    
class Transformer:
    def __init__(self, syms, maxlen, layers, embed_dim, num_heads, ff_dim):
        self.maxlen, self.syms = maxlen, syms
        self.embed = torch.Tensor.uniform_(maxlen+syms, embed_dim, requires_grad=False)
        self.tbs = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(layers)]
        self.final = torch.Tensor.uniform_(embed_dim, syms)

    def forward(self, x):
        bs = x.shape[0]

        maxlen_eye = torch.eye(x.shape[1])
        maxlen_eye = maxlen_eye.unsqueeze(0).expand([bs, *maxlen_eye.shape])

        onehot_feat = x.one_hot(self.syms)

        onehot = maxlen_eye.cat(onehot_feat, dim=2).flatten(end_dim=1)

        x = onehot.dot(self.embed).reshape((bs, x.shape[1], -1))
        x = x.sequential(self.tbs)
        x = x.reshape((-1, x.shape[-1])).dot(self.final).log_softmax()
        return x.reshape((bs, -1, x.shape[-1]))