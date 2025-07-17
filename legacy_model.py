import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, activation):
        super().__init__()
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.activation = activation
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.activation(self.up_proj(x))
        x = self.dropout1(x)
        x = self.down_proj(x)
        return self.dropout2(x)
    
class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.silu(self.up_proj(x)) * self.gate(x)
        x = self.down_proj(x)
        return self.dropout(x)

class CustomDecoderLayer(nn.Module):
        # CustomTransformerDecoderLayer
        def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float = 0.1, activation = F.relu, batch_first=True):
            super().__init__()
            # 使用GPT2结构
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
#             self.norm1 = nn.LayerNorm(d_model)
#             self.norm2 = nn.LayerNorm(d_model)
#             self.FFN = FFN(d_model, d_ffn, dropout, activation)
            self.FFN = SwiGLUFFN(d_model, d_ffn, dropout)
            self.mutihead_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=batch_first)
            
        
        def forward(self, x, attn_mask, key_padding_mask, is_causal):
            res = x
            x = self.norm1(x)
            x = self.mutihead_self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, is_causal=is_causal)[0]
            x = res + x
            res = x
            x = self.norm2(x)
            x = res + self.FFN(x)
            return x
        


class CustomGPT(nn.Module):
    def __init__(self,vocab_size, seq_max_len, total_word, d_model, d_ffn, n_heads, d_head, n_layers, dropout_rate, activation, device):        
        super().__init__()  # 调用父类的初始化方法
        # self.vocab_size = vocab_size  # 词汇表大小
        # self.d_ffn = d_ffn  # FFN嵌入维度
        # self.n_heads = n_heads  # 头数数
        # self.d_head = d_head  # 头嵌入维度
        # self.activation = activation
        # self.dropout_rate = dropout_rate  # Dropout率
        self.n_layers = n_layers  # 层数
        self.total_word = total_word  # 总词数
        self.d_model = d_model  # 模型嵌入维度
        self.seq_max_len = seq_max_len  # 序列最大长度
        self.device = device
        self.tok_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)  # Token嵌入层
        self.pos_emb = nn.Embedding(num_embeddings=seq_max_len, embedding_dim=d_model)  # 位置嵌入层
        self.decoder_layer = CustomDecoderLayer(d_model=d_model, d_ffn=d_ffn, n_heads=n_heads, dropout=dropout_rate, activation=activation)
        self.out_fc = nn.Linear(self.d_model, self.total_word, bias=False)  # 输出全连接层 
        self.last_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)

    def forward(self, x):
#         pad_mask = (x==0).to(torch.float)
#         pos = torch.arange(self.seq_max_len).to(self.device)
#         emb = self.tok_emb(x) + self.pos_emb(pos)
        emb = self.tok_emb(x)
        x, _ = self.lstm(emb)
#         causal_mask = nn.Transformer.generate_square_subsequent_mask(self.seq_max_len).to(self.device)
        
#         decoder_list = nn.ModuleList([self.decoder_layer for _ in range(self.n_layers)])
#         x = self.dropout(emb)
#         for layer in decoder_list:
#             x = layer(x, attn_mask=causal_mask, key_padding_mask=pad_mask, is_causal=False)
#         # 使用is_causal无需attn_mask
#         x = self.last_norm(x)
        
        out = self.out_fc(x) # CrossEntropyLoss输出时不需要softmax
        return out
    

