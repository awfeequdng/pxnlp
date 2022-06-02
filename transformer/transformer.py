from json import decoder, encoder
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import matplotlib.pyplot as plt
import numpy as np

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, input):
        return self.lut(input) * math.sqrt(self.d_model)


d_model = 512
vocab = 1000

# 注意，这行代码存在错误，原因是vocab的值为1000，也就是说词所代表的index值不能超过1000（必须小于1000）
# x1 = torch.LongTensor([[100, 2, 421, 52], [12, 343, 2, 1001]])
# x2 = torch.LongTensor([[100, 2, 421, 52], [12, 343, 2, 1000]])
x3 = torch.LongTensor([[100, 2, 421, 52], [12, 343, 2, 999]])

emb = Embeddings(d_model, vocab)
# embr1 = emb(x1)
# embr2 = emb(x2)
embr3 = emb(x3)
print(f'embr: {embr3}')
print(embr3.shape)

print(math.log(10000.0))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len)
        position = position.unsqueeze(1)

        pe = torch.zeros(max_len, d_model)

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

pe = PositionalEncoding(d_model, 0, vocab)


x = embr3
pe_result = pe(x)
print(pe_result)
print(pe_result.shape)


def draw_figure():
    plt.figure(figsize=(15, 5))
    vec_sz = 20
    pe = PositionalEncoding(vec_sz, 0)
    y = pe(torch.zeros(1, 100, vec_sz))
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d"%p for p in [4, 5, 6, 7]])
    plt.show()

# draw_figure()


def subsequent_mask(size):
    mask = 1 - np.triu(np.ones([1, size, size]), k=1).astype("uint8")
    return torch.from_numpy(mask)

sm = subsequent_mask(20)
print(sm)
# plt.figure(figsize=(5,5))
# plt.imshow(sm[0])
# plt.show()

x = torch.randn(5, 5)
print(x)
mask = torch.zeros(5, 5)
print(mask)
y = x.masked_fill(mask==0, 1e-9)
print(y)

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask==0, 1e-9)

    # 对scores的最后一个维度进行softmax
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    # 完成p_atten与value的乘法, 并返回query注意力表示
    return torch.matmul(p_attn, value), p_attn

query = key = value = pe_result
# mask = torch.zeros(2, 4, 4)
mask=None
attn, p_attn = attention(query, key, value, mask=mask)
print('attn:', attn)
print(attn.shape)
print('p_attn:', p_attn)
print(p_attn.shape)


import copy

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim % head == 0

        self.d_k = embedding_dim // head

        self.head = head

        self.attn = None
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 代表多头中的n个头
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
            for model, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        return self.linears[-1](x)

head = 8
embedding_dim = 512
dropout = 0.2

query = key = value = pe_result
# todo: mask该如何设置
mask = torch.zeros(2, 4, 4)

mha = MultiHeadedAttention(head, embedding_dim, dropout)
mha_result = mha(query, key, value, mask)

print(mha_result)
print(mha_result.shape)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout) -> None:
        super(PositionwiseFeedForward, self).__init__()

        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


d_model = 512
d_ff = 64
dropout = 0.2

x = mha_result
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
ff_result = ff(x)
print(ff_result)
print(ff_result.shape)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6) -> None:
        super(LayerNorm, self).__init__()

        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        # 首先对x最后一个维度上求均值操作，同时保持输出维度和输入维度一致
        std = x.std(-1, keepdim = True)

        return self.a2 * (x - mean) / (std + self.eps) + self.b2


features = d_model = 512

ln = LayerNorm(features)
x = ff_result

ln_result = ln(x)
print(ln_result)
print(ln_result.shape)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.2) -> None:
        super(SublayerConnection, self).__init__()

        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


size = 512
dropout = 0.2
head = 8
d_model = 512

x = pe_result
mask = torch.zeros(2, 4, 4)


multi_head = MultiHeadedAttention(head, d_model, dropout)
sublayer = lambda x : multi_head(x, x, x, mask)
sublayer1 = ff


sc = SublayerConnection(size)
sc_result = sc(x, sublayer)
print(sc_result)
print(sc_result.shape)

sc_result1 = sc(x, sublayer1)
print(sc_result1)
print(sc_result1.shape)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayers[1](x, self.feed_forward)


size = d_model = 512
head = 8
d_ff = 64
dropout = 0.2
x = pe_result

self_attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
mask = torch.zeros(2, 4, 4)

el = EncoderLayer(size, self_attn, ff, dropout)
el_result = el(x, mask)
print(el_result)
print(el_result.shape)


class Encoder(nn.Module):
    def __init__(self, layer, N) -> None:

        super(Encoder, self).__init__()

        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# 注意，这些对象需要重新创建，上面已经使用了这些对象，不能在用在下面了，否则报错
self_attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
mask = torch.zeros(2, 4, 4)
el = EncoderLayer(size, self_attn, ff, dropout)
N = 8
en = Encoder(el, 8)
en_result = en(x, mask)
print(en_result)
print(en_result.shape)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm = LayerNorm(size)
        self.sublayers = clones(SublayerConnection(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, source_mask, target_mask):
        m = memory
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, m, m, source_mask))
        return self.sublayers[2](x, self.feed_forward)

self_attn = MultiHeadedAttention(head, d_model)
src_attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
src_mask = torch.zeros(2, 4, 4)
target_mask = torch.zeros(2, 4, 4)
dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
x = pe_result
memory = en_result
dl_result = dl(x, memory, src_mask, target_mask)
print(dl_result)
print(dl_result.shape)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)

self_attn = MultiHeadedAttention(head, d_model)
src_attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
src_mask = torch.zeros(2, 4, 4)
target_mask = torch.zeros(2, 4, 4)
dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
x = pe_result
memory = en_result
N = 8
de = Decoder(dl, N)
de_result = de(x, memory, src_mask, target_mask)
print(de_result)
print(de_result.shape)


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)


    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)


d_model = 512
vocab_size = 1000
x = de_result

gen = Generator(d_model, vocab_size)
gen_result = gen(x)
print(gen_result)
print(gen_result.shape)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator) -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        en_result = self.encode(source, source_mask)
        de_result = self.decode(en_result, source_mask, target, target_mask)
        return self.generator(de_result)




    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


vocab_size = 1000
d_model = 512
encoder = en
decoder = de
source_embed = nn.Embedding(vocab_size, d_model)
target_embed = nn.Embedding(vocab_size, d_model)
generator = gen

source = target = torch.LongTensor([[100, 2, 234, 52], [431, 345, 234, 22]])
source_mask = targt = torch.zeros(2, 4, 4)
ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
ed_result = ed(source, target, source_mask, target_mask)
print(ed_result)
print(ed_result.shape)


def make_model(source_vocab, target_vocab, N=6,
               d_model=512, d_ff=2048, head=8, dropout=0.1):
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionalEncoding(d_model, dropout)
    position = PositionalEncoding(d_model, dropout)
    c = copy.deepcopy
    model = EncoderDecoder(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                           Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
                           nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
                           nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
                           Generator(d_model, target_vocab))

    # 模型结构完成后，接下来就是初始化模型中的参数，比如线性层中的变换矩阵
    # 这里一旦判断参数的维度大于1， 则会将其初始化成一个服从均匀分布的矩阵
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model

w = torch.empty(3, 5)
w = nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
print(w)


if __name__ == '__main__':
    source_vocab = 11
    target_vocab = 11
    N = 6
    model = make_model(source_vocab, target_vocab, N)
    print(model)