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
pe_x = pe(x)
print(x)
print(x.shape)


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

