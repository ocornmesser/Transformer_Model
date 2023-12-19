import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import pandas as pd

from all_classes import MultiHeadAttention
from all_classes import PositionWiseFeedForward
from all_classes import PositionalEncoding
from all_classes import EncoderLayer
from all_classes import DecoderLayer
from all_classes import Transformer
import data_preprocess as data

src_vocab_size = 135*(10^5)
tgt_vocab_size = 135*(10^5)
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 48
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(data.XTrain, data.yTrain)
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), data.yTrain.contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

#df = pd.DataFrame(torch.randint(1, src_vocab_size, (64, max_seq_length)))
#df.to_csv('testFile.csv', index=False)