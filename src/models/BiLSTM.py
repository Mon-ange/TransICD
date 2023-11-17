import math

import torch
import torch.nn as nn
from layers.aggregate_layer import AggregateLayer
from data.icd import icd_utility
from layers.graph_attention_layer import GraphAttentionLayer

class BiLSTMConfig:
    def __init__(self, in_dim, hidden_dim, n_layer, n_class, embed_weights, freeze_weight,embed_size, dropout_rate,device,
                 pad_idx=0, icd_description_max_length = 8, relu_alpha = 0.8):

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.n_class = n_class
        self.embed_weights = embed_weights
        self.freeze_weight = freeze_weight
        self.embed_size = embed_size
        self.dropout_rate = dropout_rate
        self.bidirectional = True
        self.pad_idx = pad_idx
        self.icd_description_max_length = icd_description_max_length
        # alpha parameter to be used in Graph Attention Layer
        self.relu_alpha = relu_alpha
        self.device=device

class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.config = config
        self.n_layer = config.n_layer
        self.hidden_dim = config.hidden_dim
        self.bidirectional = config.bidirectional
        self.embedding_layer = nn.Embedding.from_pretrained(config.embed_weights, freeze=config.freeze_weight)
        self.drop_out = nn.Dropout(config.dropout_rate)
        self.lstm = nn.LSTM(config.in_dim, config.hidden_dim, config.n_layer, batch_first=True,
                            bidirectional=config.bidirectional)
        if self.bidirectional:
            self.classifier = nn.Linear(self.hidden_dim * 2, config.n_class)
        else:
            self.classifier = nn.Linear(self.hidden_dim, config.n_class)

    def forward(self, x, targets=None):
        #attn_mask = (x != self.config.pad_idx).unsqueeze(2).to(self.config.device)
        x = self.embedding_layer(x) * math.sqrt(self.config.embed_size)
        x = self.drop_out(x)
        out, (hn, _) = self.lstm(x)
        if self.bidirectional:
            out = torch.hstack((hn[-2, :, :], hn[-1, :, :]))
        else:
            out = out[:, -1, :]
        out = self.classifier(out)
        return out, None, None