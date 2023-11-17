import math

import torch
import torch.nn as nn
from layers.aggregate_layer import AggregateLayer
from data.icd import icd_utility
from layers.graph_attention_layer import GraphAttentionLayer

class BiLSTMGraphConfig:
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

class BiLSTMGraph(nn.Module):
    def __init__(self, config):
        super(BiLSTMGraph, self).__init__()
        self.config = config
        self.n_layer = config.n_layer
        self.hidden_dim = config.hidden_dim
        self.bidirectional = config.bidirectional
        self.embedding_layer = nn.Embedding.from_pretrained(config.embed_weights, freeze=config.freeze_weight)
        self.drop_out = nn.Dropout(config.dropout_rate)
        self.lstm = nn.LSTM(config.in_dim, config.hidden_dim, config.n_layer, batch_first=True,
                            bidirectional=config.bidirectional)
        # Graph Attention Layer
        self.adjacent_graph = icd_utility.getGraph()
        label_descs = icd_utility.getIndexedICDDescriptions(self.config.icd_description_max_length)
        label_descs = torch.tensor(label_descs, dtype=torch.long)
        self.graph_attention = GraphAttentionLayer(config.embed_size, config.embed_size, config.dropout_rate,
                                                   config.relu_alpha)
        self.embedder = nn.Embedding.from_pretrained(self.config.embed_weights, freeze=True)
        self.aggregate_layer = AggregateLayer(config.embed_size * 2, config.embed_size, config.dropout_rate)
        if self.bidirectional:
            self.classifier = nn.Linear(self.hidden_dim * 2, config.n_class)
        else:
            self.classifier = nn.Linear(self.hidden_dim, config.n_class)
        self.fcs = nn.ModuleList([nn.Linear(config.embed_size * 2, 1) for code in range(config.n_class)])
        self.register_buffer('label_desc_mask', (label_descs != self.config.pad_idx) * 1.0)
        self.register_buffer('label_descs', label_descs)

    def embed_label_desc(self):
        label_embeds = self.embedder(self.label_descs).transpose(1, 2).matmul(self.label_desc_mask.unsqueeze(2))
        label_embeds = torch.div(label_embeds.squeeze(2), torch.sum(self.label_desc_mask, dim=-1).unsqueeze(1))
        return label_embeds

    def forward(self, x, targets=None):
        attn_mask = (x != self.config.pad_idx).unsqueeze(2).to(self.config.device)
        x = self.embedding_layer(x) * math.sqrt(self.config.embed_size)
        x = self.drop_out(x)
        out, (hn, _) = self.lstm(x)
        label_embeds = self.embed_label_desc()
        icd_gat = self.graph_attention(label_embeds, self.adjacent_graph)  # L x description_max_len
        weighted_outputs, attn_weights = self.aggregate_layer(out, icd_gat, attn_mask)
        outputs = torch.zeros((weighted_outputs.size(0), self.config.n_class)).to(self.config.device)
        for code, fc in enumerate(self.fcs):
            outputs[:, code:code + 1] = fc(weighted_outputs[:, code, :])
        #out = self.classifier(weighted_outputs)
        return outputs, None, None