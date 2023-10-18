import math
import torch
import torch.nn as nn
from layers.positional_encoding_layer import PositionalEncodingLayer
from layers.perlabel_attention_layer import PerLabelAttentionLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from layers.graph_attention_layer import GraphAttentionLayer
from layers.aggregate_layer import AggregateLayer
from data.icd import icd_utility


class TransGraphICDConfig:

    def __init__(self, embed_weights, embed_size, freeze_weight, text_max_len, num_layers, num_heads, forward_dimension,
                 output_size, label_attention_expansion, dropout_rate, device, icd_description_max_length = 8, relu_alpha = 0.8, pad_idx=0):
        # 提前预训练好的词向量权重数组
        self.embed_weights = embed_weights
        # 词向量embedding后的大小
        self.embed_size = embed_size
        # freeze_weight为True表示训练过程中不更新weight，等同于embedding.weight.requires_grad = False
        self.freeze_weight = freeze_weight
        self.dropout_rate = dropout_rate
        # 输入文本的最大长度
        self.text_max_len = text_max_len
        # the number of heads in the multiheadattention model
        self.num_heads = num_heads
        # the dimension of the feedforward network model in Transformer(default=2048).
        self.forward_dimension = forward_dimension
        # the number of sub-encoder-layers in the encoder in Transformer (required).
        self.num_layers = num_layers
        # output_size 输出的维度，由分类的类别数决定
        self.output_size = output_size
        # Expansion factor for attention model
        self.label_attention_expansion = label_attention_expansion
        self.device = device
        # 空白填充字符的index
        self.pad_idx = pad_idx
        self.icd_description_max_length = icd_description_max_length
        # alpha parameter to be used in Graph Attention Layer
        self.relu_alpha = relu_alpha

class TransGraphICD(nn.Module):

    def __init__(self, config):
        super(TransGraphICD, self).__init__()
        torch.cuda.manual_seed_all(271)
        if config.embed_size % config.num_heads != 0:
            raise ValueError(f"Embedding size {config.embed_size} needs to be divisible by number of heads {config.num_heads}")
        self.config = config
        self.embedding_layer = nn.Embedding.from_pretrained(config.embed_weights, freeze=config.freeze_weight)
        self.dropout = nn.Dropout(config.dropout_rate)
        label_descs = icd_utility.getIndexedICDDescriptions(self.config.icd_description_max_length)
        print(label_descs)
        label_descs = torch.tensor(label_descs, dtype=torch.long)
        self.adjacent_graph = icd_utility.getGraph()
        self.position_encoding_layer = PositionalEncodingLayer(config.embed_size, config.dropout_rate,
                                                               config.text_max_len)
        encoding_layer = TransformerEncoderLayer(d_model=config.embed_size,
                                                 nhead=config.num_heads,
                                                 dim_feedforward=config.forward_dimension*config.embed_size,
                                                 dropout=config.dropout_rate)
        self.encoder = TransformerEncoder(encoding_layer, config.num_layers)
        self.perlabel_attention_layer = PerLabelAttentionLayer(config.embed_size, config.output_size,
                                                               config.label_attention_expansion, config.dropout_rate)
        self.fcs = nn.ModuleList([nn.Linear(config.embed_size, 1) for code in range(config.output_size)])
        self.graph_attention = GraphAttentionLayer(config.embed_size ,config.embed_size, config.dropout_rate, config.relu_alpha)
        self.embedder = nn.Embedding.from_pretrained(self.config.embed_weights, freeze=True)
        self.aggregate_layer = AggregateLayer(config.embed_size, config.embed_size, config.dropout_rate)
        self.register_buffer('label_desc_mask', (label_descs != self.config.pad_idx) * 1.0)
        self.register_buffer('label_descs', label_descs)

    def embed_label_desc(self):
        label_embeds = self.embedder(self.label_descs).transpose(1, 2).matmul(self.label_desc_mask.unsqueeze(2))
        label_embeds = torch.div(label_embeds.squeeze(2), torch.sum(self.label_desc_mask, dim=-1).unsqueeze(1))
        return label_embeds

    def forward(self, inputs, targets=None):
        """
            N is the Number of input
            S is the Size of Each Text(max_len)
            E is the embedding size
        """
        # attn_mask: B x S -> B x S x 1

        attn_mask = (inputs != self.config.pad_idx).unsqueeze(2).to(self.config.device)
        src_key_padding_mask = (inputs == self.config.pad_idx).to(self.config.device)  # N x S
        embeds = self.embedding_layer(inputs) * math.sqrt(self.config.embed_size)  # N x S x E
        embeds = self.position_encoding_layer(embeds)
        embeds = self.dropout(embeds)
        embeds = embeds.permute(1, 0, 2)  # 调整维度顺序， S x N x E

        encoded_inputs = self.encoder(embeds, src_key_padding_mask=src_key_padding_mask)  # T x N x E
        encoded_inputs = encoded_inputs.permute(1, 0, 2)  # N x T x E
        weighted_outputs, attn_weights = self.perlabel_attention_layer(encoded_inputs, attn_mask) # weighted_outputs N x L ?
        # TODO: Graph Attention Layer to ICD Description
        label_embeds = self.embed_label_desc()
        icd_gat = self.graph_attention(label_embeds, self.adjacent_graph)  # L x description_max_len
        weighted_outputs, attn_weights = self.aggregate_layer(encoded_inputs, icd_gat, attn_mask)

        outputs = torch.zeros((weighted_outputs.size(0), self.config.output_size)).to(self.config.device)
        for code, fc in enumerate(self.fcs):
            outputs[:, code:code+1] = fc(weighted_outputs[:, code, :])
        return outputs, None, None
