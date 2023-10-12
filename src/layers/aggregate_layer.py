import torch.nn as nn
import torch
import torch.nn.functional as F

class AggregateLayer(nn.Module):

    """
        n_size is the patient size
        l_size is the label size
    """

    def __init__(self, hidden_size, label_embed_size, dropout_rate):
        super(AggregateLayer, self).__init__()
        self.l1 = nn.Linear(hidden_size, label_embed_size)
        self.tnh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden, label_embeds, attn_mask=None):
        # B: batch size, S sequence length, H: hidden size, L label size
        # output_1: B x S x H -> B x S x E
        output_1 = self.tnh(self.l1(hidden))
        output_1 = self.dropout(output_1)

        # output_2: (B x S x E) x (E x L) -> B x S x L
        output_2 = torch.matmul(output_1, label_embeds.t())

        # Masked fill to avoid softmaxing over padded words
        if attn_mask is not None:
            output_2 = output_2.masked_fill(attn_mask == 0, -1e9)

        # attn_weights: B x S x L -> B x L x S
        attn_weights = F.softmax(output_2, dim=1).transpose(1, 2)

        # weighted_output: (B x L x S) @ (B x S x H) -> B x L x H
        weighted_output = attn_weights @ hidden
        return weighted_output, attn_weights

