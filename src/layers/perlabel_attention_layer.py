import torch.nn as nn
import torch.nn.functional as F


class PerLabelAttentionLayer(nn.Module):
    def __init__(self, hidden_size, output_size, attn_expansion, dropout_rate):
        super(PerLabelAttentionLayer, self).__init__()
        self.l1 = nn.Linear(hidden_size, hidden_size * attn_expansion)
        self.tnh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)
        self.l2 = nn.Linear(hidden_size * attn_expansion, output_size)

    def forward(self, hidden, attn_mask=None):
        # output_1: B x S x H -> B x S x attn_expansion*H
        output_1 = self.tnh(self.l1(hidden))
        output_1 = self.dropout(output_1)

        # output_2: B x S x attn_expansion*H -> B x S x output_size(O)
        output_2 = self.l2(output_1)

        # Masked fill to avoid softmaxing over padded words
        if attn_mask is not None:
            output_2 = output_2.masked_fill(attn_mask == 0, -1e9)

        # attn_weights: B x S x output_size(O) -> B x O x S
        attn_weights = F.softmax(output_2, dim=1).transpose(1, 2)

        # weighted_output: (B x O x S) @ (B x S x H) -> B x O x H
        weighted_output = attn_weights @ hidden
        return weighted_output, attn_weights
