import torch
import torch.nn as nn
from utils import clones


class HierarchicalEncoding(nn.Module):
    def __init__(self, embdim, layer, w_layer, level):
        super(HierarchicalEncoding, self).__init__()
        self.embdim = embdim
        self.level = level
        if self.level == "paragraph":
            clone_count = 2
        elif self.level == "sentence":
            clone_count = 1
        self.curr_level = clones(layer, clone_count)
        self.w = clones(w_layer, 1)

    def forward(self, x):
        [num_batch, num_paragraph, num_sent, num_word, embdim] = x.size()
        x = torch.transpose(x, 0, 3)
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, [num_word, num_sent * num_paragraph * num_batch, embdim])
        if self.level == "paragraph":
            x = self.w[0](x).reshape(num_sent, num_paragraph * num_batch, embdim)
            x = self.curr_level[0](x).reshape(num_paragraph, num_batch, self.embdim)
            output = self.curr_level[1](x)
        elif self.level == "sentence":
            x = self.w[0](x).reshape(num_sent * num_paragraph, num_batch, embdim)
            output = self.curr_level[0](x)
        else:
            # TODO: Try other embedding level
            raise NotImplementedError('Specified Level Not Implemented')
        return output
