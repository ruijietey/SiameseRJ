import math

import torch
import torch.nn as nn
from models.encoder import TransformerEncoder


# From Seoyoon
class Transformer(nn.Module):
    def __init__(self, args, shared):
        super(Transformer, self).__init__()
        self.args = args
        self.embdim = args.embedding_size
        if shared:
            num_layer = args.num_shared_layer
        else:
            num_layer = args.num_layer
        self.tf = TransformerEncoder(self.embdim, args.enc_ff_size, args.num_head, num_layer, args.dropout)
        self.f = nn.Linear(self.embdim, self.embdim)
        self.attn = nn.Linear(self.embdim, 1)

    def forward(self, x):
        x = self.tf(x)
        output = torch.mean(x, 0)
        return output


class Mash(nn.Module):
    def __init__(self,paragraph_model, sentence_model, embedding_size, vocab):
        super(Smash, self).__init__()
        self.embedding = nn.Embedding(vocab, embedding_size)
        self.init_weights()
        self.sentence = sentence_model
        self.paragraph = paragraph_model
        self.embdim = embedding_size

    def forward(self, x):
        assert len(x.size()) == 4
        # [num_batch, num_paragraph, num_sentence, num_word] = x.size()
        x = self.embedding(x) * math.sqrt(self.embdim)
        paragraph_output = self.paragraph(x)
        sentence_output = self.sentence(x)
        return torch.cat((paragraph_output, sentence_output), dim=-1)

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)


# Modified-- siamese-triplet-master
# https://github.com/adambielski/siamese-triplet
class SiameseNet(nn.Module):
    def __init__(self, smash_model, embedding_size, dropout):
        super(SiameseNet, self).__init__()
        self.smash_model = smash_model
        self.embedding_size = embedding_size
        self.dropout = dropout

    def forward(self, x1, x2):
        output1 = self.smash_model(x1)
        output2 = self.smash_model(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.smash_model(x)

# From siamese-triplet-master
# https://github.com/adambielski/siamese-triplet
# class SiameseNet(nn.Module):
#     def __init__(self, embedding_net):
#         super(SiameseNet, self).__init__()
#         self.embedding_net = embedding_net
#
#     def forward(self, x1, x2):
#         output1 = self.embedding_net(x1)
#         output2 = self.embedding_net(x2)
#         return output1, output2
#
#     def get_embedding(self, x):
#         return self.embedding_net(x)
