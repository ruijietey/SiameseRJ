from networks import Transformer, Mash, SiameseNet
from hierarchical_model import HierarchicalEncoding
from torch import nn


def build(args, vocab, device):
    layer = Transformer(args, shared=False)
    w_layer = Transformer(args, shared=True)
    paragraph_model = HierarchicalEncoding(args.embedding_size, layer, w_layer, level="paragraph")
    sentence_model = HierarchicalEncoding(args.embedding_size, layer, w_layer, level="sentence")

    smash = Mash(paragraph_model, sentence_model, args.embedding_size, vocab)
    model = SiameseNet(smash, args.embedding_size, args.dropout)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    # model.train()

    return model.to(device)