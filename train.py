from datetime import datetime
import torch
import torch.nn as nn
import argparse
import os
from trainer import train
from models.model_builder import build

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-model_type", default='shwest', type=str, choices=['smash', 'sht', 'shwest', 'ashbert', 'triplet']) #TODO: Work on these models
    parser.add_argument("-result_path", default='./results/ann/')

    # Hyperparameters (Added following Seoyoon's Paper)
    parser.add_argument("-lr", default=0.0001, type=float) #Learning Rate
    parser.add_argument("-batch_size", default=1, type=int) # 140 originally in other paper, 1 in SY's paper
    parser.add_argument("-test_batch_size", default=200, type=int)
    parser.add_argument("-num_class", default=2, type=int)
    parser.add_argument("-num_token", default=30000, type=int)
    parser.add_argument("-embedding_size", default=256, type=int)
    parser.add_argument("-num_h_layer", default=128, type=int)
    parser.add_argument("-num_layer", default=3, type=int)
    parser.add_argument("-num_shared_layer", default=2, type=int)
    parser.add_argument("-num_head", default=2, type=int)
    parser.add_argument("-num_epochs", default=70, type=int)
    parser.add_argument("-dropout", default=0.1, type=float)
    parser.add_argument("-tolerance", default=6, type=int)
    parser.add_argument("-load_model", type=str2bool, nargs='?', const=True, default=False)

    # Encoder
    # parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    # parser.add_argument("-enc_dropout", default=0.2, type=float)
    # parser.add_argument("-enc_layers", default=6, type=int)

    # GPU / Environment setup
    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_file', default='./logs/ann.log')
    parser.add_argument('-seed', default=666, type=int)

    args = parser.parse_args()
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    device = "cpu" if args.visible_gpus == '-1' else "cuda"

    if args.load_model:
        model = torch.load(datetime.utcnow().strftime("%d-%m-%Y_%H-%M-%S") + '.pt')
        # TODO: Add Result
        # result = torch.load(datetime.utcnow().strftime("%d-%m-%Y_%H-%M-%S") + '._result.pt')
    else:
        # TODO: Add other model types
        if args.model_type == 'shwest':
            model = build(args, device)
        # TODO: Add Result
        # result = Result

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.90)

    if args.mode == 'train':
        train(model, args, optimizer, criterion, scheduler, device)

    # TODO: Add validation and test
    # else if args.mode == 'validate':
    #     validate(model, val_data, device, num_class, criterion, min_loss)


