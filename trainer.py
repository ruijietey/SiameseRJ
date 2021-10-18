from datetime import datetime


def train(model, result, epochs, train_data, val_data, test_data, optimizer, criterion, scheduler, device, n_class, name, eval_count=1):
    min_loss = 1
    tolerence = 0

    start_time = datetime.now()
    for epoch in epochs:
        model.train()
        