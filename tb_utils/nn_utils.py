'''
@File    :   nn_utils.py
@Time    :   2023/04/06 21:33:38
@Author  :   jiujiuche 
@Version :   1.0
@Contact :   jiujiuche@gmail.com
@License :   (C)Copyright 2023-2024, jiujiuche
@Desc    :   util functions for torch/nn related operations
'''

import torch

def get_device():
    """get the device

    Returns:
        torch.device: torch device to run networks
    """
    mps_available = hasattr(torch.backends, "mps") and \
        torch.backends.mps.is_available()
    if torch.cuda.is_available():
        device = "cuda"
    elif mps_available:
        device = torch.device("mps")
    else:
        device = "cpu"
    return device


def train_step(model, dataloader, loss_fn, optimizer, device):
    """train one epoch, 
    this code is from https://www.learnpytorch.io/04_pytorch_custom_datasets/

    Args:
        model (torch.nn.Module): network model
        dataloader (torch.utils.data.DataLoader): dataloader
        loss_fn (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        device (torch.device): torch device to run networks

    Returns:
        float: train loss and accuracy
    """
    model.train()
    train_loss, train_acc = 0, 0
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model, dataloader, loss_fn, device):
    """test one epoch, 
    this code is from https://www.learnpytorch.io/04_pytorch_custom_datasets/

    Args:
        model (torch.nn.Module): network model
        dataloader (torch.utils.data.DataLoader): dataloader
        loss_fn (torch.nn.Module): loss function
        device (torch.device): torch device to run networks

    Returns:
        float: test loss and accuracy
    """
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc
