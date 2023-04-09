'''
@File    :   train.py
@Time    :   2023/04/06 21:30:12
@Author  :   jiujiuche 
@Version :   1.0
@Contact :   jiujiuche@gmail.com
@License :   (C)Copyright 2023-2024, jiujiuche
@Desc    :   main entrance to load the data and train the network
'''

from datetime import datetime
from timeit import default_timer as timer 

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from torchinfo import summary
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms

from data import loader
from tb_utils import base_operation, io_utils, nn_utils

# nn settings
device = nn_utils.get_device()
batch_size = 32
n_workders = 4
input_size = 224
learn_rate = 1e-4
epochs = 100
visualize = False
torch.manual_seed(22) 
torch.cuda.manual_seed(22)

# file paths
train_dir = './data/birds510/train'
valid_dir = './data/birds510/valid'
test_dir  = './data/birds510/test'

opt_nn_train = base_operation.BaseOperation('./data/birds510/', 'MobileNetV3_train')
opt_nn_train_handle = opt_nn_train.get_handle(force_run=False, verbose=False)

@opt_nn_train_handle
def train(model, train_loader, valid_loader, optimizer, loss_fn, epochs, device):
    train_loss_list, train_acc_list = np.zeros(epochs), np.zeros(epochs)
    valid_loss_list, valid_acc_list = np.zeros(epochs), np.zeros(epochs)

    for epoch in tqdm(range(epochs)):
        start_time = timer()
        train_loss, train_acc = nn_utils.train_step(
            model=model, dataloader=train_loader, loss_fn=loss_fn, optimizer=optimizer, \
            device=device)
        valid_loss, valid_acc = nn_utils.train_step(
            model=model, dataloader=valid_loader, loss_fn=loss_fn, optimizer=optimizer, \
            device=device)
        end_time = timer()

        print(
            f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
            f"valid_loss: {valid_loss:.4f} | valid_acc: {valid_acc:.4f} | "
            f"time: {end_time-start_time:.3f}s"
        )

        train_loss_list[epoch], train_acc_list[epoch] = train_loss, train_acc
        valid_loss_list[epoch], valid_acc_list[epoch] = valid_loss, valid_acc
    
    return train_loss_list, train_acc_list, valid_loss_list, valid_loss_list

if __name__ == '__main__':
    # get dataset stats
    train_list = io_utils.dfs_files(train_dir, '.jpg')
    valid_list = io_utils.dfs_files(valid_dir, '.jpg')
    opt_ds_stats = base_operation.BaseOperation('./data/birds510/', 'train_valid_stats')
    opt_ds_stats_handle = opt_ds_stats.get_handle(force_run=False, verbose=False)
    mean, std = opt_ds_stats.run_opt(loader.running_stats, force_run=False, verbose=False, 
                                    images=train_list+valid_list)

    # transforms for image
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize(int(input_size/0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # define the dataset
    train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(root=valid_dir, transform=test_transform)
    test_data  = datasets.ImageFolder(root=test_dir,  transform=test_transform)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, \
                                num_workers=n_workders)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, \
                                num_workers=n_workders)
    test_dataloader =  DataLoader(test_data, batch_size=batch_size, shuffle=False, \
                                num_workers=n_workders)

    # train network
    network = models.mobilenet_v3_small(pretrained=True).to(device)
    if visualize:
        summary(network, input_size=[1, 3, input_size, input_size])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=network.parameters(), lr=learn_rate)
    train_loss, train_acc, valid_loss, valid_acc = train(
        network, train_dataloader, valid_dataloader, optimizer, loss_fn, epochs, device)

    torch.save(network.state_dict(), 
               f'./models/birds510/mobilenetv3_{datetime.now().strftime("%m-%d-%Y")}.pth')

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, valid_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='train_accuracy')
    plt.plot(epochs, valid_acc, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
